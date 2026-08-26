"""Matched-source random augmentation used as a recovery-data control."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from robust_tool.data.recovery_cases import (
    RecoveryDatasetConfig,
    generate_recovery_tasks,
)
from robust_tool.data.schemas import Task

RANDOM_AUGMENTATION_VERSION = "calendar-matched-random-augmentation-v2"
MATCHED_SOURCE_CONTROL = "matched_source_semantic_rewrite"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_QUERY_TEMPLATES = (
    "Please handle this calendar request: {query}",
    "Complete the following request using the provided tools when needed: {query}",
    "Work through this calendar request and report the result: {query}",
    "Follow this request exactly and use a tool only when appropriate: {query}",
)


@dataclass(frozen=True)
class RandomAugmentationConfig:
    """Frozen inputs for the equal-size random augmentation control."""

    dataset_name: str
    generator_version: str
    seed: int
    size: int
    control_type: str
    matched_recovery_config_sha256: str

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name cannot be empty")
        if self.generator_version != RANDOM_AUGMENTATION_VERSION:
            raise ValueError(
                f"unsupported generator_version {self.generator_version!r}; "
                f"expected {RANDOM_AUGMENTATION_VERSION!r}"
            )
        if self.size <= 0:
            raise ValueError("size must be a positive integer")
        if self.control_type != MATCHED_SOURCE_CONTROL:
            raise ValueError(f"unsupported control_type: {self.control_type!r}")
        if not _SHA256_PATTERN.fullmatch(self.matched_recovery_config_sha256):
            raise ValueError("matched_recovery_config_sha256 must be a lowercase SHA-256")

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RandomAugmentationConfig":
        return cls(
            dataset_name=str(record["dataset_name"]),
            generator_version=str(record["generator_version"]),
            seed=int(record["seed"]),
            size=int(record["size"]),
            control_type=str(record["control_type"]),
            matched_recovery_config_sha256=str(
                record["matched_recovery_config_sha256"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "size": self.size,
            "control_type": self.control_type,
            "matched_recovery_config_sha256": (
                self.matched_recovery_config_sha256
            ),
        }


def load_random_augmentation_config(path: str | Path) -> RandomAugmentationConfig:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load random augmentation config {source}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"random augmentation config must be an object: {source}")
    try:
        return RandomAugmentationConfig.from_dict(record)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid random augmentation config {source}: {exc}") from exc


def generate_random_augmentation_tasks(
    config: RandomAugmentationConfig,
    recovery_config: RecoveryDatasetConfig,
    source_train_tasks: Iterable[Task],
) -> list[Task]:
    """Rewrite the exact Train sources used by Recovery v2 without adding failures."""

    sources = list(source_train_tasks)
    if config.size != recovery_config.size:
        raise ValueError(
            "random control and Recovery v2 must have equal size: "
            f"{config.size} != {recovery_config.size}"
        )
    source_by_id = {task.task_id: task for task in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("source Train tasks contain duplicate IDs")
    matched_recovery = generate_recovery_tasks(recovery_config, sources)
    tasks: list[Task] = []
    for index, matched_task in enumerate(matched_recovery):
        source_id = str(matched_task.metadata["source_train_task_id"])
        source = source_by_id[source_id]
        matched_target = str(matched_task.metadata["target_robustness"])
        tasks.append(
            _rewrite_source(
                config,
                source,
                matched_target=matched_target,
                index=index,
            )
        )
    return tasks


def audit_random_augmentation_tasks(
    tasks: Iterable[Task],
    config: RandomAugmentationConfig,
    recovery_config: RecoveryDatasetConfig,
    source_splits: Mapping[str, Iterable[Task]],
) -> dict[str, Any]:
    """Reject source mismatch, failure injection, mutation drift, and split leakage."""

    materialized = list(tasks)
    if len(materialized) != config.size:
        raise ValueError(
            f"generated {len(materialized)} random controls; expected {config.size}"
        )
    ids = [task.task_id for task in materialized]
    queries = [_normalized_query(task.user_query) for task in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("random controls contain duplicate task IDs")
    if len(queries) != len(set(queries)):
        raise ValueError("random controls contain duplicate normalized user queries")

    source_by_id: dict[str, tuple[str, Task]] = {}
    source_queries: set[str] = set()
    source_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        source_tasks = list(source_splits.get(split, ()))
        source_counts[split] = len(source_tasks)
        for task in source_tasks:
            if task.task_id in source_by_id:
                raise ValueError(f"source task ID crosses splits: {task.task_id}")
            source_by_id[task.task_id] = (split, task)
            source_queries.add(_normalized_query(task.user_query))
    overlap_ids = set(ids) & set(source_by_id)
    overlap_queries = set(queries) & source_queries
    if overlap_ids:
        raise ValueError(f"random control IDs overlap sources: {sorted(overlap_ids)[:5]}")
    if overlap_queries:
        raise ValueError(
            f"random control queries overlap sources: {sorted(overlap_queries)[:5]}"
        )

    train_sources = [
        task for split, task in source_by_id.values() if split == "train"
    ]
    matched_recovery = generate_recovery_tasks(recovery_config, train_sources)
    expected_source_ids = {
        str(task.metadata["source_train_task_id"]) for task in matched_recovery
    }
    actual_source_ids: list[str] = []
    matched_targets: Counter[str] = Counter()
    source_categories: Counter[str] = Counter()
    for task in materialized:
        metadata = task.metadata
        source_id = str(metadata.get("source_train_task_id", ""))
        source_entry = source_by_id.get(source_id)
        if source_entry is None or source_entry[0] != "train":
            raise ValueError(f"random control has a non-Train source: {task.task_id}")
        source = source_entry[1]
        if metadata.get("source_train_task_sha256") != _task_sha256(source):
            raise ValueError(f"random control source hash drift: {task.task_id}")
        if metadata.get("control_type") != MATCHED_SOURCE_CONTROL:
            raise ValueError(f"random control protocol drift: {task.task_id}")
        if "robustness" in metadata or "target_robustness" in metadata:
            raise ValueError(f"random control contains failure injection: {task.task_id}")
        _validate_semantic_copy(task, source)
        actual_source_ids.append(source_id)
        matched_targets[str(metadata.get("matched_recovery_target", ""))] += 1
        source_categories[str(source.metadata.get("category", "unknown"))] += 1
    if len(actual_source_ids) != len(set(actual_source_ids)):
        raise ValueError("a Train source was reused across random controls")
    if set(actual_source_ids) != expected_source_ids:
        raise ValueError("random control does not use the exact Recovery v2 source set")
    if dict(matched_targets) != dict(recovery_config.target_counts):
        raise ValueError(
            f"matched Recovery target counts differ: {dict(matched_targets)}"
        )
    return {
        "count": len(materialized),
        "control_type": config.control_type,
        "unique_task_ids": len(set(ids)),
        "unique_user_queries": len(set(queries)),
        "unique_source_train_tasks": len(set(actual_source_ids)),
        "matched_recovery_target_counts": dict(matched_targets),
        "source_category_counts": dict(sorted(source_categories.items())),
        "source_split_counts": source_counts,
        "source_overlap": {"task_ids": 0, "normalized_user_queries": 0},
        "failure_injection_count": 0,
        "validation_or_test_content_used_for_generation": False,
    }


def _rewrite_source(
    config: RandomAugmentationConfig,
    source: Task,
    *,
    matched_target: str,
    index: int,
) -> Task:
    variant = _stable_variant(source.task_id, config.seed)
    query = _QUERY_TEMPLATES[variant].format(query=source.user_query)
    metadata = copy.deepcopy(dict(source.metadata))
    metadata.update(
        {
            "split": "train",
            "category": "random_augmentation_control",
            "dataset_name": config.dataset_name,
            "generator_version": config.generator_version,
            "seed": config.seed,
            "control_type": config.control_type,
            "matched_recovery_target": matched_target,
            "source_train_task_id": source.task_id,
            "source_train_task_sha256": _task_sha256(source),
            "source_category": source.metadata.get("category"),
            "source_difficulty": source.difficulty,
        }
    )
    return Task(
        task_id=f"calendar_random_control_v2_{index:05d}",
        domain=source.domain,
        user_query=query,
        available_tools=tuple(source.available_tools),
        initial_state=copy.deepcopy(dict(source.initial_state)),
        goal_state=copy.deepcopy(dict(source.goal_state)),
        difficulty=f"random_control_{source.difficulty}",
        failure_tags=tuple(source.failure_tags),
        expected_action=source.expected_action,
        reference_calls=tuple(source.reference_calls),
        metadata=metadata,
    )


def _validate_semantic_copy(task: Task, source: Task) -> None:
    unchanged = (
        task.domain == source.domain
        and task.available_tools == source.available_tools
        and dict(task.initial_state) == dict(source.initial_state)
        and dict(task.goal_state) == dict(source.goal_state)
        and task.failure_tags == source.failure_tags
        and task.expected_action == source.expected_action
        and task.reference_calls == source.reference_calls
        and source.user_query in task.user_query
    )
    if not unchanged:
        raise ValueError(f"random control changed task semantics: {task.task_id}")


def _stable_variant(task_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % len(_QUERY_TEMPLATES)


def _normalized_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _task_sha256(task: Task) -> str:
    payload = json.dumps(
        task.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
