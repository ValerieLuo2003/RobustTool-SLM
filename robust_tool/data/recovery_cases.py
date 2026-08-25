"""Deterministic Train-only cases for tool absence and execution recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from robust_tool.data.schemas import Task

RECOVERY_GENERATOR_VERSION = "calendar-recovery-failure-aware-v2"
RECOVERY_TARGETS = ("missing_tool", "tool_failure", "partial_tool_response")
_SELECTION_ORDER = ("partial_tool_response", "missing_tool", "tool_failure")
_READ_ONLY_TOOLS = {"list_events", "check_availability"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TARGET_OFFSETS = {
    "missing_tool": 0,
    "tool_failure": 1_000_000,
    "partial_tool_response": 2_000_000,
}
_MISSING_TOOL_SUFFIXES = (
    "Use only the tools shown; if the required capability is absent, name it "
    "and explain that the request cannot be completed.",
    "Do not invent a replacement tool. If the needed operation is unavailable, "
    "state that explicitly.",
    "Before acting, verify the required tool is available; otherwise explain "
    "which tool is missing.",
    "If none of the provided tools can perform this operation, say so rather than guessing.",
)
_TOOL_FAILURE_SUFFIXES = (
    "If a tool reports a temporary retriable timeout, retry the same operation "
    "once before answering.",
    "A transient tool failure may be retried once; never claim success from the failed response.",
    "Read the tool error carefully and make one identical retry when it explicitly "
    "permits retrying.",
    "If the first execution times out and is marked retriable, repeat that call "
    "once and use only the successful result.",
)
_PARTIAL_RESPONSE_SUFFIXES = (
    "Verify that the result contains the fields needed to answer; if it is "
    "incomplete, query once more.",
    "Do not infer missing result fields. Repeat the read-only call once when its "
    "first response is partial.",
    "Check the returned structure before answering and re-query once if the key "
    "answer field is absent.",
    "Treat a response missing its result payload as incomplete, then repeat the "
    "same read-only request once.",
)


@dataclass(frozen=True)
class RecoveryDatasetConfig:
    """Frozen inputs for the second failure-aware Train augmentation."""

    dataset_name: str
    generator_version: str
    seed: int
    source_robust_validation_sha256: str
    target_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name cannot be empty")
        if self.generator_version != RECOVERY_GENERATOR_VERSION:
            raise ValueError(
                f"unsupported generator_version {self.generator_version!r}; "
                f"expected {RECOVERY_GENERATOR_VERSION!r}"
            )
        if not _SHA256_PATTERN.fullmatch(self.source_robust_validation_sha256):
            raise ValueError("source_robust_validation_sha256 must be a lowercase SHA-256")
        if set(self.target_counts) != set(RECOVERY_TARGETS):
            raise ValueError(f"target_counts must contain exactly {RECOVERY_TARGETS}")
        if any(
            not isinstance(count, int) or count <= 0
            for count in self.target_counts.values()
        ):
            raise ValueError("all target counts must be positive integers")

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RecoveryDatasetConfig":
        counts = record.get("target_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("target_counts must be an object")
        return cls(
            dataset_name=str(record["dataset_name"]),
            generator_version=str(record["generator_version"]),
            seed=int(record["seed"]),
            source_robust_validation_sha256=str(
                record["source_robust_validation_sha256"]
            ),
            target_counts={str(key): int(value) for key, value in counts.items()},
        )

    @property
    def size(self) -> int:
        return sum(self.target_counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "source_robust_validation_sha256": self.source_robust_validation_sha256,
            "target_counts": dict(self.target_counts),
        }


def load_recovery_config(path: str | Path) -> RecoveryDatasetConfig:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load recovery config {source}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"recovery config must be a JSON object: {source}")
    try:
        return RecoveryDatasetConfig.from_dict(record)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid recovery config {source}: {exc}") from exc


def generate_recovery_tasks(
    config: RecoveryDatasetConfig,
    source_train_tasks: Iterable[Task],
) -> list[Task]:
    """Transform distinct frozen Train sources into three recovery task families."""

    sources = list(source_train_tasks)
    if not sources:
        raise ValueError("source_train_tasks cannot be empty")
    if len({task.task_id for task in sources}) != len(sources):
        raise ValueError("source Train tasks contain duplicate IDs")
    wrong_split = [
        task.task_id for task in sources if task.metadata.get("split") != "train"
    ]
    if wrong_split:
        raise ValueError(f"source tasks are not Train: {wrong_split[:5]}")

    used_source_ids: set[str] = set()
    generated: list[Task] = []
    for target_index, target in enumerate(_SELECTION_ORDER):
        candidates = [
            task
            for task in _eligible_sources(target, sources)
            if task.task_id not in used_source_ids
        ]
        selected = _stratified_selection(
            candidates,
            count=config.target_counts[target],
            seed=config.seed + target_index * 1009,
        )
        used_source_ids.update(task.task_id for task in selected)
        generated.extend(
            _transform_task(config, task, target, index)
            for index, task in enumerate(selected)
        )
    random.Random(config.seed + 99_991).shuffle(generated)
    return generated


def audit_recovery_tasks(
    tasks: Iterable[Task],
    config: RecoveryDatasetConfig,
    source_splits: Mapping[str, Iterable[Task]],
) -> dict[str, Any]:
    """Fail closed on target drift, source reuse, or Validation/Test leakage."""

    materialized = list(tasks)
    if len(materialized) != config.size:
        raise ValueError(f"generated {len(materialized)} recovery cases; expected {config.size}")
    ids = [task.task_id for task in materialized]
    queries = [_normalized_query(task) for task in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("recovery cases contain duplicate task IDs")
    if len(queries) != len(set(queries)):
        raise ValueError("recovery cases contain duplicate normalized user queries")

    source_by_id: dict[str, tuple[str, Task]] = {}
    source_queries: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        source_tasks = list(source_splits.get(split, ()))
        source_counts[split] = len(source_tasks)
        for task in source_tasks:
            if task.task_id in source_by_id:
                raise ValueError(f"source task ID crosses splits: {task.task_id}")
            source_by_id[task.task_id] = (split, task)
            source_queries.setdefault(_normalized_query(task), split)
    overlap_ids = set(ids) & set(source_by_id)
    overlap_queries = set(queries) & set(source_queries)
    if overlap_ids:
        raise ValueError(f"recovery IDs overlap source IDs: {sorted(overlap_ids)[:5]}")
    if overlap_queries:
        raise ValueError(
            f"recovery queries overlap source queries: {sorted(overlap_queries)[:5]}"
        )

    counts: Counter[str] = Counter()
    source_ids: list[str] = []
    source_categories: dict[str, Counter[str]] = defaultdict(Counter)
    for task in materialized:
        metadata = task.metadata
        target = str(metadata.get("target_robustness"))
        counts[target] += 1
        if metadata.get("split") != "train":
            raise ValueError(f"recovery case is not marked Train: {task.task_id}")
        if metadata.get("source_robust_validation_sha256") != (
            config.source_robust_validation_sha256
        ):
            raise ValueError(f"recovery source snapshot drift: {task.task_id}")
        source_id = str(metadata.get("source_train_task_id", ""))
        source = source_by_id.get(source_id)
        if source is None or source[0] != "train":
            raise ValueError(f"recovery case does not reference a Train source: {task.task_id}")
        if metadata.get("source_train_task_sha256") != _task_sha256(source[1]):
            raise ValueError(f"recovery source task hash drift: {task.task_id}")
        source_ids.append(source_id)
        source_categories[target][str(source[1].metadata.get("category"))] += 1
        _validate_target_structure(task, target)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("a source Train task was reused across recovery cases")
    if dict(counts) != dict(config.target_counts):
        raise ValueError(f"recovery target counts differ: {dict(counts)}")
    return {
        "count": len(materialized),
        "target_counts": dict(counts),
        "unique_task_ids": len(set(ids)),
        "unique_user_queries": len(set(queries)),
        "unique_source_train_tasks": len(set(source_ids)),
        "source_category_counts": {
            target: dict(sorted(values.items()))
            for target, values in sorted(source_categories.items())
        },
        "source_split_counts": source_counts,
        "source_overlap": {"task_ids": 0, "normalized_user_queries": 0},
        "validation_or_test_content_used_for_generation": False,
    }


def _eligible_sources(target: str, sources: list[Task]) -> list[Task]:
    single_call = [
        task
        for task in sources
        if task.expected_action == "call" and len(task.reference_calls) == 1
    ]
    if target == "partial_tool_response":
        return [task for task in single_call if task.reference_calls[0].name in _READ_ONLY_TOOLS]
    return single_call


def _stratified_selection(
    candidates: list[Task],
    *,
    count: int,
    seed: int,
) -> list[Task]:
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible Train sources for {count} cases")
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in sorted(candidates, key=lambda item: item.task_id):
        groups[str(task.metadata.get("category", "unknown"))].append(task)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[Task] = []
    while len(selected) < count:
        progressed = False
        for category in sorted(groups):
            if groups[category]:
                selected.append(groups[category].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"selected {len(selected)} of {count} requested Train sources")
    return selected


def _transform_task(
    config: RecoveryDatasetConfig,
    source: Task,
    target: str,
    index: int,
) -> Task:
    target_call = source.reference_calls[0]
    available_tools = list(source.available_tools)
    expected_action = source.expected_action
    reference_calls = list(source.reference_calls)
    goal_state = copy.deepcopy(dict(source.goal_state))
    failure_tags = list(source.failure_tags)
    metadata = copy.deepcopy(dict(source.metadata))
    robustness: dict[str, Any] = {
        "kind": target,
        "source_task_id": source.task_id,
        "source_task_sha256": _task_sha256(source),
        "source_expected_action": source.expected_action,
        "training_case": True,
    }

    if target == "missing_tool":
        available_tools = [name for name in available_tools if name != target_call.name]
        expected_action = "respond"
        reference_calls = []
        goal_state = {"events": {"count": len(source.initial_state.get("events", []))}}
        metadata["oracle_response"] = (
            f"The {target_call.name} tool is not available, so I cannot complete this request."
        )
        metadata["response_expectation"] = {
            "all_phrases": [target_call.name],
            "any_phrases": ["not available", "unavailable", "cannot"],
        }
        robustness["removed_tool"] = target_call.name
        failure_tags.extend(
            ("wrong_call_decision", "hallucinated_tool", "unnecessary_tool_call")
        )
        suffixes = _MISSING_TOOL_SUFFIXES
    elif target == "tool_failure":
        reference_calls = [target_call, target_call]
        robustness["faults"] = [
            {
                "tool_name": target_call.name,
                "occurrence": 1,
                "code": "timeout",
                "message": "temporary timeout from training sandbox; retry is allowed",
                "retriable": True,
            }
        ]
        failure_tags.extend(("tool_error_recovery_failure", "ignore_tool_result"))
        suffixes = _TOOL_FAILURE_SUFFIXES
    elif target == "partial_tool_response":
        reference_calls = [target_call, target_call]
        removed_fields = (
            ["events", "count"]
            if target_call.name == "list_events"
            else ["available"]
        )
        robustness["response_mutations"] = [
            {
                "tool_name": target_call.name,
                "occurrence": 1,
                "mode": "remove_fields",
                "fields": removed_fields,
            }
        ]
        robustness["partial_removed_fields"] = removed_fields
        failure_tags.extend(("tool_error_recovery_failure", "ignore_tool_result"))
        suffixes = _PARTIAL_RESPONSE_SUFFIXES
    else:
        raise ValueError(f"unsupported recovery target: {target}")

    unique = _TARGET_OFFSETS[target] + index
    metadata.update(
        {
            "split": "train",
            "category": "recovery_failure_aware",
            "dataset_name": config.dataset_name,
            "generator_version": config.generator_version,
            "seed": config.seed,
            "target_robustness": target,
            "hard_case_family": f"recovery_{target}",
            "source_train_task_id": source.task_id,
            "source_train_task_sha256": _task_sha256(source),
            "source_robust_validation_sha256": (
                config.source_robust_validation_sha256
            ),
            "robustness": robustness,
        }
    )
    query = f"{source.user_query} {suffixes[unique % len(suffixes)]}"
    return Task(
        task_id=f"calendar_recovery_v2_{target}_{index:05d}",
        domain=source.domain,
        user_query=query,
        available_tools=tuple(available_tools),
        initial_state=copy.deepcopy(dict(source.initial_state)),
        goal_state=goal_state,
        difficulty=f"recovery_{target}",
        failure_tags=tuple(dict.fromkeys(failure_tags)),
        expected_action=expected_action,
        reference_calls=tuple(reference_calls),
        metadata=metadata,
    )


def _validate_target_structure(task: Task, target: str) -> None:
    robustness = task.metadata.get("robustness")
    if not isinstance(robustness, Mapping) or robustness.get("kind") != target:
        raise ValueError(f"recovery metadata drift: {task.task_id}")
    for call in task.reference_calls:
        if call.name not in task.available_tools:
            raise ValueError(f"unavailable reference call {call.name!r}: {task.task_id}")
    if target == "missing_tool":
        removed = str(robustness.get("removed_tool", ""))
        if (
            task.expected_action != "respond"
            or task.reference_calls
            or removed in task.available_tools
        ):
            raise ValueError(f"invalid missing-tool recovery case: {task.task_id}")
    elif len(task.reference_calls) != 2 or (
        task.reference_calls[0] != task.reference_calls[1]
    ):
        raise ValueError(f"recovery case lacks an identical retry: {task.task_id}")


def _normalized_query(task: Task) -> str:
    return " ".join(task.user_query.casefold().split())


def _task_sha256(task: Task) -> str:
    payload = json.dumps(
        task.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
