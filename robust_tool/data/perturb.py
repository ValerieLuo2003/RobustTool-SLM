"""Deterministic paired perturbations for the Calendar robustness benchmark."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from robust_tool.data.schemas import Task
from robust_tool.tools.registry import calendar_registry

ROBUSTNESS_GENERATOR_VERSION = "calendar-robustness-v1"


class PerturbationKind(str, Enum):
    SIMILAR_TOOL_DISTRACTOR = "similar_tool_distractor"
    TOOL_ORDER_SHUFFLE = "tool_order_shuffle"
    TOOL_DESCRIPTION_REWRITE = "tool_description_rewrite"
    TOOL_NAME_SIMILARITY = "tool_name_similarity"
    MISSING_TOOL = "missing_tool"
    TOOL_FAILURE = "tool_failure"
    NOISY_TOOL_RESPONSE = "noisy_tool_response"
    PARTIAL_TOOL_RESPONSE = "partial_tool_response"
    AMBIGUOUS_USER_QUERY = "ambiguous_user_query"
    IRRELEVANT_TOOL_ADDED = "irrelevant_tool_added"


PERTURBATION_KINDS = tuple(kind.value for kind in PerturbationKind)
_READ_ONLY_TOOLS = {"list_events", "check_availability"}
_AMBIGUOUS_TOOLS = {"create_event", "update_event", "delete_event"}
_DESCRIPTION_REWRITES = {
    "list_events": (
        "Return scheduled entries whose start lies inside an optional half-open local-time window."
    ),
    "create_event": "Add one new calendar entry if its time does not overlap an existing entry.",
    "update_event": "Change selected fields of an existing calendar entry and retain omitted fields.",
    "delete_event": "Remove one existing calendar entry using its event identifier.",
    "check_availability": "Report whether a half-open local-time interval has calendar conflicts.",
}


@dataclass(frozen=True)
class RobustnessDatasetConfig:
    dataset_name: str
    generator_version: str
    seed: int
    source_split: str
    count_per_kind: int

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name cannot be empty")
        if self.generator_version != ROBUSTNESS_GENERATOR_VERSION:
            raise ValueError(
                f"unsupported generator_version {self.generator_version!r}; "
                f"expected {ROBUSTNESS_GENERATOR_VERSION!r}"
            )
        if self.source_split not in {"validation", "test"}:
            raise ValueError("source_split must be validation or test")
        if not isinstance(self.count_per_kind, int) or self.count_per_kind <= 0:
            raise ValueError("count_per_kind must be a positive integer")

    @property
    def size(self) -> int:
        return self.count_per_kind * len(PERTURBATION_KINDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "source_split": self.source_split,
            "count_per_kind": self.count_per_kind,
        }


def load_robustness_config(path: str | Path) -> RobustnessDatasetConfig:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load robustness config {source}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"robustness config must be a JSON object: {source}")
    try:
        return RobustnessDatasetConfig(
            dataset_name=str(record["dataset_name"]),
            generator_version=str(record["generator_version"]),
            seed=int(record["seed"]),
            source_split=str(record["source_split"]),
            count_per_kind=int(record["count_per_kind"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid robustness config {source}: {exc}") from exc


def generate_robustness_tasks(
    config: RobustnessDatasetConfig,
    source_tasks: Iterable[Task],
) -> list[Task]:
    """Create paired perturbations without changing or sampling from Train."""

    sources = list(source_tasks)
    if not sources:
        raise ValueError("source_tasks cannot be empty")
    if len({task.task_id for task in sources}) != len(sources):
        raise ValueError("source_tasks contain duplicate task IDs")
    wrong_split = [
        task.task_id
        for task in sources
        if str(task.metadata.get("split")) != config.source_split
    ]
    if wrong_split:
        raise ValueError(
            f"source tasks do not belong to {config.source_split}: {wrong_split[:5]}"
        )

    generated: list[Task] = []
    for kind_index, kind in enumerate(PERTURBATION_KINDS):
        candidates = _eligible_sources(kind, sources)
        selected = _stratified_selection(
            candidates,
            count=config.count_per_kind,
            seed=config.seed + kind_index * 1009,
        )
        generated.extend(
            _perturb_task(config, source, kind, index)
            for index, source in enumerate(selected)
        )
    random.Random(config.seed + 99_991).shuffle(generated)
    audit_robustness_tasks(generated, config, sources)
    return generated


def audit_robustness_tasks(
    tasks: Iterable[Task],
    config: RobustnessDatasetConfig,
    source_tasks: Iterable[Task],
) -> dict[str, Any]:
    materialized = list(tasks)
    sources = {task.task_id: task for task in source_tasks}
    if len(materialized) != config.size:
        raise ValueError(f"robustness task count {len(materialized)} != {config.size}")
    if len({task.task_id for task in materialized}) != len(materialized):
        raise ValueError("duplicate robustness task IDs")
    overlap = set(sources) & {task.task_id for task in materialized}
    if overlap:
        raise ValueError(f"robustness IDs overlap source IDs: {sorted(overlap)[:5]}")

    counts: Counter[str] = Counter()
    source_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_usage: Counter[str] = Counter()
    for task in materialized:
        robustness = task.metadata.get("robustness")
        if not isinstance(robustness, Mapping):
            raise ValueError(f"missing robustness metadata: {task.task_id}")
        kind = str(robustness.get("kind"))
        source_id = str(robustness.get("source_task_id"))
        if kind not in PERTURBATION_KINDS:
            raise ValueError(f"unknown perturbation kind for {task.task_id}: {kind}")
        if source_id not in sources:
            raise ValueError(f"unknown source task for {task.task_id}: {source_id}")
        counts[kind] += 1
        source_usage[source_id] += 1
        source_category_counts[kind][str(sources[source_id].metadata.get("category"))] += 1
        for call in task.reference_calls:
            if call.name not in task.available_tools:
                raise ValueError(f"reference tool {call.name!r} unavailable for {task.task_id}")
        if task.metadata.get("split") != f"robust_{config.source_split}":
            raise ValueError(f"incorrect robustness split metadata: {task.task_id}")
        if kind == PerturbationKind.MISSING_TOOL.value:
            removed = str(robustness.get("removed_tool"))
            if removed in task.available_tools or task.reference_calls or task.expected_action != "respond":
                raise ValueError(f"invalid missing-tool task: {task.task_id}")
        if kind in {
            PerturbationKind.TOOL_FAILURE.value,
            PerturbationKind.PARTIAL_TOOL_RESPONSE.value,
        }:
            if len(task.reference_calls) != 2 or task.reference_calls[0] != task.reference_calls[1]:
                raise ValueError(f"recovery task must contain an explicit retry: {task.task_id}")

    expected_counts = {kind: config.count_per_kind for kind in PERTURBATION_KINDS}
    if dict(counts) != expected_counts:
        raise ValueError(f"perturbation counts differ: {dict(counts)} != {expected_counts}")
    return {
        "count": len(materialized),
        "perturbation_counts": dict(sorted(counts.items())),
        "source_category_counts": {
            kind: dict(sorted(category_counts.items()))
            for kind, category_counts in sorted(source_category_counts.items())
        },
        "unique_source_tasks": len(source_usage),
        "maximum_variants_per_source": max(source_usage.values(), default=0),
        "source_split": config.source_split,
        "train_tasks_used": 0,
    }


def _eligible_sources(kind: str, sources: list[Task]) -> list[Task]:
    if kind in {
        PerturbationKind.MISSING_TOOL.value,
        PerturbationKind.TOOL_FAILURE.value,
    }:
        return [
            task
            for task in sources
            if task.expected_action == "call" and len(task.reference_calls) == 1
        ]
    if kind in {
        PerturbationKind.NOISY_TOOL_RESPONSE.value,
        PerturbationKind.PARTIAL_TOOL_RESPONSE.value,
    }:
        return [
            task
            for task in sources
            if task.expected_action == "call"
            and len(task.reference_calls) == 1
            and task.reference_calls[0].name in _READ_ONLY_TOOLS
        ]
    if kind == PerturbationKind.AMBIGUOUS_USER_QUERY.value:
        return [
            task
            for task in sources
            if task.expected_action == "call"
            and len(task.reference_calls) == 1
            and task.reference_calls[0].name in _AMBIGUOUS_TOOLS
        ]
    if kind in {
        PerturbationKind.SIMILAR_TOOL_DISTRACTOR.value,
        PerturbationKind.TOOL_NAME_SIMILARITY.value,
        PerturbationKind.IRRELEVANT_TOOL_ADDED.value,
    }:
        return [task for task in sources if task.expected_action == "call"]
    return list(sources)


def _stratified_selection(
    candidates: list[Task],
    *,
    count: int,
    seed: int,
) -> list[Task]:
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible source tasks for requested count {count}")
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in sorted(candidates, key=lambda item: item.task_id):
        groups[str(task.metadata.get("category", "unknown"))].append(task)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    categories = sorted(groups)
    selected: list[Task] = []
    while len(selected) < count:
        progressed = False
        for category in categories:
            if groups[category]:
                selected.append(groups[category].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"stratified selection produced {len(selected)} of {count} tasks")
    return selected


def _perturb_task(
    config: RobustnessDatasetConfig,
    source: Task,
    kind: str,
    index: int,
) -> Task:
    robustness: dict[str, Any] = {
        "kind": kind,
        "source_task_id": source.task_id,
        "source_task_sha256": _task_sha256(source),
        "source_expected_action": source.expected_action,
    }
    metadata = copy.deepcopy(dict(source.metadata))
    metadata.update(
        {
            "split": f"robust_{config.source_split}",
            "dataset_name": config.dataset_name,
            "generator_version": config.generator_version,
            "seed": config.seed,
            "robustness": robustness,
        }
    )
    available = list(source.available_tools)
    query = source.user_query
    goal = copy.deepcopy(dict(source.goal_state))
    expected_action = source.expected_action
    calls = list(source.reference_calls)
    failure_tags = list(source.failure_tags)

    if kind == PerturbationKind.TOOL_ORDER_SHUFFLE.value:
        available = _different_order(available)
    elif kind == PerturbationKind.TOOL_DESCRIPTION_REWRITE.value:
        metadata["tool_description_overrides"] = {
            name: _DESCRIPTION_REWRITES[name]
            for name in available
            if name in _DESCRIPTION_REWRITES
        }
    elif kind == PerturbationKind.SIMILAR_TOOL_DISTRACTOR.value:
        addition = {
            "name": "search_calendar_items",
            "domain": "calendar",
            "description": (
                "Search reusable calendar item templates by keyword; this does not inspect "
                "scheduled events or change the calendar."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
                "additionalProperties": False,
            },
        }
        metadata["tool_schema_additions"] = [addition]
        robustness["synthetic_tool_results"] = {
            addition["name"]: {"items": [], "count": 0, "scope": "templates"}
        }
        available = _insert_after(available, calls[0].name, addition["name"])
    elif kind == PerturbationKind.TOOL_NAME_SIMILARITY.value:
        target = calls[0].name
        target_definition = calendar_registry().get(target)
        addition = {
            "name": f"{target}_preview",
            "domain": "calendar",
            "description": (
                f"Preview a potential {target} request without reading or changing calendar state."
            ),
            "parameters": copy.deepcopy(dict(target_definition.parameters)),
        }
        metadata["tool_schema_additions"] = [addition]
        robustness["synthetic_tool_results"] = {
            addition["name"]: {"preview_only": True, "operation": target}
        }
        available = _insert_after(available, target, addition["name"])
    elif kind == PerturbationKind.IRRELEVANT_TOOL_ADDED.value:
        addition = {
            "name": "get_calendar_timezone",
            "domain": "calendar",
            "description": "Return the calendar display timezone without reading or changing events.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
        metadata["tool_schema_additions"] = [addition]
        robustness["synthetic_tool_results"] = {
            addition["name"]: {"timezone": "local", "utc_offset": None}
        }
        available.append(addition["name"])
        available = _different_order(available)
    elif kind == PerturbationKind.MISSING_TOOL.value:
        removed = calls[0].name
        available = [name for name in available if name != removed]
        query = (
            source.user_query
            + " Use only the tools currently available; if the required operation is unavailable, explain that."
        )
        goal = {"events": {"count": len(source.initial_state.get("events", []))}}
        expected_action = "respond"
        calls = []
        metadata["oracle_response"] = (
            f"The {removed} tool is not available, so I cannot complete this request."
        )
        metadata["response_expectation"] = {
            "all_phrases": [removed],
            "any_phrases": ["not available", "unavailable", "cannot"],
        }
        robustness["removed_tool"] = removed
        failure_tags.extend(("wrong_call_decision", "hallucinated_tool", "unnecessary_tool_call"))
    elif kind == PerturbationKind.TOOL_FAILURE.value:
        target = calls[0]
        calls = [target, target]
        robustness["faults"] = [
            {
                "tool_name": target.name,
                "occurrence": 1,
                "code": "timeout",
                "message": "deterministic injected timeout; retry is allowed",
                "retriable": True,
            }
        ]
        failure_tags.extend(("tool_error_recovery_failure", "ignore_tool_result"))
    elif kind == PerturbationKind.NOISY_TOOL_RESPONSE.value:
        target = calls[0]
        robustness["response_mutations"] = [
            {
                "tool_name": target.name,
                "occurrence": 1,
                "mode": "add_noise",
                "noise": {
                    "debug_hint": "cache-node-7",
                    "request_id": f"robust-{index:05d}",
                    "warning": "metadata below is unrelated to the calendar answer",
                },
            }
        ]
        failure_tags.append("ignore_tool_result")
    elif kind == PerturbationKind.PARTIAL_TOOL_RESPONSE.value:
        target = calls[0]
        calls = [target, target]
        removed_fields = (
            ["events", "count"] if target.name == "list_events" else ["available"]
        )
        robustness["response_mutations"] = [
            {
                "tool_name": target.name,
                "occurrence": 1,
                "mode": "remove_fields",
                "fields": removed_fields,
            }
        ]
        robustness["partial_removed_fields"] = removed_fields
        failure_tags.extend(("tool_error_recovery_failure", "ignore_tool_result"))
    elif kind == PerturbationKind.AMBIGUOUS_USER_QUERY.value:
        target = calls[0].name
        source_arguments = dict(source.reference_calls[0].arguments)
        initial_events = list(source.initial_state.get("events", []))
        source_date = (
            str(initial_events[0].get("start", "unknown-date")).split("T", 1)[0]
            if initial_events
            else "unknown-date"
        )
        if target == "create_event":
            query = (
                f"Please add {source_arguments.get('title', 'this meeting')} to my calendar, "
                "but I have not provided when it starts or ends."
            )
        elif target == "update_event":
            query = (
                f"Please update one of my meetings on {source_date}, but I have not identified "
                "which event or provided the changes."
            )
        else:
            query = (
                f"Please remove one of my meetings on {source_date}, but I have not identified "
                "which calendar event."
            )
        goal = {"events": {"count": len(source.initial_state.get("events", []))}}
        expected_action = "clarify"
        calls = []
        metadata["oracle_response"] = "Could you provide the event identity and the missing details?"
        metadata["missing_fields"] = ["event_identity", "required_arguments"]
        robustness["original_tool"] = target
        failure_tags.extend(("clarification_failure", "wrong_call_decision"))
    else:
        raise ValueError(f"unsupported perturbation kind: {kind}")

    return Task(
        task_id=f"calendar_robust_v1_{config.source_split}_{kind}_{index:05d}",
        domain=source.domain,
        user_query=query,
        available_tools=tuple(available),
        initial_state=copy.deepcopy(dict(source.initial_state)),
        goal_state=goal,
        difficulty=f"robust_{kind}",
        failure_tags=tuple(dict.fromkeys(failure_tags)),
        expected_action=expected_action,
        reference_calls=tuple(calls),
        metadata=metadata,
    )


def _different_order(names: list[str]) -> list[str]:
    if len(names) < 2:
        return list(names)
    changed = list(reversed(names))
    return changed if changed != names else names[1:] + names[:1]


def _insert_after(names: list[str], anchor: str, addition: str) -> list[str]:
    result = list(names)
    index = result.index(anchor) + 1 if anchor in result else len(result)
    result.insert(index, addition)
    return result


def _task_sha256(task: Task) -> str:
    payload = json.dumps(
        task.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
