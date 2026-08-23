"""Deterministic, train-only hard cases selected from Validation failure categories."""

from __future__ import annotations

import copy
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from robust_tool.data.generator import ALL_CALENDAR_TOOLS
from robust_tool.data.schemas import Task, ToolCall

HARD_CASE_GENERATOR_VERSION = "calendar-failure-aware-v1"
SUPPORTED_FAILURE_TARGETS = (
    "wrong_argument_value",
    "ignore_tool_result",
    "missing_argument",
)

_TARGET_OFFSETS = {
    "wrong_argument_value": 0,
    "ignore_tool_result": 1_000_000,
    "missing_argument": 2_000_000,
}
_LOCATIONS = ("Room D", "Room E", "East Hall", "West Hall", "Meet", "Webex")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FailureAwareDatasetConfig:
    """Frozen inputs controlling the size and seed of a train-only hard set."""

    dataset_name: str
    generator_version: str
    seed: int
    target_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.generator_version != HARD_CASE_GENERATOR_VERSION:
            raise ValueError(
                f"unsupported generator_version {self.generator_version!r}; "
                f"expected {HARD_CASE_GENERATOR_VERSION!r}"
            )
        if not self.dataset_name:
            raise ValueError("dataset_name cannot be empty")
        if not self.target_counts:
            raise ValueError("target_counts cannot be empty")
        unknown = sorted(set(self.target_counts) - set(SUPPORTED_FAILURE_TARGETS))
        if unknown:
            raise ValueError(f"unsupported failure targets: {unknown}")
        if any(not isinstance(value, int) or value <= 0 for value in self.target_counts.values()):
            raise ValueError("all target counts must be positive integers")

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "FailureAwareDatasetConfig":
        counts = record.get("target_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("target_counts must be an object")
        return cls(
            dataset_name=str(record["dataset_name"]),
            generator_version=str(record["generator_version"]),
            seed=int(record["seed"]),
            target_counts={str(key): int(value) for key, value in counts.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "target_counts": dict(self.target_counts),
        }

    @property
    def size(self) -> int:
        return sum(self.target_counts.values())


@dataclass(frozen=True)
class FailureTargetSelection:
    """Auditable category-only signal extracted from one SFT Validation run."""

    selection_protocol: str
    source_run: str
    source_git_commit: str | None
    task_snapshot_sha256: str
    validation_task_count: int
    selected_failures: tuple[str, ...]

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "FailureTargetSelection":
        selected = record.get("selected_failures")
        if not isinstance(selected, list) or not selected:
            raise ValueError("selected_failures must be a non-empty list")
        ordered: list[str] = []
        for expected_rank, item in enumerate(selected, start=1):
            if not isinstance(item, Mapping):
                raise ValueError("every selected failure must be an object")
            if int(item.get("rank", -1)) != expected_rank:
                raise ValueError("selected failure ranks must start at 1 and be contiguous")
            failure = str(item.get("failure", ""))
            if failure not in SUPPORTED_FAILURE_TARGETS:
                raise ValueError(f"selected failure is not supported: {failure!r}")
            if int(item.get("count", 0)) <= 0:
                raise ValueError(f"selected failure count must be positive: {failure!r}")
            ordered.append(failure)
        if len(ordered) != len(set(ordered)):
            raise ValueError("selected failures cannot contain duplicates")
        snapshot = str(record.get("task_snapshot_sha256", ""))
        if not _SHA256_PATTERN.fullmatch(snapshot):
            raise ValueError("task_snapshot_sha256 must be a lowercase SHA-256 digest")
        validation_count = int(record.get("validation_task_count", 0))
        if validation_count <= 0:
            raise ValueError("validation_task_count must be positive")
        protocol = str(record.get("selection_protocol", ""))
        if "Validation" not in protocol:
            raise ValueError("failure targets must explicitly come from Validation")
        return cls(
            selection_protocol=protocol,
            source_run=str(record.get("source_run", "")),
            source_git_commit=(
                None
                if record.get("source_git_commit") is None
                else str(record.get("source_git_commit"))
            ),
            task_snapshot_sha256=snapshot,
            validation_task_count=validation_count,
            selected_failures=tuple(ordered),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_protocol": self.selection_protocol,
            "source_run": self.source_run,
            "source_git_commit": self.source_git_commit,
            "task_snapshot_sha256": self.task_snapshot_sha256,
            "validation_task_count": self.validation_task_count,
            "selected_failures": list(self.selected_failures),
        }


def load_failure_aware_config(path: str | Path) -> FailureAwareDatasetConfig:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load failure-aware config {source}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"failure-aware config must be a JSON object: {source}")
    try:
        return FailureAwareDatasetConfig.from_dict(record)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid failure-aware config {source}: {exc}") from exc


def load_failure_target_selection(path: str | Path) -> FailureTargetSelection:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load failure targets {source}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"failure targets must be a JSON object: {source}")
    try:
        return FailureTargetSelection.from_dict(record)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid failure targets {source}: {exc}") from exc


def _iso(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat(timespec="seconds")


def _display(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def _day(unique: int, salt: int) -> datetime:
    return datetime(2035, 1, 1) + timedelta(days=(unique * 13 + salt * 37) % 3_600)


def _label(target: str, index: int) -> str:
    prefix = {
        "wrong_argument_value": "Quartz",
        "ignore_tool_result": "Nimbus",
        "missing_argument": "Saffron",
    }[target]
    return f"{prefix} case {index:05d}"


def _event(
    event_id: str,
    title: str,
    start: datetime,
    end: datetime,
    *,
    location: str = "",
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "title": title,
        "start": _iso(start),
        "end": _iso(end),
        "location": location,
        "description": "",
        "attendees": [],
    }


def _base_events(day: datetime, label: str) -> list[dict[str, Any]]:
    return [
        _event("evt-0001", f"Morning {label}", day.replace(hour=9), day.replace(hour=10)),
        _event("evt-0002", f"Review {label}", day.replace(hour=13), day.replace(hour=14)),
        _event("evt-0003", f"Wrap-up {label}", day.replace(hour=17), day.replace(hour=17, minute=30)),
    ]


def _tool_order(seed: int, unique: int) -> tuple[str, ...]:
    tools = list(ALL_CALENDAR_TOOLS)
    random.Random(seed + unique * 104_729).shuffle(tools)
    return tuple(tools)


def _observation(
    name: str,
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "tool_name": name,
        "arguments": copy.deepcopy(dict(arguments)),
        "result": copy.deepcopy(dict(result)),
    }


def _task(
    config: FailureAwareDatasetConfig,
    selection: FailureTargetSelection,
    *,
    target: str,
    index: int,
    family: str,
    query: str,
    events: Iterable[Mapping[str, Any]],
    goal: Mapping[str, Any],
    calls: Iterable[ToolCall],
    expected_action: str = "call",
    failure_tags: Iterable[str] = (),
    oracle_response: str | None = None,
) -> Task:
    unique = _TARGET_OFFSETS[target] + index
    metadata: dict[str, Any] = {
        "split": "train",
        "category": "failure_aware",
        "target_failure": target,
        "hard_case_family": family,
        "generator_version": config.generator_version,
        "dataset_name": config.dataset_name,
        "seed": config.seed,
        "source_validation_task_snapshot_sha256": selection.task_snapshot_sha256,
    }
    if oracle_response is not None:
        metadata["oracle_response"] = oracle_response
    return Task(
        task_id=f"calendar_failure_v1_{target}_{index:05d}",
        domain="calendar",
        user_query=query,
        available_tools=_tool_order(config.seed, unique),
        initial_state={"events": copy.deepcopy(list(events))},
        goal_state=copy.deepcopy(dict(goal)),
        difficulty=f"failure_aware_{target}",
        failure_tags=tuple(dict.fromkeys((target, *failure_tags))),
        expected_action=expected_action,
        reference_calls=tuple(calls),
        metadata=metadata,
    )


def _wrong_argument_value_task(
    config: FailureAwareDatasetConfig,
    selection: FailureTargetSelection,
    index: int,
) -> Task:
    target = "wrong_argument_value"
    unique = _TARGET_OFFSETS[target] + index
    day = _day(unique, 1)
    label = _label(target, index)
    events = _base_events(day, label)
    profile = index % 4
    if profile == 0:
        start = day.replace(hour=10)
        end = start + timedelta(minutes=(30, 45, 75)[index % 3])
        arguments = {"start": _iso(start), "end": _iso(end)}
        query = (
            f"For {label}, check exactly {_display(start)} through {_display(end)}. "
            "An event ending at the start boundary must not count as a conflict."
        )
        goal = {
            "required_observations": [
                _observation(
                    "check_availability",
                    arguments,
                    {"available": True, "conflicting_event_ids": []},
                )
            ]
        }
        calls = [ToolCall("check_availability", arguments)]
        family = "half_open_boundary"
    elif profile == 1:
        start = day.replace(hour=10, minute=(index % 4) * 10)
        end = start + timedelta(minutes=(35, 50, 80)[index % 3])
        optional: dict[str, Any] = {
            "location": _LOCATIONS[index % len(_LOCATIONS)],
            "description": f"Precision note {index:05d}",
        }
        if index % 2:
            optional["attendees"] = [
                f"case{index % 997}@example.com",
                f"owner{(index * 7) % 997}@example.com",
            ]
        arguments = {
            "title": f"Precision review {label}",
            "start": _iso(start),
            "end": _iso(end),
            **optional,
        }
        query = (
            f"Create '{arguments['title']}' from {_display(start)} to {_display(end)}, "
            f"at {arguments['location']}, with description '{arguments['description']}'"
            + (
                ", inviting " + ", ".join(arguments["attendees"])
                if "attendees" in arguments
                else ""
            )
            + ". Preserve every value exactly."
        )
        goal = {"events": {"contains": [arguments], "count": 4}}
        calls = [ToolCall("create_event", arguments)]
        family = "precise_create_values"
    elif profile == 2:
        new_start = day.replace(hour=14, minute=15)
        new_end = new_start + timedelta(minutes=45 + (index % 2) * 15)
        arguments = {
            "event_id": "evt-0002",
            "title": f"Revised {label}",
            "start": _iso(new_start),
            "end": _iso(new_end),
            "location": _LOCATIONS[(index + 2) % len(_LOCATIONS)],
        }
        query = (
            f"Update only event evt-0002 (currently 'Review {label}'): rename it to "
            f"'{arguments['title']}', move it to {_display(new_start)}–{_display(new_end)}, "
            f"and set location to {arguments['location']}."
        )
        goal = {"events": {"contains": [arguments], "count": 3}}
        calls = [ToolCall("update_event", arguments)]
        family = "similar_event_exact_patch"
    else:
        start = day.replace(hour=16, minute=59)
        end = day.replace(hour=17, minute=1)
        arguments = {"start": _iso(start), "end": _iso(end)}
        query = (
            f"Inspect the two-minute window {_display(start)}–{_display(end)} for {label}; "
            "use those exact minute values."
        )
        goal = {
            "required_observations": [
                _observation("list_events", arguments, {"count": 1})
            ]
        }
        calls = [ToolCall("list_events", arguments)]
        family = "narrow_time_window"
    return _task(
        config,
        selection,
        target=target,
        index=index,
        family=family,
        query=query,
        events=events,
        goal=goal,
        calls=calls,
        failure_tags=("missing_argument",),
    )


def _ignore_tool_result_task(
    config: FailureAwareDatasetConfig,
    selection: FailureTargetSelection,
    index: int,
) -> Task:
    target = "ignore_tool_result"
    unique = _TARGET_OFFSETS[target] + index
    day = _day(unique, 2)
    label = _label(target, index)
    events = _base_events(day, label)
    profile = index % 5
    day_range = {"start": _iso(day), "end": _iso(day + timedelta(days=1))}
    if profile == 0:
        patch = {"event_id": "evt-0002", "location": _LOCATIONS[index % len(_LOCATIONS)]}
        query = (
            f"Find the event titled 'Review {label}' on {day:%Y-%m-%d}. "
            f"Use the returned event ID to change its location to {patch['location']}."
        )
        calls = [ToolCall("list_events", day_range), ToolCall("update_event", patch)]
        goal = {
            "events": {"contains": [patch], "count": 3},
            "required_observations": [_observation("list_events", day_range, {"count": 3})],
        }
        family = "list_then_update_returned_id"
    elif profile == 1:
        delete_args = {"event_id": "evt-0003"}
        query = (
            f"List {day:%Y-%m-%d}, locate 'Wrap-up {label}', and delete the exact ID "
            "returned by the list result."
        )
        calls = [ToolCall("list_events", day_range), ToolCall("delete_event", delete_args)]
        goal = {
            "events": {"absent": [delete_args], "count": 2},
            "required_observations": [_observation("list_events", day_range, {"count": 3})],
        }
        family = "list_then_delete_returned_id"
    elif profile == 2:
        start, end = day.replace(hour=10), day.replace(hour=11)
        availability = {"start": _iso(start), "end": _iso(end)}
        create = {
            "title": f"Conditional {label}",
            "start": _iso(start),
            "end": _iso(end),
            "location": _LOCATIONS[(index + 1) % len(_LOCATIONS)],
        }
        query = (
            f"Check {_display(start)}–{_display(end)} first. Only if the tool reports free, "
            f"create '{create['title']}' at {create['location']}."
        )
        calls = [ToolCall("check_availability", availability), ToolCall("create_event", create)]
        goal = {
            "events": {"contains": [create], "count": 4},
            "required_observations": [
                _observation("check_availability", availability, {"available": True})
            ],
        }
        family = "availability_then_create"
    elif profile == 3:
        start, end = day.replace(hour=9, minute=30), day.replace(hour=10, minute=30)
        arguments = {"start": _iso(start), "end": _iso(end)}
        query = (
            f"Check {_display(start)}–{_display(end)} for {label}. If the result reports a "
            "conflict, do not create or modify anything; tell me it is blocked."
        )
        calls = [ToolCall("check_availability", arguments)]
        goal = {
            "events": {"count": 3},
            "required_observations": [
                _observation(
                    "check_availability",
                    arguments,
                    {"available": False, "conflicting_event_ids": ["evt-0001"]},
                )
            ],
        }
        family = "conflict_result_then_stop"
    else:
        start, end = day.replace(hour=10), day.replace(hour=11)
        create = {
            "title": f"Two-stage {label}",
            "start": _iso(start),
            "end": _iso(end),
        }
        update = {
            "event_id": "evt-0004",
            "location": _LOCATIONS[(index + 3) % len(_LOCATIONS)],
        }
        query = (
            f"Create '{create['title']}' from {_display(start)} to {_display(end)}. Then use "
            f"the new event ID returned by the tool to set its location to {update['location']}."
        )
        calls = [ToolCall("create_event", create), ToolCall("update_event", update)]
        goal = {
            "events": {
                "contains": [{**create, "event_id": "evt-0004", "location": update["location"]}],
                "count": 4,
            },
            "required_observations": [
                _observation("create_event", create, {"event": {"event_id": "evt-0004"}})
            ],
        }
        family = "create_then_update_returned_id"
    return _task(
        config,
        selection,
        target=target,
        index=index,
        family=family,
        query=query,
        events=events,
        goal=goal,
        calls=calls,
        failure_tags=("wrong_next_tool", "wrong_argument_value"),
    )


def _missing_argument_task(
    config: FailureAwareDatasetConfig,
    selection: FailureTargetSelection,
    index: int,
) -> Task:
    target = "missing_argument"
    unique = _TARGET_OFFSETS[target] + index
    day = _day(unique, 3)
    label = _label(target, index)
    events = _base_events(day, label)
    profile = index % 4
    if profile == 0:
        start, end = day.replace(hour=10), day.replace(hour=11, minute=index % 2 * 15)
        arguments = {
            "title": f"Required fields {label}",
            "start": _iso(start),
            "end": _iso(end),
        }
        query = (
            f"Create '{arguments['title']}' starting {_display(start)} and ending {_display(end)}. "
            "No optional fields are requested."
        )
        return _task(
            config,
            selection,
            target=target,
            index=index,
            family="all_required_create_fields",
            query=query,
            events=events,
            goal={"events": {"contains": [arguments], "count": 4}},
            calls=[ToolCall("create_event", arguments)],
            failure_tags=("wrong_argument_value",),
        )
    if profile == 1:
        patch = {
            "event_id": "evt-0001",
            "description": f"Mandatory identity note {index:05d}",
        }
        query = (
            f"For event ID evt-0001 ('Morning {label}'), set description to "
            f"'{patch['description']}'. Keep the required event ID in the call."
        )
        return _task(
            config,
            selection,
            target=target,
            index=index,
            family="required_event_id_update",
            query=query,
            events=events,
            goal={"events": {"contains": [patch], "count": 3}},
            calls=[ToolCall("update_event", patch)],
            failure_tags=("wrong_argument_value",),
        )
    if profile == 2:
        start, end = day.replace(hour=10, minute=15), day.replace(hour=10, minute=45)
        arguments = {"start": _iso(start), "end": _iso(end)}
        query = (
            f"For {label}, check availability from {_display(start)} through {_display(end)}; "
            "both interval endpoints are required."
        )
        return _task(
            config,
            selection,
            target=target,
            index=index,
            family="required_interval_endpoints",
            query=query,
            events=events,
            goal={
                "required_observations": [
                    _observation("check_availability", arguments, {"available": True})
                ]
            },
            calls=[ToolCall("check_availability", arguments)],
            failure_tags=("wrong_argument_value",),
        )

    clarify_profile = (index // 4) % 4
    if clarify_profile == 0:
        query = f"Schedule '{label}' sometime on {day:%Y-%m-%d}; I have not chosen start or end time."
        missing = ("start", "end")
    elif clarify_profile == 1:
        query = f"Create '{label}' beginning {_display(day.replace(hour=16))}; the end time is unknown."
        missing = ("end",)
    elif clarify_profile == 2:
        query = (
            f"Create a calendar event from {_display(day.replace(hour=15))} to "
            f"{_display(day.replace(hour=16))}; I have not provided its title."
        )
        missing = ("title",)
    else:
        query = f"Update my '{label}' meeting, but I have not identified which event or what to change."
        missing = ("event_id", "update_fields")
    response = "Please provide the required " + " and ".join(missing) + " before I call a tool."
    return _task(
        config,
        selection,
        target=target,
        index=index,
        family="clarify_missing_required_fields",
        query=query,
        events=events,
        goal={"events": {"count": 3}},
        calls=[],
        expected_action="clarify",
        failure_tags=("clarification_failure", "wrong_call_decision", "unnecessary_tool_call"),
        oracle_response=response,
    )


_BUILDERS = {
    "wrong_argument_value": _wrong_argument_value_task,
    "ignore_tool_result": _ignore_tool_result_task,
    "missing_argument": _missing_argument_task,
}


def generate_failure_aware_tasks(
    config: FailureAwareDatasetConfig,
    selection: FailureTargetSelection,
) -> list[Task]:
    """Generate only new Train tasks for the selected Validation failure labels."""

    configured = set(config.target_counts)
    selected = set(selection.selected_failures)
    if configured != selected:
        raise ValueError(
            "configured failure targets must exactly match the selected Validation failures; "
            f"configured={sorted(configured)}, selected={sorted(selected)}"
        )
    tasks: list[Task] = []
    for target in selection.selected_failures:
        builder = _BUILDERS[target]
        tasks.extend(builder(config, selection, index) for index in range(config.target_counts[target]))
    random.Random(config.seed).shuffle(tasks)
    return tasks


def _normalized_query(task: Task) -> str:
    return " ".join(task.user_query.casefold().split())


def audit_failure_aware_tasks(
    tasks: Iterable[Task],
    config: FailureAwareDatasetConfig,
    selection: FailureTargetSelection,
    source_splits: Mapping[str, Iterable[Task]],
) -> dict[str, Any]:
    """Fail closed on target drift, invalid references, or source-split leakage."""

    materialized = list(tasks)
    if len(materialized) != config.size:
        raise ValueError(f"generated {len(materialized)} hard cases; expected {config.size}")
    ids = [task.task_id for task in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("hard cases contain duplicate task IDs")
    queries = [_normalized_query(task) for task in materialized]
    if len(queries) != len(set(queries)):
        raise ValueError("hard cases contain duplicate normalized user queries")
    counts = Counter(str(task.metadata.get("target_failure")) for task in materialized)
    if dict(counts) != dict(config.target_counts):
        raise ValueError(f"hard-case target counts differ: {dict(counts)} != {dict(config.target_counts)}")

    source_ids: dict[str, str] = {}
    source_queries: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        source_tasks = list(source_splits.get(split, ()))
        source_counts[split] = len(source_tasks)
        for task in source_tasks:
            previous = source_ids.setdefault(task.task_id, split)
            if previous != split:
                raise ValueError(f"source task ID {task.task_id!r} crosses splits")
            query = _normalized_query(task)
            source_queries.setdefault(query, split)

    overlap_ids = sorted(set(ids) & set(source_ids))
    overlap_queries = sorted(set(queries) & set(source_queries))
    if overlap_ids:
        raise ValueError(f"hard cases overlap source task IDs: {overlap_ids[:5]}")
    if overlap_queries:
        raise ValueError(f"hard cases overlap source user queries: {overlap_queries[:5]}")

    families: Counter[str] = Counter()
    tool_orders: set[tuple[str, ...]] = set()
    clarification_count = 0
    multi_step_count = 0
    for task in materialized:
        if task.metadata.get("split") != "train":
            raise ValueError(f"hard case is not marked Train: {task.task_id}")
        target = str(task.metadata.get("target_failure"))
        if target not in selection.selected_failures or target not in task.failure_tags:
            raise ValueError(f"hard case target metadata drift: {task.task_id}")
        if task.metadata.get("source_validation_task_snapshot_sha256") != selection.task_snapshot_sha256:
            raise ValueError(f"hard case source snapshot drift: {task.task_id}")
        for call in task.reference_calls:
            if call.name not in task.available_tools:
                raise ValueError(f"unavailable reference call {call.name!r}: {task.task_id}")
        if task.expected_action == "call" and not task.reference_calls:
            raise ValueError(f"call task has no reference call: {task.task_id}")
        if task.expected_action != "call" and task.reference_calls:
            raise ValueError(f"non-call task has reference calls: {task.task_id}")
        families[str(task.metadata.get("hard_case_family"))] += 1
        tool_orders.add(task.available_tools)
        clarification_count += task.expected_action == "clarify"
        multi_step_count += len(task.reference_calls) > 1
    if len(materialized) >= 10 and len(tool_orders) < 10:
        raise ValueError("tool order randomization produced fewer than 10 permutations")
    return {
        "count": len(materialized),
        "target_counts": dict(counts),
        "family_counts": dict(families),
        "clarification_count": clarification_count,
        "multi_step_count": multi_step_count,
        "unique_task_ids": len(set(ids)),
        "unique_user_queries": len(set(queries)),
        "distinct_tool_orders": len(tool_orders),
        "source_split_counts": source_counts,
        "source_overlap": {"task_ids": 0, "normalized_user_queries": 0},
        "test_content_used_for_generation": False,
    }
