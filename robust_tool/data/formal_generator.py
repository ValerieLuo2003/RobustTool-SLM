"""Deterministic, configuration-backed Calendar dataset for formal SFT experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from robust_tool.data.generator import ALL_CALENDAR_TOOLS
from robust_tool.data.schemas import Task, ToolCall, write_tasks

FORMAL_GENERATOR_VERSION = "calendar-formal-sft-v1"
FORMAL_CATEGORIES = (
    "list_events",
    "create_event",
    "update_event",
    "delete_event",
    "check_availability",
    "clarify",
    "no_tool",
    "multi_step",
)
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class FormalDatasetConfig:
    """Frozen inputs that completely determine a formal benchmark build."""

    dataset_name: str
    generator_version: str
    seed: int
    split_category_counts: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        if self.generator_version != FORMAL_GENERATOR_VERSION:
            raise ValueError(
                f"unsupported generator_version {self.generator_version!r}; "
                f"expected {FORMAL_GENERATOR_VERSION!r}"
            )
        if not self.dataset_name:
            raise ValueError("dataset_name cannot be empty")
        if set(self.split_category_counts) != set(SPLITS):
            raise ValueError(f"split_category_counts must contain exactly {SPLITS}")
        for split in SPLITS:
            counts = self.split_category_counts[split]
            if set(counts) != set(FORMAL_CATEGORIES):
                raise ValueError(f"{split} must contain exactly {FORMAL_CATEGORIES}")
            if any(not isinstance(value, int) or value <= 0 for value in counts.values()):
                raise ValueError(f"all {split} category counts must be positive integers")

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "FormalDatasetConfig":
        counts = record.get("split_category_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("split_category_counts must be an object")
        parsed: dict[str, dict[str, int]] = {}
        for split, category_counts in counts.items():
            if not isinstance(category_counts, Mapping):
                raise ValueError(f"category counts for {split!r} must be an object")
            parsed[str(split)] = {
                str(category): int(value) for category, value in category_counts.items()
            }
        return cls(
            dataset_name=str(record["dataset_name"]),
            generator_version=str(record["generator_version"]),
            seed=int(record["seed"]),
            split_category_counts=parsed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "split_category_counts": {
                split: dict(self.split_category_counts[split]) for split in SPLITS
            },
        }

    def split_size(self, split: str) -> int:
        return sum(self.split_category_counts[split].values())


def load_formal_dataset_config(path: str | Path) -> FormalDatasetConfig:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load formal dataset config {source}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"formal dataset config must be a JSON object: {source}")
    try:
        return FormalDatasetConfig.from_dict(record)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid formal dataset config {source}: {exc}") from exc


_SPLIT_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000}
_CATEGORY_OFFSETS = {category: index * 10_000 for index, category in enumerate(FORMAL_CATEGORIES)}
_ADJECTIVES = (
    "Amber",
    "Azure",
    "Bright",
    "Calm",
    "Cedar",
    "Coral",
    "Golden",
    "Ivory",
    "Lunar",
    "Maple",
    "Nova",
    "Silver",
)
_NOUNS = (
    "Atlas",
    "Beacon",
    "Comet",
    "Delta",
    "Harbor",
    "Meadow",
    "Orchid",
    "Pioneer",
    "Summit",
    "Willow",
)
_LOCATIONS = (
    "Room A",
    "Room B",
    "Room C",
    "North Hall",
    "South Hall",
    "Zoom",
    "Teams",
    "Library",
)

_TEMPLATES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "train": {
        "list_events": (
            "List my calendar events from {start_text} through {end_text}.",
            "Show everything scheduled between {start_text} and {end_text}.",
            "What events start in the window {start_text} to {end_text}?",
            "Check my agenda for {start_text}–{end_text}.",
        ),
        "create_event": (
            "Create {title} from {start_text} to {end_text}{details}.",
            "Add an event called {title}, scheduled {start_text}–{end_text}{details}.",
            "Put {title} on my calendar between {start_text} and {end_text}{details}.",
            "Schedule {title} for {start_text} until {end_text}{details}.",
        ),
        "update_event": (
            "Update event {event_id}: {changes}.",
            "Please change calendar item {event_id} so that {changes}.",
            "Modify {event_id}; {changes}.",
            "Apply this edit to event {event_id}: {changes}.",
        ),
        "delete_event": (
            "Delete calendar event {event_id}, titled {title}.",
            "Remove {title} ({event_id}) from my calendar.",
            "Cancel the calendar item {event_id} named {title}.",
        ),
        "check_availability": (
            "Am I free from {start_text} to {end_text}?",
            "Check whether the time window {start_text}–{end_text} is available.",
            "Do I have a conflict between {start_text} and {end_text}?",
            "Is my calendar open from {start_text} until {end_text}?",
        ),
        "clarify": (
            "Schedule {title}, but I have not decided when.",
            "Create an event at {start_text}; I have not said how long it should last.",
            "Move {title} to a better time.",
            "Delete my {title} meeting; there are two meetings with that name.",
        ),
        "no_tool": (
            "For project {title}, explain what calendar help you can provide without changing anything.",
            "Tell me which kinds of calendar requests you can handle for {title}; do not edit my calendar.",
            "Without calling a tool, describe how you could help organize {title}.",
        ),
        "multi_step": (
            "If {start_text} to {end_text} is free, schedule {title}{details}.",
            "Find the event titled {title} on {date_text}, then update it so {changes}.",
            "Find {title} on {date_text}, then delete that event.",
        ),
    },
    "validation": {
        "list_events": (
            "Could you retrieve appointments beginning between {start_text} and {end_text}?",
            "Give me the agenda entries whose start falls in {start_text}–{end_text}.",
            "Review the portion of my schedule from {start_text} until {end_text}.",
        ),
        "create_event": (
            "Please book {title} on my calendar for {start_text} through {end_text}{details}.",
            "Make a new calendar entry: {title}, {start_text}–{end_text}{details}.",
            "Reserve {start_text} to {end_text} for {title}{details}.",
        ),
        "update_event": (
            "Revise appointment {event_id} as follows: {changes}.",
            "Edit the existing item {event_id}; {changes}.",
            "For calendar ID {event_id}, make this adjustment: {changes}.",
        ),
        "delete_event": (
            "Erase the appointment {title} with ID {event_id}.",
            "Take calendar entry {event_id} ({title}) off my schedule.",
        ),
        "check_availability": (
            "Verify whether I am unbooked during {start_text}–{end_text}.",
            "Would {start_text} through {end_text} clash with an existing appointment?",
            "Tell me if the slot from {start_text} to {end_text} is clear.",
        ),
        "clarify": (
            "Arrange {title}; the date and time are still undecided.",
            "Add {title} at {start_text}, although I did not provide an end time.",
            "Reschedule {title}, but I have not specified which occurrence or the new time.",
        ),
        "no_tool": (
            "Summarize your calendar capabilities for {title}, with no calendar changes.",
            "What calendar assistance is available for {title}? Please only explain.",
        ),
        "multi_step": (
            "First confirm that {start_text}–{end_text} is open; if so, add {title}{details}.",
            "Look up {title} on {date_text} and then revise it so {changes}.",
            "Locate the {title} entry on {date_text} before removing it.",
        ),
    },
    "test": {
        "list_events": (
            "Scan my diary for entries starting no earlier than {start_text} and before {end_text}.",
            "Which appointments begin inside the interval {start_text}–{end_text}?",
            "Read back the events in my schedule window from {start_text} to {end_text}.",
        ),
        "create_event": (
            "Block off {start_text}–{end_text} for {title}{details}.",
            "Set up a new appointment named {title} between {start_text} and {end_text}{details}.",
            "Record {title} in my diary for {start_text} through {end_text}{details}.",
        ),
        "update_event": (
            "Amend diary entry {event_id}: {changes}.",
            "Adjust the appointment identified by {event_id} so {changes}.",
            "Change calendar record {event_id}; {changes}.",
        ),
        "delete_event": (
            "Drop {title}, calendar ID {event_id}, from the diary.",
            "Remove the scheduled item {title} identified as {event_id}.",
        ),
        "check_availability": (
            "See whether my diary has room from {start_text} through {end_text}.",
            "Is anything already booked in the interval {start_text}–{end_text}?",
            "Confirm whether {start_text} to {end_text} remains vacant.",
        ),
        "clarify": (
            "Put {title} in my diary sometime soon.",
            "Add {title} beginning {start_text}; the finishing time is missing.",
            "Change my {title} appointment without any further details.",
        ),
        "no_tool": (
            "Describe, but do not perform, the calendar operations relevant to {title}.",
            "For {title}, what can a calendar assistant do without touching the schedule?",
        ),
        "multi_step": (
            "Check the diary slot {start_text}–{end_text}, and only when it is vacant add {title}{details}.",
            "Search the day {date_text} for {title}; after finding it, alter it so {changes}.",
            "Search for {title} on {date_text} and remove the matching diary entry.",
        ),
    },
}


def _event(
    event_id: str,
    title: str,
    start: datetime,
    end: datetime,
    *,
    location: str = "",
    description: str = "",
    attendees: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "title": title,
        "start": _iso(start),
        "end": _iso(end),
        "location": location,
        "description": description,
        "attendees": list(attendees),
    }


def _iso(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat(timespec="seconds")


def _display(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def _date_for(unique_index: int, salt: int = 0) -> datetime:
    # 1,800 days keeps every largest v1 category on a distinct date while
    # still cycling deterministically for future, larger configurations.
    day = (unique_index * 17 + salt * 29) % 1_800
    return datetime(2027, 1, 1) + timedelta(days=day)


def _label(unique_index: int) -> str:
    adjective = _ADJECTIVES[unique_index % len(_ADJECTIVES)]
    noun = _NOUNS[(unique_index // len(_ADJECTIVES)) % len(_NOUNS)]
    return f"{adjective} {noun} {unique_index:06d}"


def _base_events(day: datetime, unique_index: int, *, target_title: str | None = None) -> list[dict[str, Any]]:
    first_title = target_title or f"Morning sync {_label(unique_index)}"
    return [
        _event(
            "evt-0001",
            first_title,
            day.replace(hour=9),
            day.replace(hour=10),
            location=_LOCATIONS[unique_index % len(_LOCATIONS)],
        ),
        _event(
            "evt-0002",
            f"Lunch review {_label(unique_index + 1)}",
            day.replace(hour=13),
            day.replace(hour=14),
            location=_LOCATIONS[(unique_index + 3) % len(_LOCATIONS)],
        ),
    ]


def _tool_order(seed: int, unique_index: int) -> tuple[str, ...]:
    tools = list(ALL_CALENDAR_TOOLS)
    random.Random(seed + unique_index * 104_729).shuffle(tools)
    return tuple(tools)


def _template(split: str, category: str, index: int) -> tuple[int, str]:
    choices = _TEMPLATES[split][category]
    template_id = index % len(choices)
    return template_id, choices[template_id]


def _observation(tool_name: str, arguments: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "arguments": copy.deepcopy(dict(arguments)),
        "result": copy.deepcopy(dict(result)),
    }


def _details(arguments: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    if "location" in arguments:
        pieces.append(f"at {arguments['location']}")
    if "description" in arguments:
        pieces.append(f"with note '{arguments['description']}'")
    if "attendees" in arguments:
        pieces.append("with " + ", ".join(arguments["attendees"]))
    return "" if not pieces else ", " + ", ".join(pieces)


def _change_text(patch: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for field, value in patch.items():
        if field in {"start", "end"}:
            parts.append(f"{field} is {value.replace('T', ' ')}")
        elif field == "attendees":
            parts.append("attendees are " + ", ".join(value))
        else:
            parts.append(f"{field} is {value}")
    return "; ".join(parts)


def _optional_create_fields(unique_index: int) -> dict[str, Any]:
    profile = unique_index % 8
    fields: dict[str, Any] = {}
    if profile in {1, 4, 6, 7}:
        fields["location"] = _LOCATIONS[unique_index % len(_LOCATIONS)]
    if profile in {2, 5, 6, 7}:
        fields["description"] = f"Agenda for {_label(unique_index + 7)}"
    if profile in {3, 4, 5, 7}:
        fields["attendees"] = [
            f"alex{unique_index % 997}@example.com",
            f"sam{(unique_index * 7) % 997}@example.com",
        ]
    return fields


def _make_task(
    *,
    config: FormalDatasetConfig,
    split: str,
    category: str,
    index: int,
    query: str,
    initial_events: Iterable[Mapping[str, Any]],
    goal_state: Mapping[str, Any],
    reference_calls: Iterable[ToolCall],
    expected_action: str = "call",
    difficulty: str = "basic",
    failure_tags: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Task:
    unique_index = _SPLIT_OFFSETS[split] + _CATEGORY_OFFSETS[category] + index
    template_id, _ = _template(split, category, index)
    return Task(
        task_id=f"calendar_formal_v1_{split}_{category}_{index:05d}",
        domain="calendar",
        user_query=query,
        available_tools=_tool_order(config.seed, unique_index),
        initial_state={"events": copy.deepcopy(list(initial_events))},
        goal_state=copy.deepcopy(dict(goal_state)),
        difficulty=difficulty,
        failure_tags=tuple(failure_tags),
        expected_action=expected_action,
        reference_calls=tuple(reference_calls),
        metadata={
            "split": split,
            "category": category,
            "generator_version": config.generator_version,
            "dataset_name": config.dataset_name,
            "seed": config.seed,
            "template_id": template_id,
            **dict(metadata or {}),
        },
    )


def _build_list_task(config: FormalDatasetConfig, split: str, index: int) -> Task:
    unique = _SPLIT_OFFSETS[split] + _CATEGORY_OFFSETS["list_events"] + index
    day = _date_for(unique, 1)
    events = _base_events(day, unique)
    mode = index % 4
    if mode == 0:
        start, end = day.replace(hour=0), day.replace(hour=12)
        count = 1
    elif mode == 1:
        start, end = day.replace(hour=12), day + timedelta(days=1)
        count = 1
    elif mode == 2:
        start, end = day.replace(hour=8), day.replace(hour=15)
        count = 2
    else:
        start, end = day.replace(hour=10), day.replace(hour=13)
        count = 0
    arguments = {"start": _iso(start), "end": _iso(end)}
    _, template = _template(split, "list_events", index)
    query = template.format(start_text=_display(start), end_text=_display(end))
    return _make_task(
        config=config,
        split=split,
        category="list_events",
        index=index,
        query=query,
        initial_events=events,
        goal_state={"required_observations": [_observation("list_events", arguments, {"count": count})]},
        reference_calls=[ToolCall("list_events", arguments)],
        difficulty="empty_result" if count == 0 else "basic",
        failure_tags=("wrong_tool", "wrong_argument_value", "ignore_tool_result"),
        metadata={"parameter_profile": f"bounded_range_count_{count}"},
    )


def _build_create_task(config: FormalDatasetConfig, split: str, index: int) -> Task:
    unique = _SPLIT_OFFSETS[split] + _CATEGORY_OFFSETS["create_event"] + index
    day = _date_for(unique, 2)
    events = _base_events(day, unique)
    start_hour = 10 if index % 2 == 0 else 15
    durations = (30, 45, 60, 90)
    start = day.replace(hour=start_hour)
    end = start + timedelta(minutes=durations[index % len(durations)])
    title = f"Project {_label(unique)} review"
    optional = _optional_create_fields(unique)
    arguments = {"title": title, "start": _iso(start), "end": _iso(end), **optional}
    _, template = _template(split, "create_event", index)
    query = template.format(
        title=title,
        start_text=_display(start),
        end_text=_display(end),
        details=_details(optional),
    )
    return _make_task(
        config=config,
        split=split,
        category="create_event",
        index=index,
        query=query,
        initial_events=events,
        goal_state={"events": {"contains": [arguments], "count": 3}},
        reference_calls=[ToolCall("create_event", arguments)],
        failure_tags=("wrong_tool", "missing_argument", "wrong_argument_type", "wrong_argument_value"),
        metadata={"parameter_profile": f"optional_fields_{unique % 8}"},
    )


def _update_patch(unique: int, day: datetime, profile: int) -> dict[str, Any]:
    if profile == 0:
        return {"title": f"Revised {_label(unique)} session"}
    if profile == 1:
        return {"location": _LOCATIONS[(unique + 2) % len(_LOCATIONS)]}
    if profile == 2:
        return {"description": f"Updated notes for {_label(unique + 5)}"}
    if profile == 3:
        return {"attendees": [f"lee{unique % 991}@example.com", f"kim{unique % 983}@example.com"]}
    if profile == 4:
        return {"start": _iso(day.replace(hour=10)), "end": _iso(day.replace(hour=11))}
    return {
        "title": f"Moved {_label(unique)} workshop",
        "start": _iso(day.replace(hour=10, minute=30)),
        "end": _iso(day.replace(hour=11, minute=30)),
        "location": _LOCATIONS[(unique + 4) % len(_LOCATIONS)],
    }


def _build_update_task(config: FormalDatasetConfig, split: str, index: int) -> Task:
    unique = _SPLIT_OFFSETS[split] + _CATEGORY_OFFSETS["update_event"] + index
    day = _date_for(unique, 3)
    events = _base_events(day, unique)
    patch = _update_patch(unique, day, index % 6)
    arguments = {"event_id": "evt-0001", **patch}
    _, template = _template(split, "update_event", index)
    query = (
        template.format(event_id="evt-0001", changes=_change_text(patch))
        + f" Its current title is {events[0]['title']}."
    )
    return _make_task(
        config=config,
        split=split,
        category="update_event",
        index=index,
        query=query,
        initial_events=events,
        goal_state={"events": {"contains": [arguments], "count": 2}},
        reference_calls=[ToolCall("update_event", arguments)],
        difficulty="parameter_combination" if len(patch) > 1 else "basic",
        failure_tags=("wrong_tool", "missing_argument", "wrong_argument_value"),
        metadata={"parameter_profile": f"patch_{index % 6}"},
    )


def _build_delete_task(config: FormalDatasetConfig, split: str, index: int) -> Task:
    unique = _SPLIT_OFFSETS[split] + _CATEGORY_OFFSETS["delete_event"] + index
    day = _date_for(unique, 4)
    events = _base_events(day, unique)
    target = events[index % 2]
    arguments = {"event_id": target["event_id"]}
    _, template = _template(split, "delete_event", index)
    query = template.format(event_id=target["event_id"], title=target["title"])
    return _make_task(
        config=config,
        split=split,
        category="delete_event",
        index=index,
        query=query,
        initial_events=events,
        goal_state={"events": {"absent": [arguments], "count": 1}},
        reference_calls=[ToolCall("delete_event", arguments)],
        failure_tags=("wrong_tool", "wrong_argument_value"),
    )


def _build_availability_task(config: FormalDatasetConfig, split: str, index: int) -> Task:
    unique = _SPLIT_OFFSETS[split] + _CATEGORY_OFFSETS["check_availability"] + index
    day = _date_for(unique, 5)
    events = _base_events(day, unique)
    expected_available = index % 2 == 0
    if expected_available:
        start, end = day.replace(hour=10), day.replace(hour=11)
        conflicts: list[str] = []
    else:
        start, end = day.replace(hour=9, minute=30), day.replace(hour=10, minute=30)
        conflicts = ["evt-0001"]
    arguments = {"start": _iso(start), "end": _iso(end)}
    _, template = _template(split, "check_availability", index)
    query = template.format(start_text=_display(start), end_text=_display(end))
    return _make_task(
        config=config,
        split=split,
        category="check_availability",
        index=index,
        query=query,
        initial_events=events,
        goal_state={
            "required_observations": [
                _observation(
                    "check_availability",
                    arguments,
                    {"available": expected_available, "conflicting_event_ids": conflicts},
                )
            ]
        },
        reference_calls=[ToolCall("check_availability", arguments)],
        difficulty="tool_selection_distractor",
        failure_tags=("wrong_tool", "wrong_argument_value", "ignore_tool_result"),
        metadata={"expected_available": expected_available},
    )


def _build_clarify_task(config: FormalDatasetConfig, split: str, index: int) -> Task:
    unique = _SPLIT_OFFSETS[split] + _CATEGORY_OFFSETS["clarify"] + index
    day = _date_for(unique, 6)
    title = f"{_label(unique)} planning"
    events = _base_events(day, unique, target_title=title)
    templates = _TEMPLATES[split]["clarify"]
    profile = index % len(templates)
    query = templates[profile].format(title=title, start_text=_display(day.replace(hour=16)))
    missing_fields = (
        ("date", "start", "end"),
        ("end",),
        ("event_identity", "new_start", "new_end"),
        ("event_identity",),
    )[profile % 4]
    clarification = "Could you provide " + ", ".join(missing_fields).replace("_", " ") + "?"
    return _make_task(
        config=config,
        split=split,
        category="clarify",
        index=index,
        query=query,
        initial_events=events,
        goal_state={"events": {"count": 2}},
        reference_calls=[],
        expected_action="clarify",
        difficulty="ambiguous_request",
        failure_tags=("clarification_failure", "wrong_call_decision", "unnecessary_tool_call"),
        metadata={"missing_fields": list(missing_fields), "oracle_response": clarification},
    )


def _build_no_tool_task(config: FormalDatasetConfig, split: str, index: int) -> Task:
    unique = _SPLIT_OFFSETS[split] + _CATEGORY_OFFSETS["no_tool"] + index
    day = _date_for(unique, 7)
    title = _label(unique)
    events = _base_events(day, unique)
    _, template = _template(split, "no_tool", index)
    query = template.format(title=title)
    response = (
        "I can list, create, update, and delete calendar events, and check availability. "
        "I have not changed your calendar."
    )
    return _make_task(
        config=config,
        split=split,
        category="no_tool",
        index=index,
        query=query,
        initial_events=events,
        goal_state={"events": {"count": 2}},
        reference_calls=[],
        expected_action="respond",
        difficulty="no_tool",
        failure_tags=("unnecessary_tool_call", "wrong_call_decision"),
        metadata={"oracle_response": response},
    )


def _build_multi_step_task(config: FormalDatasetConfig, split: str, index: int) -> Task:
    unique = _SPLIT_OFFSETS[split] + _CATEGORY_OFFSETS["multi_step"] + index
    day = _date_for(unique, 8)
    subtype = index % 3
    title = f"{_label(unique)} checkpoint"
    templates = _TEMPLATES[split]["multi_step"]
    template = templates[subtype]
    if subtype == 0:
        events = _base_events(day, unique)
        start, end = day.replace(hour=10), day.replace(hour=11)
        optional = _optional_create_fields(unique)
        create_args = {"title": title, "start": _iso(start), "end": _iso(end), **optional}
        availability_args = {"start": _iso(start), "end": _iso(end)}
        query = template.format(
            start_text=_display(start),
            end_text=_display(end),
            title=title,
            details=_details(optional),
        )
        calls = [
            ToolCall("check_availability", availability_args),
            ToolCall("create_event", create_args),
        ]
        goal = {
            "events": {"contains": [create_args], "count": 3},
            "required_observations": [
                _observation("check_availability", availability_args, {"available": True})
            ],
        }
        dependency = "availability_then_create"
    elif subtype == 1:
        events = _base_events(day, unique, target_title=title)
        list_args = {"start": _iso(day), "end": _iso(day + timedelta(days=1))}
        patch = _update_patch(unique, day, 1 + index % 4)
        update_args = {"event_id": "evt-0001", **patch}
        query = template.format(title=title, date_text=day.strftime("%Y-%m-%d"), changes=_change_text(patch))
        calls = [ToolCall("list_events", list_args), ToolCall("update_event", update_args)]
        goal = {
            "events": {"contains": [update_args], "count": 2},
            "required_observations": [_observation("list_events", list_args, {"count": 2})],
        }
        dependency = "list_then_update_by_result_id"
    else:
        events = _base_events(day, unique, target_title=title)
        list_args = {"start": _iso(day), "end": _iso(day + timedelta(days=1))}
        delete_args = {"event_id": "evt-0001"}
        query = template.format(title=title, date_text=day.strftime("%Y-%m-%d"))
        calls = [ToolCall("list_events", list_args), ToolCall("delete_event", delete_args)]
        goal = {
            "events": {"absent": [delete_args], "count": 1},
            "required_observations": [_observation("list_events", list_args, {"count": 2})],
        }
        dependency = "list_then_delete_by_result_id"
    return _make_task(
        config=config,
        split=split,
        category="multi_step",
        index=index,
        query=query,
        initial_events=events,
        goal_state=goal,
        reference_calls=calls,
        difficulty="multi_step_state_dependency",
        failure_tags=("wrong_tool", "wrong_next_tool", "wrong_argument_value", "ignore_tool_result"),
        metadata={"dependency": dependency},
    )


_BUILDERS = {
    "list_events": _build_list_task,
    "create_event": _build_create_task,
    "update_event": _build_update_task,
    "delete_event": _build_delete_task,
    "check_availability": _build_availability_task,
    "clarify": _build_clarify_task,
    "no_tool": _build_no_tool_task,
    "multi_step": _build_multi_step_task,
}


def generate_calendar_formal_splits(config: FormalDatasetConfig) -> dict[str, list[Task]]:
    """Generate all splits, then fail closed on leakage or duplicate prompts."""

    splits: dict[str, list[Task]] = {}
    for split in SPLITS:
        records: list[Task] = []
        for category in FORMAL_CATEGORIES:
            count = config.split_category_counts[split][category]
            records.extend(_BUILDERS[category](config, split, index) for index in range(count))
        random.Random(config.seed + _SPLIT_OFFSETS[split]).shuffle(records)
        splits[split] = records
    audit_formal_splits(splits, config)
    return splits


def audit_formal_splits(
    splits: Mapping[str, Iterable[Task]],
    config: FormalDatasetConfig,
) -> dict[str, Any]:
    """Validate size, categories, executable references, and cross-split isolation."""

    materialized = {split: list(splits[split]) for split in SPLITS}
    seen_ids: dict[str, str] = {}
    seen_queries: dict[str, str] = {}
    category_counts: dict[str, dict[str, int]] = {}
    tool_orders: set[tuple[str, ...]] = set()
    for split in SPLITS:
        tasks = materialized[split]
        expected_size = config.split_size(split)
        if len(tasks) != expected_size:
            raise ValueError(f"{split} has {len(tasks)} tasks; expected {expected_size}")
        counts = Counter(str(task.metadata.get("category")) for task in tasks)
        expected_counts = dict(config.split_category_counts[split])
        if dict(counts) != expected_counts:
            raise ValueError(f"{split} category counts differ: {dict(counts)} != {expected_counts}")
        category_counts[split] = dict(counts)
        for task in tasks:
            if task.metadata.get("split") != split:
                raise ValueError(f"task {task.task_id} has incorrect split metadata")
            previous = seen_ids.get(task.task_id)
            if previous is not None:
                raise ValueError(f"task ID {task.task_id!r} appears in {previous!r} and {split!r}")
            seen_ids[task.task_id] = split
            normalized_query = " ".join(task.user_query.casefold().split())
            previous_query = seen_queries.get(normalized_query)
            if previous_query is not None:
                raise ValueError(
                    f"duplicate normalized user query across {previous_query!r} and {task.task_id!r}"
                )
            seen_queries[normalized_query] = task.task_id
            tool_orders.add(task.available_tools)
            for call in task.reference_calls:
                if call.name not in task.available_tools:
                    raise ValueError(f"reference tool {call.name!r} unavailable for {task.task_id}")
            if task.expected_action == "call" and not task.reference_calls:
                raise ValueError(f"call task has no reference calls: {task.task_id}")
            if task.expected_action != "call" and task.reference_calls:
                raise ValueError(f"non-call task has reference calls: {task.task_id}")
    if len(tool_orders) < 10:
        raise ValueError("tool order randomization produced fewer than 10 distinct permutations")
    return {
        "counts": {split: len(materialized[split]) for split in SPLITS},
        "category_counts": category_counts,
        "unique_task_ids": len(seen_ids),
        "unique_user_queries": len(seen_queries),
        "distinct_tool_orders": len(tool_orders),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_calendar_formal_dataset(
    output_dir: str | Path,
    config: FormalDatasetConfig,
) -> dict[str, Path]:
    output = Path(output_dir)
    splits = generate_calendar_formal_splits(config)
    paths = {
        "train": output / "train.jsonl",
        "validation": output / "validation.jsonl",
        "test": output / "test.jsonl",
        "clean_test": output / "clean_test.jsonl",
        "manifest": output / "manifest.json",
    }
    for split in SPLITS:
        write_tasks(splits[split], paths[split])
    write_tasks(splits["test"], paths["clean_test"])
    audit = audit_formal_splits(splits, config)
    manifest = {
        "config": config.to_dict(),
        "audit": audit,
        "files": {
            key: {"path": path.name, "sha256": _sha256(path)}
            for key, path in paths.items()
            if key != "manifest"
        },
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
