"""Deterministic Calendar toy benchmark generator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from robust_tool.data.schemas import Task, ToolCall, write_tasks

GENERATOR_VERSION = "calendar-toy-v2"
ALL_CALENDAR_TOOLS = (
    "list_events",
    "create_event",
    "update_event",
    "delete_event",
    "check_availability",
)


def _event(
    event_id: str,
    title: str,
    start: str,
    end: str,
    *,
    location: str = "",
    attendees: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "title": title,
        "start": start,
        "end": end,
        "location": location,
        "description": "",
        "attendees": list(attendees),
    }


BASE_EVENTS = (
    _event("evt-0001", "Design review", "2026-08-10T09:00:00", "2026-08-10T10:00:00", location="Room A"),
    _event("evt-0002", "Team lunch", "2026-08-10T12:00:00", "2026-08-10T13:00:00", location="Cafe"),
    _event("evt-0003", "Dentist", "2026-08-11T15:00:00", "2026-08-11T16:00:00"),
)


def _state(events: Iterable[Mapping[str, Any]] = BASE_EVENTS) -> dict[str, Any]:
    return {"events": copy.deepcopy(list(events))}


def _task(
    task_id: str,
    split: str,
    user_query: str,
    *,
    initial_state: Mapping[str, Any],
    goal_state: Mapping[str, Any],
    reference_call: ToolCall | None,
    available_tools: Iterable[str] = ALL_CALENDAR_TOOLS,
    difficulty: str = "basic",
    failure_tags: Iterable[str] = (),
    expected_action: str = "call",
    seed: int,
) -> Task:
    return Task(
        task_id=task_id,
        domain="calendar",
        user_query=user_query,
        available_tools=tuple(available_tools),
        initial_state=copy.deepcopy(initial_state),
        goal_state=copy.deepcopy(goal_state),
        difficulty=difficulty,
        failure_tags=tuple(failure_tags),
        expected_action=expected_action,
        reference_calls=() if reference_call is None else (reference_call,),
        metadata={"split": split, "generator_version": GENERATOR_VERSION, "seed": seed},
    )


def _observation_goal(tool_name: str, arguments: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "required_observations": [
            {"tool_name": tool_name, "arguments": dict(arguments), "result": dict(result)}
        ]
    }


def generate_calendar_toy_tasks(seed: int = 20260809) -> list[Task]:
    """Return 25 fixed tasks: 15 train, 5 validation, and 5 test."""

    specs: list[Task] = []

    list_cases = [
        ("train", "events on 2026-08-10", {"start": "2026-08-10T00:00:00", "end": "2026-08-11T00:00:00"}, 2),
        ("train", "events on the morning of 2026-08-11", {"start": "2026-08-11T00:00:00", "end": "2026-08-11T12:00:00"}, 0),
        ("train", "all calendar events", {}, 3),
        ("validation", "events before noon on 2026-08-10", {"start": "2026-08-10T00:00:00", "end": "2026-08-10T12:00:00"}, 1),
        ("test", "events on the afternoon of 2026-08-11", {"start": "2026-08-11T12:00:00", "end": "2026-08-12T00:00:00"}, 1),
    ]
    for index, (split, phrase, arguments, count) in enumerate(list_cases, start=1):
        specs.append(
            _task(
                f"calendar_list_{index:03d}",
                split,
                f"Please list {phrase}.",
                initial_state=_state(),
                goal_state=_observation_goal("list_events", arguments, {"count": count}),
                reference_call=ToolCall("list_events", arguments),
                difficulty="basic" if index != 2 else "empty_result",
                failure_tags=("wrong_tool", "wrong_argument_value"),
                seed=seed,
            )
        )

    create_cases = [
        ("train", "Focus time", "2026-08-10T10:00:00", "2026-08-10T11:00:00", {}),
        ("train", "Client call", "2026-08-10T14:00:00", "2026-08-10T14:30:00", {"location": "Zoom"}),
        ("train", "Paper reading", "2026-08-11T09:00:00", "2026-08-11T10:30:00", {"attendees": ["bob@example.com"]}),
        ("validation", "Budget review", "2026-08-11T10:30:00", "2026-08-11T11:00:00", {"location": "Room B"}),
        ("test", "Project sync", "2026-08-12T15:00:00", "2026-08-12T15:30:00", {"attendees": ["alice@example.com", "bob@example.com"]}),
    ]
    for index, (split, title, start, end, optional) in enumerate(create_cases, start=1):
        arguments = {"title": title, "start": start, "end": end, **optional}
        specs.append(
            _task(
                f"calendar_create_{index:03d}",
                split,
                f"Create '{title}' from {start} to {end}." + (f" Details: {optional}." if optional else ""),
                initial_state=_state(),
                goal_state={"events": {"contains": [arguments], "count": 4}},
                reference_call=ToolCall("create_event", arguments),
                failure_tags=("missing_argument", "wrong_argument_type", "wrong_argument_value"),
                seed=seed,
            )
        )

    update_cases = [
        ("train", "evt-0001", {"title": "Architecture review"}),
        ("train", "evt-0002", {"start": "2026-08-10T13:00:00", "end": "2026-08-10T14:00:00"}),
        ("train", "evt-0003", {"location": "Central Clinic"}),
        ("validation", "evt-0001", {"attendees": ["alice@example.com", "carol@example.com"]}),
        ("test", "evt-0003", {"title": "Dental checkup", "start": "2026-08-11T16:00:00", "end": "2026-08-11T17:00:00"}),
    ]
    for index, (split, event_id, patch) in enumerate(update_cases, start=1):
        arguments = {"event_id": event_id, **patch}
        specs.append(
            _task(
                f"calendar_update_{index:03d}",
                split,
                f"Update event {event_id} with {patch}.",
                initial_state=_state(),
                goal_state={"events": {"contains": [{"event_id": event_id, **patch}], "count": 3}},
                reference_call=ToolCall("update_event", arguments),
                difficulty="cross_reference",
                failure_tags=("wrong_tool", "missing_argument", "wrong_argument_value"),
                seed=seed,
            )
        )

    delete_cases = [
        ("train", "evt-0001"),
        ("train", "evt-0002"),
        ("train", "evt-0003"),
        ("validation", "evt-0002"),
        ("test", "evt-0001"),
    ]
    for index, (split, event_id) in enumerate(delete_cases, start=1):
        specs.append(
            _task(
                f"calendar_delete_{index:03d}",
                split,
                f"Delete calendar event {event_id}.",
                initial_state=_state(),
                goal_state={"events": {"absent": [{"event_id": event_id}], "count": 2}},
                reference_call=ToolCall("delete_event", {"event_id": event_id}),
                failure_tags=("wrong_tool", "wrong_argument_value"),
                seed=seed,
            )
        )

    availability_cases = [
        (
            "calendar_availability_001",
            "train",
            "Am I free on 2026-08-10 from 10:00 to 11:00?",
            {"start": "2026-08-10T10:00:00", "end": "2026-08-10T11:00:00"},
            True,
        ),
        (
            "calendar_availability_002",
            "train",
            "Am I free on 2026-08-10 from 09:30 to 10:30?",
            {"start": "2026-08-10T09:30:00", "end": "2026-08-10T10:30:00"},
            False,
        ),
        (
            "calendar_ambiguous_001",
            "train",
            "Schedule a meeting with the team next week.",
            None,
            None,
        ),
        (
            "calendar_availability_003",
            "validation",
            "Check whether 2026-08-11 from 14:00 to 15:00 is open.",
            {"start": "2026-08-11T14:00:00", "end": "2026-08-11T15:00:00"},
            True,
        ),
        (
            "calendar_no_tool_001",
            "test",
            "What kinds of calendar tasks can you help me with?",
            None,
            None,
        ),
    ]
    for task_id, split, query, arguments, available in availability_cases:
        if arguments is not None:
            specs.append(
                _task(
                    task_id,
                    split,
                    query,
                    initial_state=_state(),
                    goal_state=_observation_goal("check_availability", arguments, {"available": available}),
                    reference_call=ToolCall("check_availability", arguments),
                    failure_tags=("wrong_tool", "wrong_argument_value"),
                    seed=seed,
                )
            )
        else:
            expected_action = "clarify" if "ambiguous" in task_id else "respond"
            specs.append(
                _task(
                    task_id,
                    split,
                    query,
                    initial_state=_state(),
                    goal_state={"events": {"count": 3}},
                    reference_call=None,
                    difficulty=expected_action,
                    failure_tags=(
                        ("clarification_failure", "wrong_call_decision")
                        if expected_action == "clarify"
                        else ("unnecessary_tool_call", "wrong_call_decision")
                    ),
                    expected_action=expected_action,
                    seed=seed,
                )
            )

    if len(specs) != 25 or len({task.task_id for task in specs}) != 25:
        raise AssertionError("calendar-toy-v2 must contain exactly 25 unique tasks")
    return specs


def write_calendar_toy_dataset(output_dir: str | Path, seed: int = 20260809) -> dict[str, Path]:
    output = Path(output_dir)
    tasks = generate_calendar_toy_tasks(seed)
    split_tasks = {
        split: [task for task in tasks if task.metadata["split"] == split]
        for split in ("train", "validation", "test")
    }
    paths = {
        "all": output / "toy_tasks.jsonl",
        "train": output / "toy_train.jsonl",
        "validation": output / "toy_validation.jsonl",
        "test": output / "toy_test.jsonl",
        "clean_test": output / "clean_test.jsonl",
        "manifest": output / "manifest.json",
    }
    write_tasks(tasks, paths["all"])
    for split, records in split_tasks.items():
        write_tasks(records, paths[split])
    write_tasks(split_tasks["test"], paths["clean_test"])
    hashed_paths = {key: path for key, path in paths.items() if key != "manifest"}
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "counts": {key: len(records) for key, records in split_tasks.items()},
        "files": {
            key: {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in hashed_paths.items()
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
