"""JSON-serializable benchmark records with lightweight validation."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

ExpectedAction = Literal["call", "clarify", "respond"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    json_valid: bool = True
    raw: str | None = None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": self.name,
            "arguments": copy.deepcopy(dict(self.arguments)),
            "json_valid": self.json_valid,
        }
        if self.raw is not None:
            record["raw"] = self.raw
        if self.parse_error is not None:
            record["parse_error"] = self.parse_error
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ToolCall":
        arguments = record.get("arguments", {})
        if not isinstance(arguments, Mapping):
            arguments = {}
        return cls(
            name=str(record.get("name", "")),
            arguments=copy.deepcopy(dict(arguments)),
            json_valid=bool(record.get("json_valid", True)),
            raw=record.get("raw"),
            parse_error=record.get("parse_error"),
        )


@dataclass(frozen=True)
class Task:
    task_id: str
    domain: str
    user_query: str
    available_tools: tuple[str, ...]
    initial_state: Mapping[str, Any]
    goal_state: Mapping[str, Any]
    difficulty: str
    failure_tags: tuple[str, ...] = ()
    expected_action: ExpectedAction = "call"
    reference_calls: tuple[ToolCall, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if self.expected_action not in {"call", "clarify", "respond"}:
            raise ValueError(f"invalid expected_action: {self.expected_action}")
        if self.expected_action == "call" and not self.reference_calls:
            raise ValueError("call tasks require at least one reference call")
        if len(self.available_tools) != len(set(self.available_tools)):
            raise ValueError(f"task {self.task_id} contains duplicate available tools")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "user_query": self.user_query,
            "available_tools": list(self.available_tools),
            "initial_state": copy.deepcopy(dict(self.initial_state)),
            "goal_state": copy.deepcopy(dict(self.goal_state)),
            "difficulty": self.difficulty,
            "failure_tags": list(self.failure_tags),
            "expected_action": self.expected_action,
            "reference_calls": [call.to_dict() for call in self.reference_calls],
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Task":
        return cls(
            task_id=str(record["task_id"]),
            domain=str(record["domain"]),
            user_query=str(record["user_query"]),
            available_tools=tuple(str(name) for name in record.get("available_tools", [])),
            initial_state=copy.deepcopy(record.get("initial_state", {})),
            goal_state=copy.deepcopy(record.get("goal_state", {})),
            difficulty=str(record.get("difficulty", "basic")),
            failure_tags=tuple(str(tag) for tag in record.get("failure_tags", [])),
            expected_action=record.get("expected_action", "call"),
            reference_calls=tuple(ToolCall.from_dict(call) for call in record.get("reference_calls", [])),
            metadata=copy.deepcopy(record.get("metadata", {})),
        )


def load_tasks(path: str | Path) -> list[Task]:
    source = Path(path)
    tasks: list[Task] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tasks.append(Task.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid task at {source}:{line_number}: {exc}") from exc
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task IDs in {source}")
    return tasks


def write_tasks(tasks: Iterable[Task], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = [json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) for task in tasks]
    destination.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")
