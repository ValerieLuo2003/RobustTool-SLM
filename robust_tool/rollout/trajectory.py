"""Fully traceable user/assistant/tool trajectory schema."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from robust_tool.data.schemas import ToolCall


@dataclass(frozen=True)
class TrajectoryMessage:
    role: str
    content: str | None = None
    action: str | None = None
    tool_call: ToolCall | None = None
    tool_result: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"invalid trajectory role: {self.role}")
        if self.tool_call is not None and self.role != "assistant":
            raise ValueError("only assistant messages may contain tool_call")
        if self.tool_result is not None and self.role != "tool":
            raise ValueError("only tool messages may contain tool_result")

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            record["content"] = self.content
        if self.action is not None:
            record["action"] = self.action
        if self.tool_call is not None:
            record["tool_call"] = self.tool_call.to_dict()
        if self.tool_result is not None:
            record["tool_result"] = copy.deepcopy(dict(self.tool_result))
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "TrajectoryMessage":
        return cls(
            role=str(record["role"]),
            content=record.get("content"),
            action=record.get("action"),
            tool_call=(
                None if record.get("tool_call") is None else ToolCall.from_dict(record["tool_call"])
            ),
            tool_result=copy.deepcopy(record.get("tool_result")),
        )


@dataclass
class Trajectory:
    task_id: str
    messages: list[TrajectoryMessage] = field(default_factory=list)
    final_state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def tool_calls(self) -> list[ToolCall]:
        return [message.tool_call for message in self.messages if message.tool_call is not None]

    def assistant_actions(self) -> list[str]:
        return [
            message.action or ("call" if message.tool_call is not None else "respond")
            for message in self.messages
            if message.role == "assistant"
        ]

    def final_answer(self) -> str | None:
        for message in reversed(self.messages):
            if message.role == "assistant" and message.tool_call is None:
                return message.content
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "messages": [message.to_dict() for message in self.messages],
            "final_state": copy.deepcopy(dict(self.final_state)),
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Trajectory":
        return cls(
            task_id=str(record["task_id"]),
            messages=[TrajectoryMessage.from_dict(message) for message in record.get("messages", [])],
            final_state=copy.deepcopy(record.get("final_state", {})),
            metadata=copy.deepcopy(record.get("metadata", {})),
        )


def write_trajectories(trajectories: Iterable[Trajectory], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = [json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) for item in trajectories]
    destination.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")


def load_trajectories(path: str | Path) -> list[Trajectory]:
    source = Path(path)
    trajectories: list[Trajectory] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            trajectories.append(Trajectory.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid trajectory at {source}:{line_number}: {exc}") from exc
    return trajectories
