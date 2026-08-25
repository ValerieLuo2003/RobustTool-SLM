"""Convert canonical executable trajectories to ms-swift Agent SFT records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from robust_tool.data.schemas import Task
from robust_tool.rollout.trajectory import Trajectory
from robust_tool.tools.registry import ToolRegistry, calendar_registry, registry_for_task_record


def _json_string(payload: Mapping[str, Any] | list[Mapping[str, Any]]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def trajectory_to_swift_record(
    task: Task,
    trajectory: Trajectory,
    *,
    system_prompt: str,
    registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Return the standard ms-swift Agent messages/tools representation."""

    if task.task_id != trajectory.task_id:
        raise ValueError(
            f"task/trajectory ID mismatch: {task.task_id!r} != {trajectory.task_id!r}"
        )
    registry = registry or calendar_registry()
    task_registry = registry_for_task_record(task.to_dict(), registry)
    tools = task_registry.function_schemas(task.available_tools)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    assistant_targets = 0
    for message in trajectory.messages:
        if message.role == "system":
            raise ValueError("trajectory must not inject a second system message")
        if message.role == "user":
            messages.append({"role": "user", "content": message.content or ""})
        elif message.role == "assistant" and message.tool_call is not None:
            if message.tool_call.name not in task.available_tools:
                raise ValueError(
                    f"trajectory calls unavailable tool {message.tool_call.name!r} for {task.task_id}"
                )
            if not message.tool_call.json_valid:
                raise ValueError(f"SFT target contains invalid JSON for {task.task_id}")
            messages.append(
                {
                    "role": "tool_call",
                    "content": _json_string(
                        {
                            "name": message.tool_call.name,
                            "arguments": dict(message.tool_call.arguments),
                        }
                    ),
                }
            )
            assistant_targets += 1
        elif message.role == "assistant":
            content = (message.content or "").strip()
            if not content:
                raise ValueError(f"empty assistant SFT target for {task.task_id}")
            messages.append({"role": "assistant", "content": content})
            assistant_targets += 1
        elif message.role == "tool":
            if message.tool_result is None:
                raise ValueError(f"missing tool result for {task.task_id}")
            messages.append(
                {
                    "role": "tool_response",
                    "content": _json_string(dict(message.tool_result)),
                }
            )
    if not any(message["role"] == "user" for message in messages):
        raise ValueError(f"SFT record has no user message: {task.task_id}")
    if assistant_targets == 0:
        raise ValueError(f"SFT record has no assistant target: {task.task_id}")
    return {
        "task_id": task.task_id,
        "source_split": str(task.metadata.get("split", "unknown")),
        "tools": _json_string(tools),
        "messages": messages,
    }


def convert_trajectories_to_swift(
    tasks: Iterable[Task],
    trajectories: Iterable[Trajectory],
    *,
    system_prompt: str,
    registry: ToolRegistry | None = None,
) -> list[dict[str, Any]]:
    task_list = list(tasks)
    trajectory_list = list(trajectories)
    task_ids = [task.task_id for task in task_list]
    trajectory_ids = [trajectory.task_id for trajectory in trajectory_list]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("duplicate trajectory task IDs")
    if set(task_ids) != set(trajectory_ids):
        missing = sorted(set(task_ids) - set(trajectory_ids))
        extra = sorted(set(trajectory_ids) - set(task_ids))
        raise ValueError(f"task/trajectory ID mismatch; missing={missing}, extra={extra}")
    by_id = {trajectory.task_id: trajectory for trajectory in trajectory_list}
    return [
        trajectory_to_swift_record(
            task,
            by_id[task.task_id],
            system_prompt=system_prompt,
            registry=registry,
        )
        for task in task_list
    ]


def assert_disjoint_splits(named_tasks: Mapping[str, Iterable[Task]]) -> None:
    seen: dict[str, str] = {}
    for split, tasks in named_tasks.items():
        for task in tasks:
            previous = seen.get(task.task_id)
            if previous is not None:
                raise ValueError(
                    f"task {task.task_id!r} appears in both {previous!r} and {split!r}"
                )
            seen[task.task_id] = split


def write_swift_records(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(record), ensure_ascii=False, sort_keys=True) for record in records]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
