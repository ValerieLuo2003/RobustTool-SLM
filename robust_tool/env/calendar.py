"""Calendar environment lifecycle and goal evaluator."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from robust_tool.env.base import ToolEnvironment, ToolExecutionResult
from robust_tool.env.executor import ToolExecutor
from robust_tool.env.state import CalendarState, normalize_datetime
from robust_tool.tools.calendar import CalendarTools
from robust_tool.tools.registry import ToolRegistry, calendar_registry


def _task_record(task: Any) -> Mapping[str, Any]:
    if isinstance(task, Mapping):
        return task
    if hasattr(task, "to_dict"):
        return task.to_dict()
    raise TypeError("task must be a mapping or expose to_dict()")


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str) and "T" in value:
        try:
            return normalize_datetime(value)
        except ValueError:
            pass
    return value


def _partial_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _partial_match(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            _partial_match(left, right) for left, right in zip(expected, actual)
        )
    return _normalize_scalar(expected) == _normalize_scalar(actual)


class CalendarEnvironment(ToolEnvironment):
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or calendar_registry()
        self._task: Mapping[str, Any] | None = None
        self._state = CalendarState()
        self._tools = CalendarTools(self._state)
        self._executor = ToolExecutor(self.registry, self._tools.execute)
        self._history: list[dict[str, Any]] = []

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._history))

    def reset(self, task: Any) -> Mapping[str, Any]:
        record = copy.deepcopy(dict(_task_record(task)))
        if record.get("domain") != "calendar":
            raise ValueError("CalendarEnvironment only accepts calendar tasks")
        self._task = record
        self._state = CalendarState(record.get("initial_state", {}).get("events", []))
        self._tools = CalendarTools(self._state)
        self._executor = ToolExecutor(self.registry, self._tools.execute)
        self._history = []
        return self.get_state()

    def execute(self, tool_call: Mapping[str, Any]) -> ToolExecutionResult:
        if self._task is None:
            raise RuntimeError("reset(task) must be called before execute")
        name = tool_call.get("name")
        available = self._task.get("available_tools", [])
        if name not in available:
            if self.registry.has(str(name)):
                result = ToolExecutionResult.failure(
                    str(name),
                    "tool_unavailable",
                    f"tool is not available for this task: {name}",
                    retriable=False,
                )
            else:
                result = ToolExecutionResult.failure(
                    str(name),
                    "hallucinated_tool",
                    f"tool is not registered: {name}",
                )
        else:
            result = self._executor.execute(tool_call)
        self._history.append({"call": copy.deepcopy(dict(tool_call)), "result": result.to_dict()})
        return result

    def get_state(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._state.snapshot())

    def check_goal(self) -> bool:
        if self._task is None:
            raise RuntimeError("reset(task) must be called before check_goal")
        goal = self._task.get("goal_state", {})
        event_goal = goal.get("events", {})
        actual_events = self._state.snapshot()["events"]

        expected_count = event_goal.get("count")
        if expected_count is not None and len(actual_events) != expected_count:
            return False
        for expected in event_goal.get("contains", []):
            if not any(_partial_match(expected, event) for event in actual_events):
                return False
        for forbidden in event_goal.get("absent", []):
            if any(_partial_match(forbidden, event) for event in actual_events):
                return False

        for observation in goal.get("required_observations", []):
            matched = False
            for item in self._history:
                result = item["result"]
                if (
                    result["ok"]
                    and item["call"].get("name") == observation.get("tool_name")
                    and _partial_match(observation.get("arguments", {}), item["call"].get("arguments", {}))
                    and _partial_match(observation.get("result", {}), result.get("data", {}))
                ):
                    matched = True
                    break
            if not matched:
                return False
        return True
