"""Calendar environment lifecycle and goal evaluator."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from robust_tool.env.base import ToolEnvironment, ToolExecutionResult
from robust_tool.env.executor import ToolExecutor, validate_arguments
from robust_tool.env.state import CalendarState, normalize_datetime
from robust_tool.tools.calendar import CalendarTools
from robust_tool.tools.registry import ToolRegistry, calendar_registry, registry_for_task_record


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
        self._base_registry = registry or calendar_registry()
        self.registry = self._base_registry
        self._task: Mapping[str, Any] | None = None
        self._state = CalendarState()
        self._tools = CalendarTools(self._state)
        self._executor = ToolExecutor(self.registry, self._tools.execute)
        self._history: list[dict[str, Any]] = []
        self._call_counts: dict[str, int] = {}

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._history))

    def reset(self, task: Any) -> Mapping[str, Any]:
        record = copy.deepcopy(dict(_task_record(task)))
        if record.get("domain") != "calendar":
            raise ValueError("CalendarEnvironment only accepts calendar tasks")
        self._task = record
        self.registry = registry_for_task_record(record, self._base_registry)
        self._state = CalendarState(record.get("initial_state", {}).get("events", []))
        self._tools = CalendarTools(self._state)
        self._executor = ToolExecutor(self.registry, self._tools.execute)
        self._history = []
        self._call_counts = {}
        return self.get_state()

    def execute(self, tool_call: Mapping[str, Any]) -> ToolExecutionResult:
        if self._task is None:
            raise RuntimeError("reset(task) must be called before execute")
        name = tool_call.get("name")
        display_name = str(name) if name is not None else "<missing>"
        available = self._task.get("available_tools", [])
        if name not in available:
            if self.registry.has(display_name):
                result = ToolExecutionResult.failure(
                    display_name,
                    "tool_unavailable",
                    f"tool is not available for this task: {name}",
                    retriable=False,
                )
            else:
                result = ToolExecutionResult.failure(
                    display_name,
                    "hallucinated_tool",
                    f"tool is not registered: {name}",
                )
        else:
            self._call_counts[display_name] = self._call_counts.get(display_name, 0) + 1
            occurrence = self._call_counts[display_name]
            result = self._execute_available(tool_call, display_name, occurrence)
        self._history.append({"call": copy.deepcopy(dict(tool_call)), "result": result.to_dict()})
        return result

    def _execute_available(
        self,
        tool_call: Mapping[str, Any],
        name: str,
        occurrence: int,
    ) -> ToolExecutionResult:
        metadata = self._task.get("metadata", {}) if self._task is not None else {}
        robustness = metadata.get("robustness", {}) if isinstance(metadata, Mapping) else {}
        if not isinstance(robustness, Mapping):
            raise ValueError("robustness metadata must be a JSON object")

        definition = self.registry.get(name)
        issues = validate_arguments(tool_call.get("arguments", {}), definition.parameters)
        if issues:
            return ToolExecutionResult.failure(
                name,
                "invalid_parameters",
                "tool arguments failed schema validation",
                validation_issues=issues,
            )

        synthetic_results = robustness.get("synthetic_tool_results", {})
        if isinstance(synthetic_results, Mapping) and name in synthetic_results:
            return ToolExecutionResult(
                tool_name=name,
                ok=True,
                data=copy.deepcopy(synthetic_results[name]),
                state_changed=False,
            )

        faults = robustness.get("faults", [])
        if not isinstance(faults, list):
            raise ValueError("robustness faults must be a JSON array")
        for fault in faults:
            if not isinstance(fault, Mapping):
                raise ValueError("each robustness fault must be a JSON object")
            if fault.get("tool_name") == name and int(fault.get("occurrence", 1)) == occurrence:
                return ToolExecutionResult.failure(
                    name,
                    str(fault.get("code", "tool_failure")),
                    str(fault.get("message", "injected deterministic tool failure")),
                    retriable=bool(fault.get("retriable", False)),
                )

        result = self._executor.execute(tool_call)
        mutations = robustness.get("response_mutations", [])
        if not isinstance(mutations, list):
            raise ValueError("response_mutations must be a JSON array")
        for mutation in mutations:
            if not isinstance(mutation, Mapping):
                raise ValueError("each response mutation must be a JSON object")
            if mutation.get("tool_name") != name or int(mutation.get("occurrence", 1)) != occurrence:
                continue
            if not result.ok or not isinstance(result.data, Mapping):
                return result
            data = copy.deepcopy(dict(result.data))
            mode = mutation.get("mode")
            if mode == "add_noise":
                noise = mutation.get("noise", {})
                if not isinstance(noise, Mapping):
                    raise ValueError("add_noise mutation requires an object")
                data.update(copy.deepcopy(dict(noise)))
            elif mode == "remove_fields":
                fields = mutation.get("fields", [])
                if not isinstance(fields, list):
                    raise ValueError("remove_fields mutation requires a list")
                for field in fields:
                    data.pop(str(field), None)
            else:
                raise ValueError(f"unknown response mutation mode: {mode}")
            return ToolExecutionResult(
                tool_name=result.tool_name,
                ok=result.ok,
                data=data,
                error=result.error,
                validation_issues=result.validation_issues,
                state_changed=result.state_changed,
            )
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
