"""Rule-based multi-label failure classifier with auditable evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from robust_tool.data.schemas import Task
from robust_tool.eval.arguments import compare_tool_call
from robust_tool.eval.task_success import ReplayResult
from robust_tool.rollout.trajectory import Trajectory
from robust_tool.tools.registry import ToolRegistry

FAILURE_LABELS = (
    "wrong_call_decision",
    "wrong_tool",
    "missing_argument",
    "extra_argument",
    "wrong_argument_type",
    "wrong_argument_value",
    "invalid_json",
    "hallucinated_tool",
    "unnecessary_tool_call",
    "repeated_tool_call",
    "ignore_tool_result",
    "wrong_next_tool",
    "tool_error_recovery_failure",
    "clarification_failure",
    "final_answer_failure",
)


@dataclass(frozen=True)
class FailureClassification:
    failures: tuple[str, ...]
    evidence: Mapping[str, list[Mapping[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return {"failures": list(self.failures), "evidence": dict(self.evidence)}


def classify_failures(
    task: Task,
    trajectory: Trajectory,
    replay: ReplayResult,
    registry: ToolRegistry,
) -> FailureClassification:
    evidence: dict[str, list[Mapping[str, Any]]] = {}

    def add(label: str, detail: Mapping[str, Any]) -> None:
        evidence.setdefault(label, []).append(dict(detail))

    calls = trajectory.tool_calls()
    if not replay.decision_correct:
        add(
            "wrong_call_decision",
            {"expected": task.expected_action, "predicted": replay.predicted_action},
        )
    if task.expected_action != "call" and calls:
        add("unnecessary_tool_call", {"expected_action": task.expected_action, "count": len(calls)})
    if task.expected_action == "clarify" and replay.predicted_action != "clarify":
        add("clarification_failure", {"predicted": replay.predicted_action})

    seen: dict[str, int] = {}
    for index, call in enumerate(calls):
        canonical = json.dumps(
            {"name": call.name, "arguments": dict(call.arguments)},
            ensure_ascii=False,
            sort_keys=True,
        )
        if canonical in seen:
            expected_retry = (
                index < len(task.reference_calls)
                and task.reference_calls[index].name == call.name
                and dict(task.reference_calls[index].arguments) == dict(call.arguments)
            )
            if not expected_retry:
                add("repeated_tool_call", {"call_index": index, "name": call.name})
        seen[canonical] = index
        if not call.json_valid:
            add("invalid_json", {"call_index": index, "error": call.parse_error})
        if not registry.has(call.name):
            add("hallucinated_tool", {"call_index": index, "name": call.name})

        if index < len(task.reference_calls):
            expected_call = task.reference_calls[index]
            if call.name != expected_call.name:
                label = "wrong_tool" if index == 0 else "wrong_next_tool"
                add(label, {"call_index": index, "expected": expected_call.name, "predicted": call.name})
            else:
                comparison = compare_tool_call(expected_call, call)
                for field, mismatch in comparison.mismatches.items():
                    add("wrong_argument_value", {"call_index": index, "field": field, **mismatch})
        elif task.reference_calls:
            add("wrong_next_tool", {"call_index": index, "predicted": call.name, "expected": None})

    for index, result in enumerate(replay.execution_results):
        for issue in result.validation_issues:
            add(issue.code, {"call_index": index, "field": issue.field, "message": issue.message})
        if not result.ok and result.error is not None and result.error.code not in {
            "invalid_json",
            "invalid_parameters",
            "hallucinated_tool",
            "tool_unavailable",
        }:
            recovered = any(later.ok for later in replay.execution_results[index + 1 :])
            if not recovered:
                add(
                    "tool_error_recovery_failure",
                    {"call_index": index, "error_code": result.error.code},
                )
    if replay.execution_results and not replay.execution_results[-1].ok and replay.final_answer_present:
        add(
            "ignore_tool_result",
            {"error_code": replay.execution_results[-1].error.code if replay.execution_results[-1].error else None},
        )
    if not replay.final_answer_present:
        add("final_answer_failure", {"reason": "missing_or_empty"})
    elif not replay.final_answer_semantic_match:
        add("final_answer_failure", {"reason": "response_semantic_mismatch"})
    elif not replay.task_success and not evidence:
        add("final_answer_failure", {"reason": "goal_not_completed"})

    ordered = tuple(label for label in FAILURE_LABELS if label in evidence)
    return FailureClassification(ordered, evidence)
