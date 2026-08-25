"""Dataset evaluator combining replay, diagnostics, failures, and metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from robust_tool.data.schemas import Task
from robust_tool.env.executor import validate_arguments
from robust_tool.eval.arguments import compare_tool_call
from robust_tool.eval.failure_classifier import FailureClassification, classify_failures
from robust_tool.eval.metrics import MetricValue, aggregate_metrics
from robust_tool.eval.task_success import ReplayResult, replay_trajectory
from robust_tool.rollout.trajectory import Trajectory
from robust_tool.tools.registry import ToolRegistry, calendar_registry, registry_for_task_record


@dataclass(frozen=True)
class TaskEvaluation:
    task_id: str
    success: bool
    failures: FailureClassification
    replay: ReplayResult
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "failures": list(self.failures.failures),
            "failure_evidence": dict(self.failures.evidence),
            "replay": self.replay.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class EvaluationReport:
    task_count: int
    metrics: Mapping[str, MetricValue]
    failure_counts: Mapping[str, int]
    task_evaluations: tuple[TaskEvaluation, ...]

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
        }

    def failure_stats_dict(self) -> dict[str, Any]:
        failed_tasks = sum(not item.success for item in self.task_evaluations)
        return {
            "task_count": self.task_count,
            "failed_task_count": failed_tasks,
            "failure_counts": dict(self.failure_counts),
            "failure_task_rates": {
                label: (count / self.task_count if self.task_count else None)
                for label, count in self.failure_counts.items()
            },
        }


def evaluate_task(
    task: Task,
    trajectory: Trajectory,
    registry: ToolRegistry | None = None,
) -> TaskEvaluation:
    base_registry = registry or calendar_registry()
    registry = registry_for_task_record(task.to_dict(), base_registry)
    replay = replay_trajectory(task, trajectory)
    calls = trajectory.tool_calls()

    tool_selection_correct = 0
    semantic_correct = 0
    semantic_total = 0
    for index, expected_call in enumerate(task.reference_calls):
        predicted_call = calls[index] if index < len(calls) else None
        if predicted_call is not None and predicted_call.json_valid and predicted_call.name == expected_call.name:
            tool_selection_correct += 1
        comparison = compare_tool_call(expected_call, predicted_call)
        semantic_correct += comparison.correct
        semantic_total += comparison.total

    json_valid_calls = sum(call.json_valid for call in calls)
    schema_valid_calls = 0
    invalid_calls = 0
    for call in calls:
        if not call.json_valid or not registry.has(call.name):
            invalid_calls += 1
            continue
        issues = validate_arguments(call.arguments, registry.get(call.name).parameters)
        if issues:
            invalid_calls += 1
        else:
            schema_valid_calls += 1

    executable_calls = sum(result.ok for result in replay.execution_results)
    failures = classify_failures(task, trajectory, replay, registry)
    recovery_eligible = any(
        not result.ok and result.error is not None and result.error.retriable
        for result in replay.execution_results
    )
    recovery_success = recovery_eligible and replay.task_success
    diagnostics = {
        "expected_action": task.expected_action,
        "predicted_action": replay.predicted_action,
        "decision_correct": int(replay.decision_correct),
        "expected_call_count": len(task.reference_calls),
        "actual_call_count": len(calls),
        "tool_selection_correct": tool_selection_correct,
        "json_valid_calls": json_valid_calls,
        "schema_valid_calls": schema_valid_calls,
        "semantic_correct_arguments": semantic_correct,
        "semantic_total_arguments": semantic_total,
        "executable_calls": executable_calls,
        "task_success": int(replay.task_success),
        "final_answer_semantic_eligible": int(
            isinstance(task.metadata.get("response_expectation"), Mapping)
        ),
        "final_answer_semantic_correct": int(replay.final_answer_semantic_match),
        "invalid_calls": invalid_calls,
        "unnecessary_call": int(task.expected_action != "call" and bool(calls)),
        "recovery_eligible": recovery_eligible,
        "recovery_success": recovery_success,
    }
    return TaskEvaluation(task.task_id, replay.task_success, failures, replay, diagnostics)


def evaluate_dataset(
    tasks: list[Task],
    trajectories: list[Trajectory],
    registry: ToolRegistry | None = None,
) -> EvaluationReport:
    registry = registry or calendar_registry()
    task_ids = [task.task_id for task in tasks]
    trajectory_ids = [trajectory.task_id for trajectory in trajectories]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("duplicate trajectory task IDs")
    if set(task_ids) != set(trajectory_ids):
        missing = sorted(set(task_ids) - set(trajectory_ids))
        extra = sorted(set(trajectory_ids) - set(task_ids))
        raise ValueError(f"task/trajectory ID mismatch; missing={missing}, extra={extra}")
    by_task_id = {trajectory.task_id: trajectory for trajectory in trajectories}
    evaluations = tuple(
        evaluate_task(task, by_task_id[task.task_id], registry) for task in tasks
    )
    metrics = aggregate_metrics(item.diagnostics for item in evaluations)
    failure_counts = Counter(
        failure for evaluation in evaluations for failure in evaluation.failures.failures
    )
    return EvaluationReport(
        task_count=len(tasks),
        metrics=metrics,
        failure_counts=dict(sorted(failure_counts.items())),
        task_evaluations=evaluations,
    )
