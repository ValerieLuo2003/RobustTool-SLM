"""Replay trajectories in fresh environments to establish task truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from robust_tool.data.schemas import Task
from robust_tool.env.base import ToolExecutionResult
from robust_tool.env.calendar import CalendarEnvironment
from robust_tool.rollout.trajectory import Trajectory


@dataclass(frozen=True)
class ReplayResult:
    predicted_action: str
    decision_correct: bool
    execution_results: tuple[ToolExecutionResult, ...]
    final_state: Mapping[str, Any]
    environment_goal_met: bool
    final_answer_present: bool
    task_success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_action": self.predicted_action,
            "decision_correct": self.decision_correct,
            "execution_results": [result.to_dict() for result in self.execution_results],
            "final_state": dict(self.final_state),
            "environment_goal_met": self.environment_goal_met,
            "final_answer_present": self.final_answer_present,
            "task_success": self.task_success,
        }


def primary_action(trajectory: Trajectory) -> str:
    actions = trajectory.assistant_actions()
    return actions[0] if actions else "missing"


def replay_trajectory(task: Task, trajectory: Trajectory) -> ReplayResult:
    env = CalendarEnvironment()
    env.reset(task)
    results: list[ToolExecutionResult] = []
    for call in trajectory.tool_calls():
        if not call.json_valid:
            results.append(
                ToolExecutionResult.failure(
                    call.name or "<parse-error>",
                    "invalid_json",
                    call.parse_error or "tool call JSON could not be parsed",
                )
            )
            continue
        results.append(env.execute(call.to_dict()))

    predicted = primary_action(trajectory)
    decision_correct = predicted == task.expected_action
    final_answer = trajectory.final_answer()
    final_answer_present = bool(final_answer and final_answer.strip())
    environment_goal_met = env.check_goal()
    if task.expected_action == "call":
        success = decision_correct and environment_goal_met and final_answer_present
    else:
        success = decision_correct and not trajectory.tool_calls() and environment_goal_met and final_answer_present
    return ReplayResult(
        predicted_action=predicted,
        decision_correct=decision_correct,
        execution_results=tuple(results),
        final_state=env.get_state(),
        environment_goal_met=environment_goal_met,
        final_answer_present=final_answer_present,
        task_success=success,
    )
