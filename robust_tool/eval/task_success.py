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
    final_answer_semantic_match: bool
    task_success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_action": self.predicted_action,
            "decision_correct": self.decision_correct,
            "execution_results": [result.to_dict() for result in self.execution_results],
            "final_state": dict(self.final_state),
            "environment_goal_met": self.environment_goal_met,
            "final_answer_present": self.final_answer_present,
            "final_answer_semantic_match": self.final_answer_semantic_match,
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
    final_answer_semantic_match = _response_matches_expectation(
        final_answer or "",
        task.metadata.get("response_expectation"),
    )
    environment_goal_met = env.check_goal()
    if task.expected_action == "call":
        success = (
            decision_correct
            and environment_goal_met
            and final_answer_present
            and final_answer_semantic_match
        )
    else:
        success = (
            decision_correct
            and not trajectory.tool_calls()
            and environment_goal_met
            and final_answer_present
            and final_answer_semantic_match
        )
    return ReplayResult(
        predicted_action=predicted,
        decision_correct=decision_correct,
        execution_results=tuple(results),
        final_state=env.get_state(),
        environment_goal_met=environment_goal_met,
        final_answer_present=final_answer_present,
        final_answer_semantic_match=final_answer_semantic_match,
        task_success=success,
    )


def _response_matches_expectation(answer: str, expectation: Any) -> bool:
    if expectation is None:
        return True
    if not isinstance(expectation, Mapping):
        raise ValueError("response_expectation must be a JSON object")
    normalized = " ".join(answer.casefold().split())
    all_phrases = expectation.get("all_phrases", [])
    any_phrases = expectation.get("any_phrases", [])
    if not isinstance(all_phrases, list) or not isinstance(any_phrases, list):
        raise ValueError("response expectation phrases must be arrays")
    if any(str(phrase).casefold() not in normalized for phrase in all_phrases):
        return False
    return not any_phrases or any(
        str(phrase).casefold() in normalized for phrase in any_phrases
    )
