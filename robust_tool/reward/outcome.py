"""Terminal outcome reward for execution-feedback training."""

from __future__ import annotations

from typing import Any, Mapping

from robust_tool.reward.base import RewardResult


def outcome_reward(
    task_success: bool,
    *,
    success_value: float = 1.0,
    failure_value: float = 0.0,
) -> float:
    """Return a binary terminal reward.

    The small standalone function is intentionally kept compatible with the
    original placeholder API so it can also be used in lightweight tests.
    """

    return float(success_value if task_success else failure_value)


def score_outcome(
    evaluation: Any,
    config: Mapping[str, Any] | None = None,
) -> RewardResult:
    """Score an evaluator ``TaskEvaluation`` with terminal success only."""

    options = dict(config or {})
    value = outcome_reward(
        bool(evaluation.success),
        success_value=float(options.get("success_value", 1.0)),
        failure_value=float(options.get("failure_value", 0.0)),
    )
    return RewardResult(
        name="outcome",
        value=value,
        components={"task_success": float(bool(evaluation.success))},
        metadata={"success_value": float(options.get("success_value", 1.0)), "failure_value": float(options.get("failure_value", 0.0))},
    )
