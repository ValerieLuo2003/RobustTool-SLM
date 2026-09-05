"""Failure-aware dense reward derived from replay diagnostics.

The shaping signal is deliberately evaluator-facing: it uses only what the
environment observed after a sampled trajectory was executed.  It does not
change the model prompt or leak reference calls into generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from robust_tool.reward.base import RewardResult


def _ratio(numerator: Any, denominator: Any, *, empty: float = 0.0) -> float:
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return empty
    if denominator_value <= 0:
        return empty
    return max(0.0, min(1.0, numerator_value / denominator_value))


@dataclass(frozen=True)
class DenseRewardConfig:
    """Weights for progress features and penalties.

    Positive weights are renormalized over the active components.  Recovery
    progress is inactive on non-recovery tasks, so ordinary tasks are not
    unfairly penalized for having no injected failure.
    """

    decision_weight: float = 0.15
    json_weight: float = 0.10
    tool_weight: float = 0.15
    argument_weight: float = 0.15
    executable_weight: float = 0.10
    goal_weight: float = 0.15
    answer_weight: float = 0.10
    recovery_weight: float = 0.10
    invalid_penalty: float = 0.10
    unnecessary_penalty: float = 0.08
    repeated_penalty: float = 0.06
    ignored_result_penalty: float = 0.08
    clip_min: float = 0.0
    clip_max: float = 1.0

    def __post_init__(self) -> None:
        fields = asdict(self)
        for name, value in fields.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"dense reward option {name} must be numeric")
        for name in (
            "decision_weight",
            "json_weight",
            "tool_weight",
            "argument_weight",
            "executable_weight",
            "goal_weight",
            "answer_weight",
            "recovery_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "invalid_penalty",
            "unnecessary_penalty",
            "repeated_penalty",
            "ignored_result_penalty",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.clip_min > self.clip_max:
            raise ValueError("clip_min must be <= clip_max")

    @classmethod
    def from_dict(cls, config: Mapping[str, Any] | None = None) -> "DenseRewardConfig":
        known = set(cls.__dataclass_fields__)
        values = {key: value for key, value in dict(config or {}).items() if key in known}
        return cls(**values)


def score_dense(
    evaluation: Any,
    config: Mapping[str, Any] | DenseRewardConfig | None = None,
) -> RewardResult:
    """Compute a bounded progress reward plus auditable failure penalties."""

    if isinstance(config, DenseRewardConfig):
        options = config
    else:
        options = DenseRewardConfig.from_dict(config)
    diagnostics = dict(getattr(evaluation, "diagnostics", {}) or {})
    replay = getattr(evaluation, "replay", None)
    failures = tuple(getattr(getattr(evaluation, "failures", None), "failures", ()) or ())

    actual_calls = int(diagnostics.get("actual_call_count", 0) or 0)
    expected_calls = int(diagnostics.get("expected_call_count", 0) or 0)
    answer_eligible = bool(diagnostics.get("final_answer_semantic_eligible", 0))
    answer_component = (
        float(bool(diagnostics.get("final_answer_semantic_correct", 0)))
        if answer_eligible
        else float(bool(getattr(replay, "final_answer_present", False)))
    )

    components: dict[str, float] = {
        "decision": float(bool(diagnostics.get("decision_correct", 0))),
        "json_valid": _ratio(
            diagnostics.get("json_valid_calls", 0),
            actual_calls,
            empty=(1.0 if expected_calls == 0 else 0.0),
        ),
        "tool_selection": _ratio(
            diagnostics.get("tool_selection_correct", 0),
            expected_calls,
            empty=(1.0 if expected_calls == 0 else 0.0),
        ),
        "argument_semantics": _ratio(
            diagnostics.get("semantic_correct_arguments", 0),
            diagnostics.get("semantic_total_arguments", 0),
            empty=(1.0 if expected_calls == 0 else 0.0),
        ),
        "executable": _ratio(
            diagnostics.get("executable_calls", 0),
            actual_calls,
            empty=(1.0 if expected_calls == 0 else 0.0),
        ),
        "environment_goal": float(bool(getattr(replay, "environment_goal_met", False))),
        "final_answer": max(0.0, min(1.0, answer_component)),
    }
    weights: dict[str, float] = {
        "decision": options.decision_weight,
        "json_valid": options.json_weight,
        "tool_selection": options.tool_weight,
        "argument_semantics": options.argument_weight,
        "executable": options.executable_weight,
        "environment_goal": options.goal_weight,
        "final_answer": options.answer_weight,
    }
    if bool(diagnostics.get("recovery_eligible", False)):
        components["recovery"] = float(bool(diagnostics.get("recovery_success", False)))
        weights["recovery"] = options.recovery_weight

    active_weight = sum(weight for name, weight in weights.items() if weight > 0)
    base_value = (
        sum(components[name] * weight for name, weight in weights.items() if weight > 0) / active_weight
        if active_weight
        else 0.0
    )
    invalid_fraction = _ratio(diagnostics.get("invalid_calls", 0), actual_calls)
    repeated_fraction = min(1.0, failures.count("repeated_tool_call") / max(actual_calls, 1))
    penalties = {
        "invalid_calls": options.invalid_penalty * invalid_fraction,
        "unnecessary_call": options.unnecessary_penalty * float(bool(diagnostics.get("unnecessary_call", 0))),
        "repeated_call": options.repeated_penalty * repeated_fraction,
        "ignored_tool_result": options.ignored_result_penalty * float("ignore_tool_result" in failures),
    }
    value = max(options.clip_min, min(options.clip_max, base_value - sum(penalties.values())))
    return RewardResult(
        name="failure_aware_dense",
        value=float(value),
        components=components,
        penalties=penalties,
        metadata={"active_weight": active_weight, "failures": list(failures)},
    )


def dense_reward(
    evaluation: Any,
    config: Mapping[str, Any] | DenseRewardConfig | None = None,
) -> float:
    """Convenience scalar API for callers that do not need the breakdown."""

    return score_dense(evaluation, config).value
