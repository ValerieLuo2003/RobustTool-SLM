"""Named execution-reward registry used by GRPO and evaluation scripts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

from robust_tool.reward.base import RewardResult
from robust_tool.reward.dense import score_dense
from robust_tool.reward.outcome import score_outcome

RewardFunction = Callable[[Any, Mapping[str, Any] | None], RewardResult]


class RewardRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, RewardFunction] = {}

    def register(self, name: str, function: RewardFunction) -> None:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("reward name cannot be empty")
        if normalized in self._functions:
            raise ValueError(f"reward already registered: {normalized}")
        self._functions[normalized] = function

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._functions))

    def get(self, name: str) -> RewardFunction:
        try:
            return self._functions[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown reward {name!r}; available={self.names()}") from exc

    def score(
        self,
        name: str,
        evaluation: Any,
        config: Mapping[str, Any] | None = None,
    ) -> RewardResult:
        return self.get(name)(evaluation, config)


def default_reward_registry() -> RewardRegistry:
    registry = RewardRegistry()
    registry.register("outcome", score_outcome)
    registry.register("failure_aware_dense", score_dense)
    registry.register("dense", score_dense)
    return registry


def build_reward(
    name: str,
    config: Mapping[str, Any] | None = None,
    registry: RewardRegistry | None = None,
) -> Callable[[Any], RewardResult]:
    selected = (registry or default_reward_registry()).get(name)

    def score(evaluation: Any) -> RewardResult:
        return selected(evaluation, config)

    return score
