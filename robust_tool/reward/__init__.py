"""Execution rewards for trajectory-level post-training."""

from robust_tool.reward.base import RewardResult
from robust_tool.reward.dense import DenseRewardConfig, dense_reward, score_dense
from robust_tool.reward.outcome import outcome_reward, score_outcome
from robust_tool.reward.registry import RewardRegistry, build_reward, default_reward_registry

__all__ = [
    "DenseRewardConfig",
    "RewardRegistry",
    "RewardResult",
    "build_reward",
    "default_reward_registry",
    "dense_reward",
    "outcome_reward",
    "score_dense",
    "score_outcome",
]
