"""Small, auditable GRPO-style execution-feedback trainer."""

from robust_tool.grpo.config import GRPOConfig, load_grpo_config
from robust_tool.grpo.objective import compute_group_advantages

__all__ = ["GRPOConfig", "compute_group_advantages", "load_grpo_config"]
