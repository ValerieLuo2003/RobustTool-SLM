"""Outcome reward placeholder; GRPO is outside the Week 1 scope."""


def outcome_reward(task_success: bool) -> float:
    return 1.0 if task_success else 0.0
