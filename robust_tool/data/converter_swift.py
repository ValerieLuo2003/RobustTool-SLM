"""Minimal trajectory-to-message conversion boundary for future ms-swift data."""

from __future__ import annotations

from typing import Any

from robust_tool.rollout.trajectory import Trajectory


def trajectory_to_messages(trajectory: Trajectory) -> dict[str, Any]:
    """Return a generic messages record; template-specific conversion is deferred."""

    return {
        "task_id": trajectory.task_id,
        "messages": [message.to_dict() for message in trajectory.messages],
    }
