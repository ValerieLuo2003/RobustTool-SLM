"""Structured environment errors that are safe to serialize in trajectories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentErrorDetail:
    code: str
    message: str
    retriable: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "retriable": self.retriable,
        }


class ToolExecutionError(Exception):
    """Expected execution failure, such as a conflict or missing event."""

    def __init__(self, code: str, message: str, *, retriable: bool = False) -> None:
        super().__init__(message)
        self.detail = EnvironmentErrorDetail(code, message, retriable)
