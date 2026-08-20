"""Environment protocol and execution result schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from robust_tool.env.errors import EnvironmentErrorDetail


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "field": self.field, "message": self.message}


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    ok: bool
    data: Any = None
    error: EnvironmentErrorDetail | None = None
    validation_issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    state_changed: bool = False

    @classmethod
    def failure(
        cls,
        tool_name: str,
        code: str,
        message: str,
        *,
        retriable: bool = False,
        validation_issues: tuple[ValidationIssue, ...] = (),
    ) -> "ToolExecutionResult":
        return cls(
            tool_name=tool_name,
            ok=False,
            error=EnvironmentErrorDetail(code, message, retriable),
            validation_issues=validation_issues,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "data": self.data,
            "error": None if self.error is None else self.error.to_dict(),
            "validation_issues": [issue.to_dict() for issue in self.validation_issues],
            "state_changed": self.state_changed,
        }


class ToolEnvironment(ABC):
    @abstractmethod
    def reset(self, task: Any) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def execute(self, tool_call: Mapping[str, Any]) -> ToolExecutionResult:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def check_goal(self) -> bool:
        raise NotImplementedError
