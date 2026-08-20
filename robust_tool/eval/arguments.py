"""Deterministic argument normalization and semantic comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from robust_tool.data.schemas import ToolCall
from robust_tool.env.state import normalize_datetime

_SPACE_PATTERN = re.compile(r"\s+")


def normalize_value(field: str, value: Any) -> Any:
    """Normalize stable representations without using an LLM judge."""

    if isinstance(value, Mapping):
        return {key: normalize_value(str(key), item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        normalized = [normalize_value(field, item) for item in value]
        return sorted(normalized) if field == "attendees" else normalized
    if isinstance(value, str):
        cleaned = _SPACE_PATTERN.sub(" ", value.strip())
        if field in {"start", "end"}:
            try:
                return normalize_datetime(cleaned)
            except ValueError:
                return cleaned
        if field in {"event_id", "title", "location", "description", "attendees"}:
            return cleaned.casefold()
        if cleaned.casefold() == "true":
            return True
        if cleaned.casefold() == "false":
            return False
        return cleaned
    return value


@dataclass(frozen=True)
class ArgumentComparison:
    correct: int
    total: int
    mismatches: Mapping[str, Mapping[str, Any]]

    @property
    def exact(self) -> bool:
        return self.total > 0 and self.correct == self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "total": self.total,
            "exact": self.exact,
            "mismatches": dict(self.mismatches),
        }


def compare_arguments(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> ArgumentComparison:
    mismatches: dict[str, Mapping[str, Any]] = {}
    correct = 0
    for field, expected_value in expected.items():
        predicted_value = predicted.get(field, None)
        left = normalize_value(str(field), expected_value)
        right = normalize_value(str(field), predicted_value)
        if field in predicted and left == right:
            correct += 1
        else:
            mismatches[str(field)] = {"expected": left, "predicted": right}
    return ArgumentComparison(correct=correct, total=len(expected), mismatches=mismatches)


def compare_tool_call(expected: ToolCall, predicted: ToolCall | None) -> ArgumentComparison:
    if predicted is None or predicted.name != expected.name or not predicted.json_valid:
        return ArgumentComparison(
            correct=0,
            total=len(expected.arguments),
            mismatches={
                str(field): {"expected": normalize_value(str(field), value), "predicted": None}
                for field, value in expected.arguments.items()
            },
        )
    return compare_arguments(expected.arguments, predicted.arguments)
