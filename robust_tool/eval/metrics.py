"""Auditable metric aggregation with explicit numerators and denominators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    numerator: float
    denominator: int

    @classmethod
    def rate(cls, numerator: int, denominator: int) -> "MetricValue":
        return cls(None if denominator == 0 else numerator / denominator, numerator, denominator)

    @classmethod
    def average(cls, total: float, count: int) -> "MetricValue":
        return cls(None if count == 0 else total / count, total, count)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }


def aggregate_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, MetricValue]:
    items = list(records)
    task_count = len(items)
    actual_calls = sum(item["actual_call_count"] for item in items)
    expected_calls = sum(item["expected_call_count"] for item in items)
    multi_turn_items = [item for item in items if item["expected_call_count"] > 1]
    recovery_items = [item for item in items if item["recovery_eligible"]]
    return {
        "call_decision_accuracy": MetricValue.rate(sum(item["decision_correct"] for item in items), task_count),
        "tool_selection_accuracy": MetricValue.rate(sum(item["tool_selection_correct"] for item in items), expected_calls),
        "json_valid_rate": MetricValue.rate(sum(item["json_valid_calls"] for item in items), actual_calls),
        "argument_schema_accuracy": MetricValue.rate(sum(item["schema_valid_calls"] for item in items), actual_calls),
        "argument_semantic_accuracy": MetricValue.rate(sum(item["semantic_correct_arguments"] for item in items), sum(item["semantic_total_arguments"] for item in items)),
        "executable_call_rate": MetricValue.rate(sum(item["executable_calls"] for item in items), actual_calls),
        "task_success_rate": MetricValue.rate(sum(item["task_success"] for item in items), task_count),
        "multi_turn_task_success_rate": MetricValue.rate(sum(item["task_success"] for item in multi_turn_items), len(multi_turn_items)),
        "recovery_success_rate": MetricValue.rate(sum(item["recovery_success"] for item in recovery_items), len(recovery_items)),
        "invalid_tool_call_rate": MetricValue.rate(sum(item["invalid_calls"] for item in items), actual_calls),
        "unnecessary_tool_call_rate": MetricValue.rate(sum(item["unnecessary_call"] for item in items), task_count),
        "average_tool_calls_per_task": MetricValue.average(actual_calls, task_count),
    }
