from __future__ import annotations

import unittest

from robust_tool.eval.metrics import MetricValue, aggregate_metrics


class MetricsTests(unittest.TestCase):
    def test_rate_without_eligible_examples_is_null(self) -> None:
        metric = MetricValue.rate(0, 0)
        self.assertIsNone(metric.value)
        self.assertEqual(metric.denominator, 0)

    def test_metric_aggregation_keeps_counts(self) -> None:
        records = [
            {
                "decision_correct": 1,
                "expected_call_count": 1,
                "actual_call_count": 1,
                "tool_selection_correct": 1,
                "json_valid_calls": 1,
                "schema_valid_calls": 1,
                "semantic_correct_arguments": 2,
                "semantic_total_arguments": 3,
                "executable_calls": 1,
                "task_success": 1,
                "invalid_calls": 0,
                "unnecessary_call": 0,
                "recovery_eligible": False,
                "recovery_success": False,
            },
            {
                "decision_correct": 0,
                "expected_call_count": 0,
                "actual_call_count": 0,
                "tool_selection_correct": 0,
                "json_valid_calls": 0,
                "schema_valid_calls": 0,
                "semantic_correct_arguments": 0,
                "semantic_total_arguments": 0,
                "executable_calls": 0,
                "task_success": 0,
                "invalid_calls": 0,
                "unnecessary_call": 0,
                "recovery_eligible": False,
                "recovery_success": False,
            },
        ]
        metrics = aggregate_metrics(records)
        self.assertEqual(metrics["call_decision_accuracy"].value, 0.5)
        self.assertEqual(metrics["argument_semantic_accuracy"].value, 2 / 3)
        self.assertIsNone(metrics["multi_turn_task_success_rate"].value)
        self.assertIsNone(metrics["recovery_success_rate"].value)
        self.assertEqual(metrics["average_tool_calls_per_task"].value, 0.5)


if __name__ == "__main__":
    unittest.main()
