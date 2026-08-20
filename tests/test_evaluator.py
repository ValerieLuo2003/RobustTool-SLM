from __future__ import annotations

import copy
import unittest

from robust_tool.data.generator import generate_calendar_toy_tasks
from robust_tool.data.schemas import ToolCall
from robust_tool.eval.arguments import compare_arguments
from robust_tool.eval.evaluator import evaluate_dataset, evaluate_task
from robust_tool.rollout.runner import OraclePolicy, RandomPolicy, run_policy
from robust_tool.rollout.trajectory import Trajectory, TrajectoryMessage


class EvaluatorTests(unittest.TestCase):
    def test_datetime_and_text_normalization_is_deterministic(self) -> None:
        comparison = compare_arguments(
            {"start": "2026-08-10T09:00:00", "title": "Design  Review"},
            {"start": "2026-08-10 09:00", "title": " design review "},
        )
        self.assertTrue(comparison.exact)

    def test_oracle_succeeds_on_all_toy_tasks(self) -> None:
        tasks = generate_calendar_toy_tasks()
        report = evaluate_dataset(tasks, run_policy(tasks, OraclePolicy()))
        self.assertEqual(report.metrics["task_success_rate"].value, 1.0)
        self.assertEqual(report.metrics["tool_selection_accuracy"].value, 1.0)
        self.assertEqual(report.metrics["argument_schema_accuracy"].value, 1.0)
        self.assertEqual(report.failure_counts, {})

    def test_random_policy_exercises_failure_classifier(self) -> None:
        tasks = generate_calendar_toy_tasks()
        report = evaluate_dataset(tasks, run_policy(tasks, RandomPolicy(7)))
        self.assertLess(report.metrics["task_success_rate"].value, 1.0)
        self.assertIn("wrong_call_decision", report.failure_counts)
        self.assertTrue(report.failure_counts)

    def test_evaluator_replays_calls_instead_of_trusting_final_state(self) -> None:
        task = generate_calendar_toy_tasks()[5]
        trajectory = run_policy([task], OraclePolicy())[0]
        tampered = copy.deepcopy(trajectory)
        tampered.final_state = {"events": []}
        result = evaluate_task(task, tampered)
        self.assertTrue(result.success)
        self.assertNotEqual(result.replay.final_state, tampered.final_state)

    def test_invalid_json_is_classified(self) -> None:
        task = generate_calendar_toy_tasks()[0]
        trajectory = Trajectory(
            task.task_id,
            messages=[
                TrajectoryMessage("user", task.user_query),
                TrajectoryMessage(
                    "assistant",
                    action="call",
                    tool_call=ToolCall("list_events", {}, json_valid=False, raw="{", parse_error="bad JSON"),
                ),
                TrajectoryMessage("assistant", content="Done", action="respond"),
            ],
        )
        result = evaluate_task(task, trajectory)
        self.assertIn("invalid_json", result.failures.failures)
        self.assertFalse(result.success)

    def test_dataset_rejects_missing_trajectory(self) -> None:
        tasks = generate_calendar_toy_tasks()[:2]
        trajectories = run_policy(tasks[:1], OraclePolicy())
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            evaluate_dataset(tasks, trajectories)


if __name__ == "__main__":
    unittest.main()
