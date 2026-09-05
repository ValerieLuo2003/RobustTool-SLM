from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robust_tool.data.generator import generate_calendar_toy_tasks
from robust_tool.eval.evaluator import evaluate_task
from robust_tool.grpo.config import load_grpo_config
from robust_tool.grpo.objective import compute_group_advantages
from robust_tool.reward.dense import score_dense
from robust_tool.reward.outcome import score_outcome
from robust_tool.reward.registry import default_reward_registry
from robust_tool.rollout.runner import OraclePolicy, RandomPolicy, run_policy
from scripts.aggregate_seed_runs import aggregate_runs


class RewardTests(unittest.TestCase):
    def test_outcome_reward_is_terminal_and_auditable(self) -> None:
        task = generate_calendar_toy_tasks()[0]
        success = evaluate_task(task, run_policy([task], OraclePolicy())[0])
        failure = evaluate_task(task, run_policy([task], RandomPolicy(7))[0])
        self.assertEqual(score_outcome(success).value, 1.0)
        self.assertEqual(score_outcome(failure).value, 0.0)
        self.assertEqual(score_outcome(success).components["task_success"], 1.0)

    def test_dense_reward_uses_failure_penalties_and_stays_bounded(self) -> None:
        task = generate_calendar_toy_tasks()[0]
        success = evaluate_task(task, run_policy([task], OraclePolicy())[0])
        failure = evaluate_task(task, run_policy([task], RandomPolicy(7))[0])
        good = score_dense(success)
        bad = score_dense(failure)
        self.assertGreaterEqual(good.value, 0.0)
        self.assertLessEqual(good.value, 1.0)
        self.assertGreaterEqual(bad.value, 0.0)
        self.assertLessEqual(bad.value, 1.0)
        self.assertGreater(good.value, bad.value)
        self.assertTrue(bad.penalties or bad.metadata["failures"])

    def test_registry_exposes_both_training_rewards(self) -> None:
        registry = default_reward_registry()
        self.assertEqual(registry.names(), ("dense", "failure_aware_dense", "outcome"))

    def test_group_advantages_are_centered_and_zero_for_ties(self) -> None:
        advantages = compute_group_advantages([0.0, 0.5, 1.0])
        self.assertAlmostEqual(sum(advantages), 0.0, places=6)
        self.assertEqual(compute_group_advantages([1.0, 1.0]), [0.0, 0.0])

    def test_aggregate_seed_runs_reports_mean_and_sample_std(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, value in enumerate((0.8, 1.0), start=1):
                run = root / f"seed-{index}"
                run.mkdir()
                (run / "metrics.json").write_text(
                    json.dumps({"metrics": {"task_success_rate": {"value": value}}}),
                    encoding="utf-8",
                )
            report = aggregate_runs(
                [("seed-1", root / "seed-1"), ("seed-2", root / "seed-2")],
                metric="metrics.task_success_rate",
            )
            self.assertAlmostEqual(report["mean"], 0.9)
            self.assertAlmostEqual(report["std"], 0.1414213562, places=6)

    def test_grpo_configs_load_and_keep_reward_name(self) -> None:
        config = load_grpo_config(
            Path("configs/grpo/qwen2_5_1_5b_grpo_dense_smoke.json")
        )
        self.assertEqual(config.reward_name, "failure_aware_dense")
        self.assertEqual(config.group_size, 4)


if __name__ == "__main__":
    unittest.main()
