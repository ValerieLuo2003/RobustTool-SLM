from __future__ import annotations

import unittest
from pathlib import Path

from robust_tool.data.formal_generator import (
    FORMAL_CATEGORIES,
    FormalDatasetConfig,
    generate_calendar_formal_splits,
)
from robust_tool.data.perturb import (
    PERTURBATION_KINDS,
    PerturbationKind,
    RobustnessDatasetConfig,
    audit_robustness_tasks,
    generate_robustness_tasks,
    load_robustness_config,
)
from robust_tool.env.calendar import CalendarEnvironment
from robust_tool.eval.evaluator import evaluate_dataset, evaluate_task
from robust_tool.eval.robustness import compute_robustness_report
from robust_tool.eval.robustness_comparison import compare_robustness_runs
from robust_tool.rollout.runner import OraclePolicy, run_policy
from robust_tool.rollout.trajectory import Trajectory, TrajectoryMessage
from robust_tool.tools.registry import calendar_registry, registry_for_task_record


class RobustnessTests(unittest.TestCase):
    @staticmethod
    def _source_tasks():
        formal = FormalDatasetConfig(
            dataset_name="calendar-robust-source-test",
            generator_version="calendar-formal-sft-v1",
            seed=123,
            split_category_counts={
                split: {category: 3 for category in FORMAL_CATEGORIES}
                for split in ("train", "validation", "test")
            },
        )
        return generate_calendar_formal_splits(formal)["validation"]

    @staticmethod
    def _config() -> RobustnessDatasetConfig:
        return RobustnessDatasetConfig(
            dataset_name="calendar-robust-test",
            generator_version="calendar-robustness-v1",
            seed=456,
            source_split="validation",
            count_per_kind=2,
        )

    def _tasks(self):
        return generate_robustness_tasks(self._config(), self._source_tasks())

    def test_generator_is_deterministic_balanced_and_train_free(self) -> None:
        source = self._source_tasks()
        left = generate_robustness_tasks(self._config(), source)
        right = generate_robustness_tasks(self._config(), source)
        self.assertEqual([task.to_dict() for task in left], [task.to_dict() for task in right])
        audit = audit_robustness_tasks(left, self._config(), source)
        self.assertEqual(audit["count"], 20)
        self.assertEqual(audit["train_tasks_used"], 0)
        self.assertEqual(
            audit["perturbation_counts"],
            {kind: 2 for kind in PERTURBATION_KINDS},
        )
        self.assertFalse({task.task_id for task in left} & {task.task_id for task in source})

    def test_oracle_completes_all_perturbations_and_recovers(self) -> None:
        tasks = self._tasks()
        trajectories = run_policy(tasks, OraclePolicy(), max_steps=4)
        report = evaluate_dataset(tasks, trajectories)
        self.assertEqual(report.metrics["task_success_rate"].value, 1.0)
        self.assertEqual(report.metrics["recovery_success_rate"].value, 1.0)
        self.assertEqual(report.metrics["final_answer_semantic_accuracy"].value, 1.0)
        self.assertFalse(report.failure_counts)

    def test_task_registry_applies_schema_additions_and_description_rewrites(self) -> None:
        tasks = self._tasks()
        name_similarity = next(
            task
            for task in tasks
            if task.metadata["robustness"]["kind"]
            == PerturbationKind.TOOL_NAME_SIMILARITY.value
        )
        registry = registry_for_task_record(name_similarity.to_dict(), calendar_registry())
        additions = name_similarity.metadata["tool_schema_additions"]
        self.assertTrue(registry.has(additions[0]["name"]))
        self.assertIn(additions[0]["name"], name_similarity.available_tools)

        rewrite = next(
            task
            for task in tasks
            if task.metadata["robustness"]["kind"]
            == PerturbationKind.TOOL_DESCRIPTION_REWRITE.value
        )
        rewritten = registry_for_task_record(rewrite.to_dict(), calendar_registry())
        for name, description in rewrite.metadata["tool_description_overrides"].items():
            self.assertEqual(rewritten.get(name).description, description)

    def test_environment_injects_fault_noise_and_partial_result_deterministically(self) -> None:
        tasks = self._tasks()
        by_kind = {
            task.metadata["robustness"]["kind"]: task
            for task in tasks
        }

        fault = by_kind[PerturbationKind.TOOL_FAILURE.value]
        env = CalendarEnvironment()
        env.reset(fault)
        call = fault.reference_calls[0].to_dict()
        first = env.execute(call)
        second = env.execute(call)
        self.assertFalse(first.ok)
        self.assertEqual(first.error.code, "timeout")
        self.assertTrue(first.error.retriable)
        self.assertTrue(second.ok)

        noisy = by_kind[PerturbationKind.NOISY_TOOL_RESPONSE.value]
        env.reset(noisy)
        noisy_result = env.execute(noisy.reference_calls[0].to_dict())
        self.assertTrue(noisy_result.ok)
        self.assertIn("request_id", noisy_result.data)

        partial = by_kind[PerturbationKind.PARTIAL_TOOL_RESPONSE.value]
        env.reset(partial)
        partial_call = partial.reference_calls[0].to_dict()
        partial_first = env.execute(partial_call)
        self.assertFalse(env.check_goal())
        partial_second = env.execute(partial_call)
        removed = partial.metadata["robustness"]["partial_removed_fields"]
        self.assertTrue(partial_first.ok)
        self.assertTrue(all(field not in partial_first.data for field in removed))
        self.assertTrue(all(field in partial_second.data for field in removed))
        self.assertTrue(env.check_goal())

    def test_missing_tool_requires_semantically_correct_response(self) -> None:
        task = next(
            task
            for task in self._tasks()
            if task.metadata["robustness"]["kind"] == PerturbationKind.MISSING_TOOL.value
        )
        oracle = evaluate_task(task, run_policy([task], OraclePolicy())[0])
        self.assertTrue(oracle.success)
        vague = Trajectory(
            task.task_id,
            messages=[
                TrajectoryMessage("user", task.user_query),
                TrajectoryMessage("assistant", action="respond", content="Done."),
            ],
        )
        result = evaluate_task(task, vague)
        self.assertFalse(result.success)
        self.assertFalse(result.replay.final_answer_semantic_match)
        self.assertIn("final_answer_failure", result.failures.failures)

    def test_tool_failure_recovery_denominator_is_task_defined(self) -> None:
        task = next(
            task
            for task in self._tasks()
            if task.metadata["robustness"]["kind"] == PerturbationKind.TOOL_FAILURE.value
        )
        no_call = Trajectory(
            task.task_id,
            messages=[
                TrajectoryMessage("user", task.user_query),
                TrajectoryMessage("assistant", action="respond", content="I cannot help."),
            ],
        )
        result = evaluate_task(task, no_call)
        self.assertTrue(result.diagnostics["recovery_eligible"])
        self.assertFalse(result.diagnostics["recovery_success"])

    def test_checked_in_config_freezes_500_validation_perturbations(self) -> None:
        path = (
            Path(__file__).parents[1]
            / "configs"
            / "data"
            / "calendar_robustness_validation_v1.json"
        )
        config = load_robustness_config(path)
        self.assertEqual(config.source_split, "validation")
        self.assertEqual(config.count_per_kind, 50)
        self.assertEqual(config.size, 500)

    def test_checked_in_config_freezes_500_test_perturbations(self) -> None:
        path = (
            Path(__file__).parents[1]
            / "configs"
            / "data"
            / "calendar_robustness_test_v1.json"
        )
        config = load_robustness_config(path)
        self.assertEqual(config.source_split, "test")
        self.assertEqual(config.count_per_kind, 50)
        self.assertEqual(config.size, 500)

    def test_paired_robustness_report_has_zero_oracle_gap(self) -> None:
        source = self._source_tasks()
        tasks = generate_robustness_tasks(self._config(), source)
        clean_report = evaluate_dataset(source, run_policy(source, OraclePolicy(), max_steps=4))
        robust_report = evaluate_dataset(tasks, run_policy(tasks, OraclePolicy(), max_steps=4))
        report = compute_robustness_report(
            [item.to_dict() for item in clean_report.task_evaluations],
            tasks,
            [item.to_dict() for item in robust_report.task_evaluations],
        )
        self.assertEqual(report["pair_count"], 20)
        self.assertEqual(report["overall"]["robustness_gap"], 0.0)
        self.assertEqual(set(report["settings"]), set(PERTURBATION_KINDS))
        for setting in report["settings"].values():
            self.assertEqual(setting["clean_task_success"]["value"], 1.0)
            self.assertEqual(setting["perturbed_task_success"]["value"], 1.0)

    def test_cross_model_comparison_rejects_different_task_snapshots(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_paths = []
            for index, task_hash in enumerate(("same", "different")):
                run_dir = root / f"run-{index}"
                run_dir.mkdir()
                (run_dir / "config.json").write_text(
                    json.dumps({"task_snapshot_sha256": task_hash, "task_count": 10}),
                    encoding="utf-8",
                )
                (run_dir / "metrics.json").write_text(
                    json.dumps({"metrics": {"task_success_rate": {"value": 0.5}}}),
                    encoding="utf-8",
                )
                (run_dir / "failure_stats.json").write_text(
                    json.dumps({"failure_counts": {}}), encoding="utf-8"
                )
                setting = {
                    "pair_count": 1,
                    "clean_task_success": {"value": 1.0},
                    "perturbed_task_success": {"value": 0.5},
                    "robustness_gap": 0.5,
                }
                (run_dir / "robustness_gap.json").write_text(
                    json.dumps(
                        {
                            "pair_count": 10,
                            "overall": {**setting, "pair_count": 10},
                            "settings": {kind: setting for kind in PERTURBATION_KINDS},
                        }
                    ),
                    encoding="utf-8",
                )
                run_paths.append(run_dir)
            with self.assertRaisesRegex(ValueError, "different Robust task snapshots"):
                compare_robustness_runs(
                    [("Base", run_paths[0]), ("SFT", run_paths[1])]
                )

    def test_cross_model_comparison_normalizes_serialized_setting_order(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_paths = []
            for index in range(2):
                run_dir = root / f"run-{index}"
                run_dir.mkdir()
                (run_dir / "config.json").write_text(
                    json.dumps({"task_snapshot_sha256": "same", "task_count": 10}),
                    encoding="utf-8",
                )
                (run_dir / "metrics.json").write_text(
                    json.dumps({"metrics": {"task_success_rate": {"value": 0.5}}}),
                    encoding="utf-8",
                )
                (run_dir / "failure_stats.json").write_text(
                    json.dumps({"failure_counts": {}}), encoding="utf-8"
                )
                setting = {
                    "pair_count": 1,
                    "clean_task_success": {"value": 1.0},
                    "perturbed_task_success": {"value": 0.5},
                    "robustness_gap": 0.5,
                }
                reversed_settings = {
                    kind: setting for kind in reversed(PERTURBATION_KINDS)
                }
                (run_dir / "robustness_gap.json").write_text(
                    json.dumps(
                        {
                            "pair_count": 10,
                            "overall": {**setting, "pair_count": 10},
                            "settings": reversed_settings,
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                run_paths.append(run_dir)
            comparison = compare_robustness_runs(
                [("Base", run_paths[0]), ("SFT", run_paths[1])]
            )
            self.assertEqual(comparison["perturbation_order"], list(PERTURBATION_KINDS))


if __name__ == "__main__":
    unittest.main()
