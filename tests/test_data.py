from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from robust_tool.data.converter_swift import (
    assert_disjoint_splits,
    convert_trajectories_to_swift,
)
from robust_tool.data.formal_generator import (
    FORMAL_CATEGORIES,
    FormalDatasetConfig,
    generate_calendar_formal_splits,
    load_formal_dataset_config,
)
from robust_tool.data.hard_cases import (
    FailureAwareDatasetConfig,
    FailureTargetSelection,
    audit_failure_aware_tasks,
    generate_failure_aware_tasks,
    load_failure_aware_config,
)
from robust_tool.data.recovery_cases import (
    RecoveryDatasetConfig,
    audit_recovery_tasks,
    generate_recovery_tasks,
    load_recovery_config,
)
from robust_tool.data.random_augmentation import (
    MATCHED_SOURCE_CONTROL,
    RANDOM_AUGMENTATION_VERSION,
    RandomAugmentationConfig,
    audit_random_augmentation_tasks,
    generate_random_augmentation_tasks,
    load_random_augmentation_config,
)
from robust_tool.data.generator import generate_calendar_toy_tasks, write_calendar_toy_dataset
from robust_tool.eval.evaluator import evaluate_dataset
from robust_tool.data.schemas import load_tasks
from robust_tool.models.config import load_model_config
from robust_tool.rollout.runner import OraclePolicy, run_policy


class DatasetTests(unittest.TestCase):
    @staticmethod
    def _small_formal_config(seed: int = 123) -> FormalDatasetConfig:
        return FormalDatasetConfig(
            dataset_name="calendar-formal-test",
            generator_version="calendar-formal-sft-v1",
            seed=seed,
            split_category_counts={
                split: {category: 2 for category in FORMAL_CATEGORIES}
                for split in ("train", "validation", "test")
            },
        )

    @staticmethod
    def _failure_selection() -> FailureTargetSelection:
        return FailureTargetSelection.from_dict(
            {
                "selection_protocol": "top failures from SFT Validation only",
                "source_run": "validation-run",
                "source_git_commit": "abc123",
                "task_snapshot_sha256": "a" * 64,
                "validation_task_count": 5,
                "selected_failures": [
                    {"rank": 1, "failure": "wrong_argument_value", "count": 28},
                    {"rank": 2, "failure": "ignore_tool_result", "count": 15},
                    {"rank": 3, "failure": "missing_argument", "count": 11},
                ],
            }
        )

    @staticmethod
    def _small_failure_config(seed: int = 456) -> FailureAwareDatasetConfig:
        return FailureAwareDatasetConfig(
            dataset_name="calendar-failure-test",
            generator_version="calendar-failure-aware-v1",
            seed=seed,
            target_counts={
                "wrong_argument_value": 20,
                "ignore_tool_result": 20,
                "missing_argument": 20,
            },
        )

    @staticmethod
    def _small_recovery_config(seed: int = 789) -> RecoveryDatasetConfig:
        return RecoveryDatasetConfig(
            dataset_name="calendar-recovery-test",
            generator_version="calendar-recovery-failure-aware-v2",
            seed=seed,
            source_robust_validation_sha256="b" * 64,
            target_counts={
                "missing_tool": 2,
                "tool_failure": 2,
                "partial_tool_response": 2,
            },
        )

    @staticmethod
    def _small_random_config(seed: int = 789) -> RandomAugmentationConfig:
        return RandomAugmentationConfig(
            dataset_name="calendar-random-control-test",
            generator_version=RANDOM_AUGMENTATION_VERSION,
            seed=seed,
            size=6,
            control_type=MATCHED_SOURCE_CONTROL,
            matched_recovery_config_sha256="c" * 64,
        )

    def test_generator_has_fixed_size_unique_ids_and_strict_splits(self) -> None:
        tasks = generate_calendar_toy_tasks(seed=123)
        self.assertEqual(len(tasks), 25)
        self.assertEqual(len({task.task_id for task in tasks}), 25)
        counts = {
            split: sum(task.metadata["split"] == split for task in tasks)
            for split in ("train", "validation", "test")
        }
        self.assertEqual(counts, {"train": 15, "validation": 5, "test": 5})
        split_ids = {
            split: {task.task_id for task in tasks if task.metadata["split"] == split}
            for split in counts
        }
        self.assertFalse(split_ids["train"] & split_ids["validation"])
        self.assertFalse(split_ids["train"] & split_ids["test"])
        self.assertFalse(split_ids["validation"] & split_ids["test"])

    def test_generation_is_deterministic_for_same_seed(self) -> None:
        left = [task.to_dict() for task in generate_calendar_toy_tasks(seed=99)]
        right = [task.to_dict() for task in generate_calendar_toy_tasks(seed=99)]
        self.assertEqual(left, right)

    def test_recovery_generator_is_deterministic_train_only_and_disjoint(self) -> None:
        splits = generate_calendar_formal_splits(self._small_formal_config())
        config = self._small_recovery_config()
        left = generate_recovery_tasks(config, splits["train"])
        right = generate_recovery_tasks(config, splits["train"])
        self.assertEqual([task.to_dict() for task in left], [task.to_dict() for task in right])
        audit = audit_recovery_tasks(left, config, splits)
        self.assertEqual(audit["count"], 6)
        self.assertEqual(audit["unique_source_train_tasks"], 6)
        self.assertEqual(audit["source_overlap"], {"task_ids": 0, "normalized_user_queries": 0})
        self.assertFalse(audit["validation_or_test_content_used_for_generation"])
        self.assertTrue(all(task.metadata["split"] == "train" for task in left))

    def test_recovery_oracle_executes_retry_and_missing_tool_families(self) -> None:
        splits = generate_calendar_formal_splits(self._small_formal_config())
        tasks = generate_recovery_tasks(self._small_recovery_config(), splits["train"])
        report = evaluate_dataset(tasks, run_policy(tasks, OraclePolicy(), max_steps=4))
        self.assertEqual(report.metrics["task_success_rate"].value, 1.0)
        self.assertEqual(report.metrics["recovery_success_rate"].value, 1.0)
        self.assertFalse(report.failure_counts)
        by_target = {
            target: [
                task
                for task in tasks
                if task.metadata["target_robustness"] == target
            ]
            for target in ("missing_tool", "tool_failure", "partial_tool_response")
        }
        self.assertTrue(
            all(task.expected_action == "respond" for task in by_target["missing_tool"])
        )
        self.assertTrue(
            all(len(task.reference_calls) == 2 for task in by_target["tool_failure"])
        )
        self.assertTrue(
            all(
                len(task.reference_calls) == 2
                for task in by_target["partial_tool_response"]
            )
        )

    def test_checked_in_recovery_configs_freeze_smoke_and_formal_scale(self) -> None:
        root = Path(__file__).parents[1] / "configs" / "data"
        smoke = load_recovery_config(
            root / "calendar_recovery_failure_aware_v2_smoke.json"
        )
        formal = load_recovery_config(root / "calendar_recovery_failure_aware_v2.json")
        self.assertEqual(smoke.size, 12)
        self.assertEqual(formal.size, 3000)
        self.assertEqual(set(formal.target_counts), set(smoke.target_counts))
        self.assertEqual(
            formal.source_robust_validation_sha256,
            "a62c76019ad935f58d915e096cad38f02fca8701cb702be1a61fe2a7f7c9f18e",
        )

    def test_random_control_is_deterministic_and_matches_recovery_sources(self) -> None:
        splits = generate_calendar_formal_splits(self._small_formal_config())
        recovery_config = self._small_recovery_config()
        random_config = self._small_random_config()
        left = generate_random_augmentation_tasks(
            random_config,
            recovery_config,
            splits["train"],
        )
        right = generate_random_augmentation_tasks(
            random_config,
            recovery_config,
            splits["train"],
        )
        self.assertEqual([task.to_dict() for task in left], [task.to_dict() for task in right])
        audit = audit_random_augmentation_tasks(
            left,
            random_config,
            recovery_config,
            splits,
        )
        self.assertEqual(audit["count"], 6)
        self.assertEqual(audit["unique_source_train_tasks"], 6)
        self.assertEqual(audit["failure_injection_count"], 0)
        self.assertEqual(audit["source_overlap"], {"task_ids": 0, "normalized_user_queries": 0})

    def test_random_control_oracle_succeeds_without_recovery_faults(self) -> None:
        splits = generate_calendar_formal_splits(self._small_formal_config())
        tasks = generate_random_augmentation_tasks(
            self._small_random_config(),
            self._small_recovery_config(),
            splits["train"],
        )
        report = evaluate_dataset(tasks, run_policy(tasks, OraclePolicy(), max_steps=4))
        self.assertEqual(report.metrics["task_success_rate"].value, 1.0)
        self.assertIsNone(report.metrics["recovery_success_rate"].value)
        self.assertFalse(report.failure_counts)
        self.assertTrue(all("robustness" not in task.metadata for task in tasks))

    def test_checked_in_random_control_configs_match_recovery_scale_and_hash(self) -> None:
        root = Path(__file__).parents[1]
        data_configs = root / "configs" / "data"
        pairs = (
            (
                "calendar_random_augmentation_v2_smoke.json",
                "calendar_recovery_failure_aware_v2_smoke.json",
            ),
            (
                "calendar_random_augmentation_v2.json",
                "calendar_recovery_failure_aware_v2.json",
            ),
        )
        for random_name, recovery_name in pairs:
            random_config = load_random_augmentation_config(data_configs / random_name)
            recovery_config = load_recovery_config(data_configs / recovery_name)
            recovery_hash = hashlib.sha256(
                (data_configs / recovery_name).read_bytes()
            ).hexdigest()
            self.assertEqual(random_config.size, recovery_config.size)
            self.assertEqual(
                random_config.matched_recovery_config_sha256,
                recovery_hash,
            )

    def test_dated_queries_make_reference_year_explicit(self) -> None:
        tasks = generate_calendar_toy_tasks()
        for task in tasks:
            reference_values = [
                value
                for call in task.reference_calls
                for value in call.arguments.values()
                if isinstance(value, str) and value.startswith("2026-")
            ]
            if reference_values:
                self.assertIn("2026", task.user_query, task.task_id)

    def test_written_clean_test_matches_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_calendar_toy_dataset(Path(directory), seed=5)
            test = [task.to_dict() for task in load_tasks(paths["test"])]
            clean = [task.to_dict() for task in load_tasks(paths["clean_test"])]
            self.assertEqual(test, clean)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"], {"train": 15, "validation": 5, "test": 5})
            self.assertEqual(
                manifest["files"]["test"]["sha256"],
                manifest["files"]["clean_test"]["sha256"],
            )

    def test_swift_agent_records_are_json_string_encoded_without_gold_leakage(self) -> None:
        tasks = generate_calendar_toy_tasks()[:2]
        config_path = Path(__file__).parents[1] / "configs" / "models" / "qwen2_5_1_5b_instruct.json"
        system_prompt = load_model_config(config_path).system_prompt
        records = convert_trajectories_to_swift(
            tasks,
            run_policy(tasks, OraclePolicy()),
            system_prompt=system_prompt,
        )
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertIsInstance(record["tools"], str)
            self.assertIsInstance(json.loads(record["tools"]), list)
            roles = [message["role"] for message in record["messages"]]
            self.assertIn("tool_call", roles)
            self.assertIn("tool_response", roles)
            serialized = json.dumps(record, ensure_ascii=False)
            self.assertNotIn("goal_state", serialized)
            self.assertNotIn("reference_calls", serialized)
            for message in record["messages"]:
                if message["role"] in {"tool_call", "tool_response"}:
                    self.assertIsInstance(json.loads(message["content"]), dict)

    def test_split_overlap_is_rejected_before_sft_conversion(self) -> None:
        task = generate_calendar_toy_tasks()[0]
        with self.assertRaisesRegex(ValueError, "appears in both"):
            assert_disjoint_splits({"train": [task], "validation": [task]})

    def test_formal_generator_is_deterministic_and_globally_deduplicated(self) -> None:
        config = self._small_formal_config()
        left = generate_calendar_formal_splits(config)
        right = generate_calendar_formal_splits(config)
        self.assertEqual(
            {split: [task.to_dict() for task in tasks] for split, tasks in left.items()},
            {split: [task.to_dict() for task in tasks] for split, tasks in right.items()},
        )
        all_tasks = [task for tasks in left.values() for task in tasks]
        self.assertEqual(len({task.task_id for task in all_tasks}), len(all_tasks))
        self.assertEqual(
            len({" ".join(task.user_query.casefold().split()) for task in all_tasks}),
            len(all_tasks),
        )

    def test_formal_generator_covers_decisions_and_state_dependent_multi_step(self) -> None:
        splits = generate_calendar_formal_splits(self._small_formal_config())
        for split, tasks in splits.items():
            categories = {task.metadata["category"] for task in tasks}
            self.assertEqual(categories, set(FORMAL_CATEGORIES), split)
            actions = {task.expected_action for task in tasks}
            self.assertEqual(actions, {"call", "clarify", "respond"}, split)
            multi_step = [task for task in tasks if task.metadata["category"] == "multi_step"]
            self.assertTrue(all(len(task.reference_calls) == 2 for task in multi_step))
            self.assertTrue(all(task.goal_state.get("required_observations") for task in multi_step))
            self.assertGreater(len({task.available_tools for task in tasks}), 1)

    def test_formal_oracle_completes_all_generated_task_types(self) -> None:
        tasks = generate_calendar_formal_splits(self._small_formal_config())["train"]
        trajectories = run_policy(tasks, OraclePolicy(), max_steps=4)
        report = evaluate_dataset(tasks, trajectories)
        self.assertEqual(report.metrics["task_success_rate"].value, 1.0)
        self.assertEqual(report.metrics["multi_turn_task_success_rate"].value, 1.0)
        self.assertFalse(report.failure_counts)
        for trajectory in trajectories:
            self.assertTrue(trajectory.final_answer())

    def test_checked_in_formal_config_freezes_requested_scale(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "data" / "calendar_formal_sft_v1.json"
        config = load_formal_dataset_config(path)
        self.assertEqual(config.split_size("train"), 6000)
        self.assertEqual(config.split_size("validation"), 500)
        self.assertEqual(config.split_size("test"), 1000)
        self.assertEqual(config.split_category_counts["train"]["multi_step"], 450)
        self.assertEqual(config.split_category_counts["train"]["clarify"], 450)
        self.assertEqual(config.split_category_counts["train"]["no_tool"], 300)

    def test_formal_sft_config_is_epoch_based_and_uses_only_train_validation(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "sft" / "qwen2_5_1_5b_lora_formal_v1.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("max_steps", config)
        self.assertEqual(config["num_train_epochs"], 1)
        self.assertEqual(config["tuner_type"], "lora")
        self.assertEqual(config["lora_rank"], 16)
        self.assertEqual(config["gradient_accumulation_steps"], 8)
        self.assertEqual(config["dataset"], ["data/processed/calendar_formal_v1/swift/train.jsonl"])
        self.assertEqual(
            config["val_dataset"],
            ["data/processed/calendar_formal_v1/swift/validation.jsonl"],
        )
        serialized = json.dumps(config)
        self.assertNotIn("test.jsonl", serialized)
        self.assertNotIn("clean_test", serialized)

    def test_failure_aware_generator_is_deterministic_train_only_and_disjoint(self) -> None:
        config = self._small_failure_config()
        selection = self._failure_selection()
        left = generate_failure_aware_tasks(config, selection)
        right = generate_failure_aware_tasks(config, selection)
        self.assertEqual([task.to_dict() for task in left], [task.to_dict() for task in right])
        source = generate_calendar_toy_tasks()
        source_splits = {
            split: [task for task in source if task.metadata["split"] == split]
            for split in ("train", "validation", "test")
        }
        audit = audit_failure_aware_tasks(left, config, selection, source_splits)
        self.assertEqual(audit["count"], 60)
        self.assertEqual(audit["source_overlap"], {"task_ids": 0, "normalized_user_queries": 0})
        self.assertFalse(audit["test_content_used_for_generation"])
        self.assertTrue(all(task.metadata["split"] == "train" for task in left))
        self.assertTrue(all(task.metadata["target_failure"] in task.failure_tags for task in left))

    def test_failure_aware_oracle_executes_every_target_family(self) -> None:
        tasks = generate_failure_aware_tasks(
            self._small_failure_config(),
            self._failure_selection(),
        )
        report = evaluate_dataset(tasks, run_policy(tasks, OraclePolicy(), max_steps=4))
        self.assertEqual(report.metrics["task_success_rate"].value, 1.0)
        self.assertEqual(report.metrics["multi_turn_task_success_rate"].value, 1.0)
        self.assertFalse(report.failure_counts)
        targets = {task.metadata["target_failure"] for task in tasks}
        self.assertEqual(
            targets,
            {"wrong_argument_value", "ignore_tool_result", "missing_argument"},
        )
        dependencies = [
            task for task in tasks if task.metadata["target_failure"] == "ignore_tool_result"
        ]
        self.assertTrue(any(len(task.reference_calls) == 2 for task in dependencies))
        missing = [task for task in tasks if task.metadata["target_failure"] == "missing_argument"]
        self.assertTrue(any(task.expected_action == "clarify" for task in missing))

    def test_failure_aware_audit_rejects_source_overlap(self) -> None:
        config = self._small_failure_config()
        selection = self._failure_selection()
        tasks = generate_failure_aware_tasks(config, selection)
        with self.assertRaisesRegex(ValueError, "overlap source task IDs"):
            audit_failure_aware_tasks(
                tasks,
                config,
                selection,
                {"train": [tasks[0]], "validation": [], "test": []},
            )

    def test_failure_aware_targets_must_match_validation_selection(self) -> None:
        selection = self._failure_selection()
        config = FailureAwareDatasetConfig(
            dataset_name="mismatch",
            generator_version="calendar-failure-aware-v1",
            seed=1,
            target_counts={"wrong_argument_value": 2},
        )
        with self.assertRaisesRegex(ValueError, "must exactly match"):
            generate_failure_aware_tasks(config, selection)

    def test_checked_in_failure_aware_config_freezes_targeted_scale(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "data" / "calendar_failure_aware_v1.json"
        config = load_failure_aware_config(path)
        self.assertEqual(config.size, 3000)
        self.assertEqual(
            config.target_counts,
            {
                "wrong_argument_value": 1200,
                "ignore_tool_result": 1000,
                "missing_argument": 800,
            },
        )

    def test_failure_sft_config_is_equal_scale_augmentation_from_frozen_base(self) -> None:
        root = Path(__file__).parents[1]
        formal = json.loads(
            (root / "configs" / "sft" / "qwen2_5_1_5b_lora_formal_v1.json").read_text(
                encoding="utf-8"
            )
        )
        failure = json.loads(
            (root / "configs" / "sft" / "qwen2_5_1_5b_lora_failure_aware_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            failure["dataset"],
            [
                "data/processed/calendar_formal_v1/swift/train.jsonl",
                "data/processed/calendar_failure_aware_v1/swift/train.jsonl",
            ],
        )
        self.assertEqual(failure["val_dataset"], formal["val_dataset"])
        for key in (
            "model",
            "model_revision",
            "tuner_type",
            "target_modules",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "learning_rate",
            "num_train_epochs",
            "gradient_accumulation_steps",
            "max_length",
            "seed",
            "data_seed",
        ):
            self.assertEqual(failure[key], formal[key], key)
        serialized = json.dumps(failure)
        self.assertNotIn("adapter_path", serialized)
        self.assertNotIn("resume_from_checkpoint", serialized)
        self.assertNotIn("test.jsonl", serialized)

    def test_failure_sft_smoke_uses_bounded_samples_of_both_train_sources(self) -> None:
        path = (
            Path(__file__).parents[1]
            / "configs"
            / "sft"
            / "qwen2_5_1_5b_lora_failure_aware_smoke.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["max_steps"], 20)
        self.assertEqual(config["lora_rank"], 16)
        self.assertEqual(config["gradient_accumulation_steps"], 8)
        self.assertEqual(len(config["dataset"]), 2)
        self.assertTrue(all(path.endswith("#64") for path in config["dataset"]))
        self.assertEqual(config["val_dataset"], [
            "data/processed/calendar_formal_v1/swift/validation.jsonl#32"
        ])


if __name__ == "__main__":
    unittest.main()
