from __future__ import annotations

import tempfile
import unittest
import json
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


if __name__ == "__main__":
    unittest.main()
