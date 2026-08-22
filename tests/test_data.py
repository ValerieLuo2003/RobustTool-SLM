from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from robust_tool.data.converter_swift import (
    assert_disjoint_splits,
    convert_trajectories_to_swift,
)
from robust_tool.data.generator import generate_calendar_toy_tasks, write_calendar_toy_dataset
from robust_tool.data.schemas import load_tasks
from robust_tool.models.config import load_model_config
from robust_tool.rollout.runner import OraclePolicy, run_policy


class DatasetTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
