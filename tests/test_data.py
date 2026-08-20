from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from robust_tool.data.generator import generate_calendar_toy_tasks, write_calendar_toy_dataset
from robust_tool.data.schemas import load_tasks


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


if __name__ == "__main__":
    unittest.main()
