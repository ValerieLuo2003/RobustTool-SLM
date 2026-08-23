from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robust_tool.training.checkpoints import select_best_checkpoint
from scripts.run_sft import _collect_metrics, _local_dataset_path


class CheckpointSelectionTests(unittest.TestCase):
    @staticmethod
    def _write_checkpoint(path: Path, payload: bytes) -> None:
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(payload)

    def test_selects_lowest_validation_loss_and_tracks_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            trainer = run_dir / "trainer_output" / "v0"
            trainer.mkdir(parents=True)
            records = [
                {"eval_loss": 0.2, "global_step/max_steps": "250/750"},
                {"eval_loss": 0.3, "global_step/max_steps": "500/750"},
                {"eval_loss": 0.1, "global_step/max_steps": "750/750"},
            ]
            (trainer / "logging.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            self._write_checkpoint(
                run_dir / "checkpoint_candidates" / "checkpoint-250", b"early"
            )
            self._write_checkpoint(trainer / "checkpoint-500", b"middle")

            selected = select_best_checkpoint(run_dir)

            self.assertEqual(selected["selected_step"], 250)
            self.assertEqual(selected["selected_eval_loss"], 0.2)
            self.assertTrue(selected["selected_checkpoint"].endswith("checkpoint-250"))
            self.assertEqual(
                [entry["checkpoint_available"] for entry in selected["evaluations"]],
                [True, True, False],
            )
            self.assertEqual(selected["adapter_weight_bytes"], len(b"early"))

    def test_rejects_logs_without_matching_complete_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            trainer = run_dir / "trainer_output" / "v0"
            trainer.mkdir(parents=True)
            (trainer / "logging.jsonl").write_text(
                json.dumps({"eval_loss": 0.1, "step": 10}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "no complete LoRA checkpoint"):
                select_best_checkpoint(run_dir)

    def test_sft_metrics_use_final_logging_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "v0" / "checkpoint-20"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": 20, "log_history": []}),
                encoding="utf-8",
            )
            records = [
                {"loss": 0.2, "global_step/max_steps": "20/20"},
                {"eval_loss": 0.1, "global_step/max_steps": "20/20"},
                {"train_runtime": 12.5, "train_loss": 0.3},
            ]
            (output / "v0" / "logging.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            metrics = _collect_metrics(output, return_code=0)

            self.assertTrue(metrics["completed"])
            self.assertEqual(metrics["last_eval_loss"], 0.1)
            self.assertEqual(metrics["eval_history"], [records[1]])
            self.assertEqual(metrics["train_summary"], records[2])

    def test_sft_preflight_understands_swift_local_sample_suffix(self) -> None:
        self.assertEqual(
            _local_dataset_path("data/processed/train.jsonl#64"),
            Path("data/processed/train.jsonl"),
        )
        self.assertEqual(
            _local_dataset_path("data/processed/train.jsonl"),
            Path("data/processed/train.jsonl"),
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _local_dataset_path("data/processed/train.jsonl#0")


if __name__ == "__main__":
    unittest.main()
