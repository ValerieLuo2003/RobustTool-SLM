#!/usr/bin/env python
"""Build small, traceable Oracle Agent data for the SFT pipeline smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.converter_swift import (
    assert_disjoint_splits,
    convert_trajectories_to_swift,
    write_swift_records,
)
from robust_tool.data.schemas import load_tasks
from robust_tool.models.config import load_model_config
from robust_tool.rollout.runner import OraclePolicy, run_policy
from robust_tool.rollout.trajectory import write_trajectories


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tasks", type=Path, default=PROJECT_ROOT / "data" / "eval" / "toy_train.jsonl")
    parser.add_argument(
        "--validation-tasks",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "toy_validation.jsonl",
    )
    parser.add_argument("--test-tasks", type=Path, default=PROJECT_ROOT / "data" / "eval" / "clean_test.jsonl")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models" / "qwen2_5_1_5b_instruct.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "sft_smoke")
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()

    for path in (args.train_tasks, args.validation_tasks, args.test_tasks, args.model_config):
        if not path.exists():
            parser.error(f"required input does not exist: {path}")
    train_tasks = load_tasks(args.train_tasks)
    validation_tasks = load_tasks(args.validation_tasks)
    test_tasks = load_tasks(args.test_tasks)
    assert_disjoint_splits(
        {"train": train_tasks, "validation": validation_tasks, "test": test_tasks}
    )
    system_prompt = load_model_config(args.model_config).system_prompt

    train_trajectories = run_policy(train_tasks, OraclePolicy(), max_steps=args.max_steps)
    validation_trajectories = run_policy(
        validation_tasks,
        OraclePolicy(),
        max_steps=args.max_steps,
    )
    train_records = convert_trajectories_to_swift(
        train_tasks,
        train_trajectories,
        system_prompt=system_prompt,
    )
    validation_records = convert_trajectories_to_swift(
        validation_tasks,
        validation_trajectories,
        system_prompt=system_prompt,
    )

    output = args.output_dir.resolve()
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    train_trajectory_path = output / "train_trajectories.jsonl"
    validation_trajectory_path = output / "validation_trajectories.jsonl"
    write_swift_records(train_records, train_path)
    write_swift_records(validation_records, validation_path)
    write_trajectories(train_trajectories, train_trajectory_path)
    write_trajectories(validation_trajectories, validation_trajectory_path)

    manifest = {
        "dataset_name": "calendar-sft-smoke-v1",
        "purpose": "pipeline_smoke_only",
        "oracle_policy": "oracle",
        "max_steps": args.max_steps,
        "counts": {"train": len(train_records), "validation": len(validation_records)},
        "source_task_files": {
            "train": {"path": str(args.train_tasks.resolve()), "sha256": _sha256(args.train_tasks)},
            "validation": {
                "path": str(args.validation_tasks.resolve()),
                "sha256": _sha256(args.validation_tasks),
            },
            "test_for_leakage_check_only": {
                "path": str(args.test_tasks.resolve()),
                "sha256": _sha256(args.test_tasks),
            },
        },
        "outputs": {
            "train": {"path": str(train_path), "sha256": _sha256(train_path)},
            "validation": {"path": str(validation_path), "sha256": _sha256(validation_path)},
        },
        "task_ids": {
            "train": [task.task_id for task in train_tasks],
            "validation": [task.task_id for task in validation_tasks],
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "train_count": len(train_records),
                "validation_count": len(validation_records),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
