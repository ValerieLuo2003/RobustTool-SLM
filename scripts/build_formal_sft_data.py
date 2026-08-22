#!/usr/bin/env python
"""Convert the audited formal Train/Validation tasks to ms-swift Agent trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.converter_swift import (
    assert_disjoint_splits,
    convert_trajectories_to_swift,
    write_swift_records,
)
from robust_tool.data.formal_generator import (
    audit_formal_splits,
    load_formal_dataset_config,
)
from robust_tool.data.schemas import load_tasks
from robust_tool.eval.evaluator import evaluate_dataset
from robust_tool.models.config import load_model_config
from robust_tool.rollout.runner import OraclePolicy, run_policy
from robust_tool.rollout.trajectory import write_trajectories


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oracle_report(tasks: list, trajectories: list) -> dict[str, object]:
    report = evaluate_dataset(tasks, trajectories)
    failed = [item.task_id for item in report.task_evaluations if not item.success]
    if failed:
        raise RuntimeError(f"refusing to build SFT data: Oracle failed {len(failed)} tasks")
    return {
        "task_count": report.task_count,
        "metrics": {name: metric.to_dict() for name, metric in report.metrics.items()},
        "failure_counts": dict(report.failure_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "calendar_formal_sft_v1.json",
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "calendar_formal_v1" / "tasks",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models" / "qwen2_5_1_5b_instruct.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "calendar_formal_v1" / "swift",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()
    if args.max_steps < 3:
        parser.error("--max-steps must be at least 3")
    required = {
        "dataset_config": args.dataset_config,
        "model_config": args.model_config,
        "train": args.task_dir / "train.jsonl",
        "validation": args.task_dir / "validation.jsonl",
        "test": args.task_dir / "test.jsonl",
        "source_manifest": args.task_dir / "manifest.json",
    }
    for name, path in required.items():
        if not path.exists():
            parser.error(f"required {name} file does not exist: {path}")

    dataset_config = load_formal_dataset_config(args.dataset_config)
    split_tasks = {
        split: load_tasks(required[split]) for split in ("train", "validation", "test")
    }
    assert_disjoint_splits(split_tasks)
    generation_audit = audit_formal_splits(split_tasks, dataset_config)
    system_prompt = load_model_config(args.model_config).system_prompt

    train_trajectories = run_policy(
        split_tasks["train"], OraclePolicy(), max_steps=args.max_steps
    )
    validation_trajectories = run_policy(
        split_tasks["validation"], OraclePolicy(), max_steps=args.max_steps
    )
    oracle_audit = {
        "train": _oracle_report(split_tasks["train"], train_trajectories),
        "validation": _oracle_report(split_tasks["validation"], validation_trajectories),
    }
    train_records = convert_trajectories_to_swift(
        split_tasks["train"], train_trajectories, system_prompt=system_prompt
    )
    validation_records = convert_trajectories_to_swift(
        split_tasks["validation"], validation_trajectories, system_prompt=system_prompt
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output / "train.jsonl",
        "validation": output / "validation.jsonl",
        "train_trajectories": output / "train_trajectories.jsonl",
        "validation_trajectories": output / "validation_trajectories.jsonl",
    }
    write_swift_records(train_records, paths["train"])
    write_swift_records(validation_records, paths["validation"])
    write_trajectories(train_trajectories, paths["train_trajectories"])
    write_trajectories(validation_trajectories, paths["validation_trajectories"])

    manifest = {
        "dataset_name": dataset_config.dataset_name,
        "purpose": "formal_sft_v1",
        "dataset_config": dataset_config.to_dict(),
        "dataset_config_sha256": _sha256(args.dataset_config),
        "model_config_sha256": _sha256(args.model_config),
        "source_manifest_sha256": _sha256(required["source_manifest"]),
        "test_usage": "split isolation and hash verification only; never converted to SFT",
        "generation_audit": generation_audit,
        "oracle_audit": oracle_audit,
        "counts": {"train": len(train_records), "validation": len(validation_records)},
        "category_counts": {
            split: dict(Counter(str(task.metadata["category"]) for task in split_tasks[split]))
            for split in ("train", "validation")
        },
        "source_task_files": {
            split: {"path": str(required[split].resolve()), "sha256": _sha256(required[split])}
            for split in ("train", "validation", "test")
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
        },
        "train_task_ids": [task.task_id for task in split_tasks["train"]],
        "validation_task_ids": [task.task_id for task in split_tasks["validation"]],
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
                "test_converted": False,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
