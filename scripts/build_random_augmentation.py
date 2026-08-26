#!/usr/bin/env python
"""Build and Oracle-audit the matched-source random augmentation control."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.converter_swift import convert_trajectories_to_swift, write_swift_records
from robust_tool.data.random_augmentation import (
    audit_random_augmentation_tasks,
    generate_random_augmentation_tasks,
    load_random_augmentation_config,
)
from robust_tool.data.recovery_cases import load_recovery_config
from robust_tool.data.schemas import load_tasks, write_tasks
from robust_tool.eval.evaluator import evaluate_dataset
from robust_tool.models.config import load_model_config
from robust_tool.rollout.runner import OraclePolicy, run_policy
from robust_tool.rollout.trajectory import write_trajectories


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT / "configs" / "data" / "calendar_random_augmentation_v2.json"
        ),
    )
    parser.add_argument(
        "--recovery-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "data"
            / "calendar_recovery_failure_aware_v2.json"
        ),
    )
    parser.add_argument(
        "--source-task-dir",
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
        default=PROJECT_ROOT / "data" / "processed" / "calendar_random_augmentation_v2",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()
    if args.max_steps < 2:
        parser.error("--max-steps must allow a tool call and final answer")

    source_paths = {
        split: args.source_task_dir / f"{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    required = [
        args.config,
        args.recovery_config,
        args.model_config,
        *source_paths.values(),
    ]
    for path in required:
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")

    config = load_random_augmentation_config(args.config)
    recovery_config = load_recovery_config(args.recovery_config)
    recovery_config_sha256 = _sha256(args.recovery_config)
    if config.matched_recovery_config_sha256 != recovery_config_sha256:
        parser.error(
            "matched Recovery config hash differs from the frozen random control: "
            f"{recovery_config_sha256}"
        )
    source_splits = {split: load_tasks(path) for split, path in source_paths.items()}
    tasks = generate_random_augmentation_tasks(
        config,
        recovery_config,
        source_splits["train"],
    )
    generation_audit = audit_random_augmentation_tasks(
        tasks,
        config,
        recovery_config,
        source_splits,
    )
    trajectories = run_policy(tasks, OraclePolicy(), max_steps=args.max_steps)
    report = evaluate_dataset(tasks, trajectories)
    failed = [item.task_id for item in report.task_evaluations if not item.success]
    if failed:
        raise RuntimeError(
            f"refusing to write random control: Oracle failed {len(failed)} tasks "
            f"({', '.join(failed[:10])})"
        )
    system_prompt = load_model_config(args.model_config).system_prompt
    swift_records = convert_trajectories_to_swift(
        tasks,
        trajectories,
        system_prompt=system_prompt,
    )

    output = args.output_dir.resolve()
    paths = {
        "tasks": output / "tasks" / "train.jsonl",
        "trajectories": output / "trajectories" / "train.jsonl",
        "swift": output / "swift" / "train.jsonl",
    }
    write_tasks(tasks, paths["tasks"])
    write_trajectories(trajectories, paths["trajectories"])
    write_swift_records(swift_records, paths["swift"])
    manifest = {
        "dataset_name": config.dataset_name,
        "purpose": "matched-source random Train augmentation control",
        "config": config.to_dict(),
        "config_sha256": _sha256(args.config),
        "matched_recovery_config": recovery_config.to_dict(),
        "matched_recovery_config_sha256": recovery_config_sha256,
        "generation_audit": generation_audit,
        "oracle_audit": {
            "task_count": report.task_count,
            "metrics": {
                name: metric.to_dict() for name, metric in report.metrics.items()
            },
            "failure_counts": dict(report.failure_counts),
        },
        "source_data_usage": {
            "train": "same source IDs as Recovery v2; neutral query rewrite only",
            "validation": "collision audit only; content is never transformed",
            "test": "collision audit only; content is never transformed",
        },
        "source_task_files": {
            split: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for split, path in source_paths.items()
        },
        "model_config_sha256": _sha256(args.model_config),
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "count": len(tasks),
                "control_type": generation_audit["control_type"],
                "unique_source_train_tasks": generation_audit[
                    "unique_source_train_tasks"
                ],
                "matched_recovery_target_counts": generation_audit[
                    "matched_recovery_target_counts"
                ],
                "failure_injection_count": generation_audit[
                    "failure_injection_count"
                ],
                "oracle_task_success_rate": report.metrics[
                    "task_success_rate"
                ].value,
                "source_overlap": generation_audit["source_overlap"],
                "validation_or_test_outputs": False,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
