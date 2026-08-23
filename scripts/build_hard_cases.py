#!/usr/bin/env python
"""Build and Oracle-audit train-only hard cases from SFT Validation failure labels."""

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
from robust_tool.data.hard_cases import (
    audit_failure_aware_tasks,
    generate_failure_aware_tasks,
    load_failure_aware_config,
    load_failure_target_selection,
)
from robust_tool.data.schemas import load_tasks, write_tasks
from robust_tool.eval.evaluator import evaluate_dataset
from robust_tool.models.config import load_model_config
from robust_tool.rollout.runner import OraclePolicy, run_policy
from robust_tool.rollout.trajectory import write_trajectories


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "calendar_failure_aware_v1.json",
    )
    parser.add_argument(
        "--failure-targets",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "qwen2_5_1_5b_sft_formal_v1_validation_new3090"
            / "failure_targets.json"
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
        default=PROJECT_ROOT / "data" / "processed" / "calendar_failure_aware_v1",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()
    if args.max_steps < 3:
        parser.error("--max-steps must be at least 3 for two-call tasks plus the final answer")

    source_paths = {
        split: args.source_task_dir / f"{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    required = {
        "config": args.config,
        "failure_targets": args.failure_targets,
        "model_config": args.model_config,
        "source_manifest": args.source_task_dir / "manifest.json",
        **{f"source_{split}": path for split, path in source_paths.items()},
    }
    for name, path in required.items():
        if not path.exists():
            parser.error(f"required {name} file does not exist: {path}")

    config = load_failure_aware_config(args.config)
    selection = load_failure_target_selection(args.failure_targets)
    source_splits = {split: load_tasks(path) for split, path in source_paths.items()}
    if selection.validation_task_count != len(source_splits["validation"]):
        parser.error(
            "failure target Validation count does not match the frozen source split: "
            f"{selection.validation_task_count} != {len(source_splits['validation'])}"
        )
    validation_hash = _sha256(source_paths["validation"])
    if selection.task_snapshot_sha256 != validation_hash:
        parser.error(
            "failure target task snapshot does not match frozen Validation: "
            f"{selection.task_snapshot_sha256} != {validation_hash}"
        )

    tasks = generate_failure_aware_tasks(config, selection)
    generation_audit = audit_failure_aware_tasks(tasks, config, selection, source_splits)
    trajectories = run_policy(tasks, OraclePolicy(), max_steps=args.max_steps)
    report = evaluate_dataset(tasks, trajectories)
    failed = [item.task_id for item in report.task_evaluations if not item.success]
    if failed:
        raise RuntimeError(
            f"refusing to write hard cases: Oracle failed {len(failed)} tasks ({', '.join(failed[:10])})"
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
    selection_record = _json_object(args.failure_targets)
    manifest = {
        "dataset_name": config.dataset_name,
        "purpose": "failure-aware Train augmentation; no Validation or Test targets",
        "config": config.to_dict(),
        "config_sha256": _sha256(args.config),
        "failure_target_selection": selection_record,
        "failure_targets_sha256": _sha256(args.failure_targets),
        "generation_audit": generation_audit,
        "oracle_audit": {
            "task_count": report.task_count,
            "metrics": {name: metric.to_dict() for name, metric in report.metrics.items()},
            "failure_counts": dict(report.failure_counts),
        },
        "source_data_usage": {
            "train": "collision audit only; no source task is copied or rewritten",
            "validation": "failure labels/ranks/rates plus collision audit only; no failed task text is copied",
            "test": "hash and collision audit only; never used to choose or construct a hard case",
        },
        "source_task_files": {
            split: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for split, path in source_paths.items()
        },
        "source_manifest_sha256": _sha256(required["source_manifest"]),
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
                "target_counts": generation_audit["target_counts"],
                "family_counts": generation_audit["family_counts"],
                "oracle_task_success_rate": report.metrics["task_success_rate"].value,
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
