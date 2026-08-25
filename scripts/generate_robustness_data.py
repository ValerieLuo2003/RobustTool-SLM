#!/usr/bin/env python
"""Generate and Oracle-audit paired Calendar Robustness Validation tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.perturb import (
    audit_robustness_tasks,
    generate_robustness_tasks,
    load_robustness_config,
)
from robust_tool.data.schemas import load_tasks, write_tasks
from robust_tool.eval.evaluator import evaluate_dataset
from robust_tool.rollout.runner import OraclePolicy, run_policy
from robust_tool.rollout.trajectory import write_trajectories


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "calendar_robustness_validation_v1.json",
    )
    parser.add_argument(
        "--source-tasks",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "calendar_formal_v1"
            / "tasks"
            / "validation.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "calendar_robustness_validation_v1",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()
    for name, path in {"config": args.config, "source_tasks": args.source_tasks}.items():
        if not path.exists():
            parser.error(f"required {name} file does not exist: {path}")
    if args.max_steps < 3:
        parser.error("--max-steps must be at least 3 for retry plus final answer")

    config = load_robustness_config(args.config)
    source_tasks = load_tasks(args.source_tasks)
    tasks = generate_robustness_tasks(config, source_tasks)
    generation_audit = audit_robustness_tasks(tasks, config, source_tasks)
    trajectories = run_policy(tasks, OraclePolicy(), max_steps=args.max_steps)
    report = evaluate_dataset(tasks, trajectories)
    failed = [item.task_id for item in report.task_evaluations if not item.success]
    if failed or report.failure_counts:
        raise RuntimeError(
            "refusing to write robustness data: Oracle audit failed; "
            f"failed={failed[:10]}, failure_counts={dict(report.failure_counts)}"
        )

    output = args.output_dir.resolve()
    paths = {
        "tasks": output / "tasks.jsonl",
        "oracle_trajectories": output / "oracle_trajectories.jsonl",
    }
    write_tasks(tasks, paths["tasks"])
    write_trajectories(trajectories, paths["oracle_trajectories"])
    manifest = {
        "dataset_name": config.dataset_name,
        "purpose": "paired Robustness Validation only; never used as SFT Train data",
        "config": config.to_dict(),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "source_tasks": {
            "path": str(args.source_tasks.resolve()),
            "sha256": _sha256(args.source_tasks),
            "count": len(source_tasks),
            "split": config.source_split,
        },
        "generation_audit": generation_audit,
        "oracle_audit": {
            "max_steps": args.max_steps,
            "task_count": report.task_count,
            "metrics": {name: metric.to_dict() for name, metric in report.metrics.items()},
            "failure_counts": dict(report.failure_counts),
        },
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
                "perturbation_counts": generation_audit["perturbation_counts"],
                "oracle_task_success_rate": report.metrics["task_success_rate"].value,
                "oracle_recovery_success_rate": report.metrics["recovery_success_rate"].value,
                "train_tasks_used": 0,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
