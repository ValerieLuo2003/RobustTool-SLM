#!/usr/bin/env python
"""Generate and Oracle-audit the frozen Calendar formal SFT benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.formal_generator import (
    SPLITS,
    load_formal_dataset_config,
    write_calendar_formal_dataset,
)
from robust_tool.data.schemas import load_tasks
from robust_tool.eval.evaluator import evaluate_dataset
from robust_tool.rollout.runner import OraclePolicy, run_policy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "calendar_formal_sft_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "calendar_formal_v1" / "tasks",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()
    if not args.config.exists():
        parser.error(f"dataset config does not exist: {args.config}")
    if args.max_steps < 3:
        parser.error("--max-steps must be at least 3 for two-call tasks plus the final answer")

    config = load_formal_dataset_config(args.config)
    paths = write_calendar_formal_dataset(args.output_dir.resolve(), config)
    reports: dict[str, object] = {}
    for split in SPLITS:
        tasks = load_tasks(paths[split])
        trajectories = run_policy(tasks, OraclePolicy(), max_steps=args.max_steps)
        report = evaluate_dataset(tasks, trajectories)
        failed = [item.task_id for item in report.task_evaluations if not item.success]
        if failed:
            preview = ", ".join(failed[:10])
            raise RuntimeError(f"Oracle audit failed for {split}: {len(failed)} tasks ({preview})")
        reports[split] = {
            "task_count": report.task_count,
            "metrics": {name: metric.to_dict() for name, metric in report.metrics.items()},
            "failure_counts": dict(report.failure_counts),
        }

    audit_path = args.output_dir.resolve() / "oracle_audit.json"
    audit_payload = {
        "dataset_name": config.dataset_name,
        "generator_version": config.generator_version,
        "seed": config.seed,
        "max_steps": args.max_steps,
        "splits": reports,
    }
    audit_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["files"]["oracle_audit"] = {
        "path": audit_path.name,
        "sha256": _sha256(audit_path),
    }
    manifest["source_config"] = {
        "path": str(args.config.resolve()),
        "sha256": _sha256(args.config),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "counts": {split: config.split_size(split) for split in SPLITS},
                "oracle_task_success": {split: 1.0 for split in SPLITS},
                "manifest": str(paths["manifest"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
