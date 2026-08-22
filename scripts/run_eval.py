#!/usr/bin/env python
"""Evaluate an existing run by replaying its trajectories."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.schemas import load_tasks
from robust_tool.eval.evaluator import evaluate_dataset
from robust_tool.rollout.trajectory import load_trajectories

SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="toy_oracle")
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "results")
    args = parser.parse_args()
    if not SAFE_RUN_NAME.fullmatch(args.run_name):
        parser.error("--run-name may contain only letters, digits, dot, underscore, and hyphen")

    run_dir = args.output_root.resolve() / args.run_name
    trajectory_path = run_dir / "trajectories.jsonl"
    if not trajectory_path.exists():
        parser.error(f"trajectory file does not exist: {trajectory_path}; run run_baseline.py first")
    task_path = args.tasks or (
        run_dir / "tasks.jsonl"
        if (run_dir / "tasks.jsonl").exists()
        else PROJECT_ROOT / "data" / "eval" / "toy_test.jsonl"
    )
    tasks = load_tasks(task_path)
    trajectories = load_trajectories(trajectory_path)
    report = evaluate_dataset(tasks, trajectories)

    _write_json(run_dir / "metrics.json", report.metrics_dict())
    _write_json(run_dir / "failure_stats.json", report.failure_stats_dict())
    evaluation_path = run_dir / "evaluation.jsonl"
    evaluation_path.write_text(
        "\n".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True)
            for item in report.task_evaluations
        )
        + "\n",
        encoding="utf-8",
    )
    with (run_dir / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"evaluated_task_count={report.task_count}\n")
    summary = {
        name: metric.value for name, metric in report.metrics.items()
    }
    print(json.dumps({"run_dir": str(run_dir), "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
