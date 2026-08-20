#!/usr/bin/env python
"""Run a deterministic oracle or random policy without loading a model."""

from __future__ import annotations

import argparse
import json
import re
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.schemas import load_tasks
from robust_tool.rollout.runner import OraclePolicy, RandomPolicy, run_policy
from robust_tool.rollout.trajectory import write_trajectories

SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "uncommitted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=PROJECT_ROOT / "data" / "eval" / "toy_test.jsonl")
    parser.add_argument("--policy", choices=("oracle", "random"), default="oracle")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--run-name", default="toy_oracle")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "results")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not SAFE_RUN_NAME.fullmatch(args.run_name):
        parser.error("--run-name may contain only letters, digits, dot, underscore, and hyphen")
    if not args.tasks.exists():
        parser.error(f"task file does not exist: {args.tasks}; run scripts/generate_data.py first")

    run_dir = args.output_root.resolve() / args.run_name
    tracked_files = (
        run_dir / "config.json",
        run_dir / "predictions.jsonl",
        run_dir / "trajectories.jsonl",
        run_dir / "run.log",
    )
    if not args.overwrite and any(path.exists() for path in tracked_files):
        parser.error(f"run artifacts already exist in {run_dir}; choose another name or pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.tasks)
    policy = OraclePolicy() if args.policy == "oracle" else RandomPolicy(args.seed)
    trajectories = run_policy(tasks, policy, max_steps=args.max_steps)
    config = {
        "run_name": args.run_name,
        "policy": args.policy,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "tasks": str(args.tasks.resolve()),
        "task_count": len(tasks),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    _write_json(run_dir / "config.json", config)
    write_trajectories(trajectories, run_dir / "trajectories.jsonl")

    predictions = [
        {
            "task_id": trajectory.task_id,
            "assistant_actions": trajectory.assistant_actions(),
            "tool_calls": [call.to_dict() for call in trajectory.tool_calls()],
            "final_answer": trajectory.final_answer(),
            "final_state": trajectory.final_state,
        }
        for trajectory in trajectories
    ]
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in predictions) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        f"policy={args.policy} seed={args.seed} task_count={len(tasks)}\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), "task_count": len(tasks), "policy": args.policy}, indent=2))


if __name__ == "__main__":
    main()
