#!/usr/bin/env python
"""Run the configured Qwen base model in the executable tool environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.schemas import load_tasks, write_tasks
from robust_tool.models.config import load_model_config
from robust_tool.models.qwen import QwenTransformersPolicy
from robust_tool.rollout.runner import run_policy
from robust_tool.rollout.trajectory import write_trajectories

SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models" / "qwen2_5_1_5b_instruct.json",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "toy_validation.jsonl",
    )
    parser.add_argument("--run-name", default="qwen2_5_1_5b_base_smoke")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "results")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not SAFE_RUN_NAME.fullmatch(args.run_name):
        parser.error("--run-name may contain only letters, digits, dot, underscore, and hyphen")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not args.config.exists():
        parser.error(f"model config does not exist: {args.config}")
    if not args.tasks.exists():
        parser.error(f"task file does not exist: {args.tasks}; run scripts/generate_data.py first")

    run_dir = args.output_root.resolve() / args.run_name
    artifacts = (
        "config.json",
        "tasks.jsonl",
        "predictions.jsonl",
        "trajectories.jsonl",
        "run.log",
    )
    if not args.overwrite and any((run_dir / name).exists() for name in artifacts):
        parser.error(f"run artifacts already exist in {run_dir}; choose another name or pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_config = load_model_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    task_snapshot = run_dir / "tasks.jsonl"
    write_tasks(tasks, task_snapshot)

    policy = QwenTransformersPolicy(model_config)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    trajectories = run_policy(tasks, policy, max_steps=args.max_steps)
    duration = time.perf_counter() - started
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
    config_record = {
        "run_name": args.run_name,
        "run_type": "qwen_base_inference",
        "model": model_config.to_dict(),
        "runtime": dict(policy.runtime_metadata()),
        "seed": model_config.seed,
        "max_steps": args.max_steps,
        "task_source": str(args.tasks.resolve()),
        "task_source_sha256": _sha256(args.tasks),
        "task_snapshot": str(task_snapshot),
        "task_snapshot_sha256": _sha256(task_snapshot),
        "task_count": len(tasks),
        "created_at_utc": started_at.isoformat(),
        "duration_seconds": duration,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    _write_json(run_dir / "config.json", config_record)
    (run_dir / "run.log").write_text(
        f"model={model_config.model_id}\n"
        f"revision={model_config.revision}\n"
        f"task_count={len(tasks)}\n"
        f"duration_seconds={duration:.6f}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "task_count": len(tasks),
                "model": model_config.model_id,
                "duration_seconds": duration,
                "next_command": f"python scripts/run_eval.py --run-name {args.run_name}",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
