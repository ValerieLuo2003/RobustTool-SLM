#!/usr/bin/env python
"""Run execution-feedback GRPO and save auditable rollout/reward artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.schemas import load_tasks, write_tasks
from robust_tool.grpo.config import load_grpo_config
from robust_tool.grpo.trainer import GRPOTrainer


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "uncommitted"


def _runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("torch", "transformers", "peft", "modelscope"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        cuda["build_version"] = torch.version.cuda
        if cuda["available"]:
            cuda["device_name"] = torch.cuda.get_device_name(0)
            cuda["device_count"] = torch.cuda.device_count()
            cuda["total_memory_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3,
                2,
            )
    except (ImportError, RuntimeError) as exc:
        cuda["inspection_error"] = str(exc)
    return {"packages": packages, "cuda": cuda}


def _resolve_inside_results(path_text: str) -> Path:
    path = (PROJECT_ROOT / path_text).resolve()
    results_root = (PROJECT_ROOT / "experiments" / "results").resolve()
    if not path.is_relative_to(results_root):
        raise ValueError(f"output_dir must stay inside {results_root}: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "grpo" / "qwen2_5_1_5b_grpo_outcome_smoke.json",
    )
    parser.add_argument("--adapter-path", help="override the LoRA adapter used to initialize GRPO")
    parser.add_argument("--output-dir", help="override the result directory")
    parser.add_argument("--seed", type=int, help="override both rollout and model seeds")
    parser.add_argument("--limit", type=int, help="train on only the first N task records")
    parser.add_argument("--max-updates", type=int, help="override optimizer update count")
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        parser.error(f"GRPO config does not exist: {config_path}")
    try:
        config = load_grpo_config(config_path)
    except ValueError as exc:
        parser.error(str(exc))
    if args.adapter_path is not None:
        config = replace(config, model=replace(config.model, adapter_path=args.adapter_path))
    if config.model.adapter_path and not Path(config.model.adapter_path).expanduser().is_absolute():
        config = replace(
            config,
            model=replace(
                config.model,
                adapter_path=str((PROJECT_ROOT / config.model.adapter_path).resolve()),
            ),
        )
    if args.seed is not None:
        config = replace(config, seed=args.seed, model=replace(config.model, seed=args.seed))
    if args.max_updates is not None:
        if args.max_updates <= 0:
            parser.error("--max-updates must be positive")
        config = replace(config, max_updates=args.max_updates)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    try:
        output_dir = _resolve_inside_results(config.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    config = replace(config, output_dir=str(output_dir))
    if output_dir.exists() and any(
        (output_dir / name).exists() for name in ("config.json", "metrics.json", "reward_records.jsonl")
    ):
        parser.error(f"run artifacts already exist in {output_dir}; choose a new output_dir")

    task_path = (PROJECT_ROOT / config.tasks_path).resolve()
    if not task_path.exists():
        parser.error(f"task file does not exist: {task_path}")
    tasks = load_tasks(task_path)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        tasks = tasks[: args.limit]
    if not tasks:
        parser.error("no training tasks remain after applying --limit")

    random.seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tasks(tasks, output_dir / "tasks.jsonl")
    metadata = {
        "run_type": "execution-feedback-grpo",
        "source_config": str(config_path),
        "grpo_config": config.to_dict(),
        "task_source": str(task_path),
        "task_count": len(tasks),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime": _runtime_metadata(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_log = output_dir / "run.log"
    with run_log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"config": config.to_dict()}, ensure_ascii=False) + "\n")
        stream.flush()
        try:
            metrics = GRPOTrainer(config, tasks).train()
        except BaseException as exc:
            stream.write(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False) + "\n")
            stream.flush()
            raise
        stream.write(json.dumps({"status": "completed", "metrics": metrics}, ensure_ascii=False) + "\n")
    print(json.dumps({"run_dir": str(output_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
