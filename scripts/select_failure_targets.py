#!/usr/bin/env python
"""Select failure-aware data targets strictly from an SFT Validation run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.schemas import load_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top-k", type=int, choices=(2, 3), default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    task_path = run_dir / "tasks.jsonl"
    config_path = run_dir / "config.json"
    failure_path = run_dir / "failure_stats.json"
    for path in (task_path, config_path, failure_path):
        if not path.exists():
            parser.error(f"required run artifact does not exist: {path}")
    tasks = load_tasks(task_path)
    bad_splits = sorted({str(task.metadata.get("split")) for task in tasks} - {"validation"})
    if bad_splits:
        parser.error(
            "failure targets may only be selected from Validation; "
            f"found non-validation split metadata: {bad_splits}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("run_type") != "qwen_lora_adapter_inference":
        parser.error("failure targets require an evaluated LoRA adapter run")
    stats = json.loads(failure_path.read_text(encoding="utf-8"))
    counts = stats.get("failure_counts", {})
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    selected = [
        {
            "rank": rank,
            "failure": label,
            "count": count,
            "task_rate": stats.get("failure_task_rates", {}).get(label),
        }
        for rank, (label, count) in enumerate(ranked[: args.top_k], start=1)
    ]
    payload = {
        "selection_protocol": "top failures from SFT Validation only",
        "source_run": str(run_dir),
        "source_git_commit": config.get("git_commit"),
        "task_snapshot_sha256": config.get("task_snapshot_sha256"),
        "validation_task_count": len(tasks),
        "requested_top_k": args.top_k,
        "selected_failures": selected,
    }
    output = args.output or (run_dir / "failure_targets.json")
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
