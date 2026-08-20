#!/usr/bin/env python
"""Generate the deterministic Week 1 Calendar benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.generator import generate_calendar_toy_tasks, write_calendar_toy_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "eval")
    args = parser.parse_args()

    paths = write_calendar_toy_dataset(args.output_dir, args.seed)
    tasks = generate_calendar_toy_tasks(args.seed)
    counts = {
        split: sum(task.metadata["split"] == split for task in tasks)
        for split in ("train", "validation", "test")
    }
    print(json.dumps({"seed": args.seed, "counts": counts, "paths": {k: str(v) for k, v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
