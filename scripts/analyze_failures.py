#!/usr/bin/env python
"""Render failure statistics as a compact Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("failure_stats", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stats = json.loads(args.failure_stats.read_text(encoding="utf-8"))
    counts = stats.get("failure_counts", {})
    rates = stats.get("failure_task_rates", {})
    lines = ["| Failure | Count | Task rate |", "|---|---:|---:|"]
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rate = rates.get(label)
        lines.append(f"| `{label}` | {count} | {rate:.1%} |" if rate is not None else f"| `{label}` | {count} | n/a |")
    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
