#!/usr/bin/env python
"""Create traceable JSON and Markdown comparisons from evaluated run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DISPLAY_METRICS = (
    "call_decision_accuracy",
    "tool_selection_accuracy",
    "argument_semantic_accuracy",
    "executable_call_rate",
    "task_success_rate",
    "multi_turn_task_success_rate",
    "invalid_tool_call_rate",
    "unnecessary_tool_call_rate",
)


def _read_json(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"expected JSON object: {path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=run_directory")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    runs: list[dict[str, Any]] = []
    task_hash: str | None = None
    for specification in args.run:
        if "=" not in specification:
            parser.error(f"--run must use LABEL=path: {specification}")
        label, raw_path = specification.split("=", 1)
        run_dir = Path(raw_path).resolve()
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        failures_path = run_dir / "failure_stats.json"
        for path in (config_path, metrics_path, failures_path):
            if not path.exists():
                parser.error(f"evaluated run artifact does not exist: {path}")
        config = _read_json(config_path)
        metrics = _read_json(metrics_path)
        failures = _read_json(failures_path)
        current_hash = str(config.get("task_snapshot_sha256", ""))
        if task_hash is None:
            task_hash = current_hash
        elif current_hash != task_hash:
            parser.error("runs use different task snapshots and cannot be compared")
        runs.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "run_type": config.get("run_type"),
                "model": config.get("model"),
                "task_snapshot_sha256": current_hash,
                "metrics": metrics.get("metrics", {}),
                "failure_stats": failures,
            }
        )

    payload = {"task_snapshot_sha256": task_hash, "runs": runs}
    json_path = args.output_prefix.with_suffix(".json")
    markdown_path = args.output_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    header = "| Metric | " + " | ".join(run["label"] for run in runs) + " |"
    separator = "|---|" + "---:|" * len(runs)
    lines = [header, separator]
    for metric_name in DISPLAY_METRICS:
        values: list[str] = []
        for run in runs:
            metric = run["metrics"].get(metric_name, {})
            value = metric.get("value") if isinstance(metric, dict) else None
            values.append("n/a" if value is None else f"{value:.2%}")
        lines.append(f"| `{metric_name}` | " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
