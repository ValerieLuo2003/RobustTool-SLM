#!/usr/bin/env python
"""Aggregate one scalar metric over explicitly named random-seed runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"metrics file must contain an object: {path}")
    return record


def _metric_value(record: dict[str, Any], metric: str) -> float | None:
    current: Any = record
    for key in metric.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, dict) and "value" in current:
        current = current["value"]
    if current is None:
        return None
    if not isinstance(current, (int, float)):
        raise ValueError(f"metric {metric!r} is not numeric: {current!r}")
    return float(current)


def aggregate_runs(
    runs: list[tuple[str, Path]],
    *,
    metric: str,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    values: list[float] = []
    for label, run_dir in runs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json does not exist: {metrics_path}")
        value = _metric_value(_load(metrics_path), metric)
        observations.append({"label": label, "run_dir": str(run_dir), "value": value})
        if value is not None:
            values.append(value)
    mean = statistics.fmean(values) if values else None
    standard_deviation = statistics.stdev(values) if len(values) >= 2 else None
    return {
        "metric": metric,
        "count": len(values),
        "mean": mean,
        "std": standard_deviation,
        "std_definition": "sample standard deviation (ddof=1)" if len(values) >= 2 else None,
        "runs": observations,
    }


def _markdown(report: dict[str, Any]) -> str:
    mean = "-" if report["mean"] is None else f"{report['mean']:.6f}"
    std = "-" if report["std"] is None else f"{report['std']:.6f}"
    lines = [
        f"# Seed aggregation: `{report['metric']}`",
        "",
        f"Mean: **{mean}**  ",
        f"Std: **{std}** ({report.get('std_definition') or 'not available'})",
        "",
        "| Run | Value |",
        "| --- | ---: |",
    ]
    for item in report["runs"]:
        value = "-" if item["value"] is None else f"{item['value']:.6f}"
        lines.append(f"| `{item['label']}` | {value} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label=run_directory; repeat once per random seed",
    )
    parser.add_argument(
        "--metric",
        default="metrics.task_success_rate",
        help="dotted path in metrics.json; MetricValue objects use their value field",
    )
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    runs: list[tuple[str, Path]] = []
    for item in args.run:
        label, separator, path_text = item.partition("=")
        if not separator or not label or not path_text:
            parser.error(f"--run must use label=run_directory: {item}")
        runs.append((label, Path(path_text).resolve()))
    report = aggregate_runs(runs, metric=args.metric)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_prefix:
        prefix = args.output_prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".json").write_text(rendered, encoding="utf-8")
        prefix.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
