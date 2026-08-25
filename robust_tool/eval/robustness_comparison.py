"""Cross-model comparison for evaluated robustness runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

from robust_tool.data.perturb import PERTURBATION_KINDS


DISPLAY_METRICS = (
    "call_decision_accuracy",
    "tool_selection_accuracy",
    "json_valid_rate",
    "argument_schema_accuracy",
    "argument_semantic_accuracy",
    "executable_call_rate",
    "task_success_rate",
    "multi_turn_task_success_rate",
    "recovery_success_rate",
    "invalid_tool_call_rate",
    "unnecessary_tool_call_rate",
)


def compare_robustness_runs(
    runs: Iterable[tuple[str, Path]],
) -> dict[str, Any]:
    """Load and validate multiple runs evaluated on one frozen Robust snapshot."""

    loaded: list[dict[str, Any]] = []
    labels: set[str] = set()
    task_hash: str | None = None
    pair_count: int | None = None
    for label, raw_path in runs:
        if not label.strip():
            raise ValueError("run label must not be blank")
        if label in labels:
            raise ValueError(f"duplicate run label: {label}")
        labels.add(label)
        run_dir = raw_path.resolve()
        config = _json_object(run_dir / "config.json")
        metrics = _json_object(run_dir / "metrics.json")
        failures = _json_object(run_dir / "failure_stats.json")
        robustness = _json_object(run_dir / "robustness_gap.json")

        current_hash = str(config.get("task_snapshot_sha256", ""))
        if not current_hash:
            raise ValueError(f"run has no task snapshot SHA-256: {run_dir}")
        if task_hash is None:
            task_hash = current_hash
        elif current_hash != task_hash:
            raise ValueError("runs use different Robust task snapshots")

        current_pairs = int(robustness.get("pair_count", -1))
        if current_pairs <= 0 or current_pairs != int(config.get("task_count", -2)):
            raise ValueError(f"run has inconsistent task/pair counts: {run_dir}")
        if pair_count is None:
            pair_count = current_pairs
        elif current_pairs != pair_count:
            raise ValueError("runs use different Robust pair counts")
        settings = robustness.get("settings")
        if not isinstance(settings, Mapping) or set(settings) != set(PERTURBATION_KINDS):
            raise ValueError(f"run has incomplete or unknown perturbation settings: {run_dir}")

        metric_values = metrics.get("metrics")
        if not isinstance(metric_values, Mapping):
            raise ValueError(f"run metrics are malformed: {run_dir}")
        failure_counts = failures.get("failure_counts")
        if not isinstance(failure_counts, Mapping):
            raise ValueError(f"run failure statistics are malformed: {run_dir}")
        loaded.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "git_commit": config.get("git_commit"),
                "run_type": config.get("run_type"),
                "duration_seconds": config.get("duration_seconds"),
                "model": config.get("model"),
                "runtime": config.get("runtime"),
                "metrics": dict(metric_values),
                "failure_stats": failures,
                "robustness": robustness,
            }
        )
    if len(loaded) < 2:
        raise ValueError("at least two Robust runs are required")
    return {
        "task_snapshot_sha256": task_hash,
        "pair_count": pair_count,
        "perturbation_order": list(PERTURBATION_KINDS),
        "runs": loaded,
    }


def write_robustness_comparison(payload: Mapping[str, Any], prefix: Path) -> dict[str, str]:
    """Write traceable JSON, tidy CSV, and human-readable Markdown reports."""

    resolved = prefix.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    json_path = resolved.with_suffix(".json")
    csv_path = resolved.with_suffix(".csv")
    markdown_path = resolved.with_suffix(".md")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        _write_csv(payload, stream)
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required run artifact does not exist: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return record


def _metric_value(run: Mapping[str, Any], name: str) -> float | None:
    metric = run["metrics"].get(name)
    return metric.get("value") if isinstance(metric, Mapping) else None


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _write_csv(payload: Mapping[str, Any], stream: TextIO) -> None:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "label",
            "setting",
            "pairs",
            "clean_task_success",
            "perturbed_task_success",
            "robustness_gap",
        ]
    )
    for run in payload["runs"]:
        report = run["robustness"]
        rows = [("overall", report["overall"])] + [
            (kind, report["settings"][kind]) for kind in payload["perturbation_order"]
        ]
        for setting, item in rows:
            writer.writerow(
                [
                    run["label"],
                    setting,
                    item["pair_count"],
                    item["clean_task_success"]["value"],
                    item["perturbed_task_success"]["value"],
                    item["robustness_gap"],
                ]
            )


def _markdown(payload: Mapping[str, Any]) -> str:
    runs = payload["runs"]
    lines = [
        "# 鲁棒性正式验证集对比",
        "",
        f"任务快照 SHA-256：`{payload['task_snapshot_sha256']}`；配对样本数：{payload['pair_count']}。",
        "",
        "## 总体指标",
        "",
        "| 指标 | " + " | ".join(run["label"] for run in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    for metric_name in DISPLAY_METRICS:
        lines.append(
            f"| `{metric_name}` | "
            + " | ".join(_percent(_metric_value(run, metric_name)) for run in runs)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 分扰动设置",
            "",
            "表格单元格格式为 `扰动成功率（相对同源 Clean 的 Gap）`。",
            "",
            "| 设置 | " + " | ".join(run["label"] for run in runs) + " |",
            "|---|" + "---:|" * len(runs),
        ]
    )
    settings = ["overall", *payload["perturbation_order"]]
    for setting in settings:
        cells: list[str] = []
        for run in runs:
            report = run["robustness"]
            item = report["overall"] if setting == "overall" else report["settings"][setting]
            cells.append(
                f"{_percent(item['perturbed_task_success']['value'])}"
                f"（{_percent(item['robustness_gap'])}）"
            )
        lines.append(f"| {setting} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 失败分布",
            "",
            "| Failure | " + " | ".join(run["label"] for run in runs) + " |",
            "|---|" + "---:|" * len(runs),
        ]
    )
    failure_names = sorted(
        {
            name
            for run in runs
            for name in run["failure_stats"]["failure_counts"]
        }
    )
    for name in failure_names:
        lines.append(
            f"| `{name}` | "
            + " | ".join(str(run["failure_stats"]["failure_counts"].get(name, 0)) for run in runs)
            + " |"
        )
    return "\n".join(lines) + "\n"
