#!/usr/bin/env python
"""Generate paired Robustness Gap JSON, CSV, and Markdown reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robust_tool.data.schemas import load_tasks
from robust_tool.eval.robustness import compute_robustness_report


def _json_object(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return record


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        records.append(record)
    return records


def _model_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    runtime = config.get("runtime", {})
    adapter = runtime.get("adapter") if isinstance(runtime, dict) else None
    return {
        "model_id": model.get("model_id") if isinstance(model, dict) else None,
        "revision": model.get("revision") if isinstance(model, dict) else None,
        "adapter_weight_sha256": (
            adapter.get("adapter_weight_sha256") if isinstance(adapter, dict) else None
        ),
        "do_sample": model.get("do_sample") if isinstance(model, dict) else None,
        "max_new_tokens": model.get("max_new_tokens") if isinstance(model, dict) else None,
        "max_steps": config.get("max_steps"),
    }


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--robust-run", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    clean_run = args.clean_run.resolve()
    robust_run = args.robust_run.resolve()
    required = {
        "clean_config": clean_run / "config.json",
        "clean_evaluation": clean_run / "evaluation.jsonl",
        "robust_config": robust_run / "config.json",
        "robust_tasks": robust_run / "tasks.jsonl",
        "robust_evaluation": robust_run / "evaluation.jsonl",
    }
    for name, path in required.items():
        if not path.is_file():
            parser.error(f"required {name} file does not exist: {path}")

    clean_config = _json_object(required["clean_config"])
    robust_config = _json_object(required["robust_config"])
    clean_fingerprint = _model_fingerprint(clean_config)
    robust_fingerprint = _model_fingerprint(robust_config)
    if clean_fingerprint != robust_fingerprint:
        parser.error(
            "Clean and Robust runs use different model/decode fingerprints: "
            f"{clean_fingerprint} != {robust_fingerprint}"
        )

    report = compute_robustness_report(
        _jsonl(required["clean_evaluation"]),
        load_tasks(required["robust_tasks"]),
        _jsonl(required["robust_evaluation"]),
    )
    payload = {
        "model_fingerprint": clean_fingerprint,
        "clean_run": str(clean_run),
        "robust_run": str(robust_run),
        **report,
    }
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = [("overall", report["overall"]), *report["settings"].items()]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["setting", "pairs", "clean_task_success", "perturbed_task_success", "robustness_gap"]
        )
        for setting, item in rows:
            writer.writerow(
                [
                    setting,
                    item["pair_count"],
                    item["clean_task_success"]["value"],
                    item["perturbed_task_success"]["value"],
                    item["robustness_gap"],
                ]
            )

    lines = [
        "| Setting | Pairs | Clean Success | Perturbed Success | Robustness Gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for setting, item in rows:
        lines.append(
            "| {setting} | {pairs} | {clean} | {perturbed} | {gap} |".format(
                setting=setting,
                pairs=item["pair_count"],
                clean=_percent(item["clean_task_success"]["value"]),
                perturbed=_percent(item["perturbed_task_success"]["value"]),
                gap=_percent(item["robustness_gap"]),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(markdown_path),
                "pair_count": report["pair_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
