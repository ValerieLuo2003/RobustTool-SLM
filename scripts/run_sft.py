#!/usr/bin/env python
"""Run one configuration-backed ms-swift SFT experiment with traceable artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _load_json(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return record


def _latest_trainer_state(output_dir: Path) -> Path | None:
    candidates = list(output_dir.rglob("trainer_state.json"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _collect_metrics(output_dir: Path, return_code: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {"completed": return_code == 0, "return_code": return_code}
    state_path = _latest_trainer_state(output_dir)
    if state_path is None:
        return metrics
    state = _load_json(state_path)
    history = state.get("log_history", [])
    train_entries = [entry for entry in history if "loss" in entry]
    eval_entries = [entry for entry in history if "eval_loss" in entry]
    summaries = [entry for entry in history if "train_loss" in entry]
    metrics.update(
        {
            "global_step": state.get("global_step"),
            "trainer_state": str(state_path),
            "last_step_loss": train_entries[-1].get("loss") if train_entries else None,
            "last_eval_loss": eval_entries[-1].get("eval_loss") if eval_entries else None,
            "train_summary": summaries[-1] if summaries else None,
        }
    )
    return metrics


def _swift_command(config_path: Path) -> list[str]:
    sibling_name = "swift.exe" if sys.platform == "win32" else "swift"
    sibling = Path(sys.executable).with_name(sibling_name)
    executable = str(sibling) if sibling.exists() else shutil.which("swift")
    if executable is None:
        raise RuntimeError("cannot find the ms-swift CLI; install the project training extra")
    return [executable, "sft", str(config_path)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sft" / "qwen2_5_1_5b_lora_smoke.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        parser.error(f"SFT config does not exist: {config_path}")
    swift_config = _load_json(config_path)
    output_dir = (PROJECT_ROOT / str(swift_config["output_dir"])).resolve()
    experiment_root = (PROJECT_ROOT / "experiments" / "results").resolve()
    if not output_dir.is_relative_to(experiment_root):
        parser.error(f"output_dir must stay inside {experiment_root}: {output_dir}")
    for dataset_key in ("dataset", "val_dataset"):
        for dataset_path in swift_config.get(dataset_key, []):
            resolved_dataset = (PROJECT_ROOT / str(dataset_path)).resolve()
            if not resolved_dataset.exists():
                parser.error(f"{dataset_key} file does not exist: {resolved_dataset}")
    run_dir = output_dir.parent
    if any((run_dir / name).exists() for name in ("config.json", "metrics.json", "run.log")):
        parser.error(f"run artifacts already exist in {run_dir}; use a new output_dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run.log"
    metadata = {
        "run_type": "ms-swift-sft",
        "source_config": str(config_path),
        "swift_config": swift_config,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    command = _swift_command(config_path)
    with run_log.open("w", encoding="utf-8") as stream:
        stream.write(f"command={json.dumps(command, ensure_ascii=False)}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
            stream.flush()
        return_code = process.wait()

    metrics = _collect_metrics(output_dir, return_code)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False, indent=2))
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
