#!/usr/bin/env python
"""Run one configuration-backed ms-swift SFT experiment with traceable artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
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


def _local_dataset_path(dataset_spec: str) -> Path:
    """Remove an optional ms-swift ``#sample_count`` suffix from a local path."""

    path_text, separator, sample_text = dataset_spec.rpartition("#")
    if separator and sample_text.isdigit():
        if int(sample_text) <= 0:
            raise ValueError(f"dataset sample count must be positive: {dataset_spec}")
        return Path(path_text)
    return Path(dataset_spec)


def _latest_trainer_state(output_dir: Path) -> Path | None:
    candidates = list(output_dir.rglob("trainer_state.json"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _latest_logging(output_dir: Path) -> Path | None:
    candidates = list(output_dir.rglob("logging.jsonl"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _logging_history(output_dir: Path) -> list[dict[str, Any]]:
    path = _latest_logging(output_dir)
    if path is None:
        return []
    history: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict):
            history.append(record)
    return history


def _runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("torch", "transformers", "peft", "ms-swift", "modelscope"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda["build_version"] = torch.version.cuda
        cuda["available"] = torch.cuda.is_available()
        if cuda["available"]:
            properties = torch.cuda.get_device_properties(0)
            cuda.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "device_count": torch.cuda.device_count(),
                    "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                }
            )
    except (ImportError, RuntimeError) as exc:
        cuda["inspection_error"] = str(exc)
    return {"packages": packages, "cuda": cuda}


def _collect_metrics(output_dir: Path, return_code: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {"completed": return_code == 0, "return_code": return_code}
    state_path = _latest_trainer_state(output_dir)
    if state_path is None:
        return metrics
    state = _load_json(state_path)
    history = _logging_history(output_dir) or state.get("log_history", [])
    train_entries = [entry for entry in history if "loss" in entry]
    eval_entries = [entry for entry in history if "eval_loss" in entry]
    summaries = [entry for entry in history if "train_runtime" in entry]
    metrics.update(
        {
            "global_step": state.get("global_step"),
            "trainer_state": str(state_path),
            "last_step_loss": train_entries[-1].get("loss") if train_entries else None,
            "last_eval_loss": eval_entries[-1].get("eval_loss") if eval_entries else None,
            "eval_history": eval_entries,
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


def _write_console_safely(line: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_line = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
    sys.stdout.write(safe_line)
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sft" / "qwen2_5_1_5b_lora_smoke.json",
    )
    parser.add_argument("--seed", type=int, help="override both model/data seeds for a replicate")
    parser.add_argument("--output-dir", help="override output_dir for a replicate run")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        parser.error(f"SFT config does not exist: {config_path}")
    swift_config = _load_json(config_path)
    cli_overrides: dict[str, Any] = {}
    if args.seed is not None:
        swift_config["seed"] = args.seed
        swift_config["data_seed"] = args.seed
        cli_overrides.update({"seed": args.seed, "data_seed": args.seed})
    if args.output_dir is not None:
        swift_config["output_dir"] = args.output_dir
        output_path = Path(args.output_dir)
        swift_config["logging_dir"] = str(
            output_path.parent / f"{output_path.name}_tensorboard"
        )
        cli_overrides.update(
            {"output_dir": swift_config["output_dir"], "logging_dir": swift_config["logging_dir"]}
        )
    output_dir = (PROJECT_ROOT / str(swift_config["output_dir"])).resolve()
    experiment_root = (PROJECT_ROOT / "experiments" / "results").resolve()
    if not output_dir.is_relative_to(experiment_root):
        parser.error(f"output_dir must stay inside {experiment_root}: {output_dir}")
    for dataset_key in ("dataset", "val_dataset"):
        for dataset_path in swift_config.get(dataset_key, []):
            resolved_dataset = (PROJECT_ROOT / _local_dataset_path(str(dataset_path))).resolve()
            if not resolved_dataset.exists():
                parser.error(f"{dataset_key} file does not exist: {resolved_dataset}")
    run_dir = output_dir.parent
    if any((run_dir / name).exists() for name in ("config.json", "metrics.json", "run.log")):
        parser.error(f"run artifacts already exist in {run_dir}; use a new output_dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_config_path = config_path
    if cli_overrides:
        effective_config_path = run_dir / "swift_config.json"
        effective_config_path.write_text(
            json.dumps(swift_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    run_log = run_dir / "run.log"
    metadata = {
        "run_type": "ms-swift-sft",
        "source_config": str(config_path),
        "effective_config": str(effective_config_path),
        "cli_overrides": cli_overrides,
        "swift_config": swift_config,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime": _runtime_metadata(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    command = _swift_command(effective_config_path)
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
        try:
            for line in process.stdout:
                _write_console_safely(line)
                stream.write(line)
                stream.flush()
            return_code = process.wait()
        except BaseException:
            process.terminate()
            process.wait(timeout=30)
            raise

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
