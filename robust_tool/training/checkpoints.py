"""Deterministically select a LoRA checkpoint from Validation loss logs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        records.append(record)
    return records


def _global_step(record: dict[str, Any]) -> int | None:
    combined = record.get("global_step/max_steps")
    if isinstance(combined, str) and "/" in combined:
        raw_step = combined.split("/", 1)[0]
        try:
            return int(raw_step)
        except ValueError:
            return None
    step = record.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        return step
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_files(checkpoint: Path) -> tuple[Path, Path] | None:
    config = checkpoint / "adapter_config.json"
    weights = next(
        (
            path
            for path in (
                checkpoint / "adapter_model.safetensors",
                checkpoint / "adapter_model.bin",
            )
            if path.is_file()
        ),
        None,
    )
    if not config.is_file() or weights is None:
        return None
    return config, weights


def _find_checkpoint(run_dir: Path, step: int) -> Path | None:
    name = f"checkpoint-{step}"
    preferred = run_dir / "checkpoint_candidates" / name
    if _adapter_files(preferred) is not None:
        return preferred
    matches = sorted(
        path
        for path in (run_dir / "trainer_output").rglob(name)
        if path.is_dir() and _adapter_files(path) is not None
    )
    return matches[-1] if matches else None


def select_best_checkpoint(run_dir: Path) -> dict[str, Any]:
    """Return a traceable best-checkpoint record without reading Test artifacts."""

    run_dir = run_dir.resolve()
    logging_files = sorted((run_dir / "trainer_output").rglob("logging.jsonl"))
    if not logging_files:
        raise FileNotFoundError(f"no logging.jsonl found under {run_dir / 'trainer_output'}")
    log_path = max(logging_files, key=lambda path: path.stat().st_mtime)

    evaluations: list[dict[str, Any]] = []
    for record in _read_json_lines(log_path):
        if "eval_loss" not in record:
            continue
        step = _global_step(record)
        if step is None:
            raise ValueError(f"evaluation record has no valid global step: {record}")
        checkpoint = _find_checkpoint(run_dir, step)
        evaluations.append(
            {
                "step": step,
                "eval_loss": float(record["eval_loss"]),
                "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
                "checkpoint_available": checkpoint is not None,
            }
        )
    available = [entry for entry in evaluations if entry["checkpoint_available"]]
    if not available:
        raise FileNotFoundError("no complete LoRA checkpoint matches an evaluation record")
    best = min(available, key=lambda entry: (entry["eval_loss"], -entry["step"]))
    checkpoint = Path(str(best["checkpoint_path"]))
    config, weights = _adapter_files(checkpoint) or (None, None)
    assert config is not None and weights is not None
    return {
        "selection_protocol": "minimum SFT Validation loss among available LoRA checkpoints",
        "source_run": str(run_dir),
        "source_logging": str(log_path),
        "evaluations": evaluations,
        "selected_step": best["step"],
        "selected_eval_loss": best["eval_loss"],
        "selected_checkpoint": str(checkpoint),
        "adapter_config_sha256": _sha256(config),
        "adapter_weight_file": weights.name,
        "adapter_weight_bytes": weights.stat().st_size,
        "adapter_weight_sha256": _sha256(weights),
    }
