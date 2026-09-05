"""Configuration for trajectory-level execution-feedback GRPO."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from robust_tool.models.config import ModelInferenceConfig


@dataclass(frozen=True)
class GRPOConfig:
    model: ModelInferenceConfig
    tasks_path: str
    reward_name: str = "outcome"
    reward_config: Mapping[str, Any] = field(default_factory=dict)
    group_size: int = 4
    max_steps: int = 4
    max_updates: int = 128
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    clip_range: float = 0.2
    gradient_accumulation_steps: int = 1
    temperature: float = 0.8
    top_p: float = 0.95
    save_steps: int = 32
    seed: int = 20260827
    output_dir: str = "experiments/results/grpo_smoke"
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if not self.tasks_path:
            raise ValueError("tasks_path cannot be empty")
        if not self.reward_name:
            raise ValueError("reward_name cannot be empty")
        for name in ("group_size", "max_steps", "max_updates", "gradient_accumulation_steps", "save_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for group-relative advantages")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("weight_decay must be non-negative and max_grad_norm positive")
        if not 0 < self.clip_range < 1:
            raise ValueError("clip_range must be in (0, 1)")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["model"] = self.model.to_dict()
        record["reward_config"] = dict(self.reward_config)
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "GRPOConfig":
        values = dict(record)
        model_record = values.pop("model", None)
        if not isinstance(model_record, Mapping):
            raise ValueError("GRPO config requires a model object")
        reward_config = values.get("reward_config", {})
        if reward_config is None:
            values["reward_config"] = {}
        elif not isinstance(reward_config, Mapping):
            raise ValueError("reward_config must be an object")
        values["model"] = ModelInferenceConfig.from_dict(model_record)
        return cls(**values)


def load_grpo_config(path: str | Path) -> GRPOConfig:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load GRPO config {source}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"GRPO config must be a JSON object: {source}")
    try:
        return GRPOConfig.from_dict(record)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid GRPO config {source}: {exc}") from exc
