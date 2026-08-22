"""Validated, JSON-serializable inference configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

ModelSource = Literal["modelscope", "huggingface", "local"]


@dataclass(frozen=True)
class ModelInferenceConfig:
    model_id: str
    revision: str
    source: ModelSource = "modelscope"
    local_model_path: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    max_input_tokens: int = 4096
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int = 42
    system_prompt: str = (
        "You are a reliable calendar tool assistant. Use only the tools provided to you. "
        "Call at most one tool per assistant turn. Never invent a tool or argument. "
        "If required information is missing, ask one concise clarifying question without "
        "calling a tool. After receiving a tool result, use it to decide the next action. "
        "Do not claim success when a tool reports an error."
    )

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.revision:
            raise ValueError("revision cannot be empty")
        if self.source not in {"modelscope", "huggingface", "local"}:
            raise ValueError(f"unsupported model source: {self.source}")
        if self.source == "local" and not self.local_model_path:
            raise ValueError("local source requires local_model_path")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.max_input_tokens <= 0 or self.max_new_tokens <= 0:
            raise ValueError("token limits must be positive")
        if self.do_sample and self.temperature is not None and self.temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ModelInferenceConfig":
        return cls(**dict(record))


def load_model_config(path: str | Path) -> ModelInferenceConfig:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load model config {source}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"model config must be a JSON object: {source}")
    try:
        return ModelInferenceConfig.from_dict(record)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid model config {source}: {exc}") from exc
