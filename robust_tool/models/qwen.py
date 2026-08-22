"""Transformers inference adapter for a single Qwen tool-calling policy."""

from __future__ import annotations

import json
import random
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

from robust_tool.data.schemas import Task
from robust_tool.models.config import ModelInferenceConfig
from robust_tool.rollout.parser import parse_assistant_output
from robust_tool.rollout.runner import AgentAction
from robust_tool.rollout.trajectory import Trajectory
from robust_tool.tools.registry import ToolRegistry, calendar_registry


def trajectory_to_chat_messages(
    trajectory: Trajectory,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Convert canonical trajectories to the chat-template tool protocol."""

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for message in trajectory.messages:
        if message.role in {"system", "user"}:
            messages.append({"role": message.role, "content": message.content or ""})
        elif message.role == "assistant" and message.tool_call is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": message.tool_call.name,
                                "arguments": dict(message.tool_call.arguments),
                            },
                        }
                    ],
                }
            )
        elif message.role == "assistant":
            messages.append({"role": "assistant", "content": message.content or ""})
        elif message.role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        dict(message.tool_result or {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
    return messages


class QwenTransformersPolicy:
    """Generate one canonical agent action at a time with Qwen."""

    def __init__(
        self,
        config: ModelInferenceConfig,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or calendar_registry()
        self.name = f"qwen-transformers:{config.model_id}@{config.revision}"
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._resolved_model_path: str | None = None

    def load(self) -> None:
        """Load optional heavyweight dependencies and model weights once."""

        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen inference requires torch and transformers; install the project training extra"
            ) from exc

        model_path = self._resolve_model_path()
        dtype = self._resolve_dtype(torch)
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=None if self.config.source in {"modelscope", "local"} else self.config.revision,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=None if self.config.source in {"modelscope", "local"} else self.config.revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model.to(self.config.device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def _resolve_model_path(self) -> str:
        if self.config.source == "local":
            path = Path(self.config.local_model_path or "").expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"local model path does not exist: {path}")
            self._resolved_model_path = str(path)
        elif self.config.source == "modelscope":
            try:
                from modelscope import snapshot_download
            except ImportError as exc:
                raise RuntimeError("ModelScope source requires the modelscope package") from exc
            self._resolved_model_path = snapshot_download(
                self.config.model_id,
                revision=self.config.revision,
            )
        else:
            self._resolved_model_path = self.config.model_id
        return self._resolved_model_path

    def _resolve_dtype(self, torch: Any) -> Any:
        return {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.config.dtype]

    def act(self, task: Task, trajectory: Trajectory) -> AgentAction:
        self.load()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        tools = self.registry.function_schemas(task.available_tools)
        messages = trajectory_to_chat_messages(trajectory, self.config.system_prompt)
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        inputs = inputs.to(model.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if self.config.do_sample:
            if self.config.temperature is not None:
                generation_kwargs["temperature"] = self.config.temperature
            if self.config.top_p is not None:
                generation_kwargs["top_p"] = self.config.top_p

        if torch.cuda.is_available() and str(model.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        elapsed = time.perf_counter() - started
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        output_ids = generated[0, prompt_tokens:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=False).strip()
        output_text = output_text.removesuffix("<|im_end|>").strip()
        parsed = parse_assistant_output(output_text)
        peak_memory = None
        if torch.cuda.is_available() and str(model.device).startswith("cuda"):
            peak_memory = int(torch.cuda.max_memory_allocated(model.device))
        action_metadata: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": int(output_ids.shape[-1]),
            "latency_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "raw_output": output_text,
        }
        return AgentAction(
            kind=parsed.kind,
            tool_call=parsed.tool_call,
            content=parsed.content,
            metadata=action_metadata,
        )

    def runtime_metadata(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "model_id": self.config.model_id,
            "configured_revision": self.config.revision,
            "source": self.config.source,
            "resolved_model_path": self._resolved_model_path,
        }
        for package in ("torch", "transformers", "modelscope", "ms-swift"):
            try:
                result[f"{package}_version"] = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError:
                result[f"{package}_version"] = None
        if self._model is not None:
            result["device"] = str(self._model.device)
            result["model_dtype"] = str(self._model.dtype)
            commit_hash = getattr(self._model.config, "_commit_hash", None)
            result["loaded_commit_hash"] = commit_hash
        return result
