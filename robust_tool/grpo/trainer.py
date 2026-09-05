"""Trajectory-level GRPO with real tool execution feedback.

This trainer intentionally stays close to the repository's canonical rollout
and evaluator interfaces.  For each task it samples a group of complete
multi-step trajectories, executes every tool call in a fresh environment,
scores the trajectories, and applies a clipped group-relative policy update to
the sampled assistant tokens.
"""

from __future__ import annotations

import importlib.metadata
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_tool.data.schemas import Task
from robust_tool.env.calendar import CalendarEnvironment
from robust_tool.eval.evaluator import TaskEvaluation, evaluate_task
from robust_tool.grpo.config import GRPOConfig
from robust_tool.grpo.objective import clipped_grpo_loss, compute_group_advantages, sequence_logprob
from robust_tool.models.qwen import trajectory_to_chat_messages
from robust_tool.reward.base import RewardResult
from robust_tool.reward.registry import RewardRegistry, default_reward_registry
from robust_tool.rollout.parser import parse_assistant_output
from robust_tool.rollout.runner import AgentAction
from robust_tool.rollout.trajectory import Trajectory, TrajectoryMessage
from robust_tool.tools.registry import ToolRegistry, calendar_registry, registry_for_task_record


@dataclass(frozen=True)
class SampledAction:
    action: AgentAction
    prompt_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    old_logprob: float


@dataclass(frozen=True)
class RolloutRecord:
    task: Task
    trajectory: Trajectory
    evaluation: TaskEvaluation
    reward: RewardResult
    actions: tuple[SampledAction, ...]


class GRPOTrainer:
    """A minimal single-GPU trainer intended for the project's 3090 host."""

    def __init__(
        self,
        config: GRPOConfig,
        tasks: list[Task],
        *,
        reward_registry: RewardRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("GRPO requires at least one task")
        self.config = config
        self.tasks = tasks
        self.reward_registry = reward_registry or default_reward_registry()
        self.tool_registry = tool_registry or calendar_registry()
        self.output_dir = Path(config.output_dir)
        self._set_seed(config.seed)
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._optimizer: Any = None
        self._device: Any = None

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _resolve_model_path(self) -> str:
        model_config = self.config.model
        if model_config.source == "local":
            path = Path(model_config.local_model_path or "").expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"local model path does not exist: {path}")
            return str(path)
        if model_config.source == "modelscope":
            try:
                from modelscope import snapshot_download
            except ImportError as exc:
                raise RuntimeError("ModelScope source requires modelscope") from exc
            return str(snapshot_download(model_config.model_id, revision=model_config.revision))
        return model_config.model_id

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "GRPO requires torch and transformers; install the project training extra"
            ) from exc
        model_config = self.config.model
        if model_config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"configured GRPO device is unavailable: {model_config.device}")
        self._device = torch.device(model_config.device)
        model_path = self._resolve_model_path()
        dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[model_config.dtype]
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=None if model_config.source in {"modelscope", "local"} else model_config.revision,
            trust_remote_code=False,
        )
        model_kwargs: dict[str, Any] = {
            "revision": None if model_config.source in {"modelscope", "local"} else model_config.revision,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        if dtype != "auto":
            major = int(importlib.metadata.version("transformers").split(".", 1)[0])
            model_kwargs["dtype" if major >= 5 else "torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        adapter_path = model_config.adapter_path
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("LoRA GRPO requires peft") from exc
            adapter = Path(adapter_path).expanduser().resolve()
            if not adapter.is_dir():
                raise FileNotFoundError(f"GRPO adapter directory does not exist: {adapter}")
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        model.to(self._device)
        if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("GRPO model has no trainable parameters; provide a trainable LoRA adapter")
        self._optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def _prepare_inputs(self, task: Task, trajectory: Trajectory) -> Any:
        task_registry = registry_for_task_record(task.to_dict(), self.tool_registry)
        messages = trajectory_to_chat_messages(trajectory, self.config.model.system_prompt)
        return self._tokenizer.apply_chat_template(
            messages,
            tools=task_registry.function_schemas(task.available_tools),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_input_tokens,
        ).to(self._device)

    def _sample_action(self, task: Task, trajectory: Trajectory) -> SampledAction:
        torch = self._torch
        self._model.eval()
        inputs = self._prepare_inputs(task, trajectory)
        prompt_ids = tuple(int(value) for value in inputs["input_ids"][0].tolist())
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.model.max_new_tokens,
            "do_sample": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **generation_kwargs)
            output_ids_tensor = generated[0, len(prompt_ids) :]
            old_logprob = sequence_logprob(
                self._model,
                prompt_ids,
                tuple(int(value) for value in output_ids_tensor.tolist()),
                device=self._device,
            )
        output_ids = tuple(int(value) for value in output_ids_tensor.tolist())
        output_text = self._tokenizer.decode(output_ids_tensor, skip_special_tokens=False).strip()
        output_text = output_text.removesuffix("<|im_end|>").strip()
        parsed = parse_assistant_output(output_text)
        action = AgentAction(
            kind=parsed.kind,
            tool_call=parsed.tool_call,
            content=parsed.content,
            metadata={
                "prompt_tokens": len(prompt_ids),
                "output_tokens": len(output_ids),
                "raw_output": output_text,
            },
        )
        return SampledAction(action, prompt_ids, output_ids, float(old_logprob.item()))

    def _rollout(self, task: Task) -> RolloutRecord:
        env = CalendarEnvironment()
        env.reset(task)
        trajectory = Trajectory(
            task_id=task.task_id,
            messages=[TrajectoryMessage(role="user", content=task.user_query)],
            metadata={"policy": "grpo", "seed": self.config.seed},
        )
        sampled_actions: list[SampledAction] = []
        for step in range(self.config.max_steps):
            sampled = self._sample_action(task, trajectory)
            sampled_actions.append(sampled)
            action = sampled.action
            if action.kind == "call" and action.tool_call is not None:
                trajectory.messages.append(
                    TrajectoryMessage(role="assistant", action="call", tool_call=action.tool_call)
                )
                result = env.execute(action.tool_call.to_dict())
                trajectory.messages.append(TrajectoryMessage(role="tool", tool_result=result.to_dict()))
                continue
            trajectory.messages.append(
                TrajectoryMessage(role="assistant", action=action.kind, content=action.content or "")
            )
            break
        else:
            trajectory.metadata = {**trajectory.metadata, "truncated": True}
        trajectory.final_state = env.get_state()
        evaluation = evaluate_task(task, trajectory, self.tool_registry)
        reward = self.reward_registry.score(
            self.config.reward_name,
            evaluation,
            self.config.reward_config,
        )
        return RolloutRecord(task, trajectory, evaluation, reward, tuple(sampled_actions))

    def _loss_for_group(
        self,
        records: list[RolloutRecord],
        advantages: list[float],
    ) -> Any:
        losses: list[Any] = []
        self._model.train()
        for record, advantage in zip(records, advantages):
            if advantage == 0.0:
                continue
            for action in record.actions:
                if not action.output_ids:
                    continue
                new_logprob = sequence_logprob(
                    self._model,
                    action.prompt_ids,
                    action.output_ids,
                    device=self._device,
                )
                losses.append(
                    clipped_grpo_loss(
                        new_logprob,
                        action.old_logprob,
                        advantage,
                        clip_range=self.config.clip_range,
                    )
                )
        if not losses:
            return None
        return self._torch.stack(losses).mean()

    def _save_checkpoint(self, step: int) -> Path:
        checkpoint = self.output_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(checkpoint)
        self._tokenizer.save_pretrained(checkpoint)
        state = {
            "global_step": step,
            "config": self.config.to_dict(),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self._model.parameters() if parameter.requires_grad
            ),
        }
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return checkpoint

    def train(self) -> dict[str, Any]:
        self.load()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records_path = self.output_dir / "reward_records.jsonl"
        self._optimizer.zero_grad(set_to_none=True)
        task_order = list(self.tasks)
        random.Random(self.config.seed).shuffle(task_order)
        update_records: list[dict[str, Any]] = []
        start = time.perf_counter()
        optimizer_steps = 0
        group_count = 0
        skipped_groups = 0
        reward_values: list[float] = []
        trajectories_path = self.output_dir / "trajectories.jsonl"
        with (
            records_path.open("w", encoding="utf-8") as stream,
            trajectories_path.open("w", encoding="utf-8") as trajectory_stream,
        ):
            for step in range(1, self.config.max_updates + 1):
                for accumulation_index in range(self.config.gradient_accumulation_steps):
                    task = task_order[group_count % len(task_order)]
                    group_count += 1
                    group = [self._rollout(task) for _ in range(self.config.group_size)]
                    rewards = [item.reward.value for item in group]
                    reward_values.extend(rewards)
                    advantages = compute_group_advantages(rewards)
                    loss = self._loss_for_group(group, advantages)
                    if loss is None:
                        skipped_groups += 1
                    else:
                        (loss / self.config.gradient_accumulation_steps).backward()
                    for index, (item, advantage) in enumerate(zip(group, advantages)):
                        rollout_id = f"u{step:05d}_g{group_count:05d}_s{index}"
                        trajectory_record = item.trajectory.to_dict()
                        trajectory_record["rollout_id"] = rollout_id
                        trajectory_stream.write(
                            json.dumps(trajectory_record, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        record = {
                            "optimizer_step": step,
                            "accumulation_index": accumulation_index,
                            "group_index": group_count,
                            "sample_index": index,
                            "rollout_id": rollout_id,
                            "task_id": item.task.task_id,
                            "success": bool(item.evaluation.success),
                            "reward": item.reward.to_dict(),
                            "advantage": advantage,
                            "failure_labels": list(item.evaluation.failures.failures),
                            "action_count": len(item.actions),
                        }
                        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        update_records.append(record)
                torch = self._torch
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in self._model.parameters() if parameter.requires_grad],
                    self.config.max_grad_norm,
                )
                self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if step % self.config.save_steps == 0:
                    self._save_checkpoint(step)
        final_checkpoint = self._save_checkpoint(optimizer_steps)
        mean_reward = sum(reward_values) / len(reward_values) if reward_values else None
        success_count = sum(1 for record in update_records if record["success"])
        metrics = {
            "completed": True,
            "optimizer_steps": optimizer_steps,
            "groups": group_count,
            "samples": len(update_records),
            "skipped_groups": skipped_groups,
            "mean_reward": mean_reward,
            "sample_task_success_rate": success_count / len(update_records) if update_records else None,
            "runtime_seconds": time.perf_counter() - start,
            "final_checkpoint": str(final_checkpoint),
        }
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return metrics
