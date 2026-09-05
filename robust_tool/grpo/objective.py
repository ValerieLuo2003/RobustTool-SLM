"""Pure pieces of the group-relative policy objective."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def compute_group_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float = 1e-8,
) -> list[float]:
    """Normalize rewards within one prompt group as GRPO advantages.

    A group with no reward variance has no preference signal, so it returns
    zero advantages rather than inventing an arbitrary update direction.
    """

    if not rewards:
        return []
    values = [float(value) for value in rewards]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    standard_deviation = math.sqrt(variance)
    if standard_deviation < epsilon:
        return [0.0 for _ in values]
    return [(value - mean) / (standard_deviation + epsilon) for value in values]


def clipped_grpo_loss(
    new_logprob: Any,
    old_logprob: float,
    advantage: float,
    *,
    clip_range: float,
) -> Any:
    """Return one differentiable clipped GRPO/PPO-style loss term."""

    import torch

    old = torch.as_tensor(float(old_logprob), device=new_logprob.device, dtype=new_logprob.dtype)
    adv = torch.as_tensor(float(advantage), device=new_logprob.device, dtype=new_logprob.dtype)
    ratio = torch.exp(new_logprob - old)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    return -torch.minimum(ratio * adv, clipped_ratio * adv)


def sequence_logprob(
    model: Any,
    prompt_ids: Sequence[int],
    output_ids: Sequence[int],
    *,
    device: Any,
) -> Any:
    """Compute the summed log-probability of output tokens only."""

    import torch

    if not output_ids:
        return torch.zeros((), device=device, dtype=torch.float32)
    full_ids = torch.tensor(
        [list(prompt_ids) + list(output_ids)],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(full_ids)
    outputs = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    labels = full_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    target_count = len(output_ids)
    target_log_probs = log_probs[:, -target_count:, :].gather(
        dim=-1,
        index=labels[:, -target_count:].unsqueeze(-1),
    )
    return target_log_probs.squeeze(-1).sum()
