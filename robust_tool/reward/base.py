"""Shared reward result types.

Rewards are computed after a trajectory has been replayed by the benchmark
evaluator.  Keeping the result structured makes reward shaping auditable and
prevents the trainer from silently collapsing different failure modes into a
single unexplained scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardResult:
    """A scalar reward together with the components that produced it."""

    name: str
    value: float
    components: Mapping[str, float] = field(default_factory=dict)
    penalties: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "components": {str(k): float(v) for k, v in self.components.items()},
            "penalties": {str(k): float(v) for k, v in self.penalties.items()},
            "metadata": dict(self.metadata),
        }
