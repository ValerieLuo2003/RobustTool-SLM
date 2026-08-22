"""Model-free policies and a shared environment rollout loop."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from robust_tool.data.schemas import Task, ToolCall
from robust_tool.env.calendar import CalendarEnvironment
from robust_tool.rollout.trajectory import Trajectory, TrajectoryMessage


@dataclass(frozen=True)
class AgentAction:
    kind: str
    tool_call: ToolCall | None = None
    content: str | None = None
    metadata: Mapping[str, Any] | None = None


class Policy(Protocol):
    name: str

    def act(self, task: Task, trajectory: Trajectory) -> AgentAction:
        ...


class OraclePolicy:
    name = "oracle"

    def act(self, task: Task, trajectory: Trajectory) -> AgentAction:
        completed = len(trajectory.tool_calls())
        if completed < len(task.reference_calls):
            return AgentAction("call", task.reference_calls[completed])
        if task.expected_action == "clarify":
            return AgentAction("clarify", content="Could you provide the date, time, duration, and attendees?")
        return AgentAction("respond", content="Done." if completed else "I can help manage calendar events and availability.")


class RandomPolicy:
    """Seeded weak policy used to exercise evaluator failure paths."""

    name = "random"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def act(self, task: Task, trajectory: Trajectory) -> AgentAction:
        if trajectory.assistant_actions():
            return AgentAction("respond", content="Finished.")
        decision = self._rng.choice(["call", "call", "clarify", "respond"])
        if decision != "call":
            content = "What date and time should I use?" if decision == "clarify" else "I can help with that."
            return AgentAction(decision, content=content)

        candidates = list(task.available_tools) + ["invented_calendar_tool"]
        name = self._rng.choice(candidates)
        arguments = copy.deepcopy(dict(task.reference_calls[0].arguments)) if task.reference_calls else {}
        mutation = self._rng.choice(["keep", "drop", "extra", "empty"])
        if mutation == "drop" and arguments:
            arguments.pop(self._rng.choice(list(arguments)))
        elif mutation == "extra":
            arguments["unsupported"] = True
        elif mutation == "empty":
            arguments = {}
        return AgentAction("call", ToolCall(name, arguments))


def run_policy(tasks: list[Task], policy: Policy, *, max_steps: int = 4) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for task in tasks:
        env = CalendarEnvironment()
        env.reset(task)
        trajectory = Trajectory(
            task_id=task.task_id,
            messages=[TrajectoryMessage(role="user", content=task.user_query)],
            metadata={"policy": policy.name},
        )
        step_metadata: list[Mapping[str, Any]] = []
        for _step in range(max_steps):
            action = policy.act(task, trajectory)
            if action.metadata:
                step_metadata.append({"step": _step, **dict(action.metadata)})
            if action.kind == "call" and action.tool_call is not None:
                trajectory.messages.append(
                    TrajectoryMessage(role="assistant", action="call", tool_call=action.tool_call)
                )
                result = env.execute(action.tool_call.to_dict())
                trajectory.messages.append(
                    TrajectoryMessage(role="tool", tool_result=result.to_dict())
                )
                continue
            trajectory.messages.append(
                TrajectoryMessage(role="assistant", action=action.kind, content=action.content or "")
            )
            break
        else:
            trajectory.metadata = {**trajectory.metadata, "truncated": True}
        if step_metadata:
            trajectory.metadata = {**trajectory.metadata, "generation_steps": step_metadata}
        trajectory.final_state = env.get_state()
        trajectories.append(trajectory)
    return trajectories
