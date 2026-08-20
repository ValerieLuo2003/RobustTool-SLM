from __future__ import annotations

import unittest

from robust_tool.data.generator import generate_calendar_toy_tasks
from robust_tool.rollout.parser import parse_tool_call
from robust_tool.rollout.runner import OraclePolicy, RandomPolicy, run_policy


class RolloutTests(unittest.TestCase):
    def test_parser_preserves_invalid_json(self) -> None:
        call = parse_tool_call('{"name": "list_events", "arguments": ')
        self.assertFalse(call.json_valid)
        self.assertIsNotNone(call.parse_error)

    def test_oracle_rollout_traces_user_call_result_and_answer(self) -> None:
        task = generate_calendar_toy_tasks()[0]
        trajectory = run_policy([task], OraclePolicy())[0]
        self.assertEqual([message.role for message in trajectory.messages], ["user", "assistant", "tool", "assistant"])
        self.assertTrue(trajectory.messages[2].tool_result["ok"])

    def test_random_rollout_is_seed_deterministic(self) -> None:
        tasks = generate_calendar_toy_tasks()[:5]
        left = [trajectory.to_dict() for trajectory in run_policy(tasks, RandomPolicy(17))]
        right = [trajectory.to_dict() for trajectory in run_policy(tasks, RandomPolicy(17))]
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
