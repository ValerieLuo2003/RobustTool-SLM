from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robust_tool.data.generator import generate_calendar_toy_tasks
from robust_tool.data.schemas import ToolCall
from robust_tool.models.config import ModelInferenceConfig, load_model_config
from robust_tool.models.qwen import trajectory_to_chat_messages
from robust_tool.rollout.parser import parse_assistant_output
from robust_tool.rollout.trajectory import Trajectory, TrajectoryMessage


class ModelAdapterTests(unittest.TestCase):
    def test_checked_in_qwen_config_is_valid_and_pinned(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "models" / "qwen2_5_1_5b_instruct.json"
        config = load_model_config(path)
        self.assertEqual(config.model_id, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertRegex(config.revision, r"^[0-9a-f]{40}$")
        self.assertFalse(config.do_sample)

    def test_invalid_local_config_is_rejected(self) -> None:
        record = ModelInferenceConfig(
            model_id="example/model",
            revision="abc",
        ).to_dict()
        record["source"] = "local"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "local_model_path"):
                load_model_config(path)

    def test_qwen_tool_tag_is_parsed(self) -> None:
        parsed = parse_assistant_output(
            '<tool_call>\n{"name":"delete_event","arguments":{"event_id":"evt-0001"}}\n</tool_call>'
        )
        self.assertEqual(parsed.kind, "call")
        self.assertEqual(parsed.tool_call.name, "delete_event")
        self.assertEqual(parsed.tool_call.arguments["event_id"], "evt-0001")

    def test_malformed_tag_is_kept_as_invalid_call(self) -> None:
        parsed = parse_assistant_output('<tool_call>{"name":"list_events","arguments":')
        self.assertEqual(parsed.kind, "call")
        self.assertFalse(parsed.tool_call.json_valid)
        self.assertIn("<tool_call>", parsed.tool_call.raw)

    def test_plain_question_is_classified_as_clarification(self) -> None:
        parsed = parse_assistant_output("Could you provide the date and time?")
        self.assertEqual(parsed.kind, "clarify")

    def test_trajectory_conversion_includes_tool_result_without_reference_answer(self) -> None:
        task = generate_calendar_toy_tasks()[0]
        trajectory = Trajectory(
            task.task_id,
            messages=[
                TrajectoryMessage("user", task.user_query),
                TrajectoryMessage("assistant", action="call", tool_call=ToolCall("list_events", {})),
                TrajectoryMessage("tool", tool_result={"ok": True, "data": {"count": 3}}),
            ],
        )
        messages = trajectory_to_chat_messages(trajectory, "system")
        self.assertEqual(messages[0], {"role": "system", "content": "system"})
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "list_events")
        self.assertIn('"count": 3', messages[3]["content"])
        self.assertNotIn("reference_calls", json.dumps(messages))


if __name__ == "__main__":
    unittest.main()
