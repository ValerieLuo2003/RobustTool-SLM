from __future__ import annotations

import copy
import unittest

from robust_tool.env.calendar import CalendarEnvironment


INITIAL_EVENT = {
    "event_id": "evt-0001",
    "title": "Design review",
    "start": "2026-08-10T09:00:00",
    "end": "2026-08-10T10:00:00",
    "location": "Room A",
    "description": "",
    "attendees": ["alice@example.com"],
}


def make_task(goal_state=None, available_tools=None):
    return {
        "task_id": "calendar_test",
        "domain": "calendar",
        "user_query": "test",
        "available_tools": available_tools
        or ["list_events", "create_event", "update_event", "delete_event", "check_availability"],
        "initial_state": {"events": [copy.deepcopy(INITIAL_EVENT)]},
        "goal_state": goal_state or {},
    }


class CalendarEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CalendarEnvironment()
        self.env.reset(make_task())

    def test_full_create_list_update_delete_lifecycle(self) -> None:
        created = self.env.execute(
            {
                "name": "create_event",
                "arguments": {
                    "title": "Lunch",
                    "start": "2026-08-10T12:00",
                    "end": "2026-08-10T13:00",
                },
            }
        )
        self.assertTrue(created.ok)
        self.assertEqual(created.data["event"]["event_id"], "evt-0002")

        listed = self.env.execute(
            {
                "name": "list_events",
                "arguments": {"start": "2026-08-10T11:00:00", "end": "2026-08-10T14:00:00"},
            }
        )
        self.assertEqual(listed.data["count"], 1)
        self.assertEqual(listed.data["events"][0]["title"], "Lunch")

        updated = self.env.execute(
            {"name": "update_event", "arguments": {"event_id": "evt-0002", "location": "Cafe"}}
        )
        self.assertTrue(updated.ok)
        self.assertEqual(updated.data["event"]["location"], "Cafe")

        deleted = self.env.execute(
            {"name": "delete_event", "arguments": {"event_id": "evt-0002"}}
        )
        self.assertTrue(deleted.ok)
        self.assertEqual(len(self.env.get_state()["events"]), 1)

    def test_conflict_does_not_mutate_state(self) -> None:
        before = self.env.get_state()
        result = self.env.execute(
            {
                "name": "create_event",
                "arguments": {
                    "title": "Overlap",
                    "start": "2026-08-10T09:30:00",
                    "end": "2026-08-10T10:30:00",
                },
            }
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "conflict")
        self.assertEqual(self.env.get_state(), before)

    def test_adjacent_interval_is_available(self) -> None:
        result = self.env.execute(
            {
                "name": "check_availability",
                "arguments": {"start": "2026-08-10T10:00:00", "end": "2026-08-10T11:00:00"},
            }
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.data["available"])

    def test_unavailable_tool_is_distinct_from_hallucinated_tool(self) -> None:
        self.env.reset(make_task(available_tools=["list_events"]))
        unavailable = self.env.execute(
            {
                "name": "create_event",
                "arguments": {
                    "title": "Lunch",
                    "start": "2026-08-10T12:00:00",
                    "end": "2026-08-10T13:00:00",
                },
            }
        )
        hallucinated = self.env.execute({"name": "book_table", "arguments": {}})
        self.assertEqual(unavailable.error.code, "tool_unavailable")
        self.assertEqual(hallucinated.error.code, "hallucinated_tool")

    def test_validation_failure_does_not_call_tool(self) -> None:
        result = self.env.execute({"name": "create_event", "arguments": {"title": "Missing dates"}})
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "invalid_parameters")
        self.assertEqual({issue.code for issue in result.validation_issues}, {"missing_argument"})
        self.assertEqual(len(self.env.get_state()["events"]), 1)

    def test_reset_isolated_and_deterministic(self) -> None:
        task = make_task()
        self.env.reset(task)
        self.env.execute({"name": "delete_event", "arguments": {"event_id": "evt-0001"}})
        self.env.reset(task)
        self.assertEqual(self.env.get_state()["events"], [INITIAL_EVENT])
        task["initial_state"]["events"][0]["title"] = "mutated outside"
        self.assertEqual(self.env.get_state()["events"][0]["title"], "Design review")

    def test_goal_checks_state_and_observation(self) -> None:
        task = make_task(
            goal_state={
                "events": {"contains": [{"event_id": "evt-0001", "title": "Design review"}], "count": 1},
                "required_observations": [
                    {
                        "tool_name": "check_availability",
                        "arguments": {
                            "start": "2026-08-10T10:00:00",
                            "end": "2026-08-10T11:00:00",
                        },
                        "result": {"available": True},
                    }
                ],
            }
        )
        self.env.reset(task)
        self.assertFalse(self.env.check_goal())
        self.env.execute(
            {
                "name": "check_availability",
                "arguments": {"start": "2026-08-10T10:00:00", "end": "2026-08-10T11:00:00"},
            }
        )
        self.assertTrue(self.env.check_goal())


if __name__ == "__main__":
    unittest.main()
