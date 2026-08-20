from __future__ import annotations

import unittest

from robust_tool.env.executor import validate_arguments
from robust_tool.tools.registry import calendar_registry


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = calendar_registry()

    def test_calendar_registry_exposes_five_function_schemas(self) -> None:
        self.assertEqual(
            self.registry.names(),
            ("list_events", "create_event", "update_event", "delete_event", "check_availability"),
        )
        schemas = self.registry.function_schemas()
        self.assertTrue(all(schema["type"] == "function" for schema in schemas))
        self.assertEqual(schemas[1]["function"]["parameters"]["required"], ["title", "start", "end"])

    def test_schema_validation_separates_missing_extra_and_type_errors(self) -> None:
        schema = self.registry.get("create_event").parameters
        issues = validate_arguments(
            {"title": 123, "start": "not-a-date", "unexpected": True},
            schema,
        )
        self.assertEqual(
            {issue.code for issue in issues},
            {"missing_argument", "extra_argument", "wrong_argument_type", "wrong_argument_value"},
        )

    def test_boolean_is_not_accepted_as_integer(self) -> None:
        issues = validate_arguments(True, {"type": "object"})
        self.assertEqual(issues[0].code, "wrong_argument_type")


if __name__ == "__main__":
    unittest.main()
