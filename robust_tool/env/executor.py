"""Schema validation and safe tool dispatch."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from robust_tool.env.base import ToolExecutionResult, ValidationIssue
from robust_tool.env.errors import ToolExecutionError
from robust_tool.env.state import normalize_datetime
from robust_tool.tools.registry import ToolRegistry


def _matches_type(value: Any, expected: str) -> bool:
    checks = {
        "string": lambda item: isinstance(item, str),
        "boolean": lambda item: isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "object": lambda item: isinstance(item, Mapping),
        "array": lambda item: isinstance(item, list),
        "null": lambda item: item is None,
    }
    return checks.get(expected, lambda _item: True)(value)


def validate_arguments(arguments: Any, schema: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
    """Validate the JSON Schema subset used by local tools."""

    if not isinstance(arguments, Mapping):
        return (ValidationIssue("wrong_argument_type", None, "arguments must be a JSON object"),)

    issues: list[ValidationIssue] = []
    properties = schema.get("properties", {})
    for required in schema.get("required", []):
        if required not in arguments:
            issues.append(ValidationIssue("missing_argument", required, f"missing required argument: {required}"))

    if schema.get("additionalProperties") is False:
        for field in arguments:
            if field not in properties:
                issues.append(ValidationIssue("extra_argument", str(field), f"unexpected argument: {field}"))

    for field, value in arguments.items():
        field_schema = properties.get(field)
        if field_schema is None:
            continue
        expected_type = field_schema.get("type")
        if expected_type and not _matches_type(value, expected_type):
            issues.append(
                ValidationIssue(
                    "wrong_argument_type",
                    str(field),
                    f"{field} must have type {expected_type}",
                )
            )
            continue
        if "enum" in field_schema and value not in field_schema["enum"]:
            issues.append(ValidationIssue("wrong_argument_value", str(field), f"{field} is not an allowed value"))
        if expected_type == "string" and len(value) < field_schema.get("minLength", 0):
            issues.append(ValidationIssue("wrong_argument_value", str(field), f"{field} is too short"))
        if field_schema.get("format") == "date-time":
            try:
                normalize_datetime(value)
            except (TypeError, ValueError) as exc:
                issues.append(ValidationIssue("wrong_argument_value", str(field), str(exc)))
        if expected_type == "array" and "items" in field_schema:
            item_type = field_schema["items"].get("type")
            for index, item in enumerate(value):
                if item_type and not _matches_type(item, item_type):
                    issues.append(
                        ValidationIssue(
                            "wrong_argument_type",
                            f"{field}[{index}]",
                            f"array item must have type {item_type}",
                        )
                    )
    return tuple(issues)


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        dispatcher: Callable[[str, Mapping[str, Any]], tuple[Any, bool]],
    ) -> None:
        self.registry = registry
        self.dispatcher = dispatcher

    def execute(self, tool_call: Mapping[str, Any]) -> ToolExecutionResult:
        name = tool_call.get("name")
        if not isinstance(name, str) or not self.registry.has(name):
            display_name = str(name) if name is not None else "<missing>"
            return ToolExecutionResult.failure(
                display_name,
                "hallucinated_tool",
                f"tool is not registered: {display_name}",
            )
        arguments = tool_call.get("arguments", {})
        definition = self.registry.get(name)
        issues = validate_arguments(arguments, definition.parameters)
        if issues:
            return ToolExecutionResult.failure(
                name,
                "invalid_parameters",
                "tool arguments failed schema validation",
                validation_issues=issues,
            )
        try:
            data, state_changed = self.dispatcher(name, arguments)
        except ToolExecutionError as exc:
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                error=exc.detail,
            )
        except (TypeError, ValueError) as exc:
            return ToolExecutionResult.failure(name, "invalid_parameters", str(exc))
        return ToolExecutionResult(
            tool_name=name,
            ok=True,
            data=data,
            state_changed=state_changed,
        )
