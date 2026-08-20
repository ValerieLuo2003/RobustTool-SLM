"""Normalize model/tool-call payloads without hiding JSON failures."""

from __future__ import annotations

import json
from typing import Any, Mapping

from robust_tool.data.schemas import ToolCall


def parse_tool_call(payload: str | Mapping[str, Any]) -> ToolCall:
    raw: str | None = payload if isinstance(payload, str) else None
    if isinstance(payload, str):
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as exc:
            return ToolCall("", {}, json_valid=False, raw=payload, parse_error=str(exc))
    else:
        record = dict(payload)
    if not isinstance(record, Mapping):
        return ToolCall("", {}, json_valid=False, raw=raw, parse_error="tool call must be a JSON object")

    name = record.get("name")
    arguments = record.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return ToolCall(str(name or ""), {}, json_valid=False, raw=raw, parse_error=str(exc))
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        return ToolCall(
            str(name or ""),
            {},
            json_valid=False,
            raw=raw,
            parse_error="tool call requires a string name and object arguments",
        )
    return ToolCall(name, dict(arguments), json_valid=True, raw=raw)
