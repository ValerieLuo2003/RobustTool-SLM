"""Normalize model/tool-call payloads without hiding JSON failures."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from robust_tool.data.schemas import ToolCall

TOOL_CALL_PATTERNS = (
    re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL),
    re.compile(r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>", re.DOTALL),
)
UNCLOSED_TOOL_MARKERS = ("<tool_call>", "<|tool_call_start|>")


@dataclass(frozen=True)
class ParsedAssistantOutput:
    kind: str
    tool_call: ToolCall | None = None
    content: str | None = None


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


def _looks_like_clarification(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    signals = (
        "could you provide",
        "can you provide",
        "please provide",
        "what date",
        "what time",
        "which event",
        "which date",
        "need the date",
        "请提供",
        "请问",
        "需要您提供",
        "能否提供",
        "哪一个",
        "什么时间",
    )
    return text.rstrip().endswith(("?", "？")) or any(signal in normalized for signal in signals)


def parse_assistant_output(output_text: str) -> ParsedAssistantOutput:
    """Parse one generated turn while preserving malformed tool-call evidence."""

    text = output_text.strip()
    for pattern in TOOL_CALL_PATTERNS:
        match = pattern.search(text)
        if match:
            call = parse_tool_call(match.group(1).strip())
            if call.raw != text:
                call = ToolCall(
                    call.name,
                    call.arguments,
                    json_valid=call.json_valid,
                    raw=text,
                    parse_error=call.parse_error,
                )
            return ParsedAssistantOutput("call", tool_call=call)

    for marker in UNCLOSED_TOOL_MARKERS:
        if marker in text:
            payload = text.split(marker, 1)[1].split("<|im_end|>", 1)[0].strip()
            call = parse_tool_call(payload)
            return ParsedAssistantOutput(
                "call",
                tool_call=ToolCall(
                    call.name,
                    call.arguments,
                    json_valid=False,
                    raw=text,
                    parse_error=call.parse_error or "unclosed tool-call tag",
                ),
            )

    if text.startswith("{"):
        call = parse_tool_call(text)
        return ParsedAssistantOutput("call", tool_call=call)
    kind = "clarify" if _looks_like_clarification(text) else "respond"
    return ParsedAssistantOutput(kind, content=text)
