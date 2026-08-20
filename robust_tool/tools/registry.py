"""Tool registry backed by versioned JSON Schema files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ToolDefinition:
    """Serializable tool metadata used by prompts and the executor."""

    name: str
    domain: str
    description: str
    parameters: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "parameters": dict(self.parameters),
        }

    def to_function_schema(self) -> dict[str, Any]:
        """Return the common chat-template function schema shape."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


class ToolRegistry:
    """Immutable-by-convention lookup for declared tools."""

    def __init__(self, definitions: Iterable[ToolDefinition] = ()) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"duplicate tool: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._definitions

    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def definitions(self, names: Iterable[str] | None = None) -> tuple[ToolDefinition, ...]:
        selected = self.names() if names is None else tuple(names)
        return tuple(self.get(name) for name in selected)

    def function_schemas(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        return [definition.to_function_schema() for definition in self.definitions(names)]


def _load_definitions(filename: str, domain: str) -> list[ToolDefinition]:
    path = Path(__file__).with_name("schemas") / filename
    records = json.loads(path.read_text(encoding="utf-8"))
    return [
        ToolDefinition(
            name=record["name"],
            domain=domain,
            description=record["description"],
            parameters=record["parameters"],
        )
        for record in records
    ]


def calendar_registry() -> ToolRegistry:
    """Build a fresh registry containing the five Week 1 Calendar tools."""

    return ToolRegistry(_load_definitions("calendar.json", "calendar"))
