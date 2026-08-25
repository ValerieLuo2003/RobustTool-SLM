"""Tool registry backed by versioned JSON Schema files."""

from __future__ import annotations

import copy
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

    @classmethod
    def from_dict(
        cls,
        record: Mapping[str, Any],
        *,
        default_domain: str = "calendar",
    ) -> "ToolDefinition":
        parameters = record.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("tool parameters must be a JSON object")
        name = str(record.get("name", "")).strip()
        description = str(record.get("description", "")).strip()
        if not name or not description:
            raise ValueError("tool definition requires non-empty name and description")
        return cls(
            name=name,
            domain=str(record.get("domain", default_domain)),
            description=description,
            parameters=copy.deepcopy(dict(parameters)),
        )

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


def registry_for_task_record(
    task: Mapping[str, Any],
    base_registry: ToolRegistry | None = None,
) -> ToolRegistry:
    """Build an isolated registry with deterministic task-level prompt perturbations."""

    base = base_registry or calendar_registry()
    metadata = task.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("task metadata must be a JSON object")
    description_overrides = metadata.get("tool_description_overrides", {})
    additions = metadata.get("tool_schema_additions", [])
    if not isinstance(description_overrides, Mapping):
        raise ValueError("tool_description_overrides must be a JSON object")
    if not isinstance(additions, list):
        raise ValueError("tool_schema_additions must be a JSON array")

    unknown_overrides = set(description_overrides) - set(base.names())
    if unknown_overrides:
        raise ValueError(f"description overrides reference unknown tools: {sorted(unknown_overrides)}")
    definitions = []
    for definition in base.definitions():
        description = str(description_overrides.get(definition.name, definition.description)).strip()
        if not description:
            raise ValueError(f"empty description override for {definition.name}")
        definitions.append(
            ToolDefinition(
                name=definition.name,
                domain=definition.domain,
                description=description,
                parameters=copy.deepcopy(dict(definition.parameters)),
            )
        )
    registry = ToolRegistry(definitions)
    for record in additions:
        if not isinstance(record, Mapping):
            raise ValueError("each tool schema addition must be a JSON object")
        registry.register(ToolDefinition.from_dict(record))

    available_tools = task.get("available_tools", [])
    if not isinstance(available_tools, (list, tuple)):
        raise ValueError("available_tools must be a list")
    missing = [str(name) for name in available_tools if not registry.has(str(name))]
    if missing:
        raise ValueError(f"task advertises tools without schemas: {missing}")
    return registry
