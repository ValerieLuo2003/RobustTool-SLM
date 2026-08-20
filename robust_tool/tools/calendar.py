"""Deterministic implementations of the five Calendar tools."""

from __future__ import annotations

from typing import Any, Mapping

from robust_tool.env.errors import ToolExecutionError
from robust_tool.env.state import CalendarEvent, CalendarState, normalize_datetime, validate_interval


class CalendarTools:
    def __init__(self, state: CalendarState) -> None:
        self.state = state

    def execute(self, name: str, arguments: Mapping[str, Any]) -> tuple[Any, bool]:
        handler = getattr(self, name, None)
        if handler is None or name.startswith("_"):
            raise ToolExecutionError("hallucinated_tool", f"unknown Calendar tool: {name}")
        return handler(**dict(arguments))

    def list_events(self, start: str | None = None, end: str | None = None) -> tuple[dict[str, Any], bool]:
        normalized_start = normalize_datetime(start) if start is not None else None
        normalized_end = normalize_datetime(end) if end is not None else None
        if normalized_start and normalized_end:
            validate_interval(normalized_start, normalized_end)
        events = [
            event.to_dict()
            for event in self.state.events()
            if (normalized_start is None or event.start >= normalized_start)
            and (normalized_end is None or event.start < normalized_end)
        ]
        return {"events": events, "count": len(events)}, False

    def create_event(
        self,
        title: str,
        start: str,
        end: str,
        location: str = "",
        description: str = "",
        attendees: list[str] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        normalized_start, normalized_end = validate_interval(start, end)
        conflicts = self.state.conflicts(normalized_start, normalized_end)
        if conflicts:
            raise ToolExecutionError(
                "conflict",
                "requested interval overlaps: " + ", ".join(event.event_id for event in conflicts),
            )
        event = CalendarEvent(
            event_id=self.state.allocate_event_id(),
            title=title.strip(),
            start=normalized_start,
            end=normalized_end,
            location=location,
            description=description,
            attendees=list(attendees or []),
        )
        self.state.put(event)
        return {"event": event.to_dict()}, True

    def update_event(
        self,
        event_id: str,
        title: str | None = None,
        start: str | None = None,
        end: str | None = None,
        location: str | None = None,
        description: str | None = None,
        attendees: list[str] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        if all(value is None for value in (title, start, end, location, description, attendees)):
            raise ToolExecutionError("no_update_fields", "update_event requires at least one field to change")
        current = self.state.get(event_id)
        normalized_start, normalized_end = validate_interval(start or current.start, end or current.end)
        conflicts = self.state.conflicts(normalized_start, normalized_end, exclude_event_id=event_id)
        if conflicts:
            raise ToolExecutionError(
                "conflict",
                "updated interval overlaps: " + ", ".join(event.event_id for event in conflicts),
            )
        updated = CalendarEvent(
            event_id=current.event_id,
            title=current.title if title is None else title.strip(),
            start=normalized_start,
            end=normalized_end,
            location=current.location if location is None else location,
            description=current.description if description is None else description,
            attendees=current.attendees if attendees is None else list(attendees),
        )
        self.state.put(updated)
        return {"event": updated.to_dict()}, True

    def delete_event(self, event_id: str) -> tuple[dict[str, Any], bool]:
        deleted = self.state.remove(event_id)
        return {"deleted_event": deleted.to_dict()}, True

    def check_availability(
        self,
        start: str,
        end: str,
        exclude_event_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        normalized_start, normalized_end = validate_interval(start, end)
        conflicts = self.state.conflicts(
            normalized_start,
            normalized_end,
            exclude_event_id=exclude_event_id,
        )
        return {
            "available": not conflicts,
            "conflicting_event_ids": [event.event_id for event in conflicts],
            "start": normalized_start,
            "end": normalized_end,
        }, False
