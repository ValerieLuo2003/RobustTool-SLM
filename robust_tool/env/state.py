"""Serializable Calendar state with deterministic ID allocation."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

from robust_tool.env.errors import ToolExecutionError

_EVENT_ID_PATTERN = re.compile(r"^evt-(\d+)$")


def normalize_datetime(value: str) -> str:
    """Validate and canonicalize a timezone-naive ISO-8601 datetime."""

    if not isinstance(value, str):
        raise ValueError("datetime must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 datetime: {value!r}") from exc
    if parsed.tzinfo is not None:
        raise ValueError("timezone-aware datetimes are not supported in the local calendar environment")
    return parsed.replace(microsecond=0).isoformat(timespec="seconds")


def validate_interval(start: str, end: str) -> tuple[str, str]:
    normalized_start = normalize_datetime(start)
    normalized_end = normalize_datetime(end)
    if datetime.fromisoformat(normalized_end) <= datetime.fromisoformat(normalized_start):
        raise ToolExecutionError("invalid_interval", "end must be later than start")
    return normalized_start, normalized_end


@dataclass
class CalendarEvent:
    event_id: str
    title: str
    start: str
    end: str
    location: str = ""
    description: str = ""
    attendees: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "CalendarEvent":
        start, end = validate_interval(record["start"], record["end"])
        return cls(
            event_id=str(record["event_id"]),
            title=str(record["title"]),
            start=start,
            end=end,
            location=str(record.get("location", "")),
            description=str(record.get("description", "")),
            attendees=[str(value) for value in record.get("attendees", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start": self.start,
            "end": self.end,
            "location": self.location,
            "description": self.description,
            "attendees": list(self.attendees),
        }


class CalendarState:
    """In-memory state; reset creates a deep, isolated copy per task."""

    def __init__(self, events: Iterable[Mapping[str, Any]] = ()) -> None:
        parsed = [CalendarEvent.from_dict(copy.deepcopy(event)) for event in events]
        ids = [event.event_id for event in parsed]
        if len(ids) != len(set(ids)):
            raise ValueError("calendar state contains duplicate event IDs")
        self._events: dict[str, CalendarEvent] = {event.event_id: event for event in parsed}
        existing_numbers = [
            int(match.group(1))
            for event_id in ids
            if (match := _EVENT_ID_PATTERN.match(event_id)) is not None
        ]
        self._next_id = max(existing_numbers, default=0) + 1

    def snapshot(self) -> dict[str, Any]:
        events = sorted(self._events.values(), key=lambda event: (event.start, event.event_id))
        return {"events": [event.to_dict() for event in events]}

    def allocate_event_id(self) -> str:
        while True:
            candidate = f"evt-{self._next_id:04d}"
            self._next_id += 1
            if candidate not in self._events:
                return candidate

    def get(self, event_id: str) -> CalendarEvent:
        try:
            return self._events[event_id]
        except KeyError as exc:
            raise ToolExecutionError("event_not_found", f"event does not exist: {event_id}") from exc

    def put(self, event: CalendarEvent) -> None:
        self._events[event.event_id] = event

    def remove(self, event_id: str) -> CalendarEvent:
        event = self.get(event_id)
        del self._events[event_id]
        return event

    def events(self) -> tuple[CalendarEvent, ...]:
        return tuple(sorted(self._events.values(), key=lambda event: (event.start, event.event_id)))

    def conflicts(self, start: str, end: str, *, exclude_event_id: str | None = None) -> list[CalendarEvent]:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        return [
            event
            for event in self.events()
            if event.event_id != exclude_event_id
            and start_dt < datetime.fromisoformat(event.end)
            and datetime.fromisoformat(event.start) < end_dt
        ]
