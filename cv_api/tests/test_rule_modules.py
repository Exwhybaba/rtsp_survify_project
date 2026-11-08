"""Independent smoke tests for extracted rule modules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.analytics.rules.after_hours import AfterHoursPresenceHandler
from src.analytics.rules.base import Rule
from src.analytics.rules.dwell_time import DwellTimeExceededHandler
from src.analytics.rules.engine import Zone, ZoneRuleEngine
from src.analytics.rules.loitering import LoiteringHandler
from src.analytics.rules.perimeter_intrusion import PerimeterIntrusionHandler
from src.analytics.rules.zone_entry import RestrictedZoneEntryHandler
from src.analytics.schemas import EventType
from src.analytics.tracker import TrackedObject


def _object(point: tuple[float, float], class_name: str = "person") -> TrackedObject:
    x, y = point
    return TrackedObject(
        track_id=1,
        bbox=(x - 5, y - 5, x + 5, y + 5),
        confidence=0.9,
        class_name=class_name,
        centroid=point,
        age=1,
        hits=1,
    )


def test_default_registry_contains_extracted_handlers() -> None:
    zone = Zone(
        "zone",
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        rules=[Rule(EventType.RESTRICTED_ZONE_ENTRY.value)],
    )
    engine = ZoneRuleEngine([zone], [])
    assert isinstance(engine.handlers[EventType.RESTRICTED_ZONE_ENTRY.value], RestrictedZoneEntryHandler)
    assert isinstance(engine.handlers[EventType.PERIMETER_INTRUSION.value], PerimeterIntrusionHandler)
    assert isinstance(engine.handlers[EventType.DWELL_TIME_EXCEEDED.value], DwellTimeExceededHandler)
    assert isinstance(engine.handlers[EventType.LOITERING_DETECTED.value], LoiteringHandler)
    assert isinstance(engine.handlers[EventType.AFTER_HOURS_PRESENCE.value], AfterHoursPresenceHandler)


def test_custom_handler_registry_is_supported() -> None:
    handler = RestrictedZoneEntryHandler()
    engine = ZoneRuleEngine(
        [
            Zone(
                "zone",
                [(0, 0), (100, 0), (100, 100), (0, 100)],
                rules=[Rule(EventType.RESTRICTED_ZONE_ENTRY.value)],
            )
        ],
        [],
        handlers={EventType.RESTRICTED_ZONE_ENTRY.value: handler},
    )
    events = engine.evaluate(
        "cam1",
        [_object((-20, -20)), _object((50, 50))],
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert events[0].event_type == EventType.RESTRICTED_ZONE_ENTRY


def test_extracted_handlers_keep_dwell_and_after_hours_behavior() -> None:
    zone = Zone(
        "zone",
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        schedule={"timezone": "UTC", "open_hours": {"mon": ["08:00-20:00"]}},
        rules=[
            Rule(EventType.LOITERING_DETECTED.value, {"max_loiter_seconds": 5}),
            Rule(EventType.AFTER_HOURS_PRESENCE.value),
        ],
    )
    engine = ZoneRuleEngine([zone], [])
    start = datetime(2026, 1, 5, 21, tzinfo=timezone.utc)
    engine.evaluate("cam1", [_object((50, 50))], start)
    events = engine.evaluate("cam1", [_object((50, 50))], start + timedelta(seconds=5))
    assert [event.event_type for event in events] == [EventType.LOITERING_DETECTED]
