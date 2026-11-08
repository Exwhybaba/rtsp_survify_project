
"""Unit tests for the zone rule engine.

Tests geometry, restricted-zone entry, line crossing, directional movement,
people counting, occupancy and dwell-time rules.

All tests use synthetic point trajectories. No video or ML model is required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.analytics.rules.base import Rule
from src.analytics.rules.engine import (
    Line,
    Zone,
    ZoneRuleEngine,
    get_line_crossing_direction,
    is_armed,
    point_in_polygon,
    segments_intersect,
)
from src.analytics.schemas import EventType
from src.analytics.tracker import TrackedObject

CAMERA_ID = "cam-test"

SQUARE_ZONE = [
    (100.0, 100.0),
    (200.0, 100.0),
    (200.0, 200.0),
    (100.0, 200.0),
]

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def obj_at(
    point: tuple[float, float],
    track_id: int = 1,
) -> TrackedObject:
    """Create a synthetic tracked person at a given point."""
    x, y = point
    half = 5.0

    return TrackedObject(
        track_id=track_id,
        bbox=(x - half, y - half, x + half, y + half),
        confidence=0.9,
        class_name="person",
        centroid=point,
        age=1,
        hits=1,
    )


def make_zone_engine(
    rule_type: str,
    **rule_params,
) -> ZoneRuleEngine:
    """Create a rule engine containing one test polygon zone."""
    zone = Zone(
        name="restricted",
        polygon=SQUARE_ZONE,
        camera_id=CAMERA_ID,
        rules=[
            Rule(type=rule_type, params=rule_params)
        ],
    )

    return ZoneRuleEngine(zones=[zone], lines=[])


def make_line_engine(**rule_params) -> ZoneRuleEngine:
    """Create a rule engine containing one horizontal virtual line."""
    line = Line(
        name="tripwire",
        points=((0.0, 150.0), (300.0, 150.0)),
        camera_id=CAMERA_ID,
        rules=[
            Rule(
                type=EventType.LINE_CROSSING.value,
                params=rule_params,
            )
        ],
    )

    return ZoneRuleEngine(zones=[], lines=[line])


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def test_point_in_polygon_inside_and_outside() -> None:
    assert point_in_polygon((150, 150), SQUARE_ZONE) is True
    assert point_in_polygon((10, 10), SQUARE_ZONE) is False


def test_segments_intersect_true_and_false() -> None:
    # x=400 is outside the line's x-range [0, 300].
    assert segments_intersect(
        (400, 140),
        (400, 160),
        (0, 150),
        (300, 150),
    ) is False

    assert segments_intersect(
        (150, 140),
        (150, 160),
        (0, 150),
        (300, 150),
    ) is True


# ---------------------------------------------------------------------------
# Restricted zone entry
# ---------------------------------------------------------------------------

def test_restricted_zone_entry_fires_exactly_on_transition_not_every_frame() -> None:
    engine = make_zone_engine(
        EventType.RESTRICTED_ZONE_ENTRY.value
    )

    trajectory = [
        (50, 50),
        (80, 80),
        (150, 150),
        (160, 160),
        (170, 170),
    ]

    fire_flags = []

    for point in trajectory:
        events = engine.evaluate(
            CAMERA_ID,
            [obj_at(point)],
            BASE_TIME,
        )

        fired = any(
            event.event_type == EventType.RESTRICTED_ZONE_ENTRY
            for event in events
        )

        fire_flags.append(fired)

    assert fire_flags == [False, False, True, False, False]


def test_restricted_zone_entry_fires_again_after_leaving_and_reentering() -> None:
    engine = make_zone_engine(
        EventType.RESTRICTED_ZONE_ENTRY.value
    )

    trajectory = [
        (50, 50),
        (150, 150),
        (250, 250),
        (150, 150),
    ]

    fire_flags = []

    for point in trajectory:
        events = engine.evaluate(
            CAMERA_ID,
            [obj_at(point)],
            BASE_TIME,
        )

        fire_flags.append(
            any(
                event.event_type == EventType.RESTRICTED_ZONE_ENTRY
                for event in events
            )
        )

    assert fire_flags == [False, True, False, True]


def test_restricted_zone_entry_never_fires_when_object_stays_outside() -> None:
    engine = make_zone_engine(
        EventType.RESTRICTED_ZONE_ENTRY.value
    )

    trajectory = [
        (0, 0),
        (10, 400),
        (400, 10),
        (5, 5),
    ]

    for point in trajectory:
        events = engine.evaluate(
            CAMERA_ID,
            [obj_at(point)],
            BASE_TIME,
        )

        assert events == []


def test_perimeter_intrusion_supports_configured_vehicle_classes_and_severity() -> None:
    zone = Zone(
        name="perimeter",
        polygon=SQUARE_ZONE,
        camera_id=CAMERA_ID,
        rules=[
            Rule(
                type=EventType.PERIMETER_INTRUSION.value,
                params={"object_classes": ["person", "car"], "severity": "critical"},
            )
        ],
    )
    engine = ZoneRuleEngine([zone], [], site_id="site-a")
    car = obj_at((150, 150))
    car = car.__class__(
        track_id=car.track_id,
        bbox=car.bbox,
        confidence=car.confidence,
        class_name="car",
        centroid=car.centroid,
        age=car.age,
        hits=car.hits,
    )

    events = engine.evaluate(CAMERA_ID, [obj_at((50, 50)), car], BASE_TIME)
    assert len(events) == 1
    assert events[0].event_type == EventType.PERIMETER_INTRUSION
    assert events[0].metadata["severity"] == "critical"
    assert events[0].metadata["class_name"] == "car"


def test_loitering_fires_after_configured_duration() -> None:
    engine = make_zone_engine(
        EventType.LOITERING_DETECTED.value,
        max_loiter_seconds=5,
    )
    engine.evaluate(CAMERA_ID, [obj_at((150, 150))], BASE_TIME)
    assert engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 150))],
        BASE_TIME + timedelta(seconds=4),
    ) == []
    events = engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 150))],
        BASE_TIME + timedelta(seconds=5),
    )
    assert events[0].event_type == EventType.LOITERING_DETECTED


def test_after_hours_presence_defaults_to_closed_hours() -> None:
    zone = Zone(
        name="after-hours-zone",
        polygon=SQUARE_ZONE,
        camera_id=CAMERA_ID,
        schedule={
            "timezone": "UTC",
            "open_hours": {"mon": ["08:00-20:00"]},
        },
        rules=[Rule(type=EventType.AFTER_HOURS_PRESENCE.value)],
    )
    engine = ZoneRuleEngine([zone], [], site_id="site-a")
    outside = engine.evaluate(
        CAMERA_ID,
        [obj_at((50, 50))],
        datetime(2026, 1, 5, 21, tzinfo=timezone.utc),
    )
    inside = engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 150))],
        datetime(2026, 1, 5, 21, tzinfo=timezone.utc),
    )
    assert outside == []
    assert inside[0].event_type == EventType.AFTER_HOURS_PRESENCE


def test_schedule_supports_midnight_crossing_and_closed_hours() -> None:
    schedule = {
        "timezone": "Africa/Lagos",
        "open_hours": {"mon-sat": ["22:00-02:00"], "sun": []},
    }
    assert is_armed(schedule, datetime(2026, 1, 5, 23, tzinfo=timezone.utc), "open_hours")
    assert is_armed(schedule, datetime(2026, 1, 6, 0, tzinfo=timezone.utc), "open_hours")
    assert is_armed(schedule, datetime(2026, 1, 6, 12, tzinfo=timezone.utc), "closed_hours")


def test_schedule_handles_dst_transition_in_new_york() -> None:
    schedule = {
        "timezone": "America/New_York",
        "open_hours": {"sun": ["01:00-04:00"]},
    }
    before_jump = datetime(2026, 3, 8, 6, 30, tzinfo=timezone.utc)
    after_jump = datetime(2026, 3, 8, 7, 30, tzinfo=timezone.utc)
    assert is_armed(schedule, before_jump, "open_hours")
    assert is_armed(schedule, after_jump, "open_hours")


def test_rule_cooldown_counts_suppressed_events() -> None:
    zone = Zone(
        name="restricted",
        polygon=SQUARE_ZONE,
        camera_id=CAMERA_ID,
        rules=[Rule(
            type=EventType.RESTRICTED_ZONE_ENTRY.value,
            params={"cooldown_seconds": 10},
        )],
    )
    engine = ZoneRuleEngine([zone], [], site_id="site1")
    engine.evaluate(CAMERA_ID, [obj_at((50, 50))], BASE_TIME)
    first = engine.evaluate(CAMERA_ID, [obj_at((150, 150))], BASE_TIME)
    engine.evaluate(CAMERA_ID, [obj_at((250, 250))], BASE_TIME + timedelta(seconds=1))
    suppressed = engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 150))],
        BASE_TIME + timedelta(seconds=2),
    )
    assert len(first) == 1
    assert suppressed == []
    engine.evaluate(CAMERA_ID, [obj_at((250, 250))], BASE_TIME + timedelta(seconds=3))
    next_event = engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 150))],
        BASE_TIME + timedelta(seconds=12),
    )
    assert next_event[0].metadata["suppressed_count"] == 1


def test_restricted_zone_entry_is_emitted_once_while_track_remains_inside() -> None:
    engine = make_zone_engine(EventType.RESTRICTED_ZONE_ENTRY.value)

    first = engine.evaluate(CAMERA_ID, [obj_at((150, 150))], BASE_TIME)
    still_inside = engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 150))],
        BASE_TIME + timedelta(seconds=1),
    )

    assert len(first) == 1
    assert still_inside == []


# ---------------------------------------------------------------------------
# Line crossing
# ---------------------------------------------------------------------------

def test_line_crossing_fires_exactly_when_segment_crosses_the_line() -> None:
    engine = make_line_engine()

    trajectory = [
        (150, 100),
        (150, 130),
        (150, 140),
        (150, 160),
        (150, 200),
    ]

    fire_flags = []

    for point in trajectory:
        events = engine.evaluate(
            CAMERA_ID,
            [obj_at(point)],
            BASE_TIME,
        )

        fire_flags.append(
            any(
                event.event_type == EventType.LINE_CROSSING
                for event in events
            )
        )

    assert fire_flags == [False, False, False, True, False]


def test_line_crossing_does_not_fire_when_object_stays_on_one_side() -> None:
    engine = make_line_engine()

    trajectory = [
        (150, 10),
        (150, 20),
        (150, 30),
        (150, 40),
    ]

    for point in trajectory:
        events = engine.evaluate(
            CAMERA_ID,
            [obj_at(point)],
            BASE_TIME,
        )

        assert events == []


# ---------------------------------------------------------------------------
# Directional line crossing
# ---------------------------------------------------------------------------

def test_line_crossing_detects_both_directions() -> None:
    line = Line(
        name="entrance",
        points=((0.0, 150.0), (300.0, 150.0)),
    )

    # Moving down across the line.
    assert get_line_crossing_direction(
        (150, 140),
        (150, 160),
        line,
    ) == "negative_to_positive"

    # Moving up across the line.
    assert get_line_crossing_direction(
        (150, 160),
        (150, 140),
        line,
    ) == "positive_to_negative"


def test_line_crossing_returns_none_without_direction_change() -> None:
    line = Line(
        name="entrance",
        points=((0.0, 150.0), (300.0, 150.0)),
    )

    assert get_line_crossing_direction(
        (150, 100),
        (150, 120),
        line,
    ) is None


# ---------------------------------------------------------------------------
# People counting and occupancy
# ---------------------------------------------------------------------------

def test_person_entering_increases_occupancy() -> None:
    engine = make_line_engine(
        track_occupancy=True,
        entry_direction="negative_to_positive",
    )

    # Person approaches from above the line.
    engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 140), track_id=1)],
        BASE_TIME,
    )

    # Person crosses the line.
    events = engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 160), track_id=1)],
        BASE_TIME,
    )

    assert len(events) == 1
    assert events[0].metadata["movement"] == "entry"
    assert events[0].metadata["occupancy_count"] == 1


def test_same_person_entering_twice_does_not_inflate_occupancy() -> None:
    engine = make_line_engine(
        track_occupancy=True,
        entry_direction="negative_to_positive",
    )

    # First entry.
    engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 140), track_id=1)],
        BASE_TIME,
    )

    first_entry = engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 160), track_id=1)],
        BASE_TIME,
    )

    assert first_entry[0].metadata["occupancy_count"] == 1

    # Person exits.
    engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 140), track_id=1)],
        BASE_TIME + timedelta(seconds=1),
    )

    # Person enters again.
    second_entry = engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 160), track_id=1)],
        BASE_TIME + timedelta(seconds=2),
    )

    assert second_entry[0].metadata["movement"] == "entry"
    assert second_entry[0].metadata["occupancy_count"] == 1


def test_person_exiting_decreases_occupancy() -> None:
    engine = make_line_engine(
        track_occupancy=True,
        entry_direction="negative_to_positive",
    )

    # Person enters.
    engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 140), track_id=1)],
        BASE_TIME,
    )

    engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 160), track_id=1)],
        BASE_TIME,
    )

    # Person exits.
    events = engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 140), track_id=1)],
        BASE_TIME + timedelta(seconds=1),
    )

    assert len(events) == 1
    assert events[0].metadata["movement"] == "exit"
    assert events[0].metadata["occupancy_count"] == 0


def test_two_different_people_entering_produces_occupancy_of_two() -> None:
    engine = make_line_engine(
        track_occupancy=True,
        entry_direction="negative_to_positive",
    )

    # First person enters.
    engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 140), track_id=1)],
        BASE_TIME,
    )

    engine.evaluate(
        CAMERA_ID,
        [obj_at((150, 160), track_id=1)],
        BASE_TIME,
    )

    # Second person enters.
    engine.evaluate(
        CAMERA_ID,
        [obj_at((200, 140), track_id=2)],
        BASE_TIME + timedelta(seconds=1),
    )

    events = engine.evaluate(
        CAMERA_ID,
        [obj_at((200, 160), track_id=2)],
        BASE_TIME + timedelta(seconds=1),
    )

    assert events[0].metadata["movement"] == "entry"
    assert events[0].metadata["occupancy_count"] == 2


def test_confirm_exited_reconciliation_removes_uncertain_occupant() -> None:
    engine = make_line_engine(
        track_occupancy=True,
        entry_direction="negative_to_positive",
    )

    engine.evaluate(CAMERA_ID, [obj_at((150, 100))], BASE_TIME)
    engine.evaluate(CAMERA_ID, [obj_at((150, 200))], BASE_TIME)
    engine.evaluate(CAMERA_ID, [], BASE_TIME, active_track_ids=set())
    status = engine.get_occupancy_status(CAMERA_ID)
    assert status.count == 1
    assert status.uncertain is True

    reconciled = engine.reconcile_occupancy(CAMERA_ID, "confirm_exited")

    assert reconciled.count == 0
    assert reconciled.uncertain is False


def test_confirm_present_reconciliation_keeps_occupant() -> None:
    engine = make_line_engine(
        track_occupancy=True,
        entry_direction="negative_to_positive",
    )

    engine.evaluate(CAMERA_ID, [obj_at((150, 100))], BASE_TIME)
    engine.evaluate(CAMERA_ID, [obj_at((150, 200))], BASE_TIME)
    engine.evaluate(CAMERA_ID, [], BASE_TIME, active_track_ids=set())

    reconciled = engine.reconcile_occupancy(CAMERA_ID, "confirm_present")

    assert reconciled.count == 1
    assert reconciled.uncertain is False


# ---------------------------------------------------------------------------
# Dwell time exceeded
# ---------------------------------------------------------------------------

def test_dwell_time_exceeded_fires_once_after_threshold_and_resets_on_exit() -> None:
    engine = make_zone_engine(
        EventType.DWELL_TIME_EXCEEDED.value,
        max_dwell_seconds=3.0,
    )

    point_inside = (150, 150)
    point_outside = (0, 0)

    # t=0: enters zone.
    events0 = engine.evaluate(
        CAMERA_ID,
        [obj_at(point_inside)],
        BASE_TIME,
    )

    assert not any(
        event.event_type == EventType.DWELL_TIME_EXCEEDED
        for event in events0
    )

    # t=1s: still under threshold.
    events1 = engine.evaluate(
        CAMERA_ID,
        [obj_at(point_inside)],
        BASE_TIME + timedelta(seconds=1),
    )

    assert not any(
        event.event_type == EventType.DWELL_TIME_EXCEEDED
        for event in events1
    )

    # t=3.5s: exceeds 3s threshold.
    events2 = engine.evaluate(
        CAMERA_ID,
        [obj_at(point_inside)],
        BASE_TIME + timedelta(seconds=3.5),
    )

    assert sum(
        1
        for event in events2
        if event.event_type == EventType.DWELL_TIME_EXCEEDED
    ) == 1

    # t=4s: should not fire again during the same stay.
    events3 = engine.evaluate(
        CAMERA_ID,
        [obj_at(point_inside)],
        BASE_TIME + timedelta(seconds=4),
    )

    assert not any(
        event.event_type == EventType.DWELL_TIME_EXCEEDED
        for event in events3
    )

    # Leave and re-enter.
    engine.evaluate(
        CAMERA_ID,
        [obj_at(point_outside)],
        BASE_TIME + timedelta(seconds=5),
    )

    events_reenter = engine.evaluate(
        CAMERA_ID,
        [obj_at(point_inside)],
        BASE_TIME + timedelta(seconds=6),
    )

    assert not any(
        event.event_type == EventType.DWELL_TIME_EXCEEDED
        for event in events_reenter
    )

    events_reenter_exceeded = engine.evaluate(
        CAMERA_ID,
        [obj_at(point_inside)],
        BASE_TIME + timedelta(seconds=9.5),
    )

    assert sum(
        1
        for event in events_reenter_exceeded
        if event.event_type == EventType.DWELL_TIME_EXCEEDED
    ) == 1
