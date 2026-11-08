"""Central orchestration for modular analytics rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from src.analytics.rules.after_hours import AfterHoursPresenceHandler
from src.analytics.rules.base import Rule, RuleHandler, _TrackState
from src.analytics.rules.dwell_time import DwellTimeExceededHandler
from src.analytics.rules.line_crossing import LineCrossingHandler
from src.analytics.rules.loitering import LoiteringHandler
from src.analytics.rules.occupancy import OccupancyManager, OccupancyStatus
from src.analytics.rules.perimeter_intrusion import PerimeterIntrusionHandler
from src.analytics.rules.zone_entry import RestrictedZoneEntryHandler
from src.analytics.schemas import BoundingBox, EventType, VisionEvent
from src.analytics.tracker import TrackedObject

Point = tuple[float, float]


@dataclass
class Zone:
    """A polygon region and its rules."""

    name: str
    polygon: list[Point]
    camera_id: str | None = None
    rules: list[Rule] | None = None
    schedule: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.rules is None:
            self.rules = []


@dataclass
class Line:
    """A virtual line segment and its rules."""

    name: str
    points: tuple[Point, Point]
    camera_id: str | None = None
    rules: list[Rule] | None = None
    schedule: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.rules is None:
            self.rules = []


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Return whether a point is inside a polygon."""
    x, y = point
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, point: Point) -> bool:
    return (
        min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= point[1] <= max(a[1], b[1])
    )


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Return whether two line segments intersect."""
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)
    if (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
        and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    ):
        return True
    return (
        (d1 == 0 and _on_segment(p3, p4, p1))
        or (d2 == 0 and _on_segment(p3, p4, p2))
        or (d3 == 0 and _on_segment(p1, p2, p3))
        or (d4 == 0 and _on_segment(p1, p2, p4))
    )


def get_line_crossing_direction(previous_point: Point, current_point: Point, line: Line) -> str | None:
    """Return the direction in which a point crossed a directed line."""
    (x1, y1), (x2, y2) = line.points
    dx, dy = x2 - x1, y2 - y1
    previous_side = dx * (previous_point[1] - y1) - dy * (previous_point[0] - x1)
    current_side = dx * (current_point[1] - y1) - dy * (current_point[0] - x1)
    if previous_side == 0 or current_side == 0 or previous_side * current_side >= 0:
        return None
    return "negative_to_positive" if previous_side < 0 < current_side else "positive_to_negative"


_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_DAY_ALIASES = {
    "mon-sat": ("mon", "tue", "wed", "thu", "fri", "sat"),
    "mon-fri": ("mon", "tue", "wed", "thu", "fri"),
}


def _schedule_windows(schedule: dict[str, Any], weekday: str) -> list[tuple[time, time]]:
    windows: list[str] = []
    for key, values in schedule.get("open_hours", {}).items():
        if weekday in _DAY_ALIASES.get(key, (key,)):
            windows.extend(values or [])
    parsed = []
    for window in windows:
        start, end = window.split("-", maxsplit=1)
        parsed.append(
            (
                time(*(int(value) for value in start.split(":"))),
                time(*(int(value) for value in end.split(":"))),
            )
        )
    return parsed


def is_armed(schedule: dict[str, Any] | None, timestamp: datetime, armed_when: str = "always") -> bool:
    """Return whether a rule is armed at a timestamp."""
    if armed_when not in {"always", "open_hours", "closed_hours"}:
        raise ValueError(f"Unsupported armed_when value: {armed_when!r}")
    if armed_when == "always" or not schedule:
        return True
    try:
        local = timestamp.astimezone(ZoneInfo(schedule["timezone"]))
    except (KeyError, ZoneInfoNotFoundError) as exc:
        raise ValueError("schedule requires a valid IANA timezone") from exc
    weekday_index = local.weekday()
    current = local.timetz().replace(tzinfo=None)
    open_now = any(
        (start <= current < end) if start < end else (current >= start or current < end)
        for start, end in _schedule_windows(schedule, _WEEKDAYS[weekday_index])
    )
    if not open_now:
        previous = _WEEKDAYS[(weekday_index - 1) % 7]
        open_now = any(
            start >= end and current < end
            for start, end in _schedule_windows(schedule, previous)
        )
    return open_now if armed_when == "open_hours" else not open_now


def _parse_rules(raw_rules: list[dict[str, Any]]) -> list[Rule]:
    return [
        Rule(type=raw["type"], params={key: value for key, value in raw.items() if key != "type"})
        for raw in (raw_rules or [])
    ]


def load_zone_config(path: str | Path) -> tuple[list[Zone], list[Line]]:
    """Load zones and lines from YAML."""
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    zones = [
        Zone(
            name=item["name"],
            polygon=[tuple(point) for point in item["polygon"]],
            camera_id=item.get("camera_id"),
            rules=_parse_rules(item.get("rules", [])),
            schedule=item.get("schedule"),
        )
        for item in raw.get("zones", [])
    ]
    lines = [
        Line(
            name=item["name"],
            points=(tuple(item["points"][0]), tuple(item["points"][1])),
            camera_id=item.get("camera_id"),
            rules=_parse_rules(item.get("rules", [])),
            schedule=item.get("schedule"),
        )
        for item in raw.get("lines", [])
    ]
    return zones, lines


RULE_HANDLERS: dict[str, RuleHandler] = {
    EventType.RESTRICTED_ZONE_ENTRY.value: RestrictedZoneEntryHandler(),
    EventType.PERIMETER_INTRUSION.value: PerimeterIntrusionHandler(),
    EventType.DWELL_TIME_EXCEEDED.value: DwellTimeExceededHandler(),
    EventType.LOITERING_DETECTED.value: LoiteringHandler(),
    EventType.AFTER_HOURS_PRESENCE.value: AfterHoursPresenceHandler(),
    EventType.LINE_CROSSING.value: LineCrossingHandler(),
}


class ZoneRuleEngine:
    """Evaluate registered handlers against tracked objects."""

    def __init__(
        self,
        zones: list[Zone],
        lines: list[Line],
        handlers: dict[str, RuleHandler] | None = None,
        site_id: str = "default",
    ) -> None:
        self.zones = zones
        self.lines = lines
        self.site_id = site_id.strip()
        if not self.site_id:
            raise ValueError("site_id must not be blank")
        self.handlers = dict(RULE_HANDLERS if handlers is None else handlers)
        self._state: dict[tuple[str, int], _TrackState] = {}
        self._occupancy = OccupancyManager()
        self._occupancy_lock = RLock()
        self._cooldown_state: dict[tuple[str, str, str], tuple[float, int]] = {}
        self._validate_rules()

    @classmethod
    def from_config(cls, path: str | Path) -> ZoneRuleEngine:
        zones, lines = load_zone_config(path)
        with open(path, encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        return cls(zones, lines, site_id=raw.get("camera", {}).get("site_id", "default"))

    def _validate_rules(self) -> None:
        for location in [*self.zones, *self.lines]:
            for rule in location.rules or []:
                if rule.type not in self.handlers:
                    raise ValueError(
                        f"Unsupported rule type '{rule.type}' configured on '{location.name}'. "
                        f"Registered rules: {sorted(self.handlers)}"
                    )

    def _state_for(self, camera_id: str, track_id: int) -> _TrackState:
        return self._state.setdefault((camera_id, track_id), _TrackState())

    def update_occupancy(self, camera_id: str, track_id: int, movement: str) -> int:
        return self._occupancy.update(camera_id, track_id, movement)

    def get_occupancy_status(self, camera_id: str) -> OccupancyStatus:
        return self._occupancy.status(camera_id)

    def reconcile_occupancy(self, camera_id: str, action: str) -> OccupancyStatus:
        return self._occupancy.reconcile(camera_id, action)

    def _expire_inactive_tracks(self, camera_id: str, active_track_ids: set[int]) -> None:
        track_ids = {track_id for camera, track_id in self._state if camera == camera_id}
        for track_id in track_ids:
            if track_id not in active_track_ids:
                self._occupancy.mark_expired(camera_id, track_id)
                self._state.pop((camera_id, track_id), None)

    def reset_track(self, camera_id: str, track_id: int) -> None:
        self._occupancy.mark_expired(camera_id, track_id)
        self._state.pop((camera_id, track_id), None)

    def reset(self) -> None:
        self._state.clear()
        self._occupancy.reset()
        self._cooldown_state.clear()

    def evaluate(
        self,
        camera_id: str,
        tracked_objects: list[TrackedObject],
        timestamp: datetime | None = None,
        active_track_ids: set[int] | None = None,
    ) -> list[VisionEvent]:
        """Evaluate all configured locations through registered handlers."""
        current_time = timestamp or datetime.now(timezone.utc)
        if active_track_ids is not None:
            self._expire_inactive_tracks(camera_id, active_track_ids)
        events: list[VisionEvent] = []
        for obj in tracked_objects:
            state = self._state_for(camera_id, obj.track_id)
            current_point = obj.centroid
            for zone in self.zones:
                if zone.camera_id is not None and zone.camera_id != camera_id:
                    continue
                inside = point_in_polygon(current_point, zone.polygon)
                was_inside = zone.name in state.zones_inside
                for rule in zone.rules or []:
                    default_armed = (
                        "closed_hours"
                        if rule.type == EventType.AFTER_HOURS_PRESENCE.value
                        else "always"
                    )
                    if not is_armed(
                        zone.schedule,
                        current_time,
                        rule.params.get("armed_when", default_armed),
                    ):
                        continue
                    handler = self.handlers[rule.type]
                    events.extend(
                        self._filter_events(
                            handler.evaluate_zone(
                                self, camera_id, obj, zone, rule, state,
                                current_time, inside, was_inside,
                            ),
                            camera_id, zone.name, rule, current_time,
                        )
                    )
                if inside:
                    state.zones_inside.add(zone.name)
                else:
                    state.zones_inside.discard(zone.name)
                    state.zone_entry_time.pop(zone.name, None)
                    state.dwell_fired.discard(zone.name)
            if state.last_point is not None:
                for line in self.lines:
                    if line.camera_id is not None and line.camera_id != camera_id:
                        continue
                    crossed = segments_intersect(
                        state.last_point,
                        current_point,
                        line.points[0],
                        line.points[1],
                    )
                    for rule in line.rules or []:
                        if not is_armed(line.schedule, current_time, rule.params.get("armed_when", "always")):
                            continue
                        events.extend(
                            self._filter_events(
                                self.handlers[rule.type].evaluate_line(
                                    self, camera_id, obj, line, rule, state,
                                    current_time, crossed, state.last_point, current_point,
                                ),
                                camera_id, line.name, rule, current_time,
                            )
                        )
            state.last_point = current_point
        return events

    def make_event(
        self,
        camera_id: str,
        obj: TrackedObject,
        event_type: EventType,
        timestamp: datetime,
        metadata: dict[str, Any],
    ) -> VisionEvent:
        """Build the canonical event for a handler."""
        x1, y1, x2, y2 = obj.bbox
        return VisionEvent(
            camera_id=camera_id,
            site_id=self.site_id,
            event_type=event_type,
            track_id=obj.track_id,
            confidence=obj.confidence,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            timestamp=timestamp,
            metadata={**metadata, "class_name": obj.class_name},
        )

    def _filter_events(
        self,
        events: list[VisionEvent],
        camera_id: str,
        location_name: str,
        rule: Rule,
        timestamp: datetime,
    ) -> list[VisionEvent]:
        cooldown = float(rule.params.get("cooldown_seconds", 0))
        if cooldown <= 0:
            return events
        filtered = []
        for event in events:
            key = (camera_id, location_name, event.event_type.value)
            last_time, suppressed = self._cooldown_state.get(key, (-float("inf"), 0))
            if timestamp.timestamp() - last_time < cooldown:
                self._cooldown_state[key] = (last_time, suppressed + 1)
                continue
            if suppressed:
                event.metadata["suppressed_count"] = suppressed
            self._cooldown_state[key] = (timestamp.timestamp(), 0)
            filtered.append(event)
        return filtered
