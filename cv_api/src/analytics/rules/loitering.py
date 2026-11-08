"""Loitering rule."""

from __future__ import annotations

from datetime import datetime

from src.analytics.rules.base import Rule, RuleHandler, _TrackState, rule_applies_to_object
from src.analytics.schemas import EventType, VisionEvent
from src.analytics.tracker import TrackedObject


class LoiteringHandler(RuleHandler):
    """Detect prolonged presence in a configured zone."""

    def evaluate_zone(
        self,
        engine,
        camera_id: str,
        obj: TrackedObject,
        zone,
        rule: Rule,
        state: _TrackState,
        timestamp: datetime,
        is_inside: bool,
        _was_inside: bool,
    ) -> list[VisionEvent]:
        if not rule_applies_to_object(rule, obj) or not is_inside:
            return []
        threshold = float(
            rule.params.get(
                "max_loiter_seconds",
                rule.params.get("max_dwell_seconds", 60.0),
            )
        )
        entry_time = state.zone_entry_time.setdefault(zone.name, timestamp.timestamp())
        elapsed = timestamp.timestamp() - entry_time
        if elapsed < threshold or zone.name in state.dwell_fired:
            return []
        state.dwell_fired.add(zone.name)
        return [
            engine.make_event(
                camera_id,
                obj,
                EventType.LOITERING_DETECTED,
                timestamp,
                {
                    "zone": zone.name,
                    "loiter_seconds": round(elapsed, 2),
                    "max_loiter_seconds": threshold,
                },
            )
        ]
