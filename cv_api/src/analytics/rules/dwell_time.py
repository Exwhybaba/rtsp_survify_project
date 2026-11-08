"""Dwell-time rule."""

from __future__ import annotations

from datetime import datetime

from src.analytics.rules.base import Rule, RuleHandler, _TrackState, rule_applies_to_object
from src.analytics.schemas import EventType, VisionEvent
from src.analytics.tracker import TrackedObject


class DwellTimeExceededHandler(RuleHandler):
    """Detect a configured object remaining in a zone too long."""

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
        max_dwell = float(rule.params.get("max_dwell_seconds", 5.0))
        entry_time = state.zone_entry_time.setdefault(zone.name, timestamp.timestamp())
        dwell_seconds = timestamp.timestamp() - entry_time
        if dwell_seconds < max_dwell or zone.name in state.dwell_fired:
            return []
        state.dwell_fired.add(zone.name)
        return [
            engine.make_event(
                camera_id,
                obj,
                EventType.DWELL_TIME_EXCEEDED,
                timestamp,
                {
                    "zone": zone.name,
                    "dwell_seconds": round(dwell_seconds, 2),
                    "max_dwell_seconds": max_dwell,
                },
            )
        ]
