"""After-hours presence rule."""

from __future__ import annotations

from datetime import datetime

from src.analytics.rules.base import Rule, RuleHandler, _TrackState, rule_applies_to_object
from src.analytics.schemas import EventType, VisionEvent
from src.analytics.tracker import TrackedObject


class AfterHoursPresenceHandler(RuleHandler):
    """Detect entry into a zone while its schedule is closed."""

    def evaluate_zone(
        self,
        engine,
        camera_id: str,
        obj: TrackedObject,
        zone,
        rule: Rule,
        _state: _TrackState,
        timestamp: datetime,
        is_inside: bool,
        was_inside: bool,
    ) -> list[VisionEvent]:
        if not rule_applies_to_object(rule, obj) or not is_inside or was_inside:
            return []
        return [
            engine.make_event(
                camera_id,
                obj,
                EventType.AFTER_HOURS_PRESENCE,
                timestamp,
                {
                    "zone": zone.name,
                    "severity": rule.params.get("severity", "medium"),
                    "after_hours": True,
                },
            )
        ]
