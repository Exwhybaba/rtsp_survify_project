"""Restricted-zone entry rule."""

from __future__ import annotations

from datetime import datetime

from src.analytics.rules.base import Rule, RuleHandler, _TrackState, rule_applies_to_object
from src.analytics.schemas import EventType, VisionEvent
from src.analytics.tracker import TrackedObject


class RestrictedZoneEntryHandler(RuleHandler):
    """Detect a configured object entering a polygon zone."""

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
                EventType.RESTRICTED_ZONE_ENTRY,
                timestamp,
                {"zone": zone.name},
            )
        ]
