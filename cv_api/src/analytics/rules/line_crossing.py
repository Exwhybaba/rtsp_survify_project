"""Virtual-line crossing rule."""

from __future__ import annotations

from datetime import datetime

from src.analytics.rules.base import Rule, RuleHandler, _TrackState, rule_applies_to_object
from src.analytics.schemas import EventType, VisionEvent
from src.analytics.tracker import TrackedObject


class LineCrossingHandler(RuleHandler):
    """Detect directional movement across a configured line."""

    def evaluate_line(
        self,
        engine,
        camera_id: str,
        obj: TrackedObject,
        line,
        rule: Rule,
        _state: _TrackState,
        timestamp: datetime,
        crossed: bool,
        previous_point,
        current_point,
    ) -> list[VisionEvent]:
        from src.analytics.rules.engine import get_line_crossing_direction

        if not rule_applies_to_object(rule, obj) or not crossed:
            return []
        direction = get_line_crossing_direction(previous_point, current_point, line)
        if direction is None:
            return []
        allowed_direction = rule.params.get("allowed_direction")
        if allowed_direction and direction != allowed_direction:
            return []
        metadata = {"line": line.name, "direction": direction}
        if rule.params.get("track_occupancy", False):
            entry_direction = rule.params.get("entry_direction", "negative_to_positive")
            movement = "entry" if direction == entry_direction else "exit"
            occupancy_count = engine.update_occupancy(
                camera_id=camera_id,
                track_id=obj.track_id,
                movement=movement,
            )
            occupancy_status = engine.get_occupancy_status(camera_id)
            metadata.update(
                {
                    "movement": movement,
                    "occupancy_count": occupancy_count,
                    "occupancy_uncertain": occupancy_status.uncertain,
                }
            )
        return [
            engine.make_event(
                camera_id,
                obj,
                EventType.LINE_CROSSING,
                timestamp,
                metadata,
            )
        ]
