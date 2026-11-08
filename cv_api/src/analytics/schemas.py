"""Pydantic data models shared across the pipeline.

Keeping the event schema in one place means the detector, tracker, zone
engine, publisher, and dashboard all agree on exactly what a "vision event"
looks like on the wire (MQTT payload, webhook JSON body, WebSocket message).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventType(str, Enum):
    """Supported zone-rule violation types."""

    RESTRICTED_ZONE_ENTRY = "restricted_zone_entry"
    PERIMETER_INTRUSION = "perimeter_intrusion"
    LINE_CROSSING = "line_crossing"
    DWELL_TIME_EXCEEDED = "dwell_time_exceeded"
    LOITERING_DETECTED = "loitering_detected"
    AFTER_HOURS_PRESENCE = "after_hours_presence"
    SUSPECTED_SHOPLIFTING = "suspected_shoplifting"


class OccupancyControlCommand(BaseModel):
    """Validated operator command for occupancy reconciliation."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1)
    action: Literal["confirm_present", "confirm_exited"]
    operator_id: str | None = None

    @field_validator("camera_id")
    @classmethod
    def _strip_camera_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("camera_id must not be blank")
        return value

    @field_validator("operator_id")
    @classmethod
    def _strip_operator_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates, ``x1 < x2`` and ``y1 < y2``."""

    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_ordering(self) -> BoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"invalid bounding box: x2/y2 must be greater than x1/y1, got {self!r}"
            )
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class EvidenceRef(BaseModel):
    """Files produced for a vision event."""

    model_config = ConfigDict(extra="forbid")

    snapshot_path: str
    clip_path: str | None = None
    clip_start: datetime
    clip_end: datetime
    snapshot_sha256: str | None = None
    clip_sha256: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VisionEvent(BaseModel):
    """A single zone-rule violation raised for one tracked object on one frame.

    This is the canonical payload published to MQTT/webhook and streamed to
    the live dashboard over WebSocket.
    """

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1, description="Identifier of the source camera/stream.")
    site_id: str = Field(min_length=1, description="Identifier of the monitored site.")
    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    track_id: int = Field(ge=0, description="Stable multi-object-tracker ID.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detector confidence for this object.")
    bbox: BoundingBox
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: EvidenceRef | None = None

    @field_validator("camera_id")
    @classmethod
    def _strip_camera_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("camera_id must not be blank")
        return value

    @field_validator("site_id")
    @classmethod
    def _strip_site_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("site_id must not be blank")
        return value

    def to_wire_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict suitable for MQTT/webhook/WebSocket payloads."""
        payload = self.model_dump(mode="json")
        return payload


class EvidenceReadyMessage(BaseModel):
    """Wire payload sent when asynchronous evidence processing finishes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["evidence_ready"] = "evidence_ready"
    event_id: UUID
    site_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    evidence: EvidenceRef

    @field_validator("site_id", "camera_id")
    @classmethod
    def _strip_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("identifier must not be blank")
        return value
