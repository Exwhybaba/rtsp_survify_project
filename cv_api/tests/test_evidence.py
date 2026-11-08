"""Tests for optional asynchronous evidence capture."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.analytics.evidence import (
    EvidenceCapture,
    annotate_event_frame,
    event_annotation_lines,
)
from src.analytics.schemas import BoundingBox, EventType, VisionEvent


def make_event() -> VisionEvent:
    return VisionEvent(
        camera_id="cam1",
        site_id="site1",
        event_type=EventType.RESTRICTED_ZONE_ENTRY,
        track_id=1,
        confidence=0.9,
        bbox=BoundingBox(x1=2, y1=2, x2=20, y2=20),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_evidence_writes_snapshot_clip_sidecar_and_hashes(tmp_path) -> None:
    ready = []
    capture = EvidenceCapture(
        evidence_dir=tmp_path,
        site_id="site1",
        camera_id="cam1",
        source_uri="rtsp://user:secret@camera/live",
        pre_roll_seconds=1,
        post_roll_seconds=1,
        fps_cap=10,
        enabled=True,
        on_evidence_ready=ready.append,
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for event_time in (0.0, 0.5, 1.0, 2.0):
        capture.add_frame(frame, event_time)
    event = make_event()
    capture.on_events([event], frame, 1.0)
    capture.add_frame(frame, 2.1)
    capture.flush()
    capture.close()

    assert event.evidence is not None
    snapshot = tmp_path / "site1" / "cam1" / "1970-01-01" / f"{event.event_id}.jpg"
    assert snapshot.exists()
    assert event.evidence.clip_path is not None
    clip = Path(event.evidence.clip_path)
    assert clip.exists()
    sidecar = clip.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["source"] == "rtsp://***:***@camera/live"
    assert payload["clip_sha256"] == hashlib.sha256(clip.read_bytes()).hexdigest()
    assert event.evidence.snapshot_sha256 == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert ready and ready[0].event_id == event.event_id


def test_evidence_annotation_contains_event_context_and_preserves_frame() -> None:
    event = make_event()
    event.metadata.update(
        {
            "class_name": "person",
            "zone": "restricted_area",
            "severity": "high",
        }
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    original = frame.copy()

    annotated = annotate_event_frame(frame, event)

    assert np.array_equal(frame, original)
    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)
    assert event_annotation_lines(event) == [
        "EVENT: RESTRICTED_ZONE_ENTRY",
        "ZONE: restricted_area",
        "SEVERITY: HIGH",
        "CAMERA: cam1",
        "SITE: site1",
        "TIME: 2026-01-01T00:00:00+00:00",
    ]


def test_evidence_annotation_handles_missing_optional_fields_and_clips_bounds() -> None:
    event = make_event()
    event.bbox = event.bbox.model_copy(
        update={"x1": -20, "y1": -20, "x2": 100, "y2": 100}
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    annotated = annotate_event_frame(frame, event)

    assert annotated.shape == frame.shape
    assert event_annotation_lines(event) == [
        "EVENT: RESTRICTED_ZONE_ENTRY",
        "CAMERA: cam1",
        "SITE: site1",
        "TIME: 2026-01-01T00:00:00+00:00",
    ]


def test_snapshot_preserves_unmodified_incident_frame(tmp_path) -> None:
    event = make_event()
    frame = np.full((48, 64, 3), 127, dtype=np.uint8)
    capture = EvidenceCapture(
        evidence_dir=tmp_path,
        site_id="site1",
        camera_id="cam1",
        source_uri="camera",
        post_roll_seconds=0,
        enabled=True,
    )

    capture.on_events([event], frame, 1.0)
    capture.flush()
    capture.close()

    snapshot = cv2.imread(event.evidence.snapshot_path)
    assert snapshot is not None
    assert np.allclose(snapshot, frame, atol=5)


def test_shoplifting_evidence_threshold_is_separate_from_event_emission(tmp_path) -> None:
    event = make_event().model_copy(
        update={
            "event_type": EventType.SUSPECTED_SHOPLIFTING,
            "confidence": 0.65,
            "metadata": {"concealment_score": 0.65},
        }
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    capture = EvidenceCapture(
        evidence_dir=tmp_path,
        site_id="site1",
        camera_id="cam1",
        source_uri="camera",
        shoplifting_min_concealment_score=0.7,
        enabled=True,
    )

    capture.on_events([event], frame, 1.0)
    capture.flush()
    capture.close()

    assert event.evidence is None
    assert event.metadata["evidence_capture_skipped"] == "below_evidence_threshold"


def test_concealment_score_gate_does_not_filter_contextual_fallback_events(tmp_path) -> None:
    event = make_event().model_copy(
        update={
            "event_type": EventType.SUSPECTED_SHOPLIFTING,
            "confidence": 0.4,
            "metadata": {"behavior": "merchandise_interaction_then_exit"},
        }
    )
    capture = EvidenceCapture(
        evidence_dir=tmp_path,
        site_id="site1",
        camera_id="cam1",
        source_uri="camera",
        shoplifting_min_concealment_score=0.7,
        enabled=True,
    )

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    capture.add_frame(frame, 1.0)
    capture.on_events([event], frame, 1.0)
    capture.flush()
    capture.close()

    assert event.evidence is not None
    assert "evidence_capture_skipped" not in event.metadata


def test_multiple_event_snapshots_keep_event_specific_annotations(tmp_path) -> None:
    first = make_event()
    first.metadata.update({"class_name": "person", "zone": "zone-a", "severity": "low"})
    second = make_event()
    second.metadata.update({"class_name": "car", "zone": "zone-b", "severity": "critical"})
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    capture = EvidenceCapture(
        evidence_dir=tmp_path,
        site_id="site1",
        camera_id="cam1",
        source_uri="camera",
        post_roll_seconds=0,
        enabled=True,
    )
    capture.on_events([first, second], frame, 1.0)
    capture.flush()
    capture.close()

    first_image = cv2.imread(first.evidence.snapshot_path)
    second_image = cv2.imread(second.evidence.snapshot_path)
    assert first_image is not None
    assert second_image is not None
    assert np.array_equal(first_image, second_image)
    assert "ZONE: zone-a" in event_annotation_lines(first)
    assert "ZONE: zone-b" in event_annotation_lines(second)
