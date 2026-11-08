"""End-to-end pipeline coverage with synthetic video and fakes."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.analytics.evidence import EvidenceCapture
from src.analytics.pipeline import run_pipeline
from src.analytics.tracker import Detection


class FakeDetector:
    """Return one person box following a trajectory through a zone."""

    def __init__(self) -> None:
        self.index = 0

    def detect(self, _frame: np.ndarray) -> list[Detection]:
        x = (20, 45, 20, 45, 20, 45)[self.index]
        self.index += 1
        return [Detection((x, 20, x + 10, 40), 0.95, "person")]


class FakePublisher:
    def __init__(self) -> None:
        self.events = []
        self.ready = []

    def publish(self, event) -> None:
        self.events.append(event)

    def publish_evidence_ready(self, message) -> None:
        self.ready.append(message)

    def close(self) -> None:
        pass


def write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (80, 60),
    )
    try:
        for _ in range(6):
            writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
    finally:
        writer.release()


def write_config(path: Path, video: Path, rule: dict) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "camera": {
                    "id": "cam1",
                    "site_id": "site1",
                    "source": str(video),
                    "loop": False,
                },
                "tracker": {
                    "type": "bytetrack",
                    "minimum_consecutive_frames": 1,
                },
                "zones": [
                    {
                        "name": "restricted",
                        "camera_id": "cam1",
                        "polygon": [[40, 10], [60, 10], [60, 50], [40, 50]],
                        "rules": [rule],
                    }
                ],
                "lines": [],
                "visualization": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )


def test_pipeline_requires_an_explicit_video_source(tmp_path: Path) -> None:
    config = tmp_path / "pipeline.yaml"
    config.write_text('camera:\n  id: cam1\n  source: ""\n', encoding="utf-8")

    try:
        run_pipeline(config)
    except ValueError as exc:
        assert "No video source configured" in str(exc)
    else:
        raise AssertionError("pipeline should reject a missing source")


def test_pipeline_publishes_evidence_and_respects_cooldown_and_schedule(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    write_video(video)
    config = tmp_path / "pipeline.yaml"
    write_config(
        config,
        video,
        {"type": "restricted_zone_entry", "cooldown_seconds": 10},
    )
    publisher = FakePublisher()
    evidence = EvidenceCapture(
        evidence_dir=tmp_path / "evidence",
        site_id="site1",
        camera_id="cam1",
        source_uri=str(video),
        pre_roll_seconds=0.5,
        post_roll_seconds=0.5,
        fps_cap=10,
        enabled=True,
        on_evidence_ready=publisher.publish_evidence_ready,
    )

    run_pipeline(
        config,
        max_frames=6,
        detector=FakeDetector(),
        publisher=publisher,
        evidence=evidence,
    )
    evidence.flush()
    evidence.close()

    assert len(publisher.events) == 1
    event = publisher.events[0]
    assert event.site_id == "site1"
    assert event.event_id
    assert event.evidence is not None
    assert Path(event.evidence.snapshot_path).exists()
    assert event.evidence.clip_path is not None
    assert Path(event.evidence.clip_path).exists()
    sidecar = Path(event.evidence.clip_path).with_suffix(".json")
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar_payload["clip_sha256"] == event.evidence.clip_sha256
    assert publisher.ready and publisher.ready[0].event_id == event.event_id

    open_config = tmp_path / "open.yaml"
    write_config(
        open_config,
        video,
        {"type": "restricted_zone_entry", "armed_when": "closed_hours"},
    )
    raw = yaml.safe_load(open_config.read_text(encoding="utf-8"))
    raw["zones"][0]["schedule"] = {
        "timezone": "UTC",
        "open_hours": {
            day: ["00:00-23:59"]
            for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
        },
    }
    open_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    no_event_publisher = FakePublisher()
    run_pipeline(
        open_config,
        max_frames=6,
        detector=FakeDetector(),
        publisher=no_event_publisher,
    )
    assert no_event_publisher.events == []


def test_graceful_stop_flushes_confirmed_event_evidence(tmp_path: Path) -> None:
    video = tmp_path / "synthetic-stop.mp4"
    write_video(video)
    config = tmp_path / "pipeline-stop.yaml"
    write_config(config, video, {"type": "restricted_zone_entry"})
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["evidence"] = {
        "enabled": True,
        "evidence_dir": str(tmp_path / "evidence-stop"),
        "pre_roll_seconds": 0,
        "post_roll_seconds": 0,
        "fps_cap": 10,
    }
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    stop_file = tmp_path / "stop.request"
    started_file = tmp_path / "source.started"

    class StopAfterIncidentDetector(FakeDetector):
        def detect(self, frame: np.ndarray) -> list[Detection]:
            detections = super().detect(frame)
            if self.index == 5:
                stop_file.touch()
            return detections

    publisher = FakePublisher()
    run_pipeline(
        config,
        detector=StopAfterIncidentDetector(),
        publisher=publisher,
        stop_file=stop_file,
        started_file=started_file,
    )

    assert started_file.is_file()
    assert stop_file.is_file()
    assert len(publisher.events) == 1
    event = publisher.events[0]
    assert event.evidence is not None
    assert Path(event.evidence.snapshot_path).is_file()
    assert event.evidence.clip_path is not None
    assert Path(event.evidence.clip_path).is_file()
