"""Unit tests for the dashboard FastAPI app.

The MQTT client is stubbed out entirely (no real broker needed) by
monkeypatching ``_build_mqtt_client``. Events are injected directly via
``store.push_event`` to simulate what would otherwise arrive over MQTT.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from fastapi.testclient import TestClient

import src.dashboard.api as api_module
from src.dashboard.api import app, store


class _DummyMqttClient:
    """Stand-in for paho.mqtt.client.Client that never touches the network."""

    def connect_async(self, *args, **kwargs) -> None:
        pass

    def loop_start(self) -> None:
        pass

    def loop_stop(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def publish(self, *args, **kwargs):
        return type("_PublishInfo", (), {"rc": 0})()


@pytest.fixture(autouse=True)
def _stub_mqtt_and_reset_store(monkeypatch):
    monkeypatch.setattr(api_module, "_build_mqtt_client", lambda: _DummyMqttClient())
    store._buffer.clear()
    store._pending_evidence.clear()
    store._subscribers.clear()
    yield
    store._buffer.clear()
    store._pending_evidence.clear()
    store._subscribers.clear()


def sample_event(track_id: int = 1) -> dict:
    return {
        "camera_id": "cam1",
        "event_type": "line_crossing",
        "track_id": track_id,
        "confidence": 0.9,
        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "timestamp": "2026-01-01T00:00:00Z",
        "metadata": {},
    }


def evidence_event(snapshot_path: str, clip_path: str | None = None) -> dict:
    event_id = str(uuid4())
    return {
        "event_id": event_id,
        "camera_id": "cam1",
        "site_id": "default",
        "event_type": "line_crossing",
        "track_id": 1,
        "confidence": 0.9,
        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "timestamp": "2026-01-01T00:00:00Z",
        "metadata": {},
        "evidence": {
            "snapshot_path": snapshot_path,
            "clip_path": clip_path,
            "clip_start": "2026-01-01T00:00:00Z",
            "clip_end": "2026-01-01T00:00:01Z",
            "snapshot_sha256": None,
            "clip_sha256": None,
            "created_at": "2026-01-01T00:00:00Z",
        },
    }


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_events_endpoint_starts_empty() -> None:
    with TestClient(app) as client:
        resp = client.get("/events")
        assert resp.status_code == 200
        assert resp.json() == []


def test_pushed_event_shows_up_in_events_endpoint() -> None:
    with TestClient(app):
        store.push_event(sample_event(track_id=1))
        store.push_event(sample_event(track_id=2))

        client = TestClient(app)
        resp = client.get("/events")
        body = resp.json()
        assert len(body) == 2
        assert [e["track_id"] for e in body] == [1, 2]


def test_ring_buffer_respects_max_size() -> None:
    small_store = api_module.EventStore(max_events=3)
    for i in range(10):
        small_store.push_event(sample_event(track_id=i))
    recent = small_store.recent()
    assert len(recent) == 3
    assert [e["track_id"] for e in recent] == [7, 8, 9]  # only the newest 3 kept


def test_websocket_receives_events_pushed_after_it_connects() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as ws:
            store.push_event(sample_event(track_id=99))
            data = ws.receive_json()
            assert data["track_id"] == 99
            assert data["event_type"] == "line_crossing"


def test_occupancy_reconciliation_publishes_operator_command() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/occupancy/cam1/reconcile",
            params={"action": "confirm_exited"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "camera_id": "cam1",
        "action": "confirm_exited",
    }


def test_occupancy_reconciliation_rejects_unknown_action() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/occupancy/cam1/reconcile",
            params={"action": "reset"},
        )

    assert response.status_code == 400


def test_snapshot_endpoint_serves_event_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "EVIDENCE_ROOT", str(tmp_path))
    snapshot = tmp_path / "default" / "cam1" / "2026-01-01" / "event.jpg"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"jpeg-data")
    event = evidence_event(str(snapshot))
    store.push_event(event)

    with TestClient(app) as client:
        response = client.get(f"/api/events/{event['event_id']}/evidence/snapshot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"jpeg-data"


def test_snapshot_endpoint_rejects_invalid_id_and_path_escape(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "EVIDENCE_ROOT", str(tmp_path))
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"not-served")
    event = evidence_event(str(outside))
    store.push_event(event)

    with TestClient(app) as client:
        invalid = client.get("/api/events/not-a-uuid/evidence/snapshot")
        escaped = client.get(f"/api/events/{event['event_id']}/evidence/snapshot")

    assert invalid.status_code == 400
    assert escaped.status_code == 404


def test_evidence_ready_updates_matching_event() -> None:
    event_id = str(uuid4())
    event = sample_event()
    event.update({"event_id": event_id, "site_id": "default"})
    notification = {
        "type": "evidence_ready",
        "event_id": event_id,
        "site_id": "default",
        "camera_id": "cam1",
        "evidence": {"clip_path": "evidence/event.mp4"},
    }

    store.push_event(notification)
    store.push_event(event)

    assert store.recent()[0]["evidence"] == notification["evidence"]


def test_clip_endpoint_supports_full_and_range_reads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "EVIDENCE_ROOT", str(tmp_path))
    clip = tmp_path / "default" / "cam1" / "2026-01-01" / "event.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"0123456789")
    event = evidence_event(str(tmp_path / "snapshot.jpg"), str(clip))
    snapshot = Path(event["evidence"]["snapshot_path"])
    snapshot.write_bytes(b"jpeg")
    store.push_event(event)

    with TestClient(app) as client:
        full = client.get(f"/api/events/{event['event_id']}/evidence/clip")
        partial = client.get(
            f"/api/events/{event['event_id']}/evidence/clip",
            headers={"Range": "bytes=2-5"},
        )

    assert full.status_code == 200
    assert full.content == b"0123456789"
    assert partial.status_code == 206
    assert partial.headers["content-range"] == "bytes 2-5/10"
    assert partial.content == b"2345"


def test_live_stream_serves_annotated_job_frames(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "JOB_ROOT", tmp_path)
    job_id = str(uuid4())
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    frame = job_dir / "live.jpg"
    frame.write_bytes(b"jpeg-frame")
    api_module.JOBS[job_id] = {
        "job_id": job_id,
        "status": "completed",
        "live_frame_path": str(frame),
    }

    try:
        with TestClient(app) as client:
            response = client.get(f"/api/jobs/{job_id}/stream")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
        assert b"Content-Type: image/jpeg" in response.content
        assert b"jpeg-frame" in response.content
    finally:
        api_module.JOBS.pop(job_id, None)


def test_stop_job_requests_graceful_pipeline_shutdown(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "JOB_ROOT", tmp_path)
    job_id = str(uuid4())
    api_module.JOBS[job_id] = {"job_id": job_id, "status": "running"}

    try:
        with TestClient(app) as client:
            response = client.post(f"/api/jobs/{job_id}/stop")
        assert response.status_code == 200
        assert response.json()["status"] == "stopping"
        assert (tmp_path / job_id / "stop.request").is_file()
    finally:
        api_module.JOBS.pop(job_id, None)


def test_job_status_does_not_expose_source_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "JOB_ROOT", tmp_path)
    job_id = str(uuid4())
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    frame = job_dir / "live.jpg"
    frame.write_bytes(b"jpeg-frame")
    api_module.JOBS[job_id] = {
        "job_id": job_id,
        "source_kind": "rtsp",
        "status": "running",
        "source": "rtsp://operator:secret@camera/live",
        "live_frame_path": str(frame),
    }

    try:
        with TestClient(app) as client:
            response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        assert "source" not in response.json()
        assert "secret" not in response.text
        assert isinstance(response.json()["last_frame_at"], float)
    finally:
        api_module.JOBS.pop(job_id, None)


def test_rtsp_job_uses_selected_source_without_persisting_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "JOB_ROOT", tmp_path)
    monkeypatch.setattr(
        api_module,
        "_base_pipeline_config",
        lambda: {"camera": {"id": "default", "site_id": "default", "source": ""}},
    )
    started = []
    monkeypatch.setattr(
        api_module,
        "_start_job",
        lambda job_id, config_path, source: started.append((job_id, config_path, source)),
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/jobs/rtsp",
                data={
                    "source": "rtsp://operator:secret@camera/live",
                    "camera_id": "front-door",
                    "site_id": "site-a",
                },
            )
        assert response.status_code == 200
        assert response.json()["status"] == "starting"
        assert "secret" not in response.text
        assert started[0][2] == "rtsp://operator:secret@camera/live"
        persisted = yaml.safe_load(Path(started[0][1]).read_text(encoding="utf-8"))
        assert persisted["camera"]["source"] == ""
        assert "secret" not in Path(started[0][1]).read_text(encoding="utf-8")
    finally:
        if started:
            api_module.JOBS.pop(started[0][0], None)


def test_upload_job_uses_uploaded_file_as_its_pipeline_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "JOB_ROOT", tmp_path)
    monkeypatch.setattr(
        api_module,
        "_base_pipeline_config",
        lambda: {"camera": {"id": "default", "site_id": "default", "source": ""}},
    )
    started = []
    monkeypatch.setattr(
        api_module,
        "_start_job",
        lambda job_id, config_path, source: started.append((job_id, config_path, source)),
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/jobs/upload",
                data={"camera_id": "upload-test", "site_id": "site-a"},
                files={"file": ("incident.mp4", b"video-data", "video/mp4")},
            )
        assert response.status_code == 200
        assert response.json()["status"] == "starting"
        assert started[0][2].endswith("input.mp4")
        assert Path(started[0][2]).read_bytes() == b"video-data"
        persisted = yaml.safe_load(Path(started[0][1]).read_text(encoding="utf-8"))
        assert persisted["camera"]["source"] == ""
    finally:
        if started:
            api_module.JOBS.pop(started[0][0], None)
