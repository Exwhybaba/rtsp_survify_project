"""FastAPI dashboard service.

Subscribes to the MQTT topic the analytics pipeline publishes
``VisionEvent``s to, keeps the most recent N of them in an in-memory ring
buffer, and re-broadcasts every event in real time to any browser
connected over ``WS /ws/live``.

Endpoints
---------
``GET /health``
    Liveness probe.
``GET /events``
    Recent events currently held in the ring buffer (newest last).
``WS /ws/live``
    Streams every new event, as JSON, to the client as it arrives.

The MQTT connection is established on startup on a best-effort basis: if
no broker is reachable (e.g. running the dashboard alone, or under
pytest) the app still serves ``/health`` and ``/events`` -- it just never
receives anything, and events can be pushed in directly via
``push_event`` for testing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

import paho.mqtt.client as mqtt
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

MAX_EVENTS = int(os.environ.get("DASHBOARD_MAX_EVENTS", "500"))
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "analytics/events")
MQTT_OCCUPANCY_CONTROL_TOPIC = os.environ.get(
    "MQTT_OCCUPANCY_CONTROL_TOPIC",
    "analytics/control/occupancy",
)
EVIDENCE_ROOT = os.environ.get("EVIDENCE_DIR", "evidence")
JOB_ROOT = Path(os.environ.get("SUVIRFY_JOB_DIR", "runtime-jobs")).resolve()
MAX_UPLOAD_BYTES = int(os.environ.get("SUVIRFY_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_CONCURRENT_JOBS = int(os.environ.get("SUVIRFY_MAX_CONCURRENT_JOBS", "1"))

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class EventStore:
    """Thread-safe-enough ring buffer + WebSocket fan-out.

    MQTT callbacks fire on paho's own network thread while WebSocket
    consumers live on the asyncio event loop, so pushing a new event has
    to hop threads via ``run_coroutine_threadsafe``.
    """

    def __init__(self, max_events: int = MAX_EVENTS) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._pending_evidence: dict[str, dict[str, Any]] = {}
        self._subscribers: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Associate the store with the dashboard event loop."""
        self._loop = loop

    def recent(self) -> list[dict[str, Any]]:
        """Return buffered events in oldest-to-newest order."""
        return list(self._buffer)

    async def subscribe(self, web_socket: WebSocket) -> None:
        """Register a WebSocket client for live event delivery."""
        self._subscribers.add(web_socket)

    def unsubscribe(self, web_socket: WebSocket) -> None:
        """Remove a WebSocket client from live event delivery."""
        self._subscribers.discard(web_socket)

    def push_event(self, event: dict[str, Any]) -> None:
        """Add an event to the buffer and fan it out to WebSocket clients.

        Safe to call from any thread.
        """
        event_id = event.get("event_id")
        if event.get("type") == "evidence_ready" and isinstance(event_id, str):
            self._merge_evidence(event_id, event)
            return
        if isinstance(event_id, str) and event_id in self._pending_evidence:
            evidence = self._pending_evidence.pop(event_id).get("evidence")
            if isinstance(evidence, dict):
                event["evidence"] = evidence
        self._buffer.append(event)
        if self._loop is None:
            return
        for web_socket in list(self._subscribers):
            asyncio.run_coroutine_threadsafe(
                self._safe_send(web_socket, event),
                self._loop,
            )

    def _merge_evidence(self, event_id: str, notification: dict[str, Any]) -> None:
        evidence = notification.get("evidence")
        if not isinstance(evidence, dict):
            return
        for event in self._buffer:
            if event.get("event_id") == event_id:
                event["evidence"] = evidence
                self._broadcast(event)
                return
        self._pending_evidence[event_id] = notification

    def _broadcast(self, event: dict[str, Any]) -> None:
        if self._loop is None:
            return
        for web_socket in list(self._subscribers):
            asyncio.run_coroutine_threadsafe(
                self._safe_send(web_socket, event),
                self._loop,
            )

    async def _safe_send(
        self,
        web_socket: WebSocket,
        event: dict[str, Any],
    ) -> None:
        try:
            await web_socket.send_json(event)
        except (OSError, RuntimeError):
            self._subscribers.discard(web_socket)

    def find_event(self, event_id: UUID) -> dict[str, Any] | None:
        """Return a buffered event by UUID."""
        event_id_text = str(event_id)
        return next(
            (event for event in reversed(self._buffer) if event.get("event_id") == event_id_text),
            None,
        )


store = EventStore()


def _on_mqtt_message(_client: mqtt.Client, _userdata: Any, msg: mqtt.MQTTMessage) -> None:
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Dropping malformed MQTT payload on topic %s", msg.topic)
        return
    store.push_event(payload)


def _build_mqtt_client() -> mqtt.Client:
    client = mqtt.Client(client_id="vision-dashboard")
    client.on_message = _on_mqtt_message

    def _on_connect(client: mqtt.Client, _userdata: Any, _flags: Any, _rc: int) -> None:
        client.subscribe(MQTT_TOPIC)
        logger.info("Dashboard subscribed to MQTT topic %s", MQTT_TOPIC)

    client.on_connect = _on_connect
    return client


MQTT_CLIENT: mqtt.Client | None = None
JOBS: dict[str, dict[str, Any]] = {}
JOB_PROCESSES: dict[str, subprocess.Popen[str]] = {}
JOB_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start and stop the optional MQTT subscription for the dashboard."""
    global MQTT_CLIENT  # pylint: disable=global-statement
    store.bind_loop(asyncio.get_running_loop())
    try:
        MQTT_CLIENT = _build_mqtt_client()
        MQTT_CLIENT.connect_async(MQTT_HOST, MQTT_PORT)
        MQTT_CLIENT.loop_start()
        logger.info("Connecting to MQTT broker %s:%s", MQTT_HOST, MQTT_PORT)
    except (OSError, RuntimeError, ValueError):
        logger.warning(
            "Could not start MQTT client; dashboard will run without live events",
            exc_info=True,
        )
        MQTT_CLIENT = None
    yield
    if MQTT_CLIENT is not None:
        MQTT_CLIENT.loop_stop()
        try:
            MQTT_CLIENT.disconnect()
        except (OSError, RuntimeError):
            pass


app = FastAPI(title="Video Analytics Dashboard", lifespan=lifespan)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    """Return the dashboard liveness status."""
    return {"status": "ok"}


@app.get("/events")
def get_events() -> list[dict[str, Any]]:
    """Return buffered analytics events."""
    return store.recent()


def _active_job_count() -> int:
    return sum(
        1 for job in JOBS.values()
        if job["status"] in {"queued", "running"}
    )


def _write_job_config(job_id: str, config: dict[str, Any]) -> Path:
    import yaml

    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _base_pipeline_config() -> dict[str, Any]:
    import yaml

    path = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def _start_job(job_id: str, config_path: Path, source: str) -> None:
    log_path = JOB_ROOT / job_id / "pipeline.log"
    command = [
        sys.executable,
        "-m",
        "src.analytics.pipeline",
        "--config",
        str(config_path),
        "--source",
        source,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    with JOB_LOCK:
        JOB_PROCESSES[job_id] = process
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["pid"] = process.pid

    def monitor() -> None:
        return_code = process.wait()
        with JOB_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                if job["status"] != "stopped":
                    job["status"] = "completed" if return_code == 0 else "failed"
                job["return_code"] = return_code
            JOB_PROCESSES.pop(job_id, None)
        input_path = JOB_ROOT / job_id
        for candidate in input_path.glob("input.*"):
            candidate.unlink(missing_ok=True)

    threading.Thread(target=monitor, name=f"job-monitor-{job_id}", daemon=True).start()


def _create_job(config: dict[str, Any], source_kind: str) -> dict[str, Any]:
    import uuid

    with JOB_LOCK:
        if _active_job_count() >= MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="maximum concurrent jobs reached")
        job_id = str(uuid.uuid4())
        config.setdefault("visualization", {})["output_path"] = str(
            JOB_ROOT / job_id / "annotated.mp4"
        )
        config_path = _write_job_config(job_id, config)
        job = {
            "job_id": job_id,
            "source_kind": source_kind,
            "status": "queued",
            "config_path": str(config_path),
            "video_path": config.get("visualization", {}).get("output_path"),
            "source": str(config.get("camera", {}).get("source", "")),
        }
        JOBS[job_id] = job
    try:
        _start_job(job_id, config_path, job["source"])
    except OSError as exc:
        with JOB_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(exc)
        raise HTTPException(status_code=500, detail="could not start analysis job") from exc
    return job


@app.post("/api/jobs/upload")
async def upload_video(
    file: UploadFile = File(...),
    camera_id: str = Form("upload-camera"),
    site_id: str = Form("default"),
) -> dict[str, Any]:
    """Create an isolated analysis job for an uploaded video."""
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="unsupported video file type")
    if not camera_id.strip() or not site_id.strip():
        raise HTTPException(status_code=400, detail="camera_id and site_id are required")

    import uuid

    job_id = str(uuid.uuid4())
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"input{extension}"
    total = 0
    try:
        with video_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="video upload is too large")
                target.write(chunk)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    config = _base_pipeline_config()
    config["camera"] = {
        **config.get("camera", {}),
        "id": camera_id.strip(),
        "site_id": site_id.strip(),
        "source": str(video_path),
        "loop": False,
    }
    config["visualization"] = {"enabled": False}
    config["visualization"]["output_path"] = str(job_dir / "annotated.mp4")
    config["visualization"]["output_fps"] = 10
    try:
        return _create_job(config, "upload")
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise


@app.post("/api/jobs/rtsp")
def start_rtsp_job(
    source: str = Form(...),
    camera_id: str = Form("rtsp-camera"),
    site_id: str = Form("default"),
) -> dict[str, Any]:
    """Create an isolated analysis job for an RTSP source."""
    if not source.lower().startswith(("rtsp://", "rtsps://")):
        raise HTTPException(status_code=400, detail="source must be an RTSP URL")
    if not camera_id.strip() or not site_id.strip():
        raise HTTPException(status_code=400, detail="camera_id and site_id are required")
    config = _base_pipeline_config()
    config["camera"] = {
        **config.get("camera", {}),
        "id": camera_id.strip(),
        "site_id": site_id.strip(),
        "source": source,
        "loop": False,
    }
    config["visualization"] = {"enabled": False}
    config["visualization"]["output_fps"] = 10
    return _create_job(config, "rtsp")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return dict(job)


@app.get("/api/jobs/{job_id}/video")
def get_job_video(job_id: str, request: Request) -> StreamingResponse:
    """Serve the annotated detection/tracking video generated by a job."""
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        raw_path = job.get("video_path")
    if not isinstance(raw_path, str):
        raise HTTPException(status_code=404, detail="annotated video is not available")
    path = Path(raw_path).resolve()
    try:
        path.relative_to((JOB_ROOT / job_id).resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="invalid job video path") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="annotated video is not available")
    file_size = path.stat().st_size
    start = 0
    end = file_size - 1
    status_code = 200
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
    range_header = request.headers.get("range")
    if range_header:
        if not range_header.startswith("bytes=") or "," in range_header:
            raise HTTPException(status_code=416, detail="invalid range")
        requested = range_header[6:].split("-", maxsplit=1)
        try:
            if len(requested) != 2:
                raise ValueError
            if requested[0]:
                start = int(requested[0])
            else:
                suffix_length = int(requested[1])
                start = max(0, file_size - suffix_length)
            if requested[1] and requested[0]:
                end = int(requested[1])
        except ValueError as exc:
            raise HTTPException(status_code=416, detail="invalid range") from exc
        if start < 0 or start >= file_size or end < start:
            raise HTTPException(status_code=416, detail="range not satisfiable")
        end = min(end, file_size - 1)
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(end - start + 1)

    def stream():
        with path.open("rb") as video_file:
            video_file.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = video_file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                yield chunk
                remaining -= len(chunk)

    return StreamingResponse(
        stream(),
        status_code=status_code,
        media_type="video/mp4",
        headers=headers,
    )


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, str]:
    with JOB_LOCK:
        process = JOB_PROCESSES.get(job_id)
        if process is None:
            raise HTTPException(status_code=404, detail="running job not found")
        process.terminate()
        JOBS[job_id]["status"] = "stopped"
    return {"job_id": job_id, "status": "stopped"}


@app.delete("/api/evidence")
def clear_evidence() -> dict[str, int]:
    """Delete all locally stored evidence and clear buffered dashboard events."""
    root = Path(EVIDENCE_ROOT).resolve()
    if not root.is_dir():
        store._buffer.clear()
        store._pending_evidence.clear()
        return {"deleted_files": 0, "cleared_events": 0}
    deleted_files = 0
    for path in root.rglob("*"):
        if path.is_file():
            path.unlink()
            deleted_files += 1
    cleared_events = len(store._buffer)
    store._buffer.clear()
    store._pending_evidence.clear()
    return {"deleted_files": deleted_files, "cleared_events": cleared_events}


def _event_or_404(event_id: str) -> dict[str, Any]:
    try:
        parsed_id = UUID(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid event ID") from exc
    event = store.find_event(parsed_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return event


def _evidence_file(event_id: str, field: str) -> Path:
    event = _event_or_404(event_id)
    evidence = event.get("evidence")
    if not isinstance(evidence, dict):
        raise HTTPException(status_code=404, detail="evidence is not available")
    raw_path = evidence.get(field)
    if not isinstance(raw_path, str) or not raw_path:
        raise HTTPException(status_code=404, detail="evidence file is not available")

    root = Path(EVIDENCE_ROOT).resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="evidence file is outside evidence root") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="evidence file is not available")
    return candidate


@app.get("/api/events/{event_id}/evidence")
def get_event_evidence(event_id: str) -> dict[str, Any]:
    """Return evidence metadata for a buffered event."""
    event = _event_or_404(event_id)
    evidence = event.get("evidence")
    if not isinstance(evidence, dict):
        raise HTTPException(status_code=404, detail="evidence is not available")
    return {"event_id": event_id, "evidence": evidence}


@app.get("/api/events/{event_id}/evidence/snapshot")
def get_event_snapshot(event_id: str) -> FileResponse:
    """Serve the event snapshot after validating its evidence-root containment."""
    path = _evidence_file(event_id, "snapshot_path")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/events/{event_id}/evidence/clip")
def get_event_clip(event_id: str, request: Request) -> StreamingResponse:
    """Stream an event clip with support for HTTP byte ranges."""
    path = _evidence_file(event_id, "clip_path")
    file_size = path.stat().st_size
    start = 0
    end = file_size - 1
    status_code = 200
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    range_header = request.headers.get("range")
    if range_header:
        if not range_header.startswith("bytes=") or "," in range_header:
            raise HTTPException(status_code=416, detail="invalid range")
        requested = range_header[6:].split("-", maxsplit=1)
        try:
            if len(requested) != 2:
                raise ValueError
            if requested[0]:
                start = int(requested[0])
            else:
                suffix_length = int(requested[1])
                start = max(0, file_size - suffix_length)
            if len(requested) == 2 and requested[1] and requested[0]:
                end = int(requested[1])
        except ValueError as exc:
            raise HTTPException(status_code=416, detail="invalid range") from exc
        if start < 0 or start >= file_size or end < start:
            raise HTTPException(status_code=416, detail="range not satisfiable")
        end = min(end, file_size - 1)
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(end - start + 1)

    def stream():
        with path.open("rb") as stream_file:
            stream_file.seek(start)
            remaining = end - start + 1
            while chunk := stream_file.read(1024 * 1024):
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                yield chunk
                remaining -= len(chunk)
                if remaining <= 0:
                    break

    return StreamingResponse(
        stream(),
        status_code=status_code,
        media_type="video/mp4",
        headers=headers,
    )


@app.post("/occupancy/{camera_id}/reconcile")
def reconcile_occupancy(camera_id: str, action: str) -> dict[str, str]:
    """Send an operator occupancy decision to the analytics pipeline."""
    if action not in {"confirm_present", "confirm_exited"}:
        raise HTTPException(
            status_code=400,
            detail="action must be confirm_present or confirm_exited",
        )
    if MQTT_CLIENT is None:
        raise HTTPException(status_code=503, detail="MQTT client is unavailable")

    payload = json.dumps({"camera_id": camera_id, "action": action})
    info = MQTT_CLIENT.publish(
        MQTT_OCCUPANCY_CONTROL_TOPIC,
        payload=payload,
        qos=1,
    )
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        raise HTTPException(
            status_code=503,
            detail="Could not publish reconciliation command",
        )
    return {"status": "accepted", "camera_id": camera_id, "action": action}


@app.get("/")
def index() -> FileResponse:
    """Serve the dashboard HTML page."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.websocket("/ws/live")
async def websocket_live(web_socket: WebSocket) -> None:
    """Accept a client and stream newly received events to it."""
    await web_socket.accept()
    await store.subscribe(web_socket)
    try:
        while True:
            # We don't expect client -> server messages, but reading keeps
            # the connection alive and lets us detect a disconnect promptly.
            await web_socket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        store.unsubscribe(web_socket)
