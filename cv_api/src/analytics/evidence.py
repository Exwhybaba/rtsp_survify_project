"""Optional rolling evidence capture for analytics events."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np

from src.analytics.ingest import redact_uri
from src.analytics.schemas import EvidenceReadyMessage, EvidenceRef, VisionEvent

logger = logging.getLogger(__name__)

_PANEL_COLOR = (24, 28, 36)
_PANEL_BORDER_COLOR = (86, 96, 112)
_TEXT_COLOR = (242, 245, 248)
_ACCENT_COLOR = (0, 204, 255)
_BOX_COLOR = (0, 0, 255)


@dataclass(frozen=True)
class _Frame:
    event_time: float
    data: bytes


@dataclass
class _ClipJob:
    start: float
    end: float
    event_ids: list[UUID]
    events: dict[UUID, VisionEvent]
    frame_size: tuple[int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_annotation_lines(event: VisionEvent) -> list[str]:
    """Return the evidence-panel lines available for an event."""
    metadata = event.metadata
    lines = [
        f"EVENT: {event.event_type.value.upper()}",
    ]
    if metadata.get("zone") is not None:
        lines.append(f"ZONE: {metadata['zone']}")
    if metadata.get("severity") is not None:
        lines.append(f"SEVERITY: {str(metadata['severity']).upper()}")
    lines.extend(
        [
            f"CAMERA: {event.camera_id}",
            f"SITE: {event.site_id}",
            f"TIME: {event.timestamp.isoformat(timespec='seconds')}",
        ]
    )
    return lines


def annotate_event_frame(frame: np.ndarray, event: VisionEvent) -> np.ndarray:
    """Return a professional evidence image annotated for one event."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    x1, y1, x2, y2 = (
        max(0, min(width - 1, int(value)))
        for value in event.bbox.as_tuple()
    )
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 4)

    metadata = event.metadata
    object_parts = []
    if metadata.get("class_name") is not None:
        object_parts.append(str(metadata["class_name"]).upper())
    object_parts.extend(
        [
            f"ID: {event.track_id}",
            f"{event.confidence:.0%}",
        ]
    )
    object_label = " | ".join(object_parts)
    object_scale = max(0.7, min(1.4, width / 2400.0))
    object_thickness = max(2, round(object_scale * 2))
    (label_width, label_height), baseline = cv2.getTextSize(
        object_label,
        cv2.FONT_HERSHEY_SIMPLEX,
        object_scale,
        object_thickness,
    )
    label_y = y1 - 12
    if label_y - label_height - baseline < 0:
        label_y = min(height - baseline - 2, y2 + label_height + baseline + 12)
    label_x = max(0, min(x1, width - label_width - 12))
    cv2.rectangle(
        annotated,
        (label_x, label_y - label_height - baseline - 8),
        (min(width - 1, label_x + label_width + 12), min(height - 1, label_y + 4)),
        _PANEL_COLOR,
        cv2.FILLED,
    )
    cv2.putText(
        annotated,
        object_label,
        (label_x + 6, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        object_scale,
        _TEXT_COLOR,
        object_thickness,
        cv2.LINE_AA,
    )

    panel_lines = event_annotation_lines(event)
    panel_scale = max(0.65, min(1.15, width / 3000.0))
    panel_thickness = max(2, round(panel_scale * 2))
    line_height = max(28, round(38 * panel_scale))
    padding = max(18, round(28 * panel_scale))
    text_sizes = [
        cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            panel_scale,
            panel_thickness,
        )[0][0]
        for line in panel_lines
    ]
    panel_width = min(
        width - 2 * padding,
        max(text_sizes, default=0) + 2 * padding,
    )
    panel_height = min(
        height - 2 * padding,
        len(panel_lines) * line_height + 2 * padding,
    )
    panel_x = padding
    panel_y = padding
    cv2.rectangle(
        annotated,
        (panel_x, panel_y),
        (
            min(width - 1, panel_x + panel_width),
            min(height - 1, panel_y + panel_height),
        ),
        _PANEL_COLOR,
        cv2.FILLED,
    )
    cv2.rectangle(
        annotated,
        (panel_x, panel_y),
        (
            min(width - 1, panel_x + panel_width),
            min(height - 1, panel_y + panel_height),
        ),
        _PANEL_BORDER_COLOR,
        max(1, round(panel_scale)),
    )
    for index, line in enumerate(panel_lines):
        color = _ACCENT_COLOR if index == 0 else _TEXT_COLOR
        baseline_y = panel_y + padding + (index + 1) * line_height - 8
        cv2.putText(
            annotated,
            line,
            (panel_x + padding, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            panel_scale,
            color,
            panel_thickness,
            cv2.LINE_AA,
        )
    return annotated


class EvidenceCapture:
    """Capture snapshots immediately and clips asynchronously."""

    def __init__(
        self,
        evidence_dir: str | Path,
        site_id: str,
        camera_id: str,
        source_uri: str,
        pre_roll_seconds: float = 5.0,
        post_roll_seconds: float = 5.0,
        fps_cap: float = 10.0,
        enabled: bool = False,
        on_evidence_ready: Callable[[EvidenceReadyMessage], None] | None = None,
    ) -> None:
        self.enabled = enabled
        self.evidence_dir = Path(evidence_dir)
        self.site_id = site_id
        self.camera_id = camera_id
        self.source_uri = source_uri
        self.pre_roll_seconds = max(0.0, pre_roll_seconds)
        self.post_roll_seconds = max(0.0, post_roll_seconds)
        self.fps_cap = max(1.0, fps_cap)
        self._on_evidence_ready = on_evidence_ready
        self._frames: deque[_Frame] = deque()
        self._events: dict[UUID, VisionEvent] = {}
        self._job: _ClipJob | None = None
        self._condition = threading.Condition()
        self._stop = False
        self._worker: threading.Thread | None = None
        self._last_frame_time: float | None = None
        self._frame_size: tuple[int, int] | None = None
        if self.enabled:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name=f"evidence-{camera_id}",
                daemon=True,
            )
            self._worker.start()

    def add_frame(self, frame: np.ndarray, event_time: float) -> None:
        """Add one JPEG-compressed frame to the rolling buffer."""
        if not self.enabled:
            return
        if (
            self._last_frame_time is not None
            and event_time - self._last_frame_time < 1.0 / self.fps_cap
        ):
            return
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("could not JPEG-compress evidence frame")
        self._last_frame_time = event_time
        with self._condition:
            self._frames.append(_Frame(event_time, encoded.tobytes()))
            self._frame_size = (frame.shape[1], frame.shape[0])
            cutoff = event_time - self.pre_roll_seconds - self.post_roll_seconds - 1.0
            while self._frames and self._frames[0].event_time < cutoff:
                self._frames.popleft()
            self._condition.notify_all()

    def on_events(
        self,
        events: list[VisionEvent],
        frame: np.ndarray,
        event_time: float,
    ) -> None:
        """Save event snapshots and schedule a merged clip."""
        if not self.enabled or not events:
            return
        timestamp = datetime.fromtimestamp(event_time, timezone.utc)
        day = timestamp.strftime("%Y-%m-%d")
        output_dir = self.evidence_dir / self.site_id / self.camera_id / day
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._condition:
            for event in events:
                annotated = annotate_event_frame(frame, event)
                snapshot_path = output_dir / f"{event.event_id}.jpg"
                if not cv2.imwrite(str(snapshot_path), annotated):
                    raise ValueError(f"could not write evidence snapshot {snapshot_path}")
                event.evidence = EvidenceRef(
                    snapshot_path=str(snapshot_path),
                    clip_start=timestamp,
                    clip_end=datetime.fromtimestamp(
                        event_time + self.post_roll_seconds,
                        timezone.utc,
                    ),
                    snapshot_sha256=_sha256(snapshot_path),
                )
                self._events[event.event_id] = event

            start = event_time - self.pre_roll_seconds
            end = event_time + self.post_roll_seconds
            if self._job is None:
                self._job = _ClipJob(
                    start=start,
                    end=end,
                    event_ids=[event.event_id for event in events],
                    events={event.event_id: event for event in events},
                    frame_size=self._frame_size or (frame.shape[1], frame.shape[0]),
                )
            else:
                self._job.start = min(self._job.start, start)
                self._job.end = max(self._job.end, end)
                for event in events:
                    if event.event_id not in self._job.event_ids:
                        self._job.event_ids.append(event.event_id)
                    self._job.events[event.event_id] = event
            self._condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop
                    or (
                        self._job is not None
                        and self._frames
                        and self._frames[-1].event_time >= self._job.end
                    )
                )
                if self._stop and self._job is None:
                    return
                if self._job is None:
                    continue
                if not self._stop and self._frames[-1].event_time < self._job.end:
                    continue
                job = self._job
                self._job = None
                frames = list(self._frames)
            try:
                self._write_clip(job, frames)
            except Exception:  # noqa: BLE001 - evidence must not stop analytics
                logger.exception("Failed to write evidence clip for camera=%s", self.camera_id)

    def _write_clip(self, job: _ClipJob, frames: list[_Frame]) -> None:
        timestamp = datetime.fromtimestamp(job.start, timezone.utc)
        output_dir = self.evidence_dir / self.site_id / self.camera_id / timestamp.strftime("%Y-%m-%d")
        output_dir.mkdir(parents=True, exist_ok=True)
        clip_path = output_dir / f"{job.event_ids[0]}.mp4"
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps_cap,
            job.frame_size,
        )
        try:
            for item in frames:
                if job.start <= item.event_time <= job.end:
                    decoded = cv2.imdecode(np.frombuffer(item.data, np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None:
                        writer.write(decoded)
        finally:
            writer.release()

        final_path = self._maybe_reencode(clip_path)
        clip_hash = _sha256(final_path)
        end_timestamp = datetime.fromtimestamp(job.end, timezone.utc)
        sidecar = final_path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "event_ids": [str(event_id) for event_id in job.event_ids],
                    "site_id": self.site_id,
                    "camera_id": self.camera_id,
                    "source": redact_uri(self.source_uri),
                    "clip_start": timestamp.isoformat(),
                    "clip_end": end_timestamp.isoformat(),
                    "snapshot_hashes": {
                        str(event_id): job.events[event_id].evidence.snapshot_sha256
                        for event_id in job.event_ids
                    },
                    "clip_sha256": clip_hash,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        for event_id in job.event_ids:
            event = job.events[event_id]
            assert event.evidence is not None
            event.evidence.clip_path = str(final_path)
            event.evidence.clip_start = timestamp
            event.evidence.clip_end = end_timestamp
            event.evidence.clip_sha256 = clip_hash
            if self._on_evidence_ready is not None:
                self._on_evidence_ready(
                    EvidenceReadyMessage(
                        event_id=event.event_id,
                        site_id=event.site_id,
                        camera_id=event.camera_id,
                        evidence=event.evidence,
                    )
                )

    def _maybe_reencode(self, clip_path: Path) -> Path:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logger.warning("ffmpeg is unavailable; browser playback may not work for %s", clip_path)
            return clip_path
        output_path = clip_path.with_name(f"{clip_path.stem}.h264.mp4")
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(clip_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg H.264 conversion failed; keeping mp4v clip")
            return clip_path
        clip_path.unlink(missing_ok=True)
        return output_path

    def flush(self) -> None:
        """Finish any pending clip synchronously."""
        if not self.enabled:
            return
        with self._condition:
            if self._job is None:
                return
            job = self._job
            self._job = None
            frames = list(self._frames)
        self._write_clip(job, frames)

    def close(self) -> None:
        """Stop the worker after flushing pending evidence."""
        if not self.enabled:
            return
        self.flush()
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
