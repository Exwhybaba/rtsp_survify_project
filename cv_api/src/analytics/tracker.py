"""ByteTrack adapter for the CV analytics pipeline.

Accepts the project's Detection objects and returns TrackedObject objects,
preserving the interface used by the zone engine and pipeline.

Extra behaviour: when ByteTrack starts a new track shortly after another
track of the same class was lost nearby, the new track reuses the old ID.
This hides ID switches caused by large frame-to-frame jumps.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker as RoboflowByteTrackTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One detector output for a single frame."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the center point of the detection bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class TrackedObject:
    """Public immutable view of a tracked object for one frame."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str
    centroid: tuple[float, float]
    age: int
    hits: int


class ByteTrackAdapter:
    """Adapts Roboflow ByteTrack to the existing project interface."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_consecutive_frames: int = 1,
        high_conf_det_threshold: float = 0.5,
        minimum_iou_threshold: float = 0.1,
        relink_max_gap_frames: int = 15,
        relink_distance_factor: float = 1.5,
    ) -> None:

        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.high_conf_det_threshold = high_conf_det_threshold
        self.minimum_iou_threshold = minimum_iou_threshold

        # A new track may reuse the ID of a lost track if it appears within
        # this many frames and within (factor x box size) pixels of it.
        self.relink_max_gap_frames = relink_max_gap_frames
        self.relink_distance_factor = relink_distance_factor

        # ByteTrack's raw ID -> the stable ID we expose to the pipeline.
        self._raw_to_stable: dict[int, int] = {}
        # stable_id -> (last_frame, last_centroid, box_size, class_name)
        self._stable_state: dict[
            int, tuple[int, tuple[float, float], float, str]
        ] = {}

        self._tracker = self._create_tracker()

        self._frame_count = 0
        self._class_to_id: dict[str, int] = {}
        self._id_to_class: dict[int, str] = {}

        self._first_seen: dict[int, int] = {}
        self._last_seen: dict[int, int] = {}
        self._hits: dict[int, int] = {}

    def _create_tracker(self) -> RoboflowByteTrackTracker:
        return RoboflowByteTrackTracker(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            high_conf_det_threshold=self.high_conf_det_threshold,
            minimum_iou_threshold=self.minimum_iou_threshold,
        )

    @property
    def active_track_ids(self) -> set[int]:
        """Return currently visible and temporarily lost track IDs."""
        return {
            track_id
            for track_id, last_seen in self._last_seen.items()
            if self._frame_count - last_seen <= self.lost_track_buffer
        }

    def _get_class_id(self, class_name: str) -> int:
        """Return the stable numeric ID assigned to an object class."""
        if class_name not in self._class_to_id:
            class_id = len(self._class_to_id)
            self._class_to_id[class_name] = class_id
            self._id_to_class[class_id] = class_name

        return self._class_to_id[class_name]

    def _find_relink_candidate(
        self,
        class_name: str,
        bbox: tuple[float, float, float, float],
        claimed: set[int],
    ) -> int | None:
        """Find a recently lost track of the same class near this box."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        best_id: int | None = None
        best_dist: float | None = None

        for stable_id, (last_frame, (lx, ly), size, last_class) in (
            self._stable_state.items()
        ):
            # Skip IDs already in use this frame or of a different class.
            if stable_id in claimed or last_class != class_name:
                continue

            # Only consider tracks lost a few frames ago.
            gap = self._frame_count - last_frame
            if gap < 1 or gap > self.relink_max_gap_frames:
                continue

            # The new box must be close to where the old one was.
            dist = math.hypot(cx - lx, cy - ly)
            if dist > self.relink_distance_factor * size:
                continue

            if best_dist is None or dist < best_dist:
                best_id, best_dist = stable_id, dist

        return best_id

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Process one frame of detections and return tracked objects."""

        self._frame_count += 1

        if detections:
            xyxy = np.asarray(
                [det.bbox for det in detections],
                dtype=np.float32,
            ).reshape(-1, 4)

            confidence = np.asarray(
                [det.confidence for det in detections],
                dtype=np.float32,
            )

            class_ids = np.asarray(
                [self._get_class_id(det.class_name) for det in detections],
                dtype=np.int32,
            )
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int32)

        supervision_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
        )

        tracked_detections = self._tracker.update(supervision_detections)

        results: list[TrackedObject] = []

        if tracked_detections.tracker_id is None:
            self._remove_expired_metadata()
            return results

        # Pass 1: read valid rows from the tracker output.
        rows: list[
            tuple[int, tuple[float, float, float, float], float, str]
        ] = []

        for index, raw_value in enumerate(tracked_detections.tracker_id):
            raw_id = int(raw_value)

            # ByteTrack may return -1 for detections without a valid track.
            if raw_id < 0:
                continue

            bbox = tuple(
                float(value) for value in tracked_detections.xyxy[index]
            )

            confidence_value = tracked_detections.confidence
            score = (
                float(confidence_value[index])
                if confidence_value is not None
                else 0.0
            )

            class_id_value = tracked_detections.class_id
            class_id = (
                int(class_id_value[index])
                if class_id_value is not None
                else -1
            )

            class_name = self._id_to_class.get(class_id, "unknown")
            rows.append((raw_id, bbox, score, class_name))

        # Pass 2: map raw ByteTrack IDs to stable IDs, re-linking new
        # tracks to recently lost ones where possible.
        claimed = {
            self._raw_to_stable[raw_id]
            for raw_id, *_ in rows
            if raw_id in self._raw_to_stable
        }

        resolved: list[
            tuple[int, tuple[float, float, float, float], float, str]
        ] = []

        for raw_id, bbox, score, class_name in rows:
            stable_id = self._raw_to_stable.get(raw_id)

            if stable_id is None:
                stable_id = self._find_relink_candidate(
                    class_name, bbox, claimed
                )

                if stable_id is None:
                    stable_id = raw_id
                else:
                    logger.info(
                        "RELINK raw_id=%d -> id=%d frame=%d",
                        raw_id,
                        stable_id,
                        self._frame_count,
                    )

                self._raw_to_stable[raw_id] = stable_id
                claimed.add(stable_id)

            resolved.append((stable_id, bbox, score, class_name))

        # Pass 3: update bookkeeping and build the output objects.
        for track_id, bbox, score, class_name in resolved:
            if track_id not in self._first_seen:
                self._first_seen[track_id] = self._frame_count
                self._hits[track_id] = 0
                logger.info(
                    "NEW track id=%d frame=%d class=%s conf=%.2f bbox=%s",
                    track_id,
                    self._frame_count,
                    class_name,
                    score,
                    tuple(round(v) for v in bbox),
                )

            self._hits[track_id] += 1
            self._last_seen[track_id] = self._frame_count

            x1, y1, x2, y2 = bbox
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            self._stable_state[track_id] = (
                self._frame_count,
                centroid,
                max(x2 - x1, y2 - y1),
                class_name,
            )

            age = self._frame_count - self._first_seen[track_id] + 1

            results.append(
                TrackedObject(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=score,
                    class_name=class_name,
                    centroid=centroid,
                    age=age,
                    hits=self._hits[track_id],
                )
            )

        self._remove_expired_metadata()

        return results

    def _remove_expired_metadata(self) -> None:
        """Remove bookkeeping for tracks outside the lost-track window."""

        expired_ids = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if self._frame_count - last_seen > self.lost_track_buffer
        ]

        for track_id in expired_ids:
            logger.info(
                "EXPIRED track id=%d last_seen=%d frame=%d",
                track_id,
                self._last_seen[track_id],
                self._frame_count,
            )
            self._last_seen.pop(track_id, None)
            self._first_seen.pop(track_id, None)
            self._hits.pop(track_id, None)
            self._stable_state.pop(track_id, None)

        # Drop raw-ID mappings whose stable ID no longer exists.
        self._raw_to_stable = {
            raw: stable
            for raw, stable in self._raw_to_stable.items()
            if stable in self._last_seen
        }

    def reset(self) -> None:
        """Reset tracking state before processing a new video."""

        self._tracker = self._create_tracker()

        self._frame_count = 0
        self._class_to_id.clear()
        self._id_to_class.clear()
        self._first_seen.clear()
        self._last_seen.clear()
        self._hits.clear()
        self._raw_to_stable.clear()
        self._stable_state.clear()
