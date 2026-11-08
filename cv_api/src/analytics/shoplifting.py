"""Shoplifting / concealment analysis using a custom YOLO pose model.

The fine-tuned pose checkpoint (``best.pt``) is run once per frame. Each
detected person's keypoints are matched to the tracker's person boxes, a
concealment score is computed (wrist close to the torso core), and scores
are aggregated per track over a sliding time window before an event is
raised. Zone context (merchandise / checkout) is used to cut false alarms
and the old zone + exit-line heuristic remains as a fallback when no pose
model is configured.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.analytics.rules.engine import get_line_crossing_direction, point_in_polygon
from src.analytics.schemas import BoundingBox, EventType, VisionEvent
from src.analytics.tracker import TrackedObject

Point = tuple[float, float]
Box = tuple[float, float, float, float]

# COCO keypoint indices
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12

_VALID_DIRECTIONS = {"either", "negative_to_positive", "positive_to_negative"}


@dataclass(frozen=True)
class ShopliftingConfig:
    """Validated configuration for pose-based and fallback analysis."""

    enabled: bool = False
    # Path to the custom-trained pose weights (best.pt). None -> fallback only.
    pose_model_path: Path | None = None
    device: str = "cpu"
    pose_confidence: float = 0.25
    pose_image_size: int = 640
    pose_stride: int = 1  # run pose every N frames
    pose_match_iou: float = 0.3

    # Concealment heuristic (same logic as the training notebook).
    wrist_confidence: float = 0.5
    torso_confidence: float = 0.3
    concealment_ratio: float = 0.45  # wrist-to-torso distance / body scale
    classification_threshold: float = 0.5  # score cut-off (ratio hit == 0.5)

    # Temporal aggregation.
    observation_window_seconds: float = 5.0
    minimum_suspicious_observations: int = 3
    cooldown_seconds: float = 30.0

    # Zone context.
    merchandise_zones: tuple[tuple[str, tuple[Point, ...]], ...] = ()
    checkout_zones: tuple[tuple[str, tuple[Point, ...]], ...] = ()
    require_merchandise_interaction: bool = False
    merchandise_lookback_seconds: float = 30.0
    suppress_in_checkout: bool = True

    # Fallback (no pose model) behaviour.
    exit_line: tuple[Point, Point] | None = None
    allowed_direction: str = "either"
    minimum_merchandise_dwell_seconds: float = 2.0
    require_checkout_zone: bool = False

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> ShopliftingConfig:
        raw = raw or {}
        if not raw.get("enabled", False):
            return cls()

        def polygons(key: str) -> tuple[tuple[str, tuple[Point, ...]], ...]:
            result = []
            for index, item in enumerate(raw.get(key, [])):
                name = str(item.get("name", f"{key}_{index}"))
                polygon = tuple(
                    (float(point[0]), float(point[1])) for point in item["polygon"]
                )
                if len(polygon) < 3:
                    raise ValueError(f"{key} polygon '{name}' needs at least 3 points")
                result.append((name, polygon))
            return tuple(result)

        line = raw.get("exit_line")
        exit_line = None
        if line is not None:
            points = line.get("points", [])
            if len(points) != 2:
                raise ValueError("shoplifting.exit_line.points must contain two points")
            exit_line = (
                (float(points[0][0]), float(points[0][1])),
                (float(points[1][0]), float(points[1][1])),
            )

        allowed_direction = str(raw.get("allowed_direction", "either"))
        if allowed_direction not in _VALID_DIRECTIONS:
            raise ValueError("shoplifting.allowed_direction is invalid")

        def unit(key: str, default: float) -> float:
            value = float(raw.get(key, default))
            if not 0 <= value <= 1:
                raise ValueError(f"shoplifting.{key} must be between 0 and 1")
            return value

        threshold = unit("classification_threshold", 0.5)
        pose_confidence = unit("pose_confidence", 0.25)
        wrist_confidence = unit("wrist_confidence", 0.5)
        torso_confidence = unit("torso_confidence", 0.3)
        match_iou = unit("pose_match_iou", 0.3)

        ratio = float(raw.get("concealment_ratio", 0.45))
        window = float(raw.get("observation_window_seconds", 5.0))
        minimum_observations = int(raw.get("minimum_suspicious_observations", 3))
        cooldown = float(raw.get("cooldown_seconds", 30.0))
        dwell = float(raw.get("minimum_merchandise_dwell_seconds", 2.0))
        lookback = float(raw.get("merchandise_lookback_seconds", 30.0))
        stride = int(raw.get("pose_stride", 1))
        image_size = int(raw.get("pose_image_size", 640))
        if ratio <= 0:
            raise ValueError("shoplifting.concealment_ratio must be positive")
        if window <= 0 or minimum_observations < 1 or cooldown < 0 or dwell < 0:
            raise ValueError("shoplifting temporal settings are invalid")
        if lookback < 0 or stride < 1 or image_size < 32:
            raise ValueError("shoplifting pose/lookback settings are invalid")

        model_path = raw.get("pose_model_path")
        return cls(
            enabled=True,
            pose_model_path=Path(model_path) if model_path else None,
            device=str(raw.get("device", "cpu")),
            pose_confidence=pose_confidence,
            pose_image_size=image_size,
            pose_stride=stride,
            pose_match_iou=match_iou,
            wrist_confidence=wrist_confidence,
            torso_confidence=torso_confidence,
            concealment_ratio=ratio,
            classification_threshold=threshold,
            observation_window_seconds=window,
            minimum_suspicious_observations=minimum_observations,
            cooldown_seconds=cooldown,
            merchandise_zones=polygons("merchandise_zones"),
            checkout_zones=polygons("checkout_zones"),
            require_merchandise_interaction=bool(
                raw.get("require_merchandise_interaction", False)
            ),
            merchandise_lookback_seconds=lookback,
            suppress_in_checkout=bool(raw.get("suppress_in_checkout", True)),
            exit_line=exit_line,
            allowed_direction=allowed_direction,
            minimum_merchandise_dwell_seconds=dwell,
            require_checkout_zone=bool(raw.get("require_checkout_zone", False)),
        )


@dataclass
class ShopliftingObservation:
    """Latest per-track observation used by the live visualizer.

    ``probability`` is the concealment score in [0, 1]; ``None`` means no
    pose could be matched to the track on this frame.
    """

    track_id: int
    bbox: Box
    probability: float | None
    pose_available: bool


@dataclass
class _BehaviorState:
    previous_point: Point | None = None
    merchandise_started_at: float | None = None
    merchandise_zone: str | None = None
    merchandise_seconds: float = 0.0
    last_merchandise_at: float | None = None
    checkout_seen: bool = False
    last_event_at: float = -float("inf")
    predictions: deque[tuple[float, float]] = field(default_factory=deque)


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _zone_at(point: Point, zones: tuple[tuple[str, tuple[Point, ...]], ...]) -> str | None:
    for name, polygon in zones:
        if point_in_polygon(point, list(polygon)):
            return name
    return None


class ShopliftingAnalyzer:
    """Run the custom concealment pose model and aggregate scores per track."""

    def __init__(self, config: ShopliftingConfig, site_id: str) -> None:
        self.config = config
        self.site_id = site_id
        self._states: dict[tuple[str, int], _BehaviorState] = {}
        self.latest_observations: list[ShopliftingObservation] = []
        self._pose_model: Any = None
        self._frame_counter = 0
        if config.enabled and config.pose_model_path is not None:
            self._load_model()

    @property
    def classifier_loaded(self) -> bool:
        """True when the pose model is available (kept for compatibility)."""
        return self._pose_model is not None

    def _load_model(self) -> None:
        path = self.config.pose_model_path
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Shoplifting pose model not found: {path}")
        try:
            from ultralytics import YOLO  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for shoplifting pose inference") from exc
        self._pose_model = YOLO(str(path))

    def reset(self) -> None:
        self._states.clear()
        self.latest_observations.clear()
        self._frame_counter = 0

    # ------------------------------------------------------------------
    # Pose inference and scoring
    # ------------------------------------------------------------------
    def _infer_poses(self, frame: np.ndarray) -> list[tuple[Box, np.ndarray]]:
        """Run the pose model once and return (box, keypoints[K,3]) pairs."""
        results = self._pose_model.predict(
            frame,
            conf=self.config.pose_confidence,
            imgsz=self.config.pose_image_size,
            device=self.config.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        keypoints = result.keypoints.data.detach().cpu().numpy()
        count = min(len(boxes), len(keypoints))
        poses = []
        for index in range(count):
            kp = keypoints[index]
            if kp.ndim != 2 or kp.shape[0] <= _R_HIP or kp.shape[1] < 3:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[index])
            poses.append(((x1, y1, x2, y2), kp))
        return poses

    def _concealment_score(self, kp: np.ndarray) -> float:
        """Score in [0, 1]; 0.5 sits exactly on the concealment-ratio limit.

        Mirrors the notebook logic: a visible wrist closer to the torso core
        than ``concealment_ratio`` x body scale counts as concealment.
        """
        cfg = self.config
        torso = [kp[_L_SHOULDER], kp[_R_SHOULDER], kp[_L_HIP], kp[_R_HIP]]
        if any(float(point[2]) < cfg.torso_confidence for point in torso):
            return 0.0
        wrists = [
            kp[index]
            for index in (_L_WRIST, _R_WRIST)
            if float(kp[index][2]) >= cfg.wrist_confidence
        ]
        if not wrists:
            return 0.0

        xy = np.array([[float(p[0]), float(p[1])] for p in torso])
        center = xy.mean(axis=0)
        body_scale = float(
            (np.linalg.norm(xy[0] - xy[2]) + np.linalg.norm(xy[1] - xy[3])) / 2.0
        )
        if body_scale < 1e-6:
            return 0.0

        distance = min(
            float(np.linalg.norm(np.array([float(w[0]), float(w[1])]) - center))
            for w in wrists
        )
        limit = cfg.concealment_ratio * body_scale
        return float(np.clip(1.0 - distance / (2.0 * limit), 0.0, 1.0))

    def _match_poses(
        self,
        people: list[TrackedObject],
        poses: list[tuple[Box, np.ndarray]],
    ) -> dict[int, np.ndarray]:
        """Greedy IoU matching of tracked people to pose detections."""
        pairs = []
        for person in people:
            for pose_index, (pose_box, _) in enumerate(poses):
                overlap = _iou(person.bbox, pose_box)
                if overlap >= self.config.pose_match_iou:
                    pairs.append((overlap, person.track_id, pose_index))
        pairs.sort(reverse=True)
        matched: dict[int, np.ndarray] = {}
        used_poses: set[int] = set()
        for _, track_id, pose_index in pairs:
            if track_id in matched or pose_index in used_poses:
                continue
            matched[track_id] = poses[pose_index][1]
            used_poses.add(pose_index)
        return matched

    # ------------------------------------------------------------------
    # Evaluation entry point
    # ------------------------------------------------------------------
    def evaluate(
        self,
        camera_id: str,
        tracked_objects: list[TrackedObject],
        timestamp: datetime,
        active_track_ids: set[int] | None = None,
        frame: np.ndarray | None = None,
    ) -> list[VisionEvent]:
        if not self.config.enabled:
            return []
        now = timestamp.timestamp()
        if active_track_ids is not None:
            for key in list(self._states):
                if key[0] == camera_id and key[1] not in active_track_ids:
                    self._states.pop(key, None)
        if self.classifier_loaded and frame is not None:
            return self._evaluate_pose(camera_id, tracked_objects, timestamp, now, frame)
        self.latest_observations = []
        return self._evaluate_fallback(camera_id, tracked_objects, timestamp, now)

    def _evaluate_pose(
        self,
        camera_id: str,
        tracked_objects: list[TrackedObject],
        timestamp: datetime,
        now: float,
        frame: np.ndarray,
    ) -> list[VisionEvent]:
        self._frame_counter += 1
        # Skip inference on off-stride frames; keep last overlay observations.
        if (self._frame_counter - 1) % self.config.pose_stride != 0:
            return []

        people = [obj for obj in tracked_objects if obj.class_name == "person"]
        self.latest_observations = []
        if not people:
            return []

        matched = self._match_poses(people, self._infer_poses(frame))
        cfg = self.config
        events: list[VisionEvent] = []

        for obj in people:
            state = self._states.setdefault((camera_id, obj.track_id), _BehaviorState())

            # Zone context is updated for every tracked person each frame.
            in_merchandise = _zone_at(obj.centroid, cfg.merchandise_zones)
            in_checkout = _zone_at(obj.centroid, cfg.checkout_zones) is not None
            if in_merchandise is not None:
                state.last_merchandise_at = now
                state.merchandise_zone = in_merchandise

            kp = matched.get(obj.track_id)
            score = self._concealment_score(kp) if kp is not None else None
            self.latest_observations.append(
                ShopliftingObservation(obj.track_id, obj.bbox, score, score is not None)
            )
            if score is None:
                continue

            state.predictions.append((now, score))
            cutoff = now - cfg.observation_window_seconds
            while state.predictions and state.predictions[0][0] < cutoff:
                state.predictions.popleft()
            suspicious = [
                value for _, value in state.predictions
                if value >= cfg.classification_threshold
            ]

            if len(suspicious) < cfg.minimum_suspicious_observations:
                continue
            if now - state.last_event_at < cfg.cooldown_seconds:
                continue
            # Context gates: customers at a till are paying, not stealing.
            if cfg.suppress_in_checkout and in_checkout:
                continue
            if cfg.require_merchandise_interaction and cfg.merchandise_zones:
                recent = (
                    state.last_merchandise_at is not None
                    and now - state.last_merchandise_at <= cfg.merchandise_lookback_seconds
                )
                if not recent:
                    continue

            state.last_event_at = now
            events.append(
                self._make_pose_event(
                    camera_id,
                    obj,
                    timestamp,
                    sum(suspicious) / len(suspicious),
                    len(suspicious),
                    state.merchandise_zone,
                )
            )
        return events

    def _make_pose_event(
        self,
        camera_id: str,
        obj: TrackedObject,
        timestamp: datetime,
        score: float,
        observations: int,
        zone: str | None,
    ) -> VisionEvent:
        x1, y1, x2, y2 = obj.bbox
        metadata: dict[str, Any] = {
            "class_name": obj.class_name,
            "severity": "high",
            "concealment_score": round(score, 4),
            "suspicious_observations": observations,
            "behavior": "sustained_concealment_pose",
            "requires_human_review": True,
        }
        if zone is not None:
            metadata["zone"] = zone
        return VisionEvent(
            camera_id=camera_id,
            site_id=self.site_id,
            event_type=EventType.SUSPECTED_SHOPLIFTING,
            track_id=obj.track_id,
            confidence=min(1.0, max(0.0, score)),
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            timestamp=timestamp,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Fallback: zone interaction + exit-line heuristic (no pose model)
    # ------------------------------------------------------------------
    def _evaluate_fallback(
        self,
        camera_id: str,
        tracked_objects: list[TrackedObject],
        timestamp: datetime,
        now: float,
    ) -> list[VisionEvent]:
        events = []
        for obj in tracked_objects:
            if obj.class_name != "person":
                continue
            state = self._states.setdefault((camera_id, obj.track_id), _BehaviorState())
            point = obj.centroid
            merchandise_zone = _zone_at(point, self.config.merchandise_zones)
            checkout_zone = _zone_at(point, self.config.checkout_zones) is not None
            if merchandise_zone is not None:
                if state.merchandise_started_at is None:
                    state.merchandise_started_at = now
                    state.merchandise_zone = merchandise_zone
                state.merchandise_seconds = now - state.merchandise_started_at
            elif checkout_zone and state.merchandise_seconds > 0:
                state.checkout_seen = True
            crossed_direction = None
            if state.previous_point is not None and self.config.exit_line is not None:
                line = type("ExitLine", (), {"points": self.config.exit_line})()
                crossed_direction = get_line_crossing_direction(
                    state.previous_point, point, line
                )
            state.previous_point = point
            if (
                crossed_direction is None
                or (
                    self.config.allowed_direction != "either"
                    and crossed_direction != self.config.allowed_direction
                )
                or state.merchandise_seconds < self.config.minimum_merchandise_dwell_seconds
                or (self.config.require_checkout_zone and not state.checkout_seen)
                or state.checkout_seen
                or now - state.last_event_at < self.config.cooldown_seconds
            ):
                continue
            state.last_event_at = now
            x1, y1, x2, y2 = obj.bbox
            events.append(
                VisionEvent(
                    camera_id=camera_id,
                    site_id=self.site_id,
                    event_type=EventType.SUSPECTED_SHOPLIFTING,
                    track_id=obj.track_id,
                    confidence=obj.confidence,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    timestamp=timestamp,
                    metadata={
                        "class_name": obj.class_name,
                        "severity": "high",
                        "zone": state.merchandise_zone,
                        "behavior": "merchandise_interaction_then_exit",
                        "requires_human_review": True,
                    },
                )
            )
        return events