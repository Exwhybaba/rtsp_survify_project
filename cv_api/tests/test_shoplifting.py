from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import torch

from src.analytics.shoplifting import (
    ShopliftingAnalyzer,
    ShopliftingConfig,
)
from src.analytics.tracker import TrackedObject


def tracked(track_id: int, x: float) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(x, 100, x + 40, 220),
        confidence=0.9,
        class_name="person",
        centroid=(x + 20, 160),
        age=1,
        hits=1,
    )


def test_emits_suspected_shoplifting_after_merchandise_exit() -> None:
    config = ShopliftingConfig.from_config(
        {
            "enabled": True,
            "merchandise_zones": [
                {
                    "name": "shelf_area",
                    "polygon": [[0, 0], [400, 0], [400, 300], [0, 300]],
                }
            ],
            "exit_line": {"points": [[500, 0], [500, 400]]},
            "minimum_merchandise_dwell_seconds": 2,
            "cooldown_seconds": 30,
        }
    )
    analyzer = ShopliftingAnalyzer(config, "default")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert analyzer.evaluate("cam1", [tracked(7, 100)], start) == []
    assert analyzer.evaluate(
        "cam1",
        [tracked(7, 100)],
        start + timedelta(seconds=2),
    ) == []

    events = analyzer.evaluate(
        "cam1",
        [tracked(7, 520)],
        start + timedelta(seconds=3),
    )

    assert len(events) == 1
    assert events[0].event_type.value == "suspected_shoplifting"
    assert events[0].track_id == 7
    assert events[0].metadata["behavior"] == "merchandise_interaction_then_exit"


def test_checkout_observation_suppresses_suspicion() -> None:
    config = ShopliftingConfig.from_config(
        {
            "enabled": True,
            "merchandise_zones": [
                {
                    "name": "shelf_area",
                    "polygon": [[0, 0], [300, 0], [300, 300], [0, 300]],
                }
            ],
            "checkout_zones": [
                {
                    "name": "checkout",
                    "polygon": [[300, 0], [450, 0], [450, 300], [300, 300]],
                }
            ],
            "exit_line": {"points": [[500, 0], [500, 400]]},
            "minimum_merchandise_dwell_seconds": 1,
        }
    )
    analyzer = ShopliftingAnalyzer(config, "default")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    analyzer.evaluate("cam1", [tracked(3, 100)], start)
    analyzer.evaluate(
        "cam1",
        [tracked(3, 100)],
        start + timedelta(seconds=2),
    )
    assert analyzer.evaluate(
        "cam1",
        [tracked(3, 320)],
        start + timedelta(seconds=3),
    ) == []
    assert analyzer.evaluate(
        "cam1",
        [tracked(3, 520)],
        start + timedelta(seconds=4),
    ) == []


def _pose_result(wrist_x: float = 30.0):
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[5] = (20, 100, 0.9)
    keypoints[6] = (40, 100, 0.9)
    keypoints[11] = (20, 200, 0.9)
    keypoints[12] = (40, 200, 0.9)
    keypoints[9] = (wrist_x, 150, 0.9)
    keypoints[10] = (wrist_x, 150, 0.9)

    class Boxes:
        xyxy = torch.tensor([[10.0, 100.0, 50.0, 220.0]])

        def __len__(self):
            return 1

    return SimpleNamespace(
        boxes=Boxes(),
        keypoints=SimpleNamespace(data=torch.from_numpy(keypoints[None, ...])),
    )


class FakePoseModel:
    def __init__(self, results):
        self.results = list(results)
        self._last = self.results[-1]

    def predict(self, *_args, **_kwargs):
        result = self.results.pop(0) if self.results else self._last
        return [result]


def test_concealment_score_tracks_wrist_distance_from_torso() -> None:
    analyzer = ShopliftingAnalyzer(ShopliftingConfig(enabled=True), "default")
    keypoints = _pose_result().keypoints.data[0].numpy()
    assert analyzer._concealment_score(keypoints) == 1.0

    keypoints[9, 0] = 75
    keypoints[10, 0] = 75
    assert analyzer._concealment_score(keypoints) == 0.5


def test_missing_pose_is_not_classified() -> None:
    config = ShopliftingConfig.from_config(
        {
            "enabled": True,
            "minimum_suspicious_observations": 1,
        }
    )
    analyzer = ShopliftingAnalyzer(config, "default")
    class EmptyBoxes:
        xyxy = torch.empty((0, 4))

        def __len__(self):
            return 0

    analyzer._pose_model = FakePoseModel(
        [SimpleNamespace(boxes=EmptyBoxes(), keypoints=SimpleNamespace(data=torch.empty((0, 17, 3))))]
    )
    events = analyzer.evaluate(
        "cam1",
        [tracked(1, 10)],
        datetime.now(timezone.utc),
        frame=np.zeros((300, 300, 3), dtype=np.uint8),
    )
    assert events == []
    assert analyzer.latest_observations[0].pose_available is False


def test_temporal_aggregation_is_per_track_and_requires_sustained_predictions() -> None:
    config = ShopliftingConfig.from_config(
        {
            "enabled": True,
            "classification_threshold": 0.5,
            "observation_window_seconds": 5,
            "minimum_suspicious_observations": 3,
            "cooldown_seconds": 30,
        }
    )
    analyzer = ShopliftingAnalyzer(config, "default")
    analyzer._pose_model = FakePoseModel([_pose_result(), _pose_result(), _pose_result()])
    frame = np.zeros((300, 700, 3), dtype=np.uint8)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert analyzer.evaluate("cam1", [tracked(1, 10)], start, frame=frame) == []
    assert analyzer.evaluate(
        "cam1",
        [tracked(1, 10)],
        start + timedelta(seconds=1),
        frame=frame,
    ) == []
    events = analyzer.evaluate(
        "cam1",
        [tracked(1, 10)],
        start + timedelta(seconds=2),
        frame=frame,
    )
    assert len(events) == 1
    assert events[0].metadata["concealment_score"] == 1.0

    assert analyzer.evaluate(
        "cam1",
        [tracked(1, 10)],
        start + timedelta(seconds=3),
        frame=frame,
    ) == []
    analyzer.evaluate(
        "cam1",
        [],
        start + timedelta(seconds=4),
        active_track_ids=set(),
        frame=frame,
    )
    assert ("cam1", 1) not in analyzer._states
