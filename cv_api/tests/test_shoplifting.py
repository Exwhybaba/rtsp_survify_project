from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from src.analytics.shoplifting import (
    ShopliftingAnalyzer,
    ShopliftingConfig,
    ShopliftingNet,
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


def test_exact_checkpoint_matches_shoplifting_architecture() -> None:
    checkpoint = Path("models/shoplifting/shoplifting_model.pth")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ShopliftingNet()
    model.load_state_dict(state_dict)
    model.eval()
    assert tuple(model(torch.zeros((1, 34))).shape) == (1, 1)


def test_pose_features_preserve_coco_xy_order() -> None:
    class Keypoints:
        xyn = torch.arange(34, dtype=torch.float32).reshape(1, 17, 2) / 34
        conf = None

    class Result:
        keypoints = Keypoints()

    features = ShopliftingAnalyzer._pose_features(Result(), 0.0)
    assert features is not None
    np.testing.assert_allclose(features, Keypoints.xyn[0].numpy().reshape(34))


def test_missing_pose_is_not_classified() -> None:
    config = ShopliftingConfig.from_config(
        {
            "enabled": True,
            "minimum_suspicious_observations": 1,
        }
    )
    analyzer = ShopliftingAnalyzer(config, "default")
    analyzer._classifier = object()
    analyzer._pose_model = object()
    analyzer._classify = lambda _crop: None
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
    analyzer._classifier = object()
    analyzer._pose_model = object()
    values = iter([0.9, 0.9, 0.9, 0.9])
    analyzer._classify = lambda _crop: next(values)
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
    assert events[0].metadata["shoplifting_probability"] == 0.9

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
