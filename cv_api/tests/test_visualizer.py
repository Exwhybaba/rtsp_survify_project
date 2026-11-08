"""Tests for frames published from the active analytics visualizer."""

from types import SimpleNamespace

import cv2
import numpy as np

from src.analytics.visualizer import Visualizer


def test_visualizer_publishes_its_annotated_frame_for_live_monitoring(tmp_path) -> None:
    frame_path = tmp_path / "live.jpg"
    visualizer = Visualizer(
        {
            "enabled": False,
            "live_frame_path": str(frame_path),
            "live_frame_fps": 8,
        }
    )
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    tracked = SimpleNamespace(
        bbox=(10, 10, 70, 70),
        track_id=4,
        class_name="person",
        confidence=0.91,
        centroid=(40, 40),
    )
    occupancy = SimpleNamespace(count=2, uncertain=False)

    visualizer.show(frame, [tracked], [], [], occupancy, [])

    published = cv2.imread(str(frame_path))
    assert published is not None
    assert published.shape == frame.shape
    assert not np.array_equal(published, frame)
    visualizer.close()
