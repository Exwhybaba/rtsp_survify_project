"""Unit tests for the ByteTrack adapter used by the analytics pipeline."""

from __future__ import annotations

from src.analytics.tracker import ByteTrackAdapter, Detection


def box_at(x: float, y: float, w: float = 30, h: float = 60) -> tuple[float, float, float, float]:
    """Build a bounding box at the given origin."""
    return (x, y, x + w, y + h)


def detection_at(
    x: float,
    y: float = 100,
    class_name: str = "person",
    confidence: float = 0.9,
) -> Detection:
    """Build a detection for a synthetic tracked object."""
    return Detection(
        bbox=box_at(x, y),
        confidence=confidence,
        class_name=class_name,
    )


def mature_tracker() -> ByteTrackAdapter:
    """Create a tracker configured to activate after one consecutive frame."""
    return ByteTrackAdapter(minimum_consecutive_frames=1)


def test_detection_centroid_is_calculated_from_bbox() -> None:
    detection = detection_at(10, 20)
    assert detection.centroid == (25.0, 50.0)


def test_track_id_persists_for_smooth_motion() -> None:
    tracker = mature_tracker()

    tracker.update([detection_at(10)])
    first = tracker.update([detection_at(15)])[0]
    second = tracker.update([detection_at(20)])[0]

    assert first.track_id == second.track_id
    assert second.hits == 2
    assert second.age == 2


def test_different_objects_receive_distinct_ids() -> None:
    tracker = mature_tracker()

    tracker.update([detection_at(10)])
    tracker.update([detection_at(15), detection_at(400, class_name="car")])
    tracked = tracker.update([detection_at(20), detection_at(400, class_name="car")])

    assert len(tracked) == 2
    by_class = {obj.class_name: obj for obj in tracked}
    assert by_class["person"].track_id != by_class["car"].track_id


def test_track_remains_active_during_lost_track_buffer() -> None:
    tracker = ByteTrackAdapter(
        minimum_consecutive_frames=1,
        lost_track_buffer=2,
    )

    tracker.update([detection_at(10)])
    tracked = tracker.update([detection_at(15)])[0]
    track_id = tracked.track_id

    tracker.update([])
    assert track_id in tracker.active_track_ids


def test_reset_clears_tracking_state() -> None:
    tracker = mature_tracker()
    tracker.update([detection_at(10)])
    tracker.update([detection_at(15)])

    tracker.reset()

    assert tracker.active_track_ids == set()
    assert tracker.update([]) == []
