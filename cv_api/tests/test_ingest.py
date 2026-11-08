"""Tests for source URI handling and stream buffering defaults."""

from __future__ import annotations

import pytest

from src.analytics.ingest import VideoSource, redact_uri


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("rtsp://user:secret@example.invalid:554/live", "rtsp://***:***@example.invalid:554/live"),
        ("rtsps://user:secret@example.invalid/live", "rtsps://***:***@example.invalid/live"),
        ("demo-media/video.mp4", "demo-media/video.mp4"),
    ],
)
def test_redact_uri_masks_credentials(uri: str, expected: str) -> None:
    assert redact_uri(uri) == expected


def test_latest_frame_only_defaults_to_streams_and_not_files() -> None:
    assert VideoSource("rtsp://camera/live").latest_frame_only is True
    assert VideoSource("demo-media/video.mp4").latest_frame_only is False


def test_latest_frame_only_can_be_overridden() -> None:
    assert VideoSource("rtsp://camera/live", latest_frame_only=False).latest_frame_only is False
    assert VideoSource("demo-media/video.mp4", latest_frame_only=True).latest_frame_only is True


def test_open_error_redacts_source_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    class ClosedCapture:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    monkeypatch.setattr("src.analytics.ingest.cv2.VideoCapture", lambda _uri: ClosedCapture())

    with pytest.raises(ConnectionError, match=r"rtsp://\*\*\*:\*\*\*@camera"):
        VideoSource("rtsp://user:secret@camera/live").open()
