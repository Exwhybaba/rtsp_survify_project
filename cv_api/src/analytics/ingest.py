
"""Video source abstraction.

Supports local video files, webcams, USB cameras, and network streams.
The rest of the pipeline consumes FrameBundle objects without needing
to know which source produced them.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

# Pylint cannot inspect members exposed by OpenCV's compiled extension module.
# pylint: disable=no-member
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_URI_CREDENTIALS = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@",
    re.IGNORECASE,
)


def redact_uri(uri: str | int) -> str:
    """Mask credentials embedded in a source URI."""
    if isinstance(uri, int):
        return f"camera:{uri}"

    return _URI_CREDENTIALS.sub(
        r"\g<scheme>***:***@",
        uri,
    )


@dataclass
class FrameBundle:
    """One frame plus bookkeeping information."""

    frame: np.ndarray
    frame_index: int
    timestamp: float


class VideoSource:
    """Capture frames from a file, webcam, USB camera, or network stream.

    Examples:

        # Laptop webcam
        with VideoSource(0) as src:
            for bundle in src.frames():
                ...

        # External USB camera
        with VideoSource(1) as src:
            for bundle in src.frames():
                ...

        # RTSP CCTV camera
        with VideoSource(
            "rtsp://user:password@192.168.1.50:554/stream1"
        ) as src:
            for bundle in src.frames():
                ...

        # Local video
        with VideoSource("demo-media/demo.mp4", loop=True) as src:
            for bundle in src.frames():
                ...

    Parameters
    ----------
    uri:
        Integer camera index or string containing an RTSP URL,
        HTTP stream URL, or local video file path.
    loop:
        Restart a local video file when it reaches the end.
    reconnect_attempts:
        Number of reconnect attempts for live sources.
    reconnect_delay_seconds:
        Delay between reconnect attempts.
    latest_frame_only:
        Process only the newest available frame for live sources.
    """

    def __init__(
        self,
        uri: str | int,
        loop: bool = False,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 1.0,
        latest_frame_only: bool | None = None,
    ) -> None:

        if not isinstance(uri, (str, int)):
            raise TypeError(
                "Video source must be a string path/URL or integer camera index."
            )

        if isinstance(uri, int) and uri < 0:
            raise ValueError("Camera index cannot be negative.")

        if isinstance(uri, str) and not uri.strip():
            raise ValueError("Video source cannot be empty.")

        self.uri = uri
        self.loop = loop
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds

        # Integer sources represent local camera devices.
        self._is_camera = isinstance(uri, int)

        # Network streams are identified by their URL scheme.
        self._is_stream = (
            isinstance(uri, str)
            and uri.strip().lower().startswith(
                ("rtsp://", "rtsps://", "http://", "https://")
            )
        )

        # Both webcams and network streams are live sources.
        self._is_live = self._is_camera or self._is_stream

        self.latest_frame_only = (
            self._is_live
            if latest_frame_only is None
            else latest_frame_only
        )

        self._cap: cv2.VideoCapture | None = None

        self._stop_event = threading.Event()
        self._frame_condition = threading.Condition()

        self._latest_bundle: FrameBundle | None = None
        self._reader_thread: threading.Thread | None = None

        self._reader_frame_index = 0
        self._last_yielded_index = -1

        self.dropped_frames = 0

    def _open_capture(self) -> cv2.VideoCapture:
        """Create an OpenCV capture for the configured source."""

        if self._is_camera:
            # DirectShow is generally a good backend for Windows webcams.
            if hasattr(cv2, "CAP_DSHOW"):
                cap = cv2.VideoCapture(self.uri, cv2.CAP_DSHOW)

                if cap.isOpened():
                    return cap

                cap.release()

            # Fallback to OpenCV's default backend.
            return cv2.VideoCapture(self.uri)

        return cv2.VideoCapture(self.uri)

    def open(self) -> VideoSource:
        """Open the configured video source."""

        if self._cap is not None and self._cap.isOpened():
            return self

        self._stop_event.clear()

        self._cap = self._open_capture()

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None

            raise ConnectionError(
                f"Unable to open video source: {redact_uri(self.uri)!r}"
            )

        logger.info(
            "Opened video source: %s",
            redact_uri(self.uri),
        )

        # Use a background reader for live sources.
        if self._is_live and self.latest_frame_only:

            self._reader_frame_index = 0
            self._latest_bundle = None

            self._reader_thread = threading.Thread(
                target=self._read_latest_frames,
                name="video-source-reader",
                daemon=True,
            )

            self._reader_thread.start()

        return self

    def close(self) -> None:
        """Stop reading and release the capture."""

        self._stop_event.set()

        with self._frame_condition:
            self._frame_condition.notify_all()

        if (
            self._reader_thread is not None
            and self._reader_thread is not threading.current_thread()
        ):
            self._reader_thread.join(timeout=2.0)

        self._reader_thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info(
            "Closed video source: %s",
            redact_uri(self.uri),
        )

    def __enter__(self) -> VideoSource:
        """Open source using a context manager."""
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        """Release source when leaving a context manager."""
        self.close()

    @property
    def fps(self) -> float:
        """Return source frame rate, defaulting to 25 FPS."""

        if self._cap is None:
            return 0.0

        fps = self._cap.get(cv2.CAP_PROP_FPS)

        return fps if fps and fps > 0 else 25.0

    def _reconnect(self) -> bool:
        """Retry opening a failed live source."""

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        for attempt in range(1, self.reconnect_attempts + 1):

            if self._stop_event.is_set():
                return False

            logger.warning(
                "Reconnecting to %s (attempt %d/%d)",
                redact_uri(self.uri),
                attempt,
                self.reconnect_attempts,
            )

            time.sleep(self.reconnect_delay_seconds)

            if self._stop_event.is_set():
                return False

            self._cap = self._open_capture()

            if self._cap.isOpened():
                logger.info(
                    "Successfully reconnected to %s",
                    redact_uri(self.uri),
                )
                return True

            self._cap.release()
            self._cap = None

        return False

    def _read_latest_frames(self) -> None:
        """Continuously read live frames, retaining only the newest."""

        while not self._stop_event.is_set():

            cap = self._cap

            if cap is None:
                return

            frame_read, frame = cap.read()

            if not frame_read:

                if self._is_camera:
                    logger.warning(
                        "Failed to read webcam frame from camera %s",
                        self.uri,
                    )

                    # Local cameras may temporarily fail to return frames.
                    time.sleep(0.05)
                    continue

                if self._reconnect():
                    continue

                logger.error(
                    "Giving up on network stream %s after repeated failures",
                    redact_uri(self.uri),
                )

                return

            bundle = FrameBundle(
                frame=frame,
                frame_index=self._reader_frame_index,
                timestamp=time.time(),
            )

            self._reader_frame_index += 1

            with self._frame_condition:

                if self._latest_bundle is not None:
                    self.dropped_frames += 1

                self._latest_bundle = bundle

                self._frame_condition.notify_all()

    def frames(self) -> Iterator[FrameBundle]:
        """Yield frames until the source ends or is stopped."""

        if self._cap is None:
            self.open()

        assert self._cap is not None

        # Live webcam and network stream processing.
        if self._is_live and self.latest_frame_only:

            while not self._stop_event.is_set():

                with self._frame_condition:

                    self._frame_condition.wait_for(
                        lambda: (
                            self._latest_bundle is not None
                            or self._stop_event.is_set()
                            or self._reader_thread is None
                            or not self._reader_thread.is_alive()
                        ),
                        timeout=1.0,
                    )

                    bundle = self._latest_bundle
                    self._latest_bundle = None

                if bundle is None:

                    if (
                        self._reader_thread is None
                        or not self._reader_thread.is_alive()
                    ):
                        return

                    continue

                self._last_yielded_index = bundle.frame_index

                yield bundle

            return

        # Local prerecorded video processing.
        frame_index = 0

        while not self._stop_event.is_set():

            cap = self._cap

            if cap is None:
                return

            frame_read, frame = cap.read()

            if not frame_read:

                if self._is_stream:

                    if self._reconnect():
                        continue

                    logger.error(
                        "Giving up on network stream %s after repeated failures",
                        redact_uri(self.uri),
                    )

                    return

                if self.loop:

                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    frame_read, frame = cap.read()

                    if not frame_read:
                        logger.error(
                            "Unable to read video after restarting: %s",
                            redact_uri(self.uri),
                        )
                        return

                    frame_index = 0

                else:
                    logger.info(
                        "Video file finished: %s",
                        redact_uri(self.uri),
                    )
                    return

            yield FrameBundle(
                frame=frame,
                frame_index=frame_index,
                timestamp=time.time(),
            )

            frame_index += 1
