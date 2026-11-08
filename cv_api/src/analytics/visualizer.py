
"""OpenCV visualisation for the Suvirfy computer vision pipeline."""

from __future__ import annotations

from typing import Any

# Pylint cannot inspect members exposed by OpenCV's compiled extension module.
# pylint: disable=no-member
import cv2
import numpy as np

from src.analytics.tracker import TrackedObject


class Visualizer:
    """Draw detections, tracks, zones, lines and events on video frames."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}

        self.enabled = bool(config.get("enabled", False))
        self.show_detections = bool(
            config.get("show_detections", True)
        )
        self.show_track_ids = bool(
            config.get("show_track_ids", True)
        )
        self.show_zones = bool(config.get("show_zones", True))
        self.show_lines = bool(config.get("show_lines", True))
        self.show_occupancy = bool(
            config.get("show_occupancy", True)
        )
        self.show_events = bool(config.get("show_events", True))

        self.window_name = config.get(
            "window_name",
            "Suvirfy Live Monitoring",
        )
        self.output_path = config.get("output_path")
        self._writer: cv2.VideoWriter | None = None
        self._output_fps = float(config.get("output_fps", 10.0))

        self._closed = False

    def show(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        zones: list[Any],
        lines: list[Any],
        occupancy: Any,
        events: list[Any],
        shoplifting_observations: list[Any] | None = None,
    ) -> bool:
        """Render one annotated frame.

        Returns False when the operator presses Q or ESC.
        """
        if self._closed:
            return True

        display = frame.copy()

        if self.show_zones:
            self._draw_zones(display, zones)

        if self.show_lines:
            self._draw_lines(display, lines)

        if self.show_detections:
            self._draw_tracks(display, tracked_objects)

        self._draw_shoplifting_observations(
            display,
            shoplifting_observations or [],
        )

        if self.show_occupancy:
            self._draw_occupancy(display, occupancy)

        if self.show_events:
            self._draw_events(display, events)

        if self.output_path:
            if self._writer is None:
                height, width = display.shape[:2]
                self._writer = cv2.VideoWriter(
                    str(self.output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self._output_fps,
                    (width, height),
                )
                if not self._writer.isOpened():
                    raise RuntimeError(
                        f"could not open annotated video: {self.output_path}"
                    )
            self._writer.write(display)

        if not self.enabled:
            return True

        cv2.imshow(self.window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            return False

        return True

    def _draw_shoplifting_observations(
        self,
        frame: np.ndarray,
        observations: list[Any],
    ) -> None:
        """Display classifier probabilities without implying confirmed theft."""
        for observation in observations:
            x1, y1, _, _ = (
                int(value) for value in observation.bbox
            )
            if observation.probability is None:
                label = f"Track #{observation.track_id} | pose unavailable"
                color = (0, 165, 255)
            else:
                label = (
                    f"Track #{observation.track_id} | "
                    f"shoplifting probability {observation.probability:.2f}"
                )
                color = (0, 0, 255) if observation.probability >= 0.5 else (0, 255, 0)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_zones(
        self,
        frame: np.ndarray,
        zones: list[Any],
    ) -> None:
        """Draw configured polygon zones."""
        for zone in zones:
            points = np.array(
                zone.polygon,
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            cv2.polylines(
                frame,
                [points],
                isClosed=True,
                color=(0, 165, 255),
                thickness=2,
            )

            overlay = frame.copy()

            cv2.fillPoly(
                overlay,
                [points],
                color=(0, 165, 255),
            )

            cv2.addWeighted(
                overlay,
                0.12,
                frame,
                0.88,
                0,
                frame,
            )

            label_x = int(zone.polygon[0][0])
            label_y = max(20, int(zone.polygon[0][1]) - 10)

            cv2.putText(
                frame,
                zone.name,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_lines(
        self,
        frame: np.ndarray,
        lines: list[Any],
    ) -> None:
        """Draw configured virtual lines."""
        for line in lines:
            start = tuple(int(value) for value in line.points[0])
            end = tuple(int(value) for value in line.points[1])

            cv2.line(
                frame,
                start,
                end,
                color=(255, 255, 0),
                thickness=2,
            )

            label_x = start[0]
            label_y = max(20, start[1] - 10)

            cv2.putText(
                frame,
                line.name,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def _draw_tracks(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
    ) -> None:
        """Draw bounding boxes and tracking IDs."""
        for obj in tracked_objects:
            x1, y1, x2, y2 = (
                int(value) for value in obj.bbox
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color=(0, 255, 0),
                thickness=2,
            )

            if self.show_track_ids:
                label = (
                    f"{obj.class_name} #{obj.track_id} "
                    f"{obj.confidence:.2f}"
                )
            else:
                label = f"{obj.class_name} {obj.confidence:.2f}"

            label_y = max(20, y1 - 10)

            cv2.putText(
                frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cx, cy = (
                int(value) for value in obj.centroid
            )

            cv2.circle(
                frame,
                (cx, cy),
                4,
                (0, 0, 255),
                -1,
            )

    def _draw_occupancy(
        self,
        frame: np.ndarray,
        occupancy: Any,
    ) -> None:
        """Display occupancy and its uncertainty status."""
        count = occupancy.count

        if occupancy.uncertain:
            status = f"OCCUPANCY: {count} (UNCERTAIN)"
            color = (0, 165, 255)
        else:
            status = f"OCCUPANCY: {count}"
            color = (0, 255, 0)

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_events(
        self,
        frame: np.ndarray,
        events: list[Any],
    ) -> None:
        """Display events generated on the current frame."""
        if not events:
            return

        y = 70

        for event in events[-4:]:
            event_name = event.event_type.value
            label = f"ALERT: {event_name} | Track #{event.track_id}"

            cv2.putText(
                frame,
                label,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            y += 25

    def close(self) -> None:
        """Close the OpenCV monitoring window."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.enabled and not self._closed:
            cv2.destroyAllWindows()
        self._closed = True
