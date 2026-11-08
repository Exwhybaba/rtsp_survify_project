"""Visualize YOLO person detections with ByteTrack IDs on a video file."""

import argparse
import sys
from pathlib import Path

# Make the project package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Pylint cannot inspect members exposed by OpenCV's compiled extension module.
# pylint: disable=no-member,wrong-import-position,catching-non-exception
import cv2
from ultralytics import YOLO

from src.analytics.tracker import ByteTrackAdapter, Detection  # pylint: disable=import-error

DEFAULT_VIDEO_PATH = "demo-media/crosswalk.avi"
DEFAULT_MODEL_PATH = "yolov8n.pt"


def visualize(
    video_path: Path,
    model_path: Path,
    output_path: Path | None,
    no_display: bool,
) -> None:
    """Track people in a video and optionally display or save the result."""
    model = YOLO(str(model_path))
    tracker = ByteTrackAdapter()
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {output_path}")

    display_enabled = not no_display
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model.predict(
                frame,
                classes=[0],
                conf=0.25,
                verbose=False,
            )[0]
            detections = [
                Detection(
                    bbox=tuple(float(value) for value in box.xyxy[0].cpu().numpy()),
                    confidence=float(box.conf[0]),
                    class_name="person",
                )
                for box in result.boxes
            ]

            for track in tracker.update(detections):
                x1, y1, x2, y2 = map(int, track.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Person ID: {track.track_id}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            if writer is not None:
                writer.write(frame)
            if display_enabled:
                try:
                    cv2.imshow("Suvirfy - YOLO + ByteTrack", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    display_enabled = False
                    print(
                        "OpenCV has no GUI support; continuing and saving the annotated video."
                    )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if output_path is not None:
        print(f"Annotated video saved to: {output_path.resolve()}")


def main() -> None:
    """Parse arguments and start tracking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLO model path.")
    parser.add_argument(
        "--output",
        help="Optional annotated output path. Omit to avoid saving a file.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a window; only save the annotated video.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output) if args.output else None
    visualize(video_path, Path(args.model), output_path, args.no_display)


if __name__ == "__main__":
    main()
