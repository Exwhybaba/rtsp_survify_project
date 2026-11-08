"""Benchmark a YOLO detector over a directory of video clips."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

from src.analytics.detector import YoloV8Detector


def main() -> None:
    """Run the benchmark and print aggregate day/night statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", type=Path, help="Folder containing video clips.")
    parser.add_argument("--model", required=True, help="Local model path or explicit model name.")
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    detector = YoloV8Detector(model_path=args.model, image_size=args.image_size)
    counts = {"day": Counter(), "night": Counter()}
    total_frames = 0
    total_seconds = 0.0
    total_detections = 0

    for path in sorted(args.clips.iterdir()):
        if not path.is_file():
            continue
        capture = cv2.VideoCapture(str(path))
        while True:
            success, frame = capture.read()
            if not success:
                break
            start = cv2.getTickCount()
            detections = detector.detect(frame)
            total_seconds += (cv2.getTickCount() - start) / cv2.getTickFrequency()
            period = "day" if float(frame.mean()) >= 80.0 else "night"
            counts[period].update(d.class_name for d in detections)
            total_detections += len(detections)
            total_frames += 1
        capture.release()

    fps = total_frames / total_seconds if total_seconds else 0.0
    print(f"frames={total_frames} fps={fps:.2f} detections={total_detections}")
    for period in ("day", "night"):
        print(f"{period}: {dict(sorted(counts[period].items()))}")


if __name__ == "__main__":
    main()
