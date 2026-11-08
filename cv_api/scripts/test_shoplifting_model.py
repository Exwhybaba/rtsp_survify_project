"""Reproduce the notebook's frame-level shoplifting classifier inference.

This script intentionally keeps the detector and pose weights separate from
the classifier checkpoint. It refuses to substitute another YOLO model when
the requested weights are unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn


class ShopliftingNet(nn.Module):
    """The exact classifier architecture used by the training project."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_classifier(checkpoint_path: Path, device: torch.device) -> ShopliftingNet:
    """Load the exact classifier with safe PyTorch checkpoint loading."""
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch version does not support safe weights_only loading."
        ) from exc
    if not isinstance(state_dict, dict):
        raise RuntimeError("Classifier checkpoint is not a state dictionary.")

    model = ShopliftingNet().to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Classifier checkpoint does not match the verified ShopliftingNet architecture."
        ) from exc
    model.eval()
    return model


def extract_pose_features(pose_result: Any) -> np.ndarray | None:
    """Return 34 flattened xyn coordinates from one pose result."""
    keypoints = getattr(pose_result, "keypoints", None)
    if keypoints is None or keypoints.xyn is None or len(keypoints.xyn) == 0:
        return None
    coordinates = keypoints.xyn[0].detach().cpu().numpy()
    if coordinates.shape != (17, 2) or not np.isfinite(coordinates).all():
        return None
    return coordinates.astype(np.float32, copy=False).reshape(34)


def classify_features(
    classifier: ShopliftingNet,
    features: np.ndarray,
    device: torch.device,
) -> float:
    """Return the sigmoid shoplifting probability for one feature vector."""
    if features.shape != (34,):
        raise ValueError(f"Expected 34 features, got {features.shape}")
    tensor = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = classifier(tensor.unsqueeze(0))
        return float(torch.sigmoid(output).item())


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}. "
            "Provide the original compatible weight with the command-line option; "
            "this script will not substitute or download weights."
        )


def run(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "The ultralytics package is required for standalone inference."
        ) from exc

    video_path = Path(args.video)
    checkpoint_path = Path(args.checkpoint)
    detector_path = Path(args.detector_model)
    pose_path = Path(args.pose_model)
    for path, description in (
        (video_path, "Video"),
        (checkpoint_path, "Classifier checkpoint"),
        (detector_path, "Person detector model"),
        (pose_path, "Pose model"),
    ):
        _require_file(path, description)

    device = torch.device(args.device)
    classifier = load_classifier(checkpoint_path, device)
    detector = YOLO(str(detector_path))
    pose_model = YOLO(str(pose_path))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create annotated output: {args.output}")

    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            detections = detector.predict(
                frame,
                classes=[0],
                conf=args.person_confidence,
                device=args.device,
                verbose=False,
            )
            boxes = detections[0].boxes if detections else None
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = (
                        int(value)
                        for value in box.xyxy[0].detach().cpu().tolist()
                    )
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    pose_results = pose_model.predict(
                        crop,
                        conf=args.pose_confidence,
                        device=args.device,
                        verbose=False,
                    )
                    if pose_results:
                        pose_keypoints = pose_results[0].keypoints
                        if (
                            pose_keypoints is not None
                            and pose_keypoints.xy is not None
                            and len(pose_keypoints.xy) > 0
                        ):
                            for point in pose_keypoints.xy[0].detach().cpu().tolist():
                                point_x = x1 + int(point[0])
                                point_y = y1 + int(point[1])
                                cv2.circle(
                                    frame,
                                    (point_x, point_y),
                                    3,
                                    (255, 0, 0),
                                    -1,
                                )
                    features = (
                        extract_pose_features(pose_results[0])
                        if pose_results
                        else None
                    )
                    probability = (
                        classify_features(classifier, features, device)
                        if features is not None
                        else None
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = (
                        "pose unavailable"
                        if probability is None
                        else (
                            f"{'shoplifting' if probability >= args.threshold else 'normal'} "
                            f"{probability:.3f}"
                        )
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            writer.write(frame)
            frame_number += 1
            if args.max_frames is not None and frame_number >= args.max_frames:
                break
    finally:
        capture.release()
        writer.release()

    print(f"Annotated output: {args.output}")
    print(f"Processed frames: {frame_number}")
    print(f"Video geometry: {width}x{height} at {fps:.3f} FPS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/shoplifting/shoplifting_model.pth"),
    )
    parser.add_argument("--detector-model", type=Path, default=Path("yolo11x.pt"))
    parser.add_argument("--pose-model", type=Path, default=Path("yolo11x-pose.pt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/shoplifting_model_test.mp4"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--pose-confidence", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    run(arguments)
