#!/usr/bin/env python
"""Generate a short synthetic demo video for the analytics pipeline.

No real camera footage is needed (or wanted) for a portfolio/reference
repo. This script draws a handful of solid-colored rectangles ("simulated
people") moving around a plain background over a fixed number of frames
and writes them out as an .mp4 using OpenCV, along with a faint overlay
of the restricted zone / tripwire line from config/pipeline.yaml so the
video is visually self-explanatory when you watch it.

Usage
-----
    python scripts/generate_demo_video.py
    python scripts/generate_demo_video.py --output /path/to/demo.mp4 --seconds 10

The default output path is a local, git-ignored `demo-media/demo.mp4`
(matching `camera.source` in config/pipeline.yaml). Point `--output`
anywhere else -- e.g. outside the repo entirely -- to generate a copy for
manual review without ever committing video binaries to source control.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# Pylint cannot inspect members exposed by OpenCV's compiled extension module.
# pylint: disable=no-member
import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
FPS = 25

# Mirrors config/pipeline.yaml so the generated video visually matches the
# zone/line the pipeline will evaluate against it.
RESTRICTED_ZONE = [(380, 60), (620, 60), (620, 300), (380, 300)]
ENTRANCE_LINE = ((0, 240), (640, 240))

BACKGROUND_COLOR = (32, 32, 30)  # BGR


@dataclass
class MovingActor:
    """A simple rectangle bouncing around the frame at constant velocity."""

    x: float
    y: float
    vx: float
    vy: float
    w: int
    h: int
    color: tuple[int, int, int]
    label: str

    def step(self) -> None:
        """Advance the actor and bounce it off the frame boundaries."""
        self.x += self.vx
        self.y += self.vy
        if self.x <= 0 or self.x + self.w >= WIDTH:
            self.vx *= -1
            self.x = max(0.0, min(self.x, WIDTH - self.w))
        if self.y <= 0 or self.y + self.h >= HEIGHT:
            self.vy *= -1
            self.y = max(0.0, min(self.y, HEIGHT - self.h))

    def bbox(self) -> tuple[int, int, int, int]:
        """Return the actor's current bounding box."""
        return int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h)


def make_actors() -> list[MovingActor]:
    """Create the actors used in the synthetic scene."""
    return [
        # Walks left -> right across the entrance line and straight into
        # the restricted zone.
        MovingActor(x=20, y=220, vx=3.2, vy=0.4, w=28, h=60, color=(60, 200, 255), label="P1"),
        # Wanders vertically, staying outside the restricted zone.
        MovingActor(x=120, y=40, vx=0.6, vy=2.6, w=26, h=56, color=(120, 255, 120), label="P2"),
        # Diagonal path that dips through the restricted zone and back out
        # -- exercises dwell time + zone entry/exit transitions.
        MovingActor(x=560, y=340, vx=-1.6, vy=-2.0, w=30, h=58, color=(255, 130, 90), label="P3"),
    ]


def draw_overlays(frame: np.ndarray) -> None:
    """Draw the restricted zone and entrance line on a video frame."""
    overlay = frame.copy()
    pts = np.array(RESTRICTED_ZONE, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color=(0, 0, 160))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, dst=frame)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 220), thickness=2)
    cv2.line(frame, ENTRANCE_LINE[0], ENTRANCE_LINE[1], color=(220, 220, 40), thickness=2)
    cv2.putText(
        frame, "RESTRICTED", (385, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA
    )


def generate(output_path: Path, seconds: float) -> Path:
    """Generate a synthetic video and return its output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(seconds * FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    actors = make_actors()
    try:
        for frame_idx in range(total_frames):
            frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)
            draw_overlays(frame)

            for actor in actors:
                actor.step()
                x1, y1, x2, y2 = actor.bbox()
                cv2.rectangle(frame, (x1, y1), (x2, y2), actor.color, thickness=-1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)
                cv2.putText(
                    frame, actor.label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, actor.color, 1, cv2.LINE_AA,
                )

            cv2.putText(
                frame, f"frame {frame_idx:04d}", (10, HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def main() -> None:
    """Parse command-line arguments and generate the demo video."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="demo-media/demo.mp4",
        help="Where to write the generated .mp4 (default: demo-media/demo.mp4, git-ignored).",
    )
    parser.add_argument("--seconds", type=float, default=10.0, help="Video duration in seconds.")
    args = parser.parse_args()

    output_path = Path(args.output)
    result = generate(output_path, args.seconds)
    size_kb = result.stat().st_size / 1024
    print(
        f"Wrote demo video: {result.resolve()} "
        f"({size_kb:.1f} KB, {args.seconds:.0f}s @ {FPS}fps)"
    )


if __name__ == "__main__":
    main()
