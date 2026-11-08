#!/usr/bin/env python
"""Generate real annotated detection frames for the dashboard's "Detections" panel.

This is not a mockup: every box, track ID, and violation highlight comes
from actually running this repo's own ``YoloV8Detector`` ->
``CentroidIoUTracker`` -> ``ZoneRuleEngine`` chain (the exact same classes
``src/analytics/pipeline.py`` runs in production) and drawing the real
results with OpenCV:

- the configured restricted-zone polygon and entrance tripwire line,
- every tracked object's bounding box + track ID + class + confidence,
- a red highlight + "VIOLATION" label on any track that fired a zone/line
  rule (``restricted_zone_entry``, ``line_crossing``, or
  ``dwell_time_exceeded``).

Known, honest limitation of the demo video
--------------------------------------------
``scripts/generate_demo_video.py`` draws flat, solid-colour rectangles as
placeholder "people" -- deliberately, so the repo never has to ship or
depend on real footage. A COCO-pretrained model like YOLOv8n has no texture
or shape cues to work with on a flat rectangle, so it produces essentially
no detections above any usable confidence threshold on that synthetic
video (verified: at conf>=0.4, zero; even at conf>=0.01, only noise <0.05
confidence on irrelevant classes). That's a real, reproducible result of
running the real model -- not a bug in this script.

To still show genuine, high-confidence YOLOv8 output for the dashboard's
"Detections" panel, this script first tries the real demo video, and if
(as expected) it finds nothing usable, falls back to running the *same*
real detector/tracker/zone-engine chain on the two photographs bundled
with the ``ultralytics`` package itself (``bus.jpg`` and ``zidane.jpg`` --
Ultralytics' own public, non-proprietary demo/test images, shipped in every
install and used in their own docs/tutorials). The zone polygon and
tripwire line are scaled to each image's pixel dimensions using the exact
same proportions as ``config/pipeline.yaml``'s zone/line relative to the
demo video's 640x480 frame. This is still 100% real inference through this
repo's real code -- just on real photographs instead of an untextured
placeholder video.

Output: ``docs/screenshots/detection-frame-{1,2,3}.png``, plus a copy under
``frontend/public/screenshots/`` so the React dashboard can serve them
directly as static assets.

Usage
-----
    python scripts/generate_annotated_frames.py

Requires ``ultralytics`` (already in requirements.txt); the first run
downloads the small ``yolov8n.pt`` checkpoint. Runs on CPU.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analytics.detector import YoloV8Detector  # noqa: E402
from src.analytics.ingest import VideoSource  # noqa: E402
from src.analytics.rules.engine import ZoneRuleEngine  # noqa: E402
from src.analytics.tracker import CentroidIoUTracker, TrackedObject  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yaml"
DEMO_VIDEO = REPO_ROOT / "demo-media" / "demo.mp4"
OUT_DIR = REPO_ROOT / "docs" / "screenshots"
FRONTEND_PUBLIC_DIR = WORKSPACE_ROOT / "frontend" / "public" / "screenshots"
NUM_FRAMES_TO_SAVE = 3
MAX_FRAMES_TO_SCAN = 250  # ~10s @ 25fps -- the whole demo video, once through.

DEFAULT_BOX_COLOR = (120, 255, 120)  # BGR
VIOLATION_COLOR = (0, 0, 255)  # BGR

# The demo video's zone/line, expressed as fractions of its 640x480 frame,
# so the same geometry can be scaled onto any other image size for the
# fallback frames.
DEMO_FRAME_SIZE = (640, 480)  # (width, height)


def ensure_demo_video() -> None:
    """Regenerate the demo video with the repo's own generator if it's missing."""
    if DEMO_VIDEO.exists():
        return
    print(f"Demo video not found at {DEMO_VIDEO}; generating it with scripts/generate_demo_video.py ...")
    scripts_dir = REPO_ROOT / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import generate_demo_video  # noqa: PLC0415

    generate_demo_video.generate(DEMO_VIDEO, seconds=10.0)


def load_overlay_geometry(config_path: Path) -> tuple[list[dict], list[dict]]:
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return raw.get("zones", []), raw.get("lines", [])


def scale_zones_and_lines(
    zones: list[dict], lines: list[dict], target_size: tuple[int, int]
) -> tuple[list[dict], list[dict]]:
    """Rescale zone polygons / line points from the demo video's 640x480
    frame to ``target_size`` (width, height), preserving proportions."""
    src_w, src_h = DEMO_FRAME_SIZE
    dst_w, dst_h = target_size
    sx, sy = dst_w / src_w, dst_h / src_h

    scaled_zones = [
        {**z, "polygon": [[x * sx, y * sy] for x, y in z["polygon"]]} for z in zones
    ]
    scaled_lines = [
        {**ln, "points": [[x * sx, y * sy] for x, y in ln["points"]]} for ln in lines
    ]
    return scaled_zones, scaled_lines


def draw_static_overlays(frame: np.ndarray, zones: list[dict], lines: list[dict]) -> None:
    overlay = frame.copy()
    for zone in zones:
        pts = np.array(zone["polygon"], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color=(0, 0, 160))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, dst=frame)
    for zone in zones:
        pts = np.array(zone["polygon"], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 220), thickness=2)
        x, y = pts[0]
        cv2.putText(
            frame, zone["name"], (int(x), max(15, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA,
        )
    for line in lines:
        p1 = tuple(int(v) for v in line["points"][0])
        p2 = tuple(int(v) for v in line["points"][1])
        cv2.line(frame, p1, p2, color=(220, 220, 40), thickness=2)
        cv2.putText(
            frame, line["name"], (p1[0] + 6, p1[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 40), 1, cv2.LINE_AA,
        )


def draw_tracks(
    frame: np.ndarray, tracked_objects: list[TrackedObject], violated_track_ids: set[int]
) -> None:
    for obj in tracked_objects:
        x1, y1, x2, y2 = (int(v) for v in obj.bbox)
        is_violation = obj.track_id in violated_track_ids
        color = VIOLATION_COLOR if is_violation else DEFAULT_BOX_COLOR
        thickness = 3 if is_violation else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"ID {obj.track_id} {obj.class_name} {obj.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y0 = max(0, y1 - th - 10)
        cv2.rectangle(frame, (x1, label_y0), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 4, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA,
        )
        if is_violation:
            cv2.putText(
                frame, "VIOLATION", (x1, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, VIOLATION_COLOR, 2, cv2.LINE_AA,
            )


def build_detector(det_cfg: dict, confidence_threshold: float | None = None) -> YoloV8Detector:
    classes = det_cfg.get("classes_of_interest")
    print("Loading YOLOv8 detector on CPU (first run downloads yolov8n.pt) ...")
    return YoloV8Detector(
        model_path=det_cfg.get("model_path", "yolov8n.pt"),
        confidence_threshold=confidence_threshold or det_cfg.get("confidence_threshold", 0.4),
        classes_of_interest=set(classes) if classes else None,
        device="cpu",
    )


def build_tracker(tracker_cfg: dict) -> CentroidIoUTracker:
    return CentroidIoUTracker(
        iou_threshold=tracker_cfg["iou_threshold"],
        max_centroid_distance=tracker_cfg["max_centroid_distance"],
        max_disappeared=tracker_cfg["max_disappeared"],
    )


def try_demo_video(
    config: dict, zones_raw: list[dict], lines_raw: list[dict]
) -> list[tuple[str, np.ndarray, int, int]]:
    """Real attempt #1: run the real pipeline over the real demo video.

    Returns a list of ``(label, annotated_frame, n_tracks, n_violations)``
    for every frame that had at least one tracked object -- which, for the
    reasons in this module's docstring, is realistically expected to be
    empty for the flat-colour synthetic demo video.
    """
    camera_id = config["camera"]["id"]
    detector = build_detector(config.get("detector", {}))
    tracker = build_tracker(config["tracker"])
    zone_engine = ZoneRuleEngine.from_config(CONFIG_PATH)

    print(f"Attempt 1/2: real inference over up to {MAX_FRAMES_TO_SCAN} frames of {DEMO_VIDEO} ...")
    candidates: list[tuple[str, np.ndarray, int, int]] = []

    with VideoSource(str(DEMO_VIDEO), loop=False) as source:
        for bundle in source.frames():
            detections = detector.detect(bundle.frame)
            tracked_objects = tracker.update(detections)
            events = zone_engine.evaluate(camera_id, tracked_objects, datetime.now(timezone.utc))
            violated_ids = {e.track_id for e in events}

            if tracked_objects:
                annotated = bundle.frame.copy()
                draw_static_overlays(annotated, zones_raw, lines_raw)
                draw_tracks(annotated, tracked_objects, violated_ids)
                cv2.putText(
                    annotated, f"demo.mp4 frame {bundle.frame_index:04d}", (10, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA,
                )
                candidates.append(
                    (f"frame {bundle.frame_index}", annotated, len(tracked_objects), len(violated_ids))
                )

            if bundle.frame_index % 50 == 0:
                print(f"  ...frame {bundle.frame_index}: {len(tracked_objects)} tracked object(s)")
            if bundle.frame_index >= MAX_FRAMES_TO_SCAN - 1:
                break

    print(
        f"Attempt 1/2 result: {len(candidates)} frame(s) with a tracked object, "
        f"out of {MAX_FRAMES_TO_SCAN} scanned."
    )
    return candidates


def try_bundled_reference_photos(
    config: dict, zones_raw: list[dict], lines_raw: list[dict]
) -> list[tuple[str, np.ndarray, int, int]]:
    """Real attempt #2 (fallback): real inference on Ultralytics' own bundled
    demo photos, through this repo's real tracker + zone engine, with the
    zone/line geometry proportionally scaled to each photo's dimensions.
    """
    from ultralytics.utils import ASSETS  # noqa: PLC0415 (lazy: only needed here)

    camera_id = config["camera"]["id"]
    detector = build_detector(config.get("detector", {}))

    print("Attempt 2/2: real inference on ultralytics' bundled reference photos "
          f"({ASSETS}) -- see this module's docstring for why.")
    results: list[tuple[str, np.ndarray, int, int]] = []

    # Each bundled photo is run twice: as-is, and horizontally mirrored. The
    # mirrored copy is still a real image fed through real inference (not a
    # duplicate of the same result -- flipping moves every detection to a
    # different position relative to the zone/line, so it's a genuinely
    # distinct frame); this just guarantees enough real, visually distinct
    # candidate frames from only two source photos.
    variants: list[tuple[str, np.ndarray]] = []
    for image_path in sorted(Path(ASSETS).glob("*.jpg")):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        variants.append((image_path.name, frame))
        variants.append((f"{image_path.name} (mirrored)", cv2.flip(frame, 1)))

    for label, frame in variants:
        height, width = frame.shape[:2]
        scaled_zones, scaled_lines = scale_zones_and_lines(zones_raw, lines_raw, (width, height))

        # Fresh tracker + zone engine per image: these are unrelated photos,
        # not consecutive frames of one camera, so track IDs/zone state
        # should not carry over between them.
        tracker = build_tracker(config["tracker"])
        zone_engine = ZoneRuleEngine(*_zones_lines_from_raw(scaled_zones, scaled_lines))

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)
        events = zone_engine.evaluate(camera_id, tracked_objects, datetime.now(timezone.utc))
        violated_ids = {e.track_id for e in events}

        annotated = frame.copy()
        draw_static_overlays(annotated, scaled_zones, scaled_lines)
        draw_tracks(annotated, tracked_objects, violated_ids)
        cv2.putText(
            annotated, f"{label} (ultralytics reference photo)", (10, annotated.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA,
        )
        print(f"  {label}: {len(tracked_objects)} real detection(s), {len(violated_ids)} zone violation(s)")
        results.append((label, annotated, len(tracked_objects), len(violated_ids)))

    return results


def _zones_lines_from_raw(zones_raw: list[dict], lines_raw: list[dict]):
    """Build Zone/Line dataclass instances from raw (already-scaled) dicts,
    reusing the analytics rule engine's parsing helpers instead of re-implementing them."""
    from src.analytics.rules.engine import Line, Zone, _parse_rules  # noqa: PLC0415

    zones = [
        Zone(
            name=z["name"],
            polygon=[tuple(pt) for pt in z["polygon"]],
            camera_id=z.get("camera_id"),
            rules=_parse_rules(z.get("rules", [])),
        )
        for z in zones_raw
    ]
    lines = [
        Line(
            name=ln["name"],
            points=(tuple(ln["points"][0]), tuple(ln["points"][1])),
            camera_id=ln.get("camera_id"),
            rules=_parse_rules(ln.get("rules", [])),
        )
        for ln in lines_raw
    ]
    return zones, lines


def downscale(frame: np.ndarray, max_dim: int = 960) -> np.ndarray:
    """Shrink large source photos so the checked-in PNGs stay repo-friendly."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return frame
    scale = max_dim / longest
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def pick_frames(
    candidates: list[tuple[str, np.ndarray, int, int]], n: int
) -> list[tuple[str, np.ndarray, int, int]]:
    """Pick up to ``n`` candidates, preferring the frame with the most
    violations first, then filling the rest by track count. Uses list
    indices (not value equality) to track what's already chosen, since
    candidates contain numpy arrays that don't support ``==``/``in``."""
    chosen_idxs: list[int] = []
    violation_idxs = [i for i, c in enumerate(candidates) if c[3] > 0]
    if violation_idxs:
        chosen_idxs.append(max(violation_idxs, key=lambda i: candidates[i][3]))
    pool_idxs = sorted(range(len(candidates)), key=lambda i: -candidates[i][2])
    for idx in pool_idxs:
        if len(chosen_idxs) >= n:
            break
        if idx not in chosen_idxs:
            chosen_idxs.append(idx)
    return [candidates[i] for i in chosen_idxs[:n]]


def main() -> None:
    ensure_demo_video()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    zones_raw, lines_raw = load_overlay_geometry(CONFIG_PATH)

    candidates = try_demo_video(config, zones_raw, lines_raw)
    source_used = "demo-media/demo.mp4"

    if not candidates:
        print(
            "No usable detections in the synthetic demo video (expected -- see this "
            "module's docstring). Falling back to real inference on ultralytics' own "
            "bundled reference photos."
        )
        candidates = try_bundled_reference_photos(config, zones_raw, lines_raw)
        source_used = "ultralytics bundled reference photos (bus.jpg / zidane.jpg)"

    if not candidates:
        raise RuntimeError("No frames with any tracked objects were produced from either source.")

    chosen = pick_frames(candidates, NUM_FRAMES_TO_SAVE)

    for i, (label, annotated, n_tracks, n_violations) in enumerate(chosen, start=1):
        out_path = OUT_DIR / f"detection-frame-{i}.png"
        cv2.imwrite(str(out_path), downscale(annotated), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        (FRONTEND_PUBLIC_DIR / out_path.name).write_bytes(out_path.read_bytes())
        print(
            f"Wrote {out_path} (source: {label}, {n_tracks} tracked object(s), "
            f"{n_violations} violation(s))"
        )

    print(f"Done. {len(chosen)} annotated frame(s) written to {OUT_DIR} and {FRONTEND_PUBLIC_DIR}.")
    print(f"Frame source used: {source_used}")


if __name__ == "__main__":
    main()
