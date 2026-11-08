# `cv_api` File Documentation

This document describes the purpose of each functional file in `cv_api`.
The sibling [`frontend/`](../frontend/) directory contains the separate
React dashboard UI.

Generated files and local tooling directories are not listed as application
source: `.venv/`, `.pytest_cache/`, `__pycache__/`, model weights, and
`demo-media/` are local or generated artifacts.

## Project flow

```text
camera/video file
    -> src/analytics/ingest.py
    -> src/analytics/detector.py
    -> src/analytics/tracker.py
    -> src/analytics/rules/
    -> src/analytics/schemas.py
    -> src/analytics/publisher.py
    -> MQTT and/or webhook
    -> src/dashboard/api.py
    -> dashboard clients
```

## Root files

| File | Purpose |
| --- | --- |
| [`README.md`](README.md) | Main project documentation, architecture, setup instructions, design decisions, dashboard usage, testing, and production limitations. |
| [`LICENSE`](LICENSE) | License terms for the project. |
| [`requirements.txt`](requirements.txt) | Python runtime and development dependencies, including OpenCV, Ultralytics YOLO, FastAPI, MQTT, pytest, and Ruff. |
| [`pyproject.toml`](pyproject.toml) | Ruff lint configuration and pytest discovery/Python path settings. |
| [`Dockerfile`](Dockerfile) | Builds the Python image used by the analytics and dashboard services. Installs system video libraries, Python dependencies, and the application source. |
| [`docker-compose.yml`](docker-compose.yml) | Defines the Mosquitto broker, analytics pipeline, FastAPI dashboard, and sibling React frontend services. |
| [`.dockerignore`](.dockerignore) | Keeps virtual environments, caches, frontend dependencies, docs, and demo media out of the backend Docker build context. |
| [`.gitignore`](.gitignore) | Excludes Python caches, environments, secrets, video/model artifacts, frontend build output, and local editor files from version control. |
| [`FILE_DOCUMENTATION.md`](FILE_DOCUMENTATION.md) | This file; a file-by-file reference for the `cv_api` backend. |

## Configuration

| File | Purpose |
| --- | --- |
| [`config/pipeline.yaml`](config/pipeline.yaml) | Main analytics configuration: camera source, YOLO model and classes, tracker thresholds, MQTT/webhook publishing, polygon zones, tripwire lines, and rule parameters. |
| [`config/mosquitto.conf`](config/mosquitto.conf) | Local Mosquitto settings used by Compose. It enables an unauthenticated private development listener on port 1883 and disables persistence. It is not suitable for a public production broker. |

## Analytics package

### Package markers

| File | Purpose |
| --- | --- |
| [`src/__init__.py`](src/__init__.py) | Marks `src` as the top-level Python package. |
| [`src/analytics/__init__.py`](src/analytics/__init__.py) | Describes the real-time ingest, detection, tracking, rule, and publishing package. |

### Video ingestion and detection

| File | Purpose |
| --- | --- |
| [`src/analytics/ingest.py`](src/analytics/ingest.py) | Wraps OpenCV `VideoCapture` for local files and RTSP/HTTP streams. Provides frame metadata, looping for finite videos, and reconnect attempts for live streams. |
| [`src/analytics/detector.py`](src/analytics/detector.py) | Defines the detector interface and the Ultralytics YOLOv8 implementation. Converts model results into normalized `Detection` objects and filters them by confidence and class. |

### Tracking, rules, and event data

| File | Purpose |
| --- | --- |
| [`src/analytics/tracker.py`](src/analytics/tracker.py) | Implements the dependency-free multi-object tracker. It matches detections using greedy IoU first, centroid distance second, and assigns persistent non-reused track IDs. |
| [`src/analytics/rules/`](src/analytics/rules/) | Modular rule package. `engine.py` loads configured zones and lines, performs shared geometry/state/schedule/cooldown orchestration, and dispatches to independent handlers for each rule type. |
| [`src/analytics/schemas.py`](src/analytics/schemas.py) | Defines the Pydantic wire models: supported `EventType` values, validated `BoundingBox` data, and the canonical `VisionEvent` payload shared by MQTT, webhooks, and the dashboard. |
| [`src/analytics/publisher.py`](src/analytics/publisher.py) | Publishes validated vision events to MQTT and optionally sends them as JSON to a webhook. Network failures are logged without stopping the video pipeline. |

### Pipeline entry point

| File | Purpose |
| --- | --- |
| [`src/analytics/pipeline.py`](src/analytics/pipeline.py) | Connects ingestion, YOLO detection, tracking, zone rules, and event publishing into the per-frame pipeline. It also provides the command-line entry point and environment-variable overrides for MQTT/webhook settings. |

## Dashboard package

| File | Purpose |
| --- | --- |
| [`src/dashboard/__init__.py`](src/dashboard/__init__.py) | Package marker and description for the live dashboard service. |
| [`src/dashboard/api.py`](src/dashboard/api.py) | FastAPI service that subscribes to MQTT, stores recent events in an in-memory ring buffer, exposes `/health` and `/events`, serves the plain HTML dashboard, and streams events through `/ws/live`. |
| [`src/dashboard/static/index.html`](src/dashboard/static/index.html) | Self-contained plain HTML/JavaScript dashboard. Displays recent events and listens for live event updates from the WebSocket endpoint. |

## Utility scripts

| File | Purpose |
| --- | --- |
| [`scripts/generate_demo_video.py`](scripts/generate_demo_video.py) | Creates a synthetic video with moving colored rectangles. It provides a repeatable local video source without requiring customer footage or a physical camera. |
| [`scripts/generate_annotated_frames.py`](scripts/generate_annotated_frames.py) | Runs the real detector, tracker, and zone engine on demo/reference images and writes annotated detection images to `docs/screenshots/` and the sibling frontend's public screenshot directory. |
| [`scripts/capture_dashboard_screenshot.py`](scripts/capture_dashboard_screenshot.py) | End-to-end screenshot harness. Starts the required local services, runs the pipeline, builds/previews the sibling frontend, uses Playwright, and saves a rendered dashboard screenshot. |

## Tests

| File | Purpose |
| --- | --- |
| [`tests/__init__.py`](tests/__init__.py) | Package marker for the test suite. |
| [`tests/test_tracker.py`](tests/test_tracker.py) | Tests IoU and centroid matching, stable IDs, new-object IDs, disappearance tolerance, ID non-reuse, and matching priority. |
| [`tests/test_zones.py`](tests/test_zones.py) | Tests polygon containment, line intersection, zone-entry transitions, line crossings, and one-shot dwell-time events/reset behavior. |
| [`tests/test_schemas.py`](tests/test_schemas.py) | Tests valid event serialization, enum handling, invalid event rejection, bounding-box validation, and centroid calculation. |
| [`tests/test_publisher.py`](tests/test_publisher.py) | Tests MQTT publishing, connection reuse, optional webhook delivery, and best-effort behavior when either destination fails. |
| [`tests/test_api.py`](tests/test_api.py) | Tests FastAPI health/events endpoints, ring-buffer limits, MQTT stubbing, and WebSocket event delivery. |

## Documentation assets

| File | Purpose |
| --- | --- |
| [`docs/screenshots/dashboard.png`](docs/screenshots/dashboard.png) | Captured example of the rendered dashboard. |
| [`docs/screenshots/detection-frame-1.png`](docs/screenshots/detection-frame-1.png) | Example annotated detection frame generated by the frame utility. |
| [`docs/screenshots/detection-frame-2.png`](docs/screenshots/detection-frame-2.png) | Second example annotated detection frame. |
| [`docs/screenshots/detection-frame-3.png`](docs/screenshots/detection-frame-3.png) | Third example annotated detection frame. |

## Common commands

Run these commands from `cv_api`:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the backend tests
pytest -q

# Check lint
ruff check .

# Generate a local demo video
python scripts/generate_demo_video.py --output demo-media/demo.mp4

# Run the analytics pipeline
python -m src.analytics.pipeline --config config/pipeline.yaml

# Start all Compose services
docker compose up --build
```
