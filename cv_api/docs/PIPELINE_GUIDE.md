# Suvirfy Analytics Pipeline: How It Works

This document explains the MVP pipeline from camera input to operator action.
It describes the runtime sequence, configuration, event formats, evidence
files, MQTT topics, schedules, cooldowns, and the test strategy.

The pipeline is intentionally a single-camera process. It does not implement
tenants, authentication roles, an incident database, an outbox, retention
automation, face recognition, or biometric identification.

## 1. What the system does

For each configured camera, the process:

1. Opens an RTSP stream or a video file.
2. Reads frames.
3. Detects objects with YOLOv8.
4. Assigns stable IDs with the ByteTrack adapter.
5. Evaluates polygon-zone and virtual-line rules.
6. Creates a validated `VisionEvent`.
7. Optionally captures a snapshot and schedules a video clip.
8. Publishes the event over MQTT and optionally through a webhook.
9. Displays the annotated frame if visualization is enabled.
10. Accepts operator occupancy reconciliation commands over MQTT.

The main implementation is in
[`src/analytics/pipeline.py`](../src/analytics/pipeline.py). The pipeline
coordinates the components; detection, tracking, rules, evidence, and
publishing remain separate so each can be replaced or tested independently.

### Rule package layout

Rule orchestration is implemented under
[`src/analytics/rules/`](../src/analytics/rules/):

- `engine.py`: configuration loading, geometry, schedules, cooldowns, event
  creation, and handler dispatch.
- `base.py`: shared `Rule`, track-state, handler interface, and object-class
  filtering.
- `zone_entry.py`: restricted-zone entry.
- `line_crossing.py`: directional line crossing and occupancy integration.
- `dwell_time.py`: dwell-time threshold handling.
- `loitering.py`: loitering threshold handling.
- `perimeter_intrusion.py`: perimeter entry for configured classes.
- `after_hours.py`: closed-schedule presence.
- `occupancy.py`: occupancy state and reconciliation.
- `queue_monitoring.py`: reserved placeholder; no queue feature is enabled.

The rule engine is implemented exclusively under
[`src/analytics/rules/`](../src/analytics/rules/). `engine.py` coordinates
configuration, shared state, schedules, cooldowns, and handler dispatch, while
each rule handler lives in its own module.

## 2. Runtime architecture

```text
RTSP/file
   |
   v
VideoSource --------------------------+
   |                                  |
   v                                  |
YOLO detector                         | add_frame(frame, event_time)
   |                                  v
   v                            EvidenceCapture
ByteTrack adapter                     |
   |                                  | snapshot immediately
   v                                  | clip asynchronously
ZoneRuleEngine                         v
   |                            evidence_ready
   v                                  |
VisionEvent --------------------------+
   |
   v
EventPublisher ---> MQTT broker ---> viewer/dashboard/other consumers
       |
       +----------------------------> optional webhook
```

The pipeline processes one frame at a time. Evidence capture is the one
intentional asynchronous part: clip encoding runs in a background thread so
the detector and rule loop does not wait for video encoding.

## 3. Starting a process

Run one process per camera:

```powershell
python -m src.analytics.pipeline --config config\cam1.yaml
```

The default configuration path is `config\pipeline.yaml`. A process reads
its YAML once at startup. The `camera.id`, `camera.site_id`, and
`camera.source` values identify the process's camera and source.

Useful command-line option:

```powershell
python -m src.analytics.pipeline --config config\cam1.yaml --max-frames 300
```

`--max-frames` is useful for demos and tests.

## 4. Frame ingestion

`VideoSource` wraps OpenCV's `VideoCapture`.

### Streams

RTSP, RTSPS, HTTP, and HTTPS sources are treated as live streams. By default,
`latest_frame_only=True` for streams:

- A daemon reader thread continuously reads the camera.
- It stores only the newest frame.
- If processing is slower than the camera, intermediate frames are dropped.
- `dropped_frames` exposes the number of replaced buffered frames.
- A failed read causes reconnect attempts.
- `close()` signals the reader thread, wakes waiting consumers, joins the
  thread, and releases OpenCV.

This prevents the analytics loop from processing stale frames that were
buffered minutes earlier.

Credentials are never emitted in source logs. For example:

```text
rtsp://operator:secret@camera.example/live
```

is logged as:

```text
rtsp://***:***@camera.example/live
```

The helper is `redact_uri()` in
[`src/analytics/ingest.py`](../src/analytics/ingest.py).

### Files

Video files default to `latest_frame_only=False`. They are read sequentially
and no frames are dropped. `loop: true` restarts a file at frame zero when it
ends.

The pipeline uses different time bases:

- Live streams use each frame's wall-clock timestamp.
- Files use the first frame timestamp plus
  `frame_index / source_fps`.

This makes dwell rules and cooldowns deterministic for file demonstrations.

## 5. Detection and tracking

The detector interface is:

```python
detections = detector.detect(frame)
```

The production implementation is `YoloV8Detector`. Its Ultralytics import
is deferred until construction, which allows tests to use a fake detector
without installing or downloading model weights.

Each detection contains:

- `bbox`: `(x1, y1, x2, y2)`
- `confidence`
- `class_name`

The pipeline passes detections to `ByteTrackAdapter.update()`. The tracker
returns immutable `TrackedObject` values containing the stable `track_id`,
box, centroid, confidence, class, age, and hit count.

Only objects with `class_name == "person"` are sent to the zone engine. The
detector may still keep vehicles and other configured classes for display or
future rules.

## 6. Rule evaluation

Rules are configured under `zones` and `lines`. Supported event types are:

- `restricted_zone_entry`
- `perimeter_intrusion`
- `line_crossing`
- `dwell_time_exceeded`
- `loitering_detected`
- `after_hours_presence`

### Polygon zones

A zone contains a polygon. The engine compares the tracked object's centroid
with the polygon:

- Outside to inside produces `restricted_zone_entry`.
- Remaining inside does not repeat the entry event.
- Leaving clears the entry state.
- Re-entering can produce another entry event.

`dwell_time_exceeded` starts its timer when the object enters and uses the
pipeline event timestamp rather than `time.time()`.

`perimeter_intrusion` is an entry rule intended for protected boundaries. It
supports `object_classes` such as `person`, `car`, and `truck`, and includes a
configurable `severity` metadata value. Rules without `object_classes` remain
person-only for backward compatibility.

`loitering_detected` is a dwell-style rule with a separate event type. Set
`max_loiter_seconds` to the required duration. It emits once per continuous
track presence and resets after the object leaves.

`after_hours_presence` emits on entry when the zone schedule is closed. Its
default `armed_when` is `closed_hours`; it can be overridden explicitly.

### Virtual lines

A line contains two points. The engine compares the previous and current
centroid positions. When the movement segment intersects the line, the engine
calculates one of:

- `negative_to_positive`
- `positive_to_negative`

`allowed_direction` can suppress the opposite direction. With
`track_occupancy: true`, the rule also updates the occupancy set and includes
occupancy metadata in the event.

### Site identity

`ZoneRuleEngine.from_config()` reads:

```yaml
camera:
  site_id: site-a
```

If `site_id` is absent, it uses `"default"` for backward compatibility.
`site_id` is stored on the engine and added to every event. It is not an
argument to `evaluate()`.

## 7. Schedules

A zone or line can define a schedule:

```yaml
schedule:
  timezone: Africa/Lagos
  open_hours:
    mon-sat: ["08:00-20:00"]
    sun: []
```

Supported weekday keys are `mon`, `tue`, `wed`, `thu`, `fri`, `sat`, `sun`,
plus the aliases `mon-fri` and `mon-sat`.

Windows use local time in the configured IANA timezone. A window may cross
midnight:

```yaml
open_hours:
  fri: ["22:00-02:00"]
```

This is open from Friday 22:00 through Saturday 02:00. Timezone conversion is
performed with `zoneinfo`, so DST changes are handled using the timezone
database.

Each rule can select when it is armed:

```yaml
rules:
  - type: restricted_zone_entry
    armed_when: closed_hours
```

Values:

- `always` (default): evaluate regardless of schedule.
- `open_hours`: evaluate only inside an open window.
- `closed_hours`: evaluate only outside open windows.

The schedule gates event generation. It does not stop detection or tracking.

## 8. Cooldowns

Cooldowns suppress repeated alerts without stopping rule state updates:

```yaml
rules:
  - type: restricted_zone_entry
    cooldown_seconds: 30
```

The key is:

```text
(camera_id, zone-or-line-name, event_type)
```

The engine compares event timestamps, not the machine clock. This is
important for deterministic video-file runs.

When an event is suppressed, the engine increments an internal counter. The
next event that fires after the cooldown includes:

```json
{
  "metadata": {
    "suppressed_count": 2
  }
}
```

`cooldown_seconds: 0` disables cooldown suppression.

## 9. Event schema and wire format

The canonical model is `VisionEvent` in
[`src/analytics/schemas.py`](../src/analytics/schemas.py).

Important fields:

| Field | Meaning |
|---|---|
| `site_id` | Required trimmed site identifier |
| `camera_id` | Required trimmed camera identifier |
| `event_id` | UUID generated for this event |
| `event_type` | Rule event type |
| `track_id` | Stable tracker identity |
| `confidence` | Detector confidence from 0 to 1 |
| `bbox` | Bounding box in pixels |
| `timestamp` | UTC event timestamp |
| `metadata` | Rule-specific fields |
| `evidence` | Snapshot reference, and later clip reference |

Example:

```json
{
  "camera_id": "cam1",
  "site_id": "site-a",
  "event_id": "7c6f2a35-9e03-4a2a-a7a2-3c2d5a5a2c01",
  "event_type": "restricted_zone_entry",
  "track_id": 12,
  "confidence": 0.94,
  "bbox": {"x1": 380, "y1": 60, "x2": 450, "y2": 180},
  "timestamp": "2026-11-06T19:00:00Z",
  "metadata": {"zone": "restricted_area", "class_name": "person"},
  "evidence": null
}
```

Pydantic rejects unknown top-level fields on the event model. This prevents
silent wire-format drift.

## 10. MQTT publishing

`EventPublisher` creates a paho client compatible with paho-mqtt 1.x and 2.x.
It applies optional credentials and TLS CA configuration before connecting.

Configuration precedence is:

1. Environment variable.
2. YAML publisher value.
3. Built-in default.

Supported environment variables:

```text
MQTT_HOST
MQTT_PORT
MQTT_TOPIC
MQTT_USERNAME
MQTT_PASSWORD
MQTT_TLS_CA
MQTT_OCCUPANCY_CONTROL_TOPIC
WEBHOOK_URL
```

### Event topic

If `mqtt_topic` is `analytics/events`, an event for site `site-a` and camera
`cam1` is published to:

```text
analytics/events/site-a/cam1
```

Consumers that want all events subscribe to:

```text
analytics/events/#
```

The publisher sends the same JSON payload to the optional webhook, but the
webhook is not used for evidence-ready notifications.

## 11. Evidence capture

Evidence is disabled by default:

```yaml
evidence:
  enabled: false
```

Enable it with:

```yaml
evidence:
  enabled: true
  evidence_dir: "evidence"
  pre_roll_seconds: 5
  post_roll_seconds: 5
  fps_cap: 10
```

### Frame flow

For every processed frame, the pipeline calls:

```python
evidence.add_frame(frame, event_time)
```

`EvidenceCapture` JPEG-compresses frames and keeps only the configured
time window. `fps_cap` limits how many frames are retained per second.
The buffer uses the pipeline event time, so files and streams follow the
same API.

After rule evaluation, the pipeline calls:

```python
evidence.on_events(events, frame, event_time)
```

If there are events:

1. The original current frame is written as the evidence JPEG; event metadata
   stays alongside the image in the dashboard.
2. The event is updated with an `EvidenceRef` containing the snapshot path and
   hash.
3. A clip job is created for pre-roll through post-roll.
4. The event can be published immediately with the snapshot reference.

The pipeline catches evidence errors and logs them rather than stopping
analytics.

### Clip flow

The worker waits until frames reach the post-roll time, then:

1. Decodes buffered JPEG frames.
2. Writes an MP4 using the `mp4v` codec.
3. If `ffmpeg` is available, re-encodes using H.264, `yuv420p`, and
   `+faststart`.
4. Computes the final clip SHA-256.
5. Writes a JSON sidecar beside the clip.
6. Updates `EvidenceRef.clip_path` and `clip_sha256`.
7. Publishes an `evidence_ready` message.

Overlapping events on the same camera are merged into one clip job. The
sidecar lists all event IDs.

### Evidence directory

Files are stored as:

```text
{evidence_dir}/{site_id}/{camera_id}/{YYYY-MM-DD}/
```

Typical contents:

```text
evidence/
  site-a/
    cam1/
      2026-11-06/
        7c6f2a35-....jpg
        7c6f2a35-....h264.mp4
        7c6f2a35-....h264.json
```

The sidecar contains event IDs, site, camera, redacted source URI, UTC clip
times, snapshot hashes, and the final clip hash. Source credentials are not
stored in the sidecar.

### Evidence-ready message

The follow-up is a separate JSON object on the same event topic:

```json
{
  "type": "evidence_ready",
  "event_id": "7c6f2a35-9e03-4a2a-a7a2-3c2d5a5a2c01",
  "site_id": "site-a",
  "camera_id": "cam1",
  "evidence": {
    "snapshot_path": "evidence/site-a/cam1/2026-11-06/....jpg",
    "clip_path": "evidence/site-a/cam1/2026-11-06/....mp4",
    "clip_start": "2026-11-06T18:59:55Z",
    "clip_end": "2026-11-06T19:00:05Z",
    "snapshot_sha256": "....",
    "clip_sha256": "....",
    "created_at": "2026-11-06T19:00:05Z"
  }
}
```

Consumers match this message to the original alert using `event_id`.

## 12. Operator viewer and occupancy commands

[`tools/viewer.html`](../tools/viewer.html) is a static page. It uses MQTT
over WebSockets and:

- subscribes to `{base_topic}/#`;
- displays newest alerts first;
- shows site, camera, event type, time, and snapshot;
- replaces the evidence cell with a clip link when `evidence_ready` arrives;
- publishes `confirm_present` and `confirm_exited` buttons.

Start a static evidence server:

```powershell
python -m http.server 8000 --directory evidence
```

Mosquitto needs both listeners:

```text
listener 1883
listener 9001
protocol websockets
```

Open the viewer with:

```text
tools/viewer.html?broker=ws://localhost:9001&evidence=http://localhost:8000
```

The viewer publishes commands to `analytics/control/occupancy` by default:

```json
{
  "camera_id": "cam1",
  "action": "confirm_exited"
}
```

The pipeline validates the command with `OccupancyControlCommand`. Unknown
fields and unsupported actions are rejected. A valid command for another
camera is ignored by the current single-camera process. Accepted commands
are logged with camera, action, operator ID when supplied, and resulting
occupancy count.

## 13. Configuration reference

The main sections of `config/pipeline.yaml` are:

| Section | Purpose |
|---|---|
| `camera` | ID, site ID, source, and file looping |
| `detector` | YOLO model, confidence, classes, device, image size |
| `tracker` | ByteTrack thresholds and relinking |
| `publisher` | MQTT, webhook, credentials, and TLS |
| `evidence` | Evidence directory and rolling capture settings |
| `zones` | Polygon geometry and zone rules |
| `lines` | Tripwire geometry and line rules |
| `visualization` | OpenCV display settings |

`evidence.min_event_confidence` is an evidence-only cutoff applied to all event
types. The dashboard pipeline sets it to `0.7`, so events below 70% confidence
are still reported but do not receive a snapshot or clip. Set it to `0.0` to
capture evidence for every event. `evidence.shoplifting_min_concealment_score`
is a separate optional evidence-only cutoff for confirmed shoplifting events;
the concealment score is a geometric heuristic, not a calibrated probability.
Neither evidence cutoff suppresses the event or MQTT alert.

Minimal camera configuration:

```yaml
camera:
  id: cam1
  site_id: site-a
  source: "rtsp://camera.example/live"
  loop: false
```

For a file demo:

```yaml
camera:
  id: demo
  site_id: demo-site
  source: "demo-media/grocery-store.mp4"
  loop: false
```

## 14. Tests and dependency boundaries

The tests are designed not to require a real camera, broker, network, or
Ultralytics model:

- MQTT clients are mocked.
- HTTP sessions are mocked.
- Detector behavior uses fakes.
- Evidence tests use synthetic NumPy frames and temporary directories.
- The end-to-end test creates a synthetic OpenCV video.
- Schedule tests use fixed UTC timestamps, including a DST transition.

Run the suite:

```powershell
pytest -q
```

Run lint:

```powershell
ruff check src tests scripts/benchmark_detector.py
```

Run the detector benchmark separately; it is not part of pytest:

```powershell
python scripts/benchmark_detector.py demo-media --model .\yolov8n.pt
```

The benchmark prints total FPS, total detections, and class counts split by
the simple mean-brightness day/night heuristic. A model is only loaded when
the benchmark command is explicitly run.

## 15. Failure behavior and operational boundaries

- A failed MQTT publish is logged and does not terminate the processing loop.
- A failed webhook request is logged and does not terminate the processing
  loop.
- Evidence failures are logged and do not terminate analytics.
- RTSP read failures trigger bounded reconnect attempts.
- File read failures end the file source.
- Invalid control commands are logged and ignored.
- The process does not persist an outbox or retry queue.
- Evidence files are not automatically deleted; retention is an operational
  responsibility outside this MVP.
