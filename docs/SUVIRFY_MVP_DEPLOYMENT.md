# Suvirfy MVP Deployment and Live-Camera Guide

This guide explains how the current Suvirfy MVP works, how to start it on
Windows, and how to connect one live RTSP camera for a controlled use-case
evaluation.

The MVP is a local, single-camera deployment. It is suitable for validating
whether the detection, shoplifting-classification, evidence, MQTT, and
dashboard workflow fits a site. It is not yet a production multi-tenant VMS.

## 1. Current architecture

```text
Video file / RTSP camera
        |
        v
OpenCV VideoSource
        |
        v
Existing YOLOv8 object detector
        |
        v
ByteTrack stable person IDs
        |
        +--> ZoneRuleEngine
        |       restricted zones, perimeter, dwell, loitering,
        |       after-hours, line crossing, occupancy
        |
        +--> Shoplifting analyzer
                person pose keypoints
                -> wrist-to-torso concealment heuristic
                -> per-track temporal confirmation
                -> suspected_shoplifting

VisionEvent
        |
        +--> EvidenceCapture
        |       original snapshot immediately
        |       clip asynchronously
        |
        +--> MQTT publisher
        |       analytics/events
        |
        +--> FastAPI dashboard service
                in-memory event buffer
                WebSocket /ws/live
                React dashboard
```

The main orchestration is in
[`cv_api/src/analytics/pipeline.py`](../cv_api/src/analytics/pipeline.py).
The live source and reconnect behavior are implemented in
[`cv_api/src/analytics/ingest.py`](../cv_api/src/analytics/ingest.py).

## 2. Current MVP capabilities

The existing YOLOv8 detector remains responsible for person/object
detection. ByteTrack assigns stable IDs. The shoplifting classifier does not
replace either component.

For each tracked person:

1. The existing tracked bounding box is cropped from the original frame.
2. YOLO11x-pose runs on that crop.
3. The first pose result's 17 COCO keypoints are read from `keypoints.xyn`.
4. The 17 `(x, y)` pairs are flattened to 34 `float32` values.
5. The classifier output is passed through sigmoid.
6. Probabilities are aggregated separately by ByteTrack ID.
7. Sustained suspicious observations create a `suspected_shoplifting` event.

Missing pose/keypoint results are skipped, not treated as normal activity.
The event means possible suspicious activity requiring human review; it does
not prove theft.

The rule engine also supports restricted-zone entry, perimeter intrusion,
dwell-time exceeded, loitering, after-hours presence, directional line
crossing, occupancy tracking, and operator occupancy reconciliation.

## 3. Prerequisites

Verify the local runtime:

```powershell
cd C:\Users\HomePC\Documents\suvirfy\rtsp-yolo\cv_api
Test-Path .venv\Scripts\python.exe
Test-Path models\shoplifting\shoplifting_model.pth
Test-Path yolo11x.pt
Test-Path yolo11x-pose.pt
```

The project also requires Node.js/npm and Mosquitto. The current launcher
expects Mosquitto at:

```text
C:\Program Files\mosquitto\mosquitto.exe
```

## 4. Start the local MVP

Run the single launcher:

[`start_suvirfy.ps1`](../start_suvirfy.ps1)

```powershell
cd C:\Users\HomePC\Documents\suvirfy\rtsp-yolo
Set-ExecutionPolicy -Scope Process Bypass
.\start_suvirfy.ps1
```

It starts Mosquitto on `1883`, FastAPI on `8000`, and Vite/React on `5173`.
It does not start a video source. It checks duplicate listeners, verifies
FastAPI health, opens the dashboard, and writes logs to `runtime-logs\`.

Open the dashboard manually at:

```text
http://127.0.0.1:5173
```

Option:

```powershell
.\start_suvirfy.ps1 -NoBrowser
```

Press `Ctrl+C` in the launcher window to stop processes started by it.
Services that were already running are left untouched.

The Vite proxy intentionally uses `127.0.0.1:8000` instead of
`localhost:8000` because Windows may resolve `localhost` to IPv6 `::1`.

## 5. Connect a live RTSP camera

Copy the configuration before editing:

```powershell
cd C:\Users\HomePC\Documents\suvirfy\rtsp-yolo\cv_api
Copy-Item config\pipeline.yaml config\pipeline-live-camera.yaml
```

Change the camera section:

```yaml
camera:
  id: "entrance-cam"
  site_id: "evaluation-site"
  source: "rtsp://192.168.1.50:554/Streaming/Channels/101"
  loop: false
```

For credentials:

```yaml
source: "rtsp://username:password@192.168.1.50:554/stream"
```

URL-encode reserved characters in credentials. Do not commit real
credentials to source control.

Test the stream in VLC or FFplay first. Then test OpenCV:

```powershell
cd C:\Users\HomePC\Documents\suvirfy\rtsp-yolo\cv_api
.\.venv\Scripts\python.exe -c "import cv2; u='rtsp://CAMERA_URL'; c=cv2.VideoCapture(u); print('opened=', c.isOpened()); ok, frame=c.read(); print('first_frame=', ok, None if frame is None else frame.shape); c.release()"
```

Start the services. They remain idle until a source is started from the UI:

```powershell
cd C:\Users\HomePC\Documents\suvirfy\rtsp-yolo
.\start_suvirfy.ps1
```

In the dashboard, choose **RTSP camera**, enter the URL and camera/site IDs,
then select **Connect camera**. The monitor shows annotated frames from that
same pipeline. **Disconnect** requests a graceful stop and evidence flush.

For a command-line-only pipeline, explicitly pass its source:

```powershell
cd C:\Users\HomePC\Documents\suvirfy\rtsp-yolo\cv_api
.\.venv\Scripts\python.exe -m src.analytics.pipeline `
  --config .\config\pipeline-live-camera.yaml `
  --source "rtsp://192.168.1.50:554/Streaming/Channels/101"
```

For an initial headless stability test:

```yaml
visualization:
  enabled: false
```

## 6. Calibrate zones and lines

Do not reuse demo-video coordinates. Zone and line coordinates use the
original camera-frame pixel dimensions. `detector.image_size` is only the
YOLO inference size.

Calibrate these for the real camera:

- merchandise/shelf zones;
- checkout zone;
- exit line;
- restricted areas;
- perimeter;
- entrance and loading zones.

The current shoplifting polygons are demo-camera geometry for the 640×480
shoplifting video. Replace them before interpreting live alerts.

Recommended initial settings:

```yaml
shoplifting:
  enabled: true
  classification_threshold: 0.5
  observation_window_seconds: 5.0
  minimum_suspicious_observations: 3
  cooldown_seconds: 30.0
```

The detector's YOLO confidence threshold, the pose model's keypoint
confidence, and the shoplifting concealment score are separate values. The
shoplifting score is a geometric heuristic in the 0–1 range, not a calibrated
probability. An event currently requires at least three scores at or above
`classification_threshold` within the five-second observation window.
Evidence is captured after that event is confirmed. The independent
`evidence.shoplifting_min_concealment_score` defaults to `0.0`, so it does not discard
confirmed incidents. A `0.7` evidence cutoff may hide lower-scoring clips and
snapshots without stopping the corresponding event alert; validate a threshold
against labeled footage before raising it.

## 7. Validate the use case

Check the CV log for:

```text
Opened video source: rtsp://...
NEW track id=... class=person
Event generated: suspected_shoplifting track_id=...
MQTT event published topic=analytics/events
```

The stream URL is redacted in logs. The dashboard should receive events
through MQTT and display them through FastAPI WebSocket `/ws/live`.

Evidence is stored under:

```text
evidence\<site_id>\<camera_id>\YYYY-MM-DD\
```

Snapshots are created immediately. Clips finish asynchronously after the
post-roll period. The current local MVP has no automatic retention policy.

## 8. Logs and troubleshooting

### Dashboard-controlled video and RTSP jobs

Start the API and frontend; no source starts automatically:

```powershell
.\start_suvirfy.ps1
```

Open `http://127.0.0.1:5173`, then use **Start an analysis** to choose
**Upload video** or **RTSP camera**. Enter a camera ID and site ID, start the
job, and watch its status while events continue through the existing
dashboard WebSocket and MQTT flow.

Uploads are stored under a unique `runtime-jobs` directory and analyzed with a
generated copy of the current pipeline configuration. The shared
`config/pipeline.yaml` is not modified, and the uploaded input is removed
after processing. Uploads are limited to 500 MB by default
(`SUVIRFY_MAX_UPLOAD_BYTES`), and one source job runs at a time by default
(`SUVIRFY_MAX_CONCURRENT_JOBS=1`).

Each dashboard job passes its selected source directly to the pipeline
process with `--source`; an empty source in the shared YAML prevents fallback
to a demo video. The standalone pipeline is opt-in in Docker Compose and in
the launch scripts.

RTSP jobs accept only `rtsp://` and `rtsps://` URLs. This is a local MVP
feature: it has no authentication, tenant isolation, persistent job store, or
production-grade secret management. Do not expose these endpoints publicly;
credentials in an RTSP URL are temporarily present in the generated job
configuration.

Launcher logs are in:

```text
runtime-logs\
```

Important files include `fastapi.err.log`, `frontend.err.log`, and
`cv-pipeline.err.log`.

```powershell
Get-NetTCPConnection -LocalPort 1883,8000,5173 -State Listen
Invoke-WebRequest http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:8000/events
```

| Symptom | Likely cause |
|---|---|
| Dashboard disconnected | FastAPI unavailable or Vite proxy failure |
| `ECONNREFUSED ::1:8000` | Windows IPv6 `localhost` resolution |
| Camera cannot open | Wrong RTSP URL, credentials, firewall, or codec |
| No rules fire | Zones/lines do not match live frame geometry |
| No shoplifting probability | Pose model returned no valid keypoints |
| No dashboard events | MQTT broker/topic mismatch |
| Browser clip does not play | FFmpeg or browser-compatible encoding unavailable |

## 9. MVP limitations

The current MVP does not provide MongoDB persistence, authentication,
tenant isolation, production access control, automatic evidence retention,
high availability, or multi-camera orchestration in one process.

The dashboard event store is in memory; restarting FastAPI clears recent
events while existing evidence files remain on disk.

The classifier was trained on frame-level pose features, not as a complete
video action-recognition model. Temporal smoothing reduces isolated alerts
but does not establish that theft occurred. Every event requires human
review.

## 10. Recommended evaluation

1. Select one camera with a clear shelf and exit view.
2. Confirm RTSP independently with VLC or FFplay.
3. Copy the pipeline configuration and set the real source.
4. Calibrate polygons and the exit line for the real resolution.
5. Run headless and verify reconnect, MQTT, and evidence behavior.
6. Run with visualization for operator review.
7. Test normal shopping, checkout, occlusion, and exit cases.
8. Record false positives, missed events, latency, resource usage, and
   reconnect behavior.
9. Decide whether accuracy is sufficient before production hardening.

The success criterion is not merely that an alert is produced. Alerts must
be timely, explainable, and accurate enough for operators to review.
