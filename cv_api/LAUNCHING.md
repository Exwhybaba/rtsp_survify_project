# Launching `cv_api`

`cv_api` contains the video analytics pipeline and a FastAPI dashboard API.
It can publish events through MQTT and optionally send each event to an HTTP
webhook.

## Option 1: Run with Docker Compose

Open PowerShell:

```powershell
cd C:\Users\HomePC\Documents\suvirfy\rtsp-yolo\cv_api

python scripts\generate_demo_video.py --output demo-media\demo.mp4
docker compose up --build
```

The services will be available at:

- React dashboard: <http://localhost:5173>
- FastAPI dashboard: <http://localhost:8000>
- Interactive API documentation: <http://localhost:8000/docs>

Stop the services with:

```powershell
docker compose down
```

## Option 2: Run locally

Create and activate a virtual environment:

```powershell
cd C:\Users\HomePC\Documents\suvirfy\rtsp-yolo\cv_api

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For Windows CPU-only inference, if `import torch` reports a DLL error, repair
the PyTorch installation with:

```powershell
python -m pip install --force-reinstall `
  --index-url https://download.pytorch.org/whl/cpu `
  torch==2.5.1 torchvision==0.20.1
```

Verify it with:

```powershell
python -c "import torch; print(torch.__version__)"
```

The tracking visualizer needs GUI-enabled `opencv-python`, not
`opencv-python-headless`.

### Run a recorded video file

To start the local broker, evidence server, viewer, and analytics pipeline
together, run this from `cv_api`:

```powershell
.\scripts\start_local.ps1
```

The script uses `.venv\Scripts\python.exe` when available, finds
`mosquitto.exe` on `PATH`, opens the viewer, and writes logs to `logs\`.
Use `-NoBrowser` to skip opening a browser, or pass `-MosquittoPath` if
Mosquitto is not on `PATH`. The default ports are MQTT `1883`, MQTT
WebSockets `9001`, evidence `8001`, and viewer `8080`.

The pipeline accepts local video files such as `.mp4`, `.avi`, and `.mov`.
For the included synthetic video, generate it with:

```powershell
python scripts\generate_demo_video.py --output demo-media\demo.mp4
```

For your own video, copy it into `demo-media` and set `camera.source` in
`config\pipeline.yaml` to its path:

```yaml
camera:
  id: "recorded-video"
  source: "demo-media/my-video.mp4"
  loop: false
```

Set `loop: true` if the file should restart automatically when it reaches the
end. A finite video with `loop: false` stops when all frames have been read.

Start the dashboard API in one PowerShell window:

```powershell
uvicorn src.dashboard.api:app --reload --port 8000
```

Start the analytics pipeline in a second PowerShell window:

```powershell
cd C:\Users\HomePC\Documents\suvirfy\rtsp-yolo\cv_api
.\.venv\Scripts\Activate.ps1

python -m src.analytics.pipeline --config config\pipeline.yaml --max-frames 200
```

Remove `--max-frames 200` to process the entire file, or to continue running
forever when `loop: true`:

```powershell
python -m src.analytics.pipeline --config config\pipeline.yaml
```

The first pipeline run may download the `yolov8n.pt` model automatically.

### Important: synthetic demo versus real people

The generated `crosswalk.avi`/`demo.mp4` is a test scene made from three
colored rectangles. The generator writes the visible `P1`, `P2`, and `P3`
labels directly into the frames; they are not YOLO or ByteTrack IDs. The
pretrained YOLO model may not classify these simple rectangles as people.

To see real `Person ID: ...` tracking boxes, use a video containing real
people:

```powershell
python scripts\visualize_tracking.py `
  --video .\path\to\real-video.mp4 `
  --output .\demo-media\real-video-tracked.mp4
```

The visualizer runs YOLO person detection first and then assigns persistent
ByteTrack IDs to detections. If YOLO finds no person, no tracking box is
drawn; any labels already present in the source video remain visible.

## Run with a camera

An IP camera is used through its RTSP stream. First make sure the computer
running `cv_api` can reach the camera over the network and that the camera's
RTSP stream is enabled.

Edit `config\pipeline.yaml`:

```yaml
camera:
  id: "front-door"
  source: "rtsp://USERNAME:PASSWORD@192.168.1.50:554/stream1"
  loop: false
```

Then start the dashboard and pipeline using the same commands from the local
run section:

```powershell
uvicorn src.dashboard.api:app --reload --port 8000
```

In another PowerShell window:

```powershell
python -m src.analytics.pipeline --config config\pipeline.yaml
```

The pipeline reconnects to an RTSP stream after a failed read. Common stream
URL formats include:

```text
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/Streaming/Channels/101
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/cam/realmonitor?channel=1&subtype=0
```

The exact path depends on the camera manufacturer. Test the URL with a player
such as VLC before starting the pipeline.

Do not commit camera passwords to `config\pipeline.yaml`. Use a secrets
manager or generate the configuration at deployment time.

### Run a camera with Docker Compose

For Docker, the RTSP camera must be reachable from inside the `analytics`
container. Set the RTSP URL in `config\pipeline.yaml`, then restart the
analytics service:

```powershell
docker compose up --build analytics dashboard dashboard-ui mosquitto
```

If the camera is on the host machine, use an address reachable from the
container rather than assuming that `localhost` refers to the host.

## API endpoints

### Health check

```powershell
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Get recent events

```powershell
curl http://localhost:8000/events
```

This returns the recent analytics events held in the dashboard's in-memory
buffer.

### Live events over WebSocket

Connect a WebSocket client to:

```text
ws://localhost:8000/ws/live
```

The server sends each new analytics event as JSON.

### Interactive API documentation

Open <http://localhost:8000/docs> in a browser to inspect and call the
available HTTP endpoints interactively.

### Reconcile uncertain occupancy

If a tracked person disappears without crossing an exit line, occupancy is
marked uncertain instead of being silently decremented. An operator can
reconcile it through the dashboard API:

Keep the occupant counted:

```powershell
curl -X POST "http://localhost:8000/occupancy/cam1/reconcile?action=confirm_present"
```

Confirm that the uncertain occupant has exited:

```powershell
curl -X POST "http://localhost:8000/occupancy/cam1/reconcile?action=confirm_exited"
```

The command is sent over MQTT to the running analytics pipeline. The pipeline
must be running and connected to the same broker for the command to take
effect.

## Event publishing

The pipeline publishes events to MQTT using the settings in
`config/pipeline.yaml`:

```yaml
publisher:
  mqtt_host: "localhost"
  mqtt_port: 1883
  mqtt_topic: "analytics/events"
```

With Docker Compose, the analytics and dashboard containers use the
`mosquitto` service as the MQTT host automatically.

## Optional HTTP webhook

To POST every detected event to another system, set `webhook_url` in
`config/pipeline.yaml`:

```yaml
publisher:
  webhook_url: "http://your-server/events"
```

Each event is sent as a JSON HTTP `POST` request with the
`Content-Type: application/json` header.

Leave `webhook_url` empty to disable webhook delivery:

```yaml
publisher:
  webhook_url: ""
```

## Using a real RTSP camera

Edit `camera.source` in `config/pipeline.yaml`:

```yaml
camera:
  source: "rtsp://username:password@192.168.1.50:554/stream1"
```

Do not commit real camera credentials. Use environment variables or a
secrets manager for production deployments.
