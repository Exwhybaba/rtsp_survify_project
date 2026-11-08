import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";
import {
  EVENT_TYPE_LABELS,
  EVENT_TYPES,
  type EvidenceReadyMessage,
  type EventType,
  type VisionEvent,
} from "./types";

const MAX_ROWS = 200;
const WS_RECONNECT_DELAY_MS = 2000;

// Empty string => same-origin relative paths ("/events", "/ws/live"), which
// works both in dev (via the Vite proxy configured in vite.config.ts) and in
// production (via the nginx reverse-proxy baked into the dashboard-ui image
// — see frontend/nginx.conf). Set VITE_API_BASE to call a different backend
// origin directly instead (e.g. during `vite dev` without the proxy).
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

const EVENT_TYPE_COLORS: Record<EventType, string> = {
  restricted_zone_entry: "#ff5d6c",
  perimeter_intrusion: "#ff7a59",
  line_crossing: "#4da3ff",
  dwell_time_exceeded: "#ffb454",
  loitering_detected: "#c084fc",
  after_hours_presence: "#f472b6",
  suspected_shoplifting: "#ef4444",
};
const UNKNOWN_EVENT_COLOR = "#8b93a7";

type ConnectionState = "connecting" | "connected" | "disconnected";

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString();
  } catch {
    return iso;
  }
}

function formatConfidence(confidence: number): string {
  return confidence.toFixed(2);
}

function isKnownEventType(value: string): value is EventType {
  return EVENT_TYPES.includes(value as EventType);
}

function eventLabel(eventType: string): string {
  return isKnownEventType(eventType)
    ? EVENT_TYPE_LABELS[eventType]
    : `Unknown event (${eventType})`;
}

function eventColor(eventType: string): string {
  return isKnownEventType(eventType)
    ? EVENT_TYPE_COLORS[eventType]
    : UNKNOWN_EVENT_COLOR;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isBoundingBox(value: unknown): boolean {
  if (!isRecord(value)) return false;
  return ["x1", "y1", "x2", "y2"].every(
    (key) => typeof value[key] === "number" && Number.isFinite(value[key]),
  );
}

function isVisionEvent(value: unknown): value is VisionEvent {
  if (!isRecord(value)) return false;
  return (
    typeof value.event_id === "string" &&
    typeof value.camera_id === "string" &&
    typeof value.site_id === "string" &&
    typeof value.event_type === "string" &&
    value.event_type.length > 0 &&
    typeof value.track_id === "number" &&
    Number.isInteger(value.track_id) &&
    typeof value.confidence === "number" &&
    Number.isFinite(value.confidence) &&
    isBoundingBox(value.bbox) &&
    typeof value.timestamp === "string" &&
    isRecord(value.metadata)
  );
}

function isEvidenceReadyMessage(value: unknown): value is EvidenceReadyMessage {
  return (
    isRecord(value) &&
    value.type === "evidence_ready" &&
    typeof value.event_id === "string" &&
    typeof value.site_id === "string" &&
    typeof value.camera_id === "string" &&
    isRecord(value.evidence)
  );
}

function App() {
  const [events, setEvents] = useState<VisionEvent[]>([]);
  const [sourceMode, setSourceMode] = useState<"upload" | "rtsp">("upload");
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [rtspSource, setRtspSource] = useState("");
  const [sourceCameraId, setSourceCameraId] = useState("dashboard-camera");
  const [sourceSiteId, setSourceSiteId] = useState("default");
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [startingJob, setStartingJob] = useState(false);
  const [clearingEvidence, setClearingEvidence] = useState(false);
  const [videoOpen, setVideoOpen] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [unavailableSnapshots, setUnavailableSnapshots] = useState<Set<string>>(
    new Set(),
  );
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | undefined>(undefined);
  const pendingEvidenceRef = useRef<Record<string, EvidenceReadyMessage["evidence"]>>({});

  const appendEvent = useCallback((event: VisionEvent) => {
    setEvents((prev) => {
      const pendingEvidence = pendingEvidenceRef.current[event.event_id];
      if (pendingEvidence) {
        delete pendingEvidenceRef.current[event.event_id];
      }
      const nextEvent = pendingEvidence ? { ...event, evidence: pendingEvidence } : event;
      const next = [
        nextEvent,
        ...prev.filter((item) => item.event_id !== event.event_id),
      ];
      return next.length > MAX_ROWS ? next.slice(0, MAX_ROWS) : next;
    });
  }, []);

  const applyEvidenceReady = useCallback((message: EvidenceReadyMessage) => {
    pendingEvidenceRef.current[message.event_id] = message.evidence;
    setEvents((prev) =>
      prev.map((event) =>
        event.event_id === message.event_id
          ? { ...event, evidence: message.evidence }
          : event,
      ),
    );
  }, []);

  const startSourceJob = useCallback(async () => {
    setStartingJob(true);
    setJobError(null);
    setJobStatus(null);
    try {
      const body = new FormData();
      body.append("camera_id", sourceCameraId);
      body.append("site_id", sourceSiteId);
      let endpoint = `${API_BASE}/api/jobs/upload`;
      if (sourceMode === "upload") {
        if (!sourceFile) throw new Error("Choose a video file first.");
        body.append("file", sourceFile);
      } else {
        if (!rtspSource.trim()) throw new Error("Enter an RTSP URL first.");
        endpoint = `${API_BASE}/api/jobs/rtsp`;
        body.append("source", rtspSource.trim());
      }
      const response = await fetch(endpoint, { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Could not start analysis.");
      }
      setJobStatus(payload.status ?? "queued");
      const jobId = payload.job_id as string;
      setJobId(jobId);
      const poll = window.setInterval(async () => {
        const statusResponse = await fetch(`${API_BASE}/api/jobs/${jobId}`);
        if (!statusResponse.ok) return;
        const status = await statusResponse.json();
        setJobStatus(status.status);
        if (["completed", "failed", "stopped"].includes(status.status)) {
          window.clearInterval(poll);
          if (status.status === "failed") {
            setJobError("Analysis job failed. Check the backend job log.");
          }
        }
      }, 2000);
    } catch (error) {
      setJobError(error instanceof Error ? error.message : "Could not start analysis.");
    } finally {
      setStartingJob(false);
    }
  }, [rtspSource, sourceCameraId, sourceFile, sourceMode, sourceSiteId]);

  const clearEvidence = useCallback(async () => {
    if (!window.confirm("Clear all stored evidence and dashboard events?")) return;
    setClearingEvidence(true);
    try {
      const response = await fetch(`${API_BASE}/api/evidence`, { method: "DELETE" });
      if (!response.ok) throw new Error("Could not clear evidence.");
      setEvents([]);
      setUnavailableSnapshots(new Set());
      pendingEvidenceRef.current = {};
    } catch (error) {
      setJobError(error instanceof Error ? error.message : "Could not clear evidence.");
    } finally {
      setClearingEvidence(false);
    }
  }, []);

  const stopSourceJob = useCallback(async () => {
    if (!jobId || !["queued", "running"].includes(jobStatus ?? "")) return;
    try {
      const response = await fetch(`${API_BASE}/api/jobs/${jobId}/stop`, {
        method: "POST",
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Could not stop analysis.");
      setJobStatus(payload.status);
    } catch (error) {
      setJobError(error instanceof Error ? error.message : "Could not stop analysis.");
    }
  }, [jobId, jobStatus]);

  // Load recent events already sitting in the backend's ring buffer.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/events`);
        if (!res.ok) return;
        const payload: unknown = await res.json();
        const data = Array.isArray(payload)
          ? payload.filter(isVisionEvent)
          : [];
        if (!cancelled) {
          // /events returns oldest-first; the table renders newest-first.
          setEvents(data.slice(-MAX_ROWS).reverse());
        }
      } catch (err) {
        console.warn("Could not load recent events", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Live updates over WS /ws/live, with auto-reconnect.
  useEffect(() => {
    let stopped = false;

    function connect() {
      if (stopped) return;
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      const base = API_BASE
        ? API_BASE.replace(/^http/, "ws")
        : `${proto}://${window.location.host}`;
      const ws = new WebSocket(`${base}/ws/live`);
      wsRef.current = ws;
      setConnectionState("connecting");

      ws.onopen = () => setConnectionState("connected");
      ws.onclose = () => {
        setConnectionState("disconnected");
        if (!stopped) {
          reconnectTimerRef.current = window.setTimeout(
            connect,
            WS_RECONNECT_DELAY_MS,
          );
        }
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (msg) => {
        try {
          const payload: unknown = JSON.parse(msg.data);
          if (isEvidenceReadyMessage(payload)) {
            applyEvidenceReady(payload);
            return;
          }
          if (isVisionEvent(payload)) {
            appendEvent(payload);
            return;
          }
          console.warn("Ignoring malformed event payload");
        } catch (err) {
          console.warn("Bad event payload", err);
        }
      };
    }

    connect();
    return () => {
      stopped = true;
      window.clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [appendEvent, applyEvidenceReady]);

  const counts = useMemo(() => {
    const byType = Object.fromEntries(
      EVENT_TYPES.map((type) => [type, 0]),
    ) as Record<EventType, number>;
    for (const evt of events) {
      if (isKnownEventType(evt.event_type)) {
        byType[evt.event_type] += 1;
      }
    }
    return byType;
  }, [events]);

  const chartData = useMemo(
    () =>
      EVENT_TYPES.map((type) => ({
        type,
        label: eventLabel(type),
        count: counts[type],
      })),
    [counts],
  );

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div>
          <h1>Video Analytics Dashboard</h1>
          <p className="subtitle">
            RTSP + YOLOv8 vision events, live from the analytics pipeline
          </p>
        </div>
        <span className={`status-badge status-${connectionState}`}>
          {connectionState === "connected" && "● connected"}
          {connectionState === "connecting" && "connecting…"}
          {connectionState === "disconnected" && "disconnected — retrying"}
        </span>
      </header>

      <main className="dashboard-main">
        <section className="panel source-panel" aria-label="Analysis source">
          <h2>Start an analysis</h2>
          <p className="panel-subtitle">
            Upload a recorded video or connect an RTSP camera without editing YAML.
          </p>
          <div className="source-tabs">
            <button
              className={sourceMode === "upload" ? "source-tab active" : "source-tab"}
              onClick={() => setSourceMode("upload")}
              type="button"
            >
              Upload video
            </button>
            <button
              className={sourceMode === "rtsp" ? "source-tab active" : "source-tab"}
              onClick={() => setSourceMode("rtsp")}
              type="button"
            >
              RTSP camera
            </button>
          </div>
          <div className="source-fields">
            <label>
              Camera ID
              <input value={sourceCameraId} onChange={(event) => setSourceCameraId(event.target.value)} />
            </label>
            <label>
              Site ID
              <input value={sourceSiteId} onChange={(event) => setSourceSiteId(event.target.value)} />
            </label>
            {sourceMode === "upload" ? (
              <label>
                Video file
                <input
                  type="file"
                  accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm"
                  onChange={(event) => setSourceFile(event.target.files?.[0] ?? null)}
                />
              </label>
            ) : (
              <label>
                RTSP URL
                <input
                  type="url"
                  placeholder="rtsp://user:password@camera/stream"
                  value={rtspSource}
                  onChange={(event) => setRtspSource(event.target.value)}
                />
              </label>
            )}
          </div>
          <div className="source-actions">
            <button className="primary-button" disabled={startingJob} onClick={startSourceJob} type="button">
              {startingJob ? "Starting…" : "Start analysis"}
            </button>
            {jobId && ["queued", "running"].includes(jobStatus ?? "") && (
              <button className="danger-button" onClick={stopSourceJob} type="button">
                Stop analysis
              </button>
            )}
            {jobStatus && <span className="job-status">Job: {jobStatus}</span>}
            {jobStatus && sourceMode === "upload" && sourceFile && (
              <span className="job-source">{sourceFile.name}</span>
            )}
            {jobError && <span className="job-error">{jobError}</span>}
          </div>
          {jobId && jobStatus === "completed" && (
            <div className="job-video">
              <h3>Annotated analysis video</h3>
              <p className="panel-subtitle">
                Includes detections, ByteTrack IDs, zones, lines, probabilities, and alerts.
              </p>
              <button
                className="primary-button"
                onClick={() => {
                  setVideoError(false);
                  setVideoOpen(true);
                }}
                type="button"
              >
                Open video player
              </button>
            </div>
          )}
        </section>

        <section className="summary-strip" aria-label="Event summary">
          <div className="summary-card total">
            <div className="summary-value">{events.length}</div>
            <div className="summary-label">Total events</div>
          </div>
          {EVENT_TYPES.map((type) => (
            <div className="summary-card" key={type}>
              <div
                className="summary-value"
                style={{ color: eventColor(type) }}
              >
                {counts[type]}
              </div>
              <div className="summary-label">{EVENT_TYPE_LABELS[type]}</div>
            </div>
          ))}
        </section>

        <section className="panel chart-panel" aria-label="Events by type">
          <h2>Events by type</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262b38" />
              <XAxis type="number" allowDecimals={false} stroke="#8b93a7" />
              <YAxis
                type="category"
                dataKey="label"
                width={170}
                stroke="#8b93a7"
                tick={{ fill: "#e6e8ee", fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{
                  background: "#161923",
                  border: "1px solid #262b38",
                  borderRadius: 8,
                  color: "#e6e8ee",
                }}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {chartData.map((entry) => (
                  <Cell key={entry.type} fill={EVENT_TYPE_COLORS[entry.type]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="panel" aria-label="Evidence snapshots">
          <div className="panel-heading">
            <div>
              <h2>Live evidence</h2>
              <p className="panel-subtitle">
                Annotated snapshots generated for events by the vision pipeline.
              </p>
            </div>
            <button
              className="danger-button"
              disabled={clearingEvidence}
              onClick={clearEvidence}
              type="button"
            >
              {clearingEvidence ? "Clearing…" : "Clear evidence"}
            </button>
          </div>
          {events.length === 0 ? (
            <div className="empty-state">Waiting for evidence events&hellip;</div>
          ) : (
            <div className="detections-grid">
              {events.slice(0, 12).map((evt) => {
                const zone =
                  typeof evt.metadata.zone === "string"
                    ? evt.metadata.zone
                    : undefined;
                const objectClass =
                  typeof evt.metadata.class_name === "string"
                    ? evt.metadata.class_name
                    : undefined;
                return (
                  <figure key={evt.event_id} className="detection-frame">
                    {evt.evidence ? (
                      unavailableSnapshots.has(evt.event_id) ? (
                        <div className="evidence-placeholder">
                          Snapshot is unavailable.
                        </div>
                      ) : (
                        <img
                          src={`${API_BASE}/api/events/${evt.event_id}/evidence/snapshot`}
                          alt={`${eventLabel(evt.event_type)} for ${objectClass ?? "object"} on ${evt.camera_id}`}
                          loading="lazy"
                          onError={() =>
                            setUnavailableSnapshots((previous) => {
                              const next = new Set(previous);
                              next.add(evt.event_id);
                              return next;
                            })
                          }
                        />
                      )
                    ) : (
                      <div className="evidence-placeholder">
                        Evidence is being generated&hellip;
                      </div>
                    )}
                    {evt.evidence?.clip_path ? (
                      <video
                        controls
                        preload="metadata"
                        src={`${API_BASE}/api/events/${evt.event_id}/evidence/clip`}
                      />
                    ) : (
                      <div className="clip-status">Clip is still being generated.</div>
                    )}
                    <figcaption>
                      <strong>{eventLabel(evt.event_type)}</strong>
                      <span>
                        {evt.camera_id} / {evt.site_id} &middot;{" "}
                        {formatTime(evt.timestamp)}
                      </span>
                      <span>
                        {objectClass ?? "Object"} &middot; Track {evt.track_id}
                        {zone ? ` \u00b7 ${zone}` : ""}
                        {typeof evt.metadata.severity === "string"
                          ? ` \u00b7 ${evt.metadata.severity}`
                          : ""}
                      </span>
                    </figcaption>
                  </figure>
                );
              })}
            </div>
          )}
        </section>

        <section className="panel" aria-label="Live event table">
          <h2>Live events</h2>
          {events.length === 0 ? (
            <div className="empty-state">Waiting for events&hellip;</div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Camera</th>
                    <th>Event</th>
                    <th>Track ID</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((evt, idx) => (
                    <tr key={`${evt.camera_id}-${evt.track_id}-${evt.timestamp}-${idx}`}>
                      <td>{formatTime(evt.timestamp)}</td>
                      <td>{evt.camera_id}</td>
                      <td>
                                <span
                                  className={`badge badge-${evt.event_type}`}
                                  style={{ color: eventColor(evt.event_type) }}
                                >
                                  {eventLabel(evt.event_type)}
                        </span>
                      </td>
                      <td>{evt.track_id}</td>
                      <td>{formatConfidence(evt.confidence)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>

      {videoOpen && jobId && (
        <div className="video-modal-backdrop" role="presentation" onClick={() => setVideoOpen(false)}>
          <section
            aria-label="Annotated analysis video player"
            className="video-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="video-modal-header">
              <h2>Annotated analysis video</h2>
              <button
                aria-label="Close video player"
                className="modal-close"
                onClick={() => setVideoOpen(false)}
                type="button"
              >
                ×
              </button>
            </div>
            {videoError ? (
              <div className="evidence-placeholder">
                This video could not be decoded by the browser. The generated MP4
                may use an unsupported codec; install FFmpeg and encode the output
                as H.264/AAC for browser playback.
              </div>
            ) : (
              <video
                autoPlay
                controls
                playsInline
                preload="metadata"
                src={`${API_BASE}/api/jobs/${jobId}/video`}
                onError={() => setVideoError(true)}
              />
            )}
          </section>
        </div>
      )}

      <footer className="dashboard-footer">
        Data source: <code>GET /events</code> on load, then{" "}
        <code>WS /ws/live</code> for real-time updates.
      </footer>
    </div>
  );
}

export default App;
