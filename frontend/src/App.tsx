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
  type EvidenceRef,
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
// Note: browsers ignore the `download` attribute on cross-origin links, so
// downloads work best in the same-origin setup.
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
type ViewMode = "clip" | "snapshot";

function formatTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleTimeString([], { hour12: false });
}

function formatDateTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString([], {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function timeAgo(iso: string): string {
  const seconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (!Number.isFinite(seconds) || seconds < 0) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function formatConfidence(confidence: number): string {
  return confidence.toFixed(2);
}

function formatClipWindow(evidence?: EvidenceRef | null): string | null {
  if (!evidence) return null;
  const start = new Date(evidence.clip_start);
  const end = new Date(evidence.clip_end);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return null;
  const seconds = Math.max(0, Math.round((end.getTime() - start.getTime()) / 1000));
  const opts: Intl.DateTimeFormatOptions = {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  };
  return `${start.toLocaleTimeString([], opts)} – ${end.toLocaleTimeString([], opts)} (${seconds}s)`;
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

const snapshotUrl = (eventId: string) =>
  `${API_BASE}/api/events/${eventId}/evidence/snapshot`;
const clipUrl = (eventId: string) =>
  `${API_BASE}/api/events/${eventId}/evidence/clip`;

function downloadName(evt: VisionEvent, extension: string): string {
  return `${evt.event_type}_${evt.camera_id}_${evt.event_id.slice(0, 8)}.${extension}`;
}

type IconName = "play" | "download" | "image" | "close" | "film" | "clock";

function Icon({ name }: { name: IconName }) {
  const paths: Record<IconName, JSX.Element> = {
    play: <path d="M7 4.5v11l9-5.5-9-5.5z" fill="currentColor" stroke="none" />,
    download: (
      <>
        <path d="M10 3v9.5" />
        <path d="M6 9l4 4 4-4" />
        <path d="M4 16h12" />
      </>
    ),
    image: (
      <>
        <rect x="3" y="4" width="14" height="12" rx="1.5" />
        <circle cx="7.5" cy="8.5" r="1.2" />
        <path d="M3.5 14l4-3.5 3 2.5 2.5-2 3.5 3" />
      </>
    ),
    close: <path d="M5 5l10 10M15 5L5 15" />,
    film: (
      <>
        <rect x="3" y="4" width="14" height="12" rx="1.5" />
        <path d="M7 4v12M13 4v12M3 8.5h4M13 8.5h4M3 11.5h4M13 11.5h4" />
      </>
    ),
    clock: (
      <>
        <circle cx="10" cy="10" r="7" />
        <path d="M10 6v4.2l2.8 1.6" />
      </>
    ),
  };
  return (
    <svg
      className="icon"
      viewBox="0 0 20 20"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}

function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return (
    <div className="clock" aria-label="Current time">
      <span className="clock-time">{now.toLocaleTimeString([], { hour12: false })}</span>
      <span className="clock-date">
        {now.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" })}
      </span>
    </div>
  );
}

function ClipStatus({ event }: { event: VisionEvent }) {
  const evidence = event.evidence;
  if (event.metadata.evidence_capture_skipped === "below_evidence_threshold") {
    return <span className="chip chip-pending">Evidence threshold not met</span>;
  }
  if (!evidence) return <span className="chip chip-pending">Preparing</span>;
  if (!evidence.clip_path) return <span className="chip chip-pending">Clip processing</span>;
  return <span className="chip chip-ready">Clip ready</span>;
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
  const [noRecentFrames, setNoRecentFrames] = useState(false);
  const [startingJob, setStartingJob] = useState(false);
  const [clearingEvidence, setClearingEvidence] = useState(false);
  const [videoOpen, setVideoOpen] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [liveZoom, setLiveZoom] = useState(1);
  const [livePan, setLivePan] = useState({ x: 0, y: 0 });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("clip");
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set());
  const [clipErrors, setClipErrors] = useState<Set<string>>(new Set());
  const [unavailableSnapshots, setUnavailableSnapshots] = useState<Set<string>>(
    new Set(),
  );
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | undefined>(undefined);
  const pendingEvidenceRef = useRef<Record<string, EvidenceReadyMessage["evidence"]>>({});
  const liveDragRef = useRef<{ pointerId: number; x: number; y: number; panX: number; panY: number } | null>(null);

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
    setJobStatus("connecting");
    setJobId(null);
    setVideoError(false);
    setLiveZoom(1);
    setLivePan({ x: 0, y: 0 });
    setNoRecentFrames(false);
    setSelectedId(null);
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
        const frameAgeSeconds = typeof status.last_frame_at === "number"
          ? Date.now() / 1000 - status.last_frame_at
          : Number.POSITIVE_INFINITY;
        setNoRecentFrames(status.status === "running" && frameAgeSeconds > 8);
        if (["completed", "failed", "stopped"].includes(status.status)) {
          window.clearInterval(poll);
          if (status.status === "failed") {
            setJobError(sourceMode === "rtsp"
              ? "Camera disconnected or could not connect. Check the RTSP URL, credentials, network, and camera codec."
              : "Analysis job failed. Check the backend job log.");
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
      setClipErrors(new Set());
      setSelectedId(null);
      pendingEvidenceRef.current = {};
    } catch (error) {
      setJobError(error instanceof Error ? error.message : "Could not clear evidence.");
    } finally {
      setClearingEvidence(false);
    }
  }, []);

  const stopSourceJob = useCallback(async () => {
    if (!jobId || !["starting", "queued", "running", "stopping"].includes(jobStatus ?? "")) return;
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

  const selectEvent = useCallback((event: VisionEvent) => {
    setSelectedId(event.event_id);
    setViewMode("clip");
  }, []);

  const toggleType = useCallback((type: string) => {
    setActiveTypes((previous) => {
      const next = new Set(previous);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  }, []);

  const markSnapshotUnavailable = useCallback((eventId: string) => {
    setUnavailableSnapshots((previous) => new Set(previous).add(eventId));
  }, []);

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
          // /events returns oldest-first; the feed renders newest-first.
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

  useEffect(() => {
    if (!videoOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setVideoOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [videoOpen]);

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

  const filtered = useMemo(
    () =>
      activeTypes.size === 0
        ? events
        : events.filter((evt) => activeTypes.has(evt.event_type)),
    [events, activeTypes],
  );

  // Derived from live state, so the viewer updates when a clip finishes.
  const selected = useMemo(
    () => {
      const explicit = events.find((evt) => evt.event_id === selectedId);
      if (explicit) return explicit;
      const sourceEvents = jobId
        ? filtered.filter(
            (evt) => evt.camera_id === sourceCameraId && evt.site_id === sourceSiteId,
          )
        : filtered;
      return sourceEvents[0] ?? null;
    },
    [events, filtered, jobId, selectedId, sourceCameraId, sourceSiteId],
  );

  const clipReady = Boolean(selected?.evidence?.clip_path);
  const clipBroken = selected ? clipErrors.has(selected.event_id) : false;
  const snapshotOk = selected
    ? Boolean(selected.evidence) && !unavailableSnapshots.has(selected.event_id)
    : false;
  const showClip = viewMode === "clip" && clipReady && !clipBroken;
  const jobRunning = ["connecting", "starting", "queued", "running", "stopping", "finalizing"].includes(jobStatus ?? "");
  const jobCanStop = ["starting", "queued", "running", "stopping"].includes(jobStatus ?? "");
  const jobHasRecording = Boolean(jobId && ["completed", "stopped"].includes(jobStatus ?? ""));

  return (
    <div className="console">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>Suvirfy Monitoring</h1>
            <p>Live video analytics</p>
          </div>
        </div>
        <div className="topbar-stats">
          <div>
            <span className="stat-value">{events.length}</span>
            <span className="stat-label">Events</span>
          </div>
          <div>
            <span className="stat-value">
              {events.filter((evt) => evt.evidence?.clip_path).length}
            </span>
            <span className="stat-label">Clips</span>
          </div>
        </div>
        <div className="topbar-right">
          <Clock />
          <span className={`link-state link-${connectionState}`}>
            <span className="link-dot" aria-hidden="true" />
            {connectionState === "connected" && "Events live"}
            {connectionState === "connecting" && "Connecting to events"}
            {connectionState === "disconnected" && "Events offline, retrying"}
          </span>
        </div>
      </header>

      <div className="workspace">
        {/* ---------- Left rail: source + filters ---------- */}
        <aside className="rail" aria-label="Controls">
          <section className="rail-section" aria-label="Analysis source">
            <h2>Analysis source</h2>
            <div className="segmented" role="tablist">
              <button
                className={sourceMode === "upload" ? "active" : ""}
                disabled={jobRunning}
                onClick={() => setSourceMode("upload")}
                role="tab"
                aria-selected={sourceMode === "upload"}
                type="button"
              >
                Video file
              </button>
              <button
                className={sourceMode === "rtsp" ? "active" : ""}
                disabled={jobRunning}
                onClick={() => setSourceMode("rtsp")}
                role="tab"
                aria-selected={sourceMode === "rtsp"}
                type="button"
              >
                RTSP camera
              </button>
            </div>
            <div className="field-stack">
              <label>
                Camera ID
                <input disabled={jobRunning} value={sourceCameraId} onChange={(event) => setSourceCameraId(event.target.value)} />
              </label>
              <label>
                Site ID
                <input disabled={jobRunning} value={sourceSiteId} onChange={(event) => setSourceSiteId(event.target.value)} />
              </label>
              {sourceMode === "upload" ? (
                <label>
                  Video file
                  <input
                    type="file"
                  disabled={jobRunning}
                    accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm"
                    onChange={(event) => setSourceFile(event.target.files?.[0] ?? null)}
                  />
                </label>
              ) : (
                <label>
                  RTSP URL
                  <input
                    type="url"
                  disabled={jobRunning}
                    placeholder="rtsp://user:password@camera/stream"
                    value={rtspSource}
                    onChange={(event) => setRtspSource(event.target.value)}
                  />
                </label>
              )}
            </div>
            <div className="rail-actions">
              <button className="btn btn-primary" disabled={startingJob || jobRunning} onClick={startSourceJob} type="button">
                {startingJob ? "Starting…" : sourceMode === "rtsp" ? "Connect camera" : "Start monitoring"}
              </button>
              {jobId && jobCanStop && (
                <button className="btn btn-danger" onClick={stopSourceJob} type="button">
                  {sourceMode === "rtsp" ? "Disconnect" : "Stop analysis"}
                </button>
              )}
            </div>
            {(jobStatus || jobError) && (
              <div className="job-readout">
                {jobStatus && (
                  <span className={`job-status job-${jobStatus}`}>
                    <span className="link-dot" aria-hidden="true" />
                    {jobStatus === "failed" && sourceMode === "rtsp" ? "Disconnected" : `Job ${jobStatus}`}
                    {sourceMode === "upload" && sourceFile ? `: ${sourceFile.name}` : ""}
                  </span>
                )}
                {jobError && <span className="job-error">{jobError}</span>}
              </div>
            )}
            {jobId && ["completed", "stopped"].includes(jobStatus ?? "") && (
              <div className="recording">
                <h3>Full analysis recording</h3>
                <p>
                  The whole run with detections and alerts drawn on. For footage of one
                  incident, use its event clip.
                </p>
                <div className="rail-actions">
                  <button
                    className="btn btn-ghost"
                    onClick={() => {
                      setVideoError(false);
                      setVideoOpen(true);
                    }}
                    type="button"
                  >
                    <Icon name="play" />
                    Open recording
                  </button>
                  <a
                    className="btn btn-ghost"
                    href={`${API_BASE}/api/jobs/${jobId}/video`}
                    download={`analysis_${jobId.slice(0, 8)}.mp4`}
                  >
                    <Icon name="download" />
                    Download
                  </a>
                </div>
              </div>
            )}
          </section>

          <section className="rail-section" aria-label="Event filters">
            <div className="rail-title-row">
              <h2>Event types</h2>
              {activeTypes.size > 0 && (
                <button className="text-button" type="button" onClick={() => setActiveTypes(new Set())}>
                  Show all
                </button>
              )}
            </div>
            <ul className="filter-list">
              {EVENT_TYPES.map((type) => {
                const on = activeTypes.has(type);
                return (
                  <li key={type}>
                    <button
                      type="button"
                      className={on ? "filter on" : "filter"}
                      aria-pressed={on}
                      onClick={() => toggleType(type)}
                    >
                      <span className="swatch" style={{ background: EVENT_TYPE_COLORS[type] }} />
                      <span className="filter-name">{EVENT_TYPE_LABELS[type]}</span>
                      <span className="filter-count">{counts[type]}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        </aside>

        {/* ---------- Center: incident viewer + analysis ---------- */}
        <main className="stage">
          <section className="viewer" aria-label={jobRunning ? "Live analysis monitor" : "Selected incident"}>
            {jobRunning && jobId ? (
              <>
                <div className="viewer-head">
                  <div className="viewer-title">
                    <span className="swatch lg" style={{ background: "var(--live)" }} />
                    <div>
                      <h2>Live analysis</h2>
                      <p>{sourceCameraId} / {sourceSiteId} &middot; {jobStatus === "stopping" ? "Stopping" : jobStatus === "queued" || jobStatus === "connecting" || jobStatus === "starting" ? "Connecting / loading" : jobStatus === "finalizing" ? "Preparing playback" : noRecentFrames ? "No recent processed frames" : "Monitoring"}</p>
                    </div>
                  </div>
                  <span className={`job-status job-${jobStatus}`}>{jobStatus === "stopping" ? "Stopping" : jobStatus === "queued" || jobStatus === "connecting" || jobStatus === "starting" ? "Connecting / loading" : jobStatus === "finalizing" ? "Preparing playback" : noRecentFrames ? "No recent frames" : "Monitoring"}</span>
                </div>
                <div
                  className={`screen live-screen${liveZoom > 1 ? " is-zoomed" : ""}${liveDragRef.current ? " is-dragging" : ""}`}
                  onPointerDown={(event) => {
                    if (liveZoom <= 1) return;
                    event.preventDefault();
                    event.currentTarget.setPointerCapture(event.pointerId);
                    liveDragRef.current = {
                      pointerId: event.pointerId,
                      x: event.clientX,
                      y: event.clientY,
                      panX: livePan.x,
                      panY: livePan.y,
                    };
                    event.currentTarget.classList.add("is-dragging");
                  }}
                  onPointerMove={(event) => {
                    const drag = liveDragRef.current;
                    if (!drag || drag.pointerId !== event.pointerId) return;
                    const bounds = event.currentTarget.getBoundingClientRect();
                    const maxX = bounds.width * (liveZoom - 1) / 2;
                    const maxY = bounds.height * (liveZoom - 1) / 2;
                    const x = drag.panX + event.clientX - drag.x;
                    const y = drag.panY + event.clientY - drag.y;
                    setLivePan({
                      x: Math.max(-maxX, Math.min(maxX, x)),
                      y: Math.max(-maxY, Math.min(maxY, y)),
                    });
                  }}
                  onPointerUp={(event) => {
                    if (liveDragRef.current?.pointerId !== event.pointerId) return;
                    liveDragRef.current = null;
                    event.currentTarget.classList.remove("is-dragging");
                  }}
                  onPointerCancel={(event) => {
                    liveDragRef.current = null;
                    event.currentTarget.classList.remove("is-dragging");
                  }}
                >
                  <span className="corner tl" />
                  <span className="corner tr" />
                  <span className="corner bl" />
                  <span className="corner br" />
                  <img
                    className="live-frame"
                    src={`${API_BASE}/api/jobs/${jobId}/stream`}
                    alt={`Processed video from ${sourceCameraId}`}
                    draggable={false}
                    style={{ transform: `translate(${livePan.x}px, ${livePan.y}px) scale(${liveZoom})` }}
                  />
                  <span className="screen-tag mono">LIVE &middot; YOLO + ByteTrack</span>
                </div>
                <div className="viewer-bar live-zoom-controls" aria-label="Live view zoom controls">
                  <div className="viewer-bar-left">
                    <span className="zoom-hint">{liveZoom > 1 ? "Drag the image to reposition" : "Adjust the live view"}</span>
                  </div>
                  <div className="viewer-bar-right">
                    <button className="btn btn-ghost small zoom-button" type="button" aria-label="Zoom out" title="Zoom out" disabled={liveZoom <= 0.5} onClick={() => {
                      const nextZoom = Math.max(0.5, Math.round((liveZoom - 0.25) * 100) / 100);
                      setLiveZoom(nextZoom);
                      if (nextZoom <= 1) setLivePan({ x: 0, y: 0 });
                    }}>−</button>
                    <span className="zoom-level" aria-live="polite">{Math.round(liveZoom * 100)}%</span>
                    <button className="btn btn-ghost small zoom-button" type="button" aria-label="Zoom in" title="Zoom in" disabled={liveZoom >= 4} onClick={() => setLiveZoom((zoom) => Math.min(4, Math.round((zoom + 0.25) * 100) / 100))}>+</button>
                    <button className="btn btn-ghost small" type="button" disabled={liveZoom === 1 && livePan.x === 0 && livePan.y === 0} onClick={() => { setLiveZoom(1); setLivePan({ x: 0, y: 0 }); }}>Reset</button>
                  </div>
                </div>
                <p className="block-note">Showing frames after detection, tracking, and analytics from this source.</p>
              </>
            ) : !selected && jobHasRecording && jobId ? (
              <>
                <div className="viewer-head">
                  <div className="viewer-title">
                    <span className="swatch lg" style={{ background: "var(--accent)" }} />
                    <div>
                      <h2>Processed recording</h2>
                      <p>{sourceCameraId} / {sourceSiteId} &middot; {jobStatus === "stopped" ? "Stopped" : "Complete"}</p>
                    </div>
                  </div>
                  <a className="btn btn-ghost" href={`${API_BASE}/api/jobs/${jobId}/video`} download={`analysis_${jobId.slice(0, 8)}.mp4`}>
                    <Icon name="download" /> Download
                  </a>
                </div>
                {videoError ? (
                  <div className="screen-empty boxed">This recording could not be decoded by the browser. Try downloading it or check that FFmpeg is installed for H.264 conversion.</div>
                ) : (
                  <div className="screen">
                    <video
                      controls
                      playsInline
                      preload="metadata"
                      src={`${API_BASE}/api/jobs/${jobId}/video`}
                      onError={() => setVideoError(true)}
                    />
                  </div>
                )}
                <p className="block-note">This is the annotated output produced by the same analysis run. Use the controls to play, pause, or seek.</p>
              </>
            ) : selected ? (
              <>
                <div className="viewer-head">
                  <div className="viewer-title">
                    <span className="swatch lg" style={{ background: eventColor(selected.event_type) }} />
                    <div>
                      <h2>{eventLabel(selected.event_type)}</h2>
                      <p>
                        {selected.camera_id} / {selected.site_id} &middot;{" "}
                        <span className="mono">{formatDateTime(selected.timestamp)}</span>
                      </p>
                    </div>
                  </div>
                  <div className="segmented compact" role="tablist" aria-label="Evidence type">
                    <button
                      className={viewMode === "clip" ? "active" : ""}
                      onClick={() => setViewMode("clip")}
                      role="tab"
                      aria-selected={viewMode === "clip"}
                      type="button"
                    >
                      <Icon name="film" />
                      Event clip
                    </button>
                    <button
                      className={viewMode === "snapshot" ? "active" : ""}
                      onClick={() => setViewMode("snapshot")}
                      role="tab"
                      aria-selected={viewMode === "snapshot"}
                      type="button"
                    >
                      <Icon name="image" />
                      Snapshot
                    </button>
                  </div>
                </div>

                <div className="screen">
                  <span className="corner tl" />
                  <span className="corner tr" />
                  <span className="corner bl" />
                  <span className="corner br" />
                  {showClip ? (
                    <video
                      key={selected.event_id}
                      controls
                      playsInline
                      preload="metadata"
                      poster={snapshotOk ? snapshotUrl(selected.event_id) : undefined}
                      src={clipUrl(selected.event_id)}
                      onError={() =>
                        setClipErrors((previous) => new Set(previous).add(selected.event_id))
                      }
                    />
                  ) : snapshotOk ? (
                    <img
                      src={snapshotUrl(selected.event_id)}
                      alt={`Snapshot of ${eventLabel(selected.event_type)} on ${selected.camera_id}`}
                      onError={() => markSnapshotUnavailable(selected.event_id)}
                    />
                  ) : (
                    <div className="screen-empty">
                      {selected.evidence ? "Snapshot not available." : "Evidence is being generated…"}
                    </div>
                  )}
                  {viewMode === "clip" && !showClip && (
                    <div className="screen-note">
                      {clipBroken
                        ? "This browser can't play the clip. Download it to watch in another player."
                        : "Clip is still processing. This view updates when it is ready."}
                    </div>
                  )}
                  <span className="screen-tag mono">
                    {selected.camera_id} &middot; TRK {selected.track_id}
                  </span>
                </div>

                <div className="viewer-bar">
                  <div className="viewer-bar-left">
                    <ClipStatus event={selected} />
                    <span className="clip-window mono">
                      <Icon name="clock" />
                      {formatClipWindow(selected.evidence) ?? "Time range pending"}
                    </span>
                  </div>
                  <div className="viewer-bar-right">
                    {clipReady && (
                      <a
                        className="btn btn-primary"
                        href={clipUrl(selected.event_id)}
                        download={downloadName(selected, "mp4")}
                      >
                        <Icon name="download" />
                        Download clip
                      </a>
                    )}
                    {snapshotOk && (
                      <a
                        className="btn btn-ghost"
                        href={snapshotUrl(selected.event_id)}
                        download={downloadName(selected, "jpg")}
                      >
                        <Icon name="download" />
                        Snapshot
                      </a>
                    )}
                  </div>
                </div>

                <dl className="facts">
                  <div>
                    <dt>Object</dt>
                    <dd>
                      {typeof selected.metadata.class_name === "string"
                        ? selected.metadata.class_name
                        : "Unknown"}
                    </dd>
                  </div>
                  <div>
                    <dt>Track ID</dt>
                    <dd className="mono">{selected.track_id}</dd>
                  </div>
                  <div>
                    <dt>{selected.event_type === "suspected_shoplifting" ? "Concealment score" : "Confidence"}</dt>
                    <dd className="mono">{formatConfidence(selected.confidence)}</dd>
                  </div>
                  <div>
                    <dt>Zone</dt>
                    <dd>
                      {typeof selected.metadata.zone === "string" ? selected.metadata.zone : "None"}
                    </dd>
                  </div>
                  <div>
                    <dt>Severity</dt>
                    <dd>
                      {typeof selected.metadata.severity === "string"
                        ? selected.metadata.severity
                        : "Not set"}
                    </dd>
                  </div>
                  <div>
                    <dt>Event ID</dt>
                    <dd className="mono">{selected.event_id.slice(0, 8)}</dd>
                  </div>
                </dl>
              </>
            ) : (
              <div className="viewer-empty">
                <div className="screen idle">
                  <span className="corner tl" />
                  <span className="corner tr" />
                  <span className="corner bl" />
                  <span className="corner br" />
                  <div className="screen-empty">
                    <strong>No incidents yet</strong>
                    <span>
                      Start an analysis from the panel on the left. Incidents appear in the
                      feed as they are detected, and the newest one opens here.
                    </span>
                  </div>
                </div>
              </div>
            )}
          </section>

          <section className="block" aria-label="Events by type">
            <div className="block-head">
              <h2>Events by type</h2>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2430" />
                <XAxis type="number" allowDecimals={false} stroke="#7d8699" />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={170}
                  stroke="#7d8699"
                  tick={{ fill: "#dfe3ea", fontSize: 12 }}
                />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  contentStyle={{
                    background: "#11141a",
                    border: "1px solid #2a3140",
                    borderRadius: 6,
                    color: "#dfe3ea",
                  }}
                />
                <Bar dataKey="count" radius={[0, 3, 3, 0]}>
                  {chartData.map((entry) => (
                    <Cell key={entry.type} fill={EVENT_TYPE_COLORS[entry.type]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </section>

          <section className="block" aria-label="Event log">
            <div className="block-head">
              <h2>Event log</h2>
              <span className="block-note">{filtered.length} shown</span>
            </div>
            {filtered.length === 0 ? (
              <div className="empty-state">Waiting for events…</div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Camera</th>
                      <th>Event</th>
                      <th>Track</th>
                      <th>Confidence</th>
                      <th>Evidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((evt) => (
                      <tr
                        key={evt.event_id}
                        className={selected?.event_id === evt.event_id ? "row-active" : ""}
                      >
                        <td className="mono">{formatTime(evt.timestamp)}</td>
                        <td>{evt.camera_id}</td>
                        <td>
                          <span className="log-type">
                            <span className="swatch" style={{ background: eventColor(evt.event_type) }} />
                            {eventLabel(evt.event_type)}
                          </span>
                        </td>
                        <td className="mono">{evt.track_id}</td>
                        <td className="mono">{formatConfidence(evt.confidence)}</td>
                        <td>
                          <button
                            className="text-button"
                            type="button"
                            onClick={() => {
                              selectEvent(evt);
                              window.scrollTo({ top: 0, behavior: "smooth" });
                            }}
                          >
                            {evt.evidence?.clip_path ? "Play clip" : "Open"}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </main>

        {/* ---------- Right: incident feed ---------- */}
        <aside className="feed" aria-label="Incident feed">
          <div className="feed-head">
            <div>
              <h2>Incident feed</h2>
              <p>
                {activeTypes.size > 0 ? `${filtered.length} of ${events.length}` : events.length} incidents, newest first
              </p>
            </div>
            <button
              className="btn btn-danger small"
              disabled={clearingEvidence || events.length === 0}
              onClick={clearEvidence}
              type="button"
            >
              {clearingEvidence ? "Clearing…" : "Clear"}
            </button>
          </div>
          {filtered.length === 0 ? (
            <div className="empty-state">
              {events.length === 0 ? "No incidents yet." : "No incidents match the filter."}
            </div>
          ) : (
            <ul className="feed-list">
              {filtered.slice(0, 40).map((evt) => {
                const ready = Boolean(evt.evidence?.clip_path);
                const thumbOk = evt.evidence && !unavailableSnapshots.has(evt.event_id);
                const objectClass =
                  typeof evt.metadata.class_name === "string" ? evt.metadata.class_name : "object";
                return (
                  <li key={evt.event_id}>
                    <button
                      type="button"
                      className={selected?.event_id === evt.event_id ? "feed-item active" : "feed-item"}
                      style={{ borderLeftColor: eventColor(evt.event_type) }}
                      onClick={() => selectEvent(evt)}
                    >
                      <span className="thumb">
                        {thumbOk ? (
                          <img
                            src={snapshotUrl(evt.event_id)}
                            alt=""
                            loading="lazy"
                            onError={() => markSnapshotUnavailable(evt.event_id)}
                          />
                        ) : (
                          <span className="thumb-empty" />
                        )}
                        {ready && (
                          <span className="thumb-play" aria-hidden="true">
                            <Icon name="play" />
                          </span>
                        )}
                      </span>
                      <span className="feed-text">
                        <span className="feed-type">{eventLabel(evt.event_type)}</span>
                        <span className="feed-sub">
                          {evt.camera_id} &middot; {objectClass} #{evt.track_id}
                        </span>
                        <span className="feed-foot">
                          <span className="mono">{formatTime(evt.timestamp)}</span>
                          <span className="feed-ago">{timeAgo(evt.timestamp)}</span>
                          <span className={ready ? "dot-ready" : "dot-pending"}>
                            {ready ? "Clip" : "Processing"}
                          </span>
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </aside>
      </div>

      {videoOpen && jobId && (
        <div className="modal-backdrop" role="presentation" onClick={() => setVideoOpen(false)}>
          <section
            aria-label="Full analysis recording player"
            aria-modal="true"
            role="dialog"
            className="modal"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-head">
              <div>
                <h2>Full analysis recording</h2>
                <p>The whole run with overlays. Incident footage is in the event clips.</p>
              </div>
              <button
                aria-label="Close video player"
                className="icon-button"
                onClick={() => setVideoOpen(false)}
                type="button"
              >
                <Icon name="close" />
              </button>
            </div>
            {videoError ? (
              <div className="screen-empty boxed">
                This video could not be decoded by the browser. The generated MP4
                may use an unsupported codec; install FFmpeg and encode the output
                as H.264/AAC for browser playback.
              </div>
            ) : (
              <video
                controls
                playsInline
                preload="metadata"
                src={`${API_BASE}/api/jobs/${jobId}/video`}
                onError={() => setVideoError(true)}
              />
            )}
            <div className="modal-foot">
              <a
                className="btn btn-primary"
                href={`${API_BASE}/api/jobs/${jobId}/video`}
                download={`analysis_${jobId.slice(0, 8)}.mp4`}
              >
                <Icon name="download" />
                Download recording
              </a>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;
