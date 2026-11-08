/**
 * TypeScript mirror of the backend Pydantic models in
 * `src/analytics/schemas.py`. Keeping this in one place means the
 * dashboard agrees with the analytics pipeline on exactly what a
 * "vision event" looks like on the wire (the same JSON shape used for
 * MQTT, the webhook, `GET /events`, and `WS /ws/live`).
 */

/** Mirrors `EventType` (str Enum) in schemas.py. */
export type EventType =
  | "restricted_zone_entry"
  | "perimeter_intrusion"
  | "line_crossing"
  | "dwell_time_exceeded"
  | "loitering_detected"
  | "after_hours_presence"
  | "suspected_shoplifting";

export const EVENT_TYPES: EventType[] = [
  "restricted_zone_entry",
  "perimeter_intrusion",
  "line_crossing",
  "dwell_time_exceeded",
  "loitering_detected",
  "after_hours_presence",
  "suspected_shoplifting",
];

/** Mirrors `BoundingBox` in schemas.py: axis-aligned, pixel coordinates. */
export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Mirrors `EvidenceRef` in schemas.py. */
export interface EvidenceRef {
  snapshot_path: string;
  clip_path: string | null;
  clip_start: string;
  clip_end: string;
  snapshot_sha256: string | null;
  clip_sha256: string | null;
  created_at: string;
}

export interface EvidenceReadyMessage {
  type: "evidence_ready";
  event_id: string;
  site_id: string;
  camera_id: string;
  evidence: EvidenceRef;
}

/** Known rule metadata, with room for future rule-specific values. */
export interface EventMetadata {
  class_name?: string;
  zone?: string;
  severity?: string;
  line?: string;
  direction?: string;
  movement?: "entry" | "exit";
  occupancy_count?: number;
  occupancy_uncertain?: boolean;
  dwell_seconds?: number;
  loiter_seconds?: number;
  behavior?: string;
  merchandise_dwell_seconds?: number;
  checkout_observed?: boolean;
  exit_direction?: string;
  confidence_basis?: string;
  suppressed_count?: number;
  [key: string]: unknown;
}

/** Mirrors `VisionEvent` in schemas.py. */
export interface VisionEvent {
  event_id: string;
  camera_id: string;
  site_id: string;
  /** Known backend values are typed, while unknown future values remain renderable. */
  event_type: EventType | (string & {});
  track_id: number;
  confidence: number;
  bbox: BoundingBox;
  /** ISO-8601 timestamp string (pydantic `datetime` serialised to JSON). */
  timestamp: string;
  metadata: EventMetadata;
  evidence?: EvidenceRef | null;
}

/** Human-readable labels for the event-type badges/chart. */
export const EVENT_TYPE_LABELS: Record<EventType, string> = {
  restricted_zone_entry: "Restricted zone entry",
  perimeter_intrusion: "Perimeter intrusion",
  line_crossing: "Line crossing",
  dwell_time_exceeded: "Dwell time exceeded",
  loitering_detected: "Loitering detected",
  after_hours_presence: "After-hours presence",
  suspected_shoplifting: "Suspected shoplifting",
};
