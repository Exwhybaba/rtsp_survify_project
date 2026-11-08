
"""Wires ingest -> detect -> track -> zone-rules -> publish -> visualize."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt
import yaml

from src.analytics.detector import Detector, YoloV8Detector
from src.analytics.evidence import EvidenceCapture
from src.analytics.ingest import VideoSource, redact_uri
from src.analytics.publisher import (
    EventPublisher,
    PublisherConfig,
    configure_mqtt_client,
    create_mqtt_client,
)
from src.analytics.rules.engine import ZoneRuleEngine
from src.analytics.schemas import OccupancyControlCommand
from src.analytics.shoplifting import ShopliftingAnalyzer, ShopliftingConfig
from src.analytics.tracker import ByteTrackAdapter
from src.analytics.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

OCCUPANCY_CONTROL_TOPIC = os.environ.get(
    "MQTT_OCCUPANCY_CONTROL_TOPIC",
    "analytics/control/occupancy",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load pipeline settings from a YAML file."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_detector(config: dict[str, Any]) -> Detector:
    """Construct the configured object detector."""
    det_cfg = config.get("detector", {})
    classes = det_cfg.get("classes_of_interest")

    return YoloV8Detector(
        model_path=det_cfg.get("model_path", "yolov8n.pt"),
        confidence_threshold=det_cfg.get(
            "confidence_threshold", 0.4
        ),
        classes_of_interest=set(classes) if classes else None,
        device=det_cfg.get("device", "cpu"),
        image_size=int(det_cfg.get("image_size", 640)),
    )


def build_tracker(config: dict[str, Any]) -> ByteTrackAdapter:
    """Build the ByteTrack adapter using YAML configuration."""
    trk_cfg = config.get("tracker", {})
    tracker_type = trk_cfg.get("type", "bytetrack").lower()

    if tracker_type != "bytetrack":
        raise ValueError(
            f"Unsupported tracker type: {tracker_type}. "
            "This pipeline currently supports bytetrack only."
        )

    return ByteTrackAdapter(
        track_activation_threshold=trk_cfg.get(
            "track_activation_threshold", 0.25
        ),
        lost_track_buffer=trk_cfg.get("lost_track_buffer", 30),
        minimum_consecutive_frames=trk_cfg.get(
            "minimum_consecutive_frames", 1
        ),
        high_conf_det_threshold=trk_cfg.get(
            "high_conf_det_threshold", 0.5
        ),
        minimum_iou_threshold=trk_cfg.get(
            "minimum_iou_threshold", 0.1
        ),

        relink_max_gap_frames=int(
            trk_cfg.get("relink_max_gap_frames", 15)
        ),
        relink_distance_factor=float(
            trk_cfg.get("relink_distance_factor", 1.5)
        ),
    )


def build_publisher(config: dict[str, Any]) -> EventPublisher:
    """Build the event publisher."""
    pub_cfg = config.get("publisher", {})

    return EventPublisher(
        PublisherConfig(
            mqtt_host=os.environ.get(
                "MQTT_HOST",
                pub_cfg.get("mqtt_host", "localhost"),
            ),
            mqtt_port=int(
                os.environ.get(
                    "MQTT_PORT",
                    pub_cfg.get("mqtt_port", 1883),
                )
            ),
            mqtt_topic=os.environ.get(
                "MQTT_TOPIC",
                pub_cfg.get("mqtt_topic", "analytics/events"),
            ),
            mqtt_qos=pub_cfg.get("mqtt_qos", 1),
            webhook_url=(
                os.environ.get("WEBHOOK_URL")
                or pub_cfg.get("webhook_url")
                or None
            ),
            mqtt_username=os.environ.get(
                "MQTT_USERNAME",
                pub_cfg.get("mqtt_username"),
            ),
            mqtt_password=os.environ.get(
                "MQTT_PASSWORD",
                pub_cfg.get("mqtt_password"),
            ),
            mqtt_tls_ca=os.environ.get(
                "MQTT_TLS_CA",
                pub_cfg.get("mqtt_tls_ca"),
            ),
        )
    )


def run_pipeline(
    config_path: str | Path,
    max_frames: int | None = None,
    source_override: str | None = None,
    detector: Detector | None = None,
    source: VideoSource | None = None,
    publisher: EventPublisher | None = None,
    evidence: EvidenceCapture | None = None,
    stop_file: str | Path | None = None,
    started_file: str | Path | None = None,
) -> None:
    """Run detection, tracking, rules, publishing and visualisation."""
    config = load_config(config_path)

    camera_id = config["camera"]["id"]
    source_uri = source_override or config["camera"].get("source")
    if not isinstance(source_uri, str) or not source_uri.strip():
        raise ValueError("No video source configured. Pass --source or set camera.source.")
    source_uri = source_uri.strip()
    loop_video = bool(config["camera"].get("loop", False))

    detector = detector or build_detector(config)
    tracker = build_tracker(config)
    zone_engine = ZoneRuleEngine.from_config(config_path)
    site_id = str(config["camera"].get("site_id", "default")).strip() or "default"
    shoplifting = ShopliftingAnalyzer(
        ShopliftingConfig.from_config(config.get("shoplifting")),
        site_id,
    )
    publisher = publisher or build_publisher(config)

    visualizer = Visualizer(
        config.get("visualization", {})
    )

    control_config = PublisherConfig(
        mqtt_host=os.environ.get(
            "MQTT_HOST",
            config.get("publisher", {}).get("mqtt_host", "localhost"),
        ),
        mqtt_port=int(
            os.environ.get(
                "MQTT_PORT",
                config.get("publisher", {}).get("mqtt_port", 1883),
            )
        ),
        mqtt_username=os.environ.get(
            "MQTT_USERNAME",
            config.get("publisher", {}).get("mqtt_username"),
        ),
        mqtt_password=os.environ.get(
            "MQTT_PASSWORD",
            config.get("publisher", {}).get("mqtt_password"),
        ),
        mqtt_tls_ca=os.environ.get(
            "MQTT_TLS_CA",
            config.get("publisher", {}).get("mqtt_tls_ca"),
        ),
    )
    control_client: mqtt.Client | None = configure_mqtt_client(
        create_mqtt_client(f"vision-occupancy-{camera_id}"),
        control_config,
    )

    def handle_control(
        _client: mqtt.Client,
        _userdata: Any,
        message: mqtt.MQTTMessage,
    ) -> None:
        """Handle operator occupancy reconciliation commands."""
        try:
            command = OccupancyControlCommand.model_validate(
                json.loads(message.payload.decode("utf-8"))
            )

            if command.camera_id != camera_id:
                return

            status = zone_engine.reconcile_occupancy(
                camera_id,
                command.action,
            )

            logger.info(
                "Occupancy reconciled for camera=%s action=%s operator_id=%s count=%d",
                camera_id,
                command.action,
                command.operator_id or "unknown",
                status.count,
            )

        except (
            KeyError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
        ):
            logger.warning(
                "Ignoring invalid occupancy reconciliation command",
                exc_info=True,
            )

    control_client.on_message = handle_control

    try:
        control_client.connect(
            control_config.mqtt_host,
            control_config.mqtt_port,
        )

        control_client.subscribe(
            OCCUPANCY_CONTROL_TOPIC,
            qos=1,
        )

        control_client.loop_start()

    except (OSError, ValueError):
        logger.warning(
            "Could not enable occupancy reconciliation commands; "
            "analytics will continue without operator control",
            exc_info=True,
        )

        control_client = None

    logger.info(
        "Starting pipeline for camera_id=%s source=%s visualization=%s",
        camera_id,
        redact_uri(str(source_uri)),
        visualizer.enabled,
    )

    is_stream = str(source_uri).strip().lower().startswith(
        ("rtsp://", "rtsps://", "http://", "https://")
    )
    base_timestamp: float | None = None

    frame_count = 0
    evidence_owned = evidence is None
    if evidence is None:
        evidence_cfg = config.get("evidence", {})
        evidence = EvidenceCapture(
            evidence_dir=evidence_cfg.get("evidence_dir", "evidence"),
            site_id=site_id,
            camera_id=camera_id,
            source_uri=str(source_uri),
            pre_roll_seconds=float(evidence_cfg.get("pre_roll_seconds", 5.0)),
            post_roll_seconds=float(evidence_cfg.get("post_roll_seconds", 5.0)),
            fps_cap=float(evidence_cfg.get("fps_cap", 10.0)),
            min_event_confidence=float(
                evidence_cfg.get("min_event_confidence", 0.0)
            ),
            shoplifting_min_concealment_score=float(
                evidence_cfg.get("shoplifting_min_concealment_score", 0.0)
            ),
            enabled=bool(evidence_cfg.get("enabled", False)),
            on_evidence_ready=publisher.publish_evidence_ready,
        )

    try:
        active_source = source or VideoSource(source_uri, loop=loop_video)
        with active_source:
            source_started = False
            for bundle in active_source.frames():
                if stop_file is not None and Path(stop_file).is_file():
                    logger.info("Analysis stop requested for camera_id=%s", camera_id)
                    break
                if not source_started and started_file is not None:
                    Path(started_file).parent.mkdir(parents=True, exist_ok=True)
                    Path(started_file).touch()
                    source_started = True

                # 1. Detect objects in the original frame.
                detections = detector.detect(bundle.frame)
                event_time = bundle.timestamp
                if base_timestamp is None:
                    base_timestamp = bundle.timestamp
                if not is_stream:
                    event_time = base_timestamp + bundle.frame_index / active_source.fps
                try:
                    evidence.add_frame(bundle.frame, event_time)
                except Exception:  # noqa: BLE001 - pylint: disable=broad-exception-caught
                    logger.exception("Evidence frame capture failed")

                # 2. Assign persistent tracking IDs.
                tracked_objects = tracker.update(detections)

                # Live streams use the real clock. Files use video time
                # (frame index / fps), so dwell results don't depend on
                # how fast your computer processes the video.
                timestamp = datetime.fromtimestamp(
                    event_time,
                    tz=timezone.utc,
                )
                # 3. Evaluate configured analytics rules. Rule parameters
                # decide which object classes each rule accepts.
                rule_objects = [
                    obj for obj in tracked_objects
                    if obj.bbox[2] > obj.bbox[0]
                    and obj.bbox[3] > obj.bbox[1]
                ]

                events = zone_engine.evaluate(
                    camera_id=camera_id,
                    tracked_objects=rule_objects,
                    timestamp=timestamp,
                    active_track_ids=tracker.active_track_ids,
                )
                events.extend(
                    shoplifting.evaluate(
                        camera_id=camera_id,
                        tracked_objects=rule_objects,
                        timestamp=timestamp,
                        active_track_ids=tracker.active_track_ids,
                        frame=bundle.frame,
                    )
                )
                try:
                    evidence.on_events(events, bundle.frame, event_time)
                except Exception:  # noqa: BLE001 - pylint: disable=broad-exception-caught
                    logger.exception("Evidence event capture failed")

                # 4. Publish generated events.
                for event in events:
                    logger.info(
                        "Event generated: %s track_id=%s",
                        event.event_type.value,
                        event.track_id,
                    )

                    if publisher.publish(event) is False:
                        logger.error(
                            "MQTT event was not published event_type=%s track_id=%s",
                            event.event_type.value,
                            event.track_id,
                        )

                # 5. Retrieve current occupancy status.
                occupancy = zone_engine.get_occupancy_status(
                    camera_id
                )

                if occupancy.uncertain:
                    logger.warning(
                        "Occupancy uncertain for camera=%s. "
                        "Last known count=%d, expired occupant tracks=%s",
                        camera_id,
                        occupancy.count,
                        occupancy.uncertain_track_ids,
                    )

                # 6. Display the original frame with analytics overlays.
                # No second YOLO inference is performed here.
                should_continue = visualizer.show(
                    frame=bundle.frame,
                    tracked_objects=tracked_objects,
                    zones=zone_engine.zones,
                    lines=zone_engine.lines,
                    occupancy=occupancy,
                    events=events,
                    shoplifting_observations=shoplifting.latest_observations,
                )

                frame_count += 1

                if not should_continue:
                    logger.info(
                        "Visual monitoring stopped by operator."
                    )
                    break

                if (
                    max_frames is not None
                    and frame_count >= max_frames
                ):
                    break

    finally:
        visualizer.close()
        if evidence_owned:
            evidence.close()

        if control_client is not None:
            control_client.loop_stop()
            control_client.disconnect()

        publisher.close()

    logger.info(
        "Pipeline stopped after %d frames",
        frame_count,
    )


def main() -> None:
    """Parse command-line options and start the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the RTSP/YOLO video analytics pipeline."
    )

    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to the pipeline YAML config.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to process.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source override, independent of the YAML camera source.",
    )
    parser.add_argument(
        "--stop-file",
        default=None,
        help="Optional marker file used to request a graceful stop.",
    )
    parser.add_argument(
        "--started-file",
        default=None,
        help="Optional marker file created once the video source opens.",
    )

    args = parser.parse_args()

    run_pipeline(
        args.config,
        max_frames=args.max_frames,
        source_override=args.source,
        stop_file=args.stop_file,
        started_file=args.started_file,
    )


if __name__ == "__main__":
    main()
