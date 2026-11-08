"""Unit tests for EventPublisher, with the MQTT client and HTTP session
fully mocked out -- no real broker or network calls involved.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.analytics.publisher import (
    EventPublisher,
    PublisherConfig,
    configure_mqtt_client,
)
from src.analytics.schemas import (
    BoundingBox,
    EventType,
    EvidenceReadyMessage,
    EvidenceRef,
    VisionEvent,
)


def make_event() -> VisionEvent:
    return VisionEvent(
        camera_id="cam1",
        site_id="site1",
        event_type=EventType.LINE_CROSSING,
        track_id=7,
        confidence=0.75,
        bbox=BoundingBox(x1=1, y1=2, x2=30, y2=40),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"line": "entrance_line"},
    )


def test_publish_sends_to_mqtt_with_configured_topic_and_qos() -> None:
    fake_mqtt = MagicMock()
    config = PublisherConfig(mqtt_topic="analytics/events", mqtt_qos=1)
    publisher = EventPublisher(config, mqtt_client=fake_mqtt)

    publisher.publish(make_event())

    fake_mqtt.connect.assert_called_once()
    assert fake_mqtt.publish.call_count == 1
    _, kwargs = fake_mqtt.publish.call_args
    assert kwargs["qos"] == 1
    payload = json.loads(kwargs["payload"])
    assert payload["camera_id"] == "cam1"
    assert payload["site_id"] == "site1"
    assert payload["track_id"] == 7
    assert payload["event_type"] == "line_crossing"
    assert fake_mqtt.publish.call_args.args[0] == "analytics/events"


def test_publish_only_connects_to_mqtt_once_across_multiple_events() -> None:
    fake_mqtt = MagicMock()
    publisher = EventPublisher(PublisherConfig(), mqtt_client=fake_mqtt)

    publisher.publish(make_event())
    publisher.publish(make_event())
    publisher.publish(make_event())

    fake_mqtt.connect.assert_called_once()
    assert fake_mqtt.publish.call_count == 3


def test_webhook_is_not_called_when_not_configured() -> None:
    fake_mqtt = MagicMock()
    fake_session = MagicMock()
    publisher = EventPublisher(
        PublisherConfig(webhook_url=None), mqtt_client=fake_mqtt, http_session=fake_session
    )

    publisher.publish(make_event())

    fake_session.post.assert_not_called()


def test_webhook_is_posted_with_json_body_when_configured() -> None:
    fake_mqtt = MagicMock()
    fake_session = MagicMock()
    publisher = EventPublisher(
        PublisherConfig(webhook_url="https://example.invalid/hook"),
        mqtt_client=fake_mqtt,
        http_session=fake_session,
    )

    publisher.publish(make_event())

    fake_session.post.assert_called_once()
    _, kwargs = fake_session.post.call_args
    assert kwargs["headers"]["Content-Type"] == "application/json"
    payload = json.loads(kwargs["data"])
    assert payload["track_id"] == 7


def test_mqtt_failure_does_not_raise_and_still_attempts_webhook() -> None:
    fake_mqtt = MagicMock()
    fake_mqtt.connect.side_effect = ConnectionRefusedError("no broker")
    fake_session = MagicMock()
    publisher = EventPublisher(
        PublisherConfig(webhook_url="https://example.invalid/hook"),
        mqtt_client=fake_mqtt,
        http_session=fake_session,
    )

    # Must not raise even though the "broker" is unreachable.
    publisher.publish(make_event())

    fake_session.post.assert_called_once()


def test_publish_returns_false_when_mqtt_rejects_message() -> None:
    fake_mqtt = MagicMock()
    fake_mqtt.publish.return_value = SimpleNamespace(rc=1)
    publisher = EventPublisher(PublisherConfig(), mqtt_client=fake_mqtt)

    assert publisher.publish(make_event()) is False


def test_publish_returns_true_when_mqtt_accepts_message() -> None:
    fake_mqtt = MagicMock()
    fake_mqtt.publish.return_value = SimpleNamespace(rc=0)
    publisher = EventPublisher(PublisherConfig(), mqtt_client=fake_mqtt)

    assert publisher.publish(make_event()) is True


def test_webhook_failure_does_not_raise() -> None:
    fake_mqtt = MagicMock()
    fake_session = MagicMock()
    fake_session.post.side_effect = TimeoutError("webhook down")
    publisher = EventPublisher(
        PublisherConfig(webhook_url="https://example.invalid/hook"),
        mqtt_client=fake_mqtt,
        http_session=fake_session,
    )

    publisher.publish(make_event())  # must not raise


def test_mqtt_auth_and_tls_are_applied() -> None:
    fake_mqtt = MagicMock()
    config = PublisherConfig(
        mqtt_username="operator",
        mqtt_password="secret",
        mqtt_tls_ca="ca.pem",
    )

    configure_mqtt_client(fake_mqtt, config)

    fake_mqtt.username_pw_set.assert_called_once_with("operator", "secret")
    fake_mqtt.tls_set.assert_called_once_with(ca_certs="ca.pem")


def test_publish_evidence_ready_uses_event_topic() -> None:
    fake_mqtt = MagicMock()
    publisher = EventPublisher(PublisherConfig(mqtt_topic="analytics/events"), mqtt_client=fake_mqtt)
    evidence = EvidenceRef(
        snapshot_path="evidence/snapshot.jpg",
        clip_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        clip_end=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )
    message = EvidenceReadyMessage(
        event_id=make_event().event_id,
        site_id="site1",
        camera_id="cam1",
        evidence=evidence,
    )

    publisher.publish_evidence_ready(message)

    payload = json.loads(fake_mqtt.publish.call_args.kwargs["payload"])
    assert fake_mqtt.publish.call_args.args[0] == "analytics/events"
    assert payload["type"] == "evidence_ready"
