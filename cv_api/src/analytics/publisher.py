"""Event publishing: MQTT (always) and an optional webhook (HTTP POST).

Keeping this as its own module means the zone engine never touches
networking directly -- it just produces ``VisionEvent`` objects, and
whatever is configured to consume them (MQTT broker, webhook receiver,
both, or in tests, neither) is a deployment detail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import paho.mqtt.client as mqtt
import requests

from src.analytics.schemas import EvidenceReadyMessage, VisionEvent

logger = logging.getLogger(__name__)


@dataclass
class PublisherConfig:
    """Configuration for MQTT and optional webhook publishing."""

    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic: str = "analytics/events"
    mqtt_client_id: str = "vision-analytics-publisher"
    mqtt_qos: int = 1
    webhook_url: str | None = None
    webhook_timeout_seconds: float = 3.0
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    mqtt_tls_ca: str | None = None


def create_mqtt_client(client_id: str) -> mqtt.Client:
    """Create a paho client compatible with paho-mqtt 1.x and 2.x."""
    callback_api_version = getattr(mqtt, "CallbackAPIVersion", None)
    if callback_api_version is not None:
        return mqtt.Client(
            callback_api_version=callback_api_version.VERSION2,
            client_id=client_id,
        )  # pylint: disable=unexpected-keyword-arg
    return mqtt.Client(client_id=client_id)


def configure_mqtt_client(client: mqtt.Client, config: PublisherConfig) -> mqtt.Client:
    """Apply optional MQTT credentials and TLS settings to a client."""
    if config.mqtt_username is not None:
        client.username_pw_set(config.mqtt_username, config.mqtt_password)
    if config.mqtt_tls_ca:
        client.tls_set(ca_certs=config.mqtt_tls_ca)
    return client


class EventPublisher:
    """Publishes :class:`VisionEvent`s to MQTT and, optionally, a webhook.

    The MQTT client is connected lazily on first publish so constructing a
    publisher (e.g. in tests, with a stubbed client) never touches the
    network.
    """

    def __init__(
        self,
        config: PublisherConfig,
        mqtt_client: mqtt.Client | None = None,
        http_session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self._mqtt_client = mqtt_client
        self._http_session = http_session or requests.Session()
        self._connected = False
        logger.info(
            "MQTT publisher initialized host=%s port=%s topic=%s qos=%s",
            config.mqtt_host,
            config.mqtt_port,
            config.mqtt_topic,
            config.mqtt_qos,
        )

    def _ensure_mqtt_connected(self) -> mqtt.Client:
        if self._mqtt_client is None:
            self._mqtt_client = create_mqtt_client(self.config.mqtt_client_id)
        configure_mqtt_client(self._mqtt_client, self.config)
        if not self._connected:
            logger.info(
                "Connecting to MQTT broker %s:%s",
                self.config.mqtt_host,
                self.config.mqtt_port,
            )
            result = self._mqtt_client.connect(
                self.config.mqtt_host,
                self.config.mqtt_port,
            )
            if isinstance(result, int) and result != mqtt.MQTT_ERR_SUCCESS:
                raise mqtt.MQTTException(
                    f"MQTT connect returned error code {result}"
                )
            self._mqtt_client.loop_start()
            self._connected = True
            logger.info(
                "MQTT connected successfully to %s:%s",
                self.config.mqtt_host,
                self.config.mqtt_port,
            )
        return self._mqtt_client

    def publish(self, event: VisionEvent) -> bool:
        """Publish one event to MQTT, then (best-effort) to the webhook."""
        payload = json.dumps(event.to_wire_dict())
        published = self._publish_mqtt(payload, self._event_topic(event.site_id, event.camera_id))
        if self.config.webhook_url:
            self._publish_webhook(payload)
        return published

    def publish_evidence_ready(self, message: EvidenceReadyMessage) -> None:
        """Publish a completed evidence notification to the event topic."""
        payload = json.dumps(message.model_dump(mode="json"))
        self._publish_mqtt(payload, self._event_topic(message.site_id, message.camera_id))

    def _event_topic(self, site_id: str, camera_id: str) -> str:
        del site_id, camera_id
        return self.config.mqtt_topic.rstrip("/")

    def _publish_mqtt(self, payload: str, topic: str) -> bool:
        try:
            client = self._ensure_mqtt_connected()
            info = client.publish(
                topic,
                payload=payload,
                qos=self.config.mqtt_qos,
            )
            result = getattr(info, "rc", mqtt.MQTT_ERR_SUCCESS)
            if isinstance(result, int) and result != mqtt.MQTT_ERR_SUCCESS:
                self._connected = False
                logger.error(
                    "MQTT publish failed topic=%s event_payload_size=%d rc=%s",
                    topic,
                    len(payload),
                    result,
                )
                return False
            logger.info(
                "MQTT event published topic=%s qos=%s payload_size=%d",
                topic,
                self.config.mqtt_qos,
                len(payload),
            )
            return True
        except Exception as exc:  # noqa: BLE001 - network boundary
            self._connected = False
            logger.exception(
                "MQTT publish failed broker=%s:%s topic=%s error=%s",
                self.config.mqtt_host,
                self.config.mqtt_port,
                topic,
                exc,
            )
            return False

    def _publish_webhook(self, payload: str) -> None:
        try:
            self._http_session.post(
                self.config.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.config.webhook_timeout_seconds,
            )
        except Exception:  # noqa: BLE001 - pylint: disable=broad-exception-caught
            logger.exception("Failed to POST event to webhook %s", self.config.webhook_url)

    def close(self) -> None:
        """Stop the MQTT loop and disconnect when a client is connected."""

        if self._mqtt_client is not None and self._connected:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception:  # noqa: BLE001 - cleanup boundary
                logger.exception("MQTT publisher cleanup failed")
            self._connected = False
