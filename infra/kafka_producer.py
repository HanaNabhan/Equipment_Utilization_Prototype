"""
infra/kafka_producer.py

"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC",     "equipment_telemetry")


class EquipmentProducer:
    """
    Wraps KafkaProducer with graceful fallback.
    If Kafka is unavailable, silently skips (pipeline continues).
    """

    def __init__(self, bootstrap: str = KAFKA_BOOTSTRAP,
                 topic: str = KAFKA_TOPIC):
        self._topic    = topic
        self._producer = None
        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers    = [bootstrap],
                value_serializer     = lambda v: json.dumps(v).encode("utf-8"),
                acks                 = 1,
                retries              = 3,
                linger_ms            = 10,     # batch for 10ms
                request_timeout_ms   = 5000,
            )
            log.info("Kafka producer connected: %s → %s", bootstrap, topic)
        except Exception as e:
            log.warning("Kafka unavailable (%s) — running without streaming", e)

    def send(self, payload: dict):
        """Send payload to Kafka. Silently skips if not connected."""
        if self._producer is None:
            return
        try:
            self._producer.send(self._topic, payload)
        except Exception as e:
            log.debug("Kafka send failed: %s", e)

    def flush(self):
        if self._producer:
            try:
                self._producer.flush(timeout=5)
            except Exception:
                pass

    def close(self):
        if self._producer:
            try:
                self._producer.close(timeout=5)
            except Exception:
                pass
