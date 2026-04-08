"""
infra/kafka_consumer.py
─────────────────────────────────────────────────────────────
Kafka → TimescaleDB consumer.

Reads equipment telemetry payloads from Kafka topic
and writes them to TimescaleDB (PostgreSQL).

Run standalone (outside Docker):
    python infra/kafka_consumer.py

Environment variables:
    KAFKA_BOOTSTRAP   Kafka broker address (default: localhost:9092)
    TIMESCALE_DSN     PostgreSQL DSN (default: sqlite fallback)
    KAFKA_TOPIC       Topic name (default: equipment_telemetry)
"""
from __future__ import annotations

import json
import logging
import os
import time

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP",  "localhost:9092")
TIMESCALE_DSN   = os.getenv("TIMESCALE_DSN",    "")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC",       "equipment_telemetry")
KAFKA_GROUP     = os.getenv("KAFKA_GROUP",        "equipment_consumer")


def get_pg_conn():
    import psycopg2
    return psycopg2.connect(TIMESCALE_DSN)


def insert_row(conn, payload: dict, offset: int, partition: int):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO equipment_telemetry (
                ts, frame_id, equipment_id, equipment_class,
                current_state, current_activity, motion_source,
                confidence, total_tracked_sec, total_active_sec,
                total_idle_sec, utilization_pct,
                kafka_offset, kafka_partition
            ) VALUES (
                %(ts)s, %(frame_id)s, %(equipment_id)s, %(equipment_class)s,
                %(current_state)s, %(current_activity)s, %(motion_source)s,
                %(confidence)s, %(total_tracked_sec)s, %(total_active_sec)s,
                %(total_idle_sec)s, %(utilization_pct)s,
                %(kafka_offset)s, %(kafka_partition)s
            )
        """, {
            "ts":               payload.get("timestamp"),
            "frame_id":         payload.get("frame_id"),
            "equipment_id":     payload.get("equipment_id"),
            "equipment_class":  payload.get("equipment_class"),
            "current_state":    payload["utilization"]["current_state"],
            "current_activity": payload["utilization"]["current_activity"],
            "motion_source":    payload["utilization"].get("motion_source"),
            "confidence":       payload["bbox"].get("confidence"),
            "total_tracked_sec":payload["time_analytics"]["total_tracked_seconds"],
            "total_active_sec": payload["time_analytics"]["total_active_seconds"],
            "total_idle_sec":   payload["time_analytics"]["total_idle_seconds"],
            "utilization_pct":  payload["time_analytics"]["utilization_percent"],
            "kafka_offset":     offset,
            "kafka_partition":  partition,
        })
    conn.commit()


def run():
    from kafka import KafkaConsumer

    log.info("Connecting to Kafka: %s  topic: %s", KAFKA_BOOTSTRAP, KAFKA_TOPIC)
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers = [KAFKA_BOOTSTRAP],
        group_id          = KAFKA_GROUP,
        auto_offset_reset = "earliest",
        value_deserializer= lambda b: json.loads(b.decode("utf-8")),
        enable_auto_commit= True,
    )

    if TIMESCALE_DSN:
        log.info("Connecting to TimescaleDB: %s", TIMESCALE_DSN[:40])
        pg = get_pg_conn()
    else:
        log.warning("No TIMESCALE_DSN — printing payloads only")
        pg = None

    log.info("Consumer running. Waiting for messages...")
    for msg in consumer:
        try:
            payload = msg.value
            log.debug("Received: %s frame=%s activity=%s",
                      payload.get("equipment_id"),
                      payload.get("frame_id"),
                      payload.get("utilization", {}).get("current_activity"))
            if pg:
                insert_row(pg, payload, msg.offset, msg.partition)
        except Exception as e:
            log.error("Failed to process message: %s", e)


if __name__ == "__main__":
    run()
