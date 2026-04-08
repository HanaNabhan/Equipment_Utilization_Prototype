-- TimescaleDB initialization
-- Creates hypertable for equipment telemetry (time-series optimized)

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Main telemetry table
CREATE TABLE IF NOT EXISTS equipment_telemetry (
    ts                TIMESTAMPTZ      NOT NULL,
    frame_id          INTEGER          NOT NULL,
    equipment_id      TEXT             NOT NULL,
    equipment_class   TEXT             NOT NULL,
    current_state     TEXT             NOT NULL,   -- ACTIVE | INACTIVE
    current_activity  TEXT             NOT NULL,   -- DIGGING | SWINGING | DUMPING | WAITING | MOVING
    motion_source     TEXT,
    confidence        DOUBLE PRECISION,
    total_tracked_sec DOUBLE PRECISION,
    total_active_sec  DOUBLE PRECISION,
    total_idle_sec    DOUBLE PRECISION,
    utilization_pct   DOUBLE PRECISION,
    -- Kafka metadata
    kafka_offset      BIGINT,
    kafka_partition   INTEGER
);

-- Convert to hypertable (TimescaleDB time-series optimization)
SELECT create_hypertable(
    'equipment_telemetry', 'ts',
    if_not_exists => TRUE
);

-- Index for fast queries per machine
CREATE INDEX IF NOT EXISTS idx_eq_id_ts
    ON equipment_telemetry (equipment_id, ts DESC);

-- Index for activity queries
CREATE INDEX IF NOT EXISTS idx_activity
    ON equipment_telemetry (current_activity, ts DESC);

-- Processing status table
CREATE TABLE IF NOT EXISTS processing_status (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Continuous aggregate: 1-minute utilization rollup
CREATE MATERIALIZED VIEW IF NOT EXISTS utilization_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', ts) AS bucket,
    equipment_id,
    AVG(utilization_pct)        AS avg_util,
    MAX(total_active_sec)       AS active_sec,
    MAX(total_idle_sec)         AS idle_sec,
    COUNT(*)                    AS frames
FROM equipment_telemetry
GROUP BY bucket, equipment_id
WITH NO DATA;

-- Auto-refresh the aggregate
SELECT add_continuous_aggregate_policy('utilization_1min',
    start_offset  => INTERVAL '10 minutes',
    end_offset    => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

COMMENT ON TABLE equipment_telemetry IS
    'Per-frame equipment telemetry from CV pipeline via Kafka';
