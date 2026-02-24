-- schema.sql
-- --------------------------------------------------------------------
-- PostgreSQL schema for the Rotating Machine dataset (Mendeley DOI: 10.17632/ztmf3m7h5x.6)
-- Data types (raw time-series):
--   - Vibration:  Time + xA,yA,xB,yB   (unit: g)
--   - Acoustic:   Time + value        (unit: Pa)   [only load=0 in this dataset]
--   - Current/Temp: Time + TempA,TempB + U,V,W     (units: °C and A)
--
-- This schema is designed to:
--   1) Preserve the dataset structure faithfully (raw time-series tables)
--   2) Support ML training efficiently (1-second feature tables + vibration wide view)
--
-- Notes:
-- - In the paper, vibration/current/temp are sampled at 25.6 kHz; acoustic at 51.2 kHz.
-- - Normal runs are typically 120 s; faulty runs 60 s.
-- - Severity can be "03", "10", "30", "01", "05" OR "0583mg" (unbalance).
-- --------------------------------------------------------------------

BEGIN;

CREATE SCHEMA IF NOT EXISTS iiot;

-- -----------------------------
-- Enum types (strict vocabulary)
-- -----------------------------
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname='iiot' AND t.typname='condition_t'
  ) THEN
    CREATE TYPE iiot.condition_t AS ENUM ('normal','bpfi','bpfo','misalign','unbalance');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname='iiot' AND t.typname='modality_t'
  ) THEN
    CREATE TYPE iiot.modality_t AS ENUM ('current_temp','vibration','acoustic');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname='iiot' AND t.typname='axis_t'
  ) THEN
    CREATE TYPE iiot.axis_t AS ENUM ('xA','yA','xB','yB');
  END IF;
END $$;

-- -----------------------------
-- Assets
-- -----------------------------
CREATE TABLE IF NOT EXISTS iiot.assets (
  asset_id    SERIAL PRIMARY KEY,
  asset_code  TEXT NOT NULL UNIQUE
);

-- -----------------------------
-- Sessions (one per file)
-- -----------------------------
CREATE TABLE IF NOT EXISTS iiot.sessions (
  session_id   BIGSERIAL PRIMARY KEY,
  asset_id     INT NOT NULL REFERENCES iiot.assets(asset_id) ON DELETE CASCADE,

  modality     iiot.modality_t NOT NULL,
  source_file  TEXT NOT NULL,   -- store relative path or full path (recommended to avoid name collisions)
  file_ext     TEXT NOT NULL,   -- 'tdms' or 'mat'

  load_nm      INT NOT NULL CHECK (load_nm IN (0,2,4)),
  condition    iiot.condition_t NOT NULL,

  -- Examples: '03','10','30','01','05','0583mg' ; for normal you may store '00' or '-'
  severity     TEXT NOT NULL CHECK (severity ~ '^[0-9]+(mg)?$' OR severity IN ('00','-')),

  -- Sampling rate (Hz) for sanity/audit
  fs_hz        DOUBLE PRECISION NOT NULL CHECK (fs_hz > 0),

  -- Synthetic or real timeline (UTC recommended)
  start_ts     TIMESTAMPTZ NOT NULL,
  end_ts       TIMESTAMPTZ NOT NULL,
  CHECK (end_ts > start_ts),

  -- Prevent duplicates per asset + modality + source
  UNIQUE(asset_id, modality, source_file)
);

-- Helps enforce consistency in child tables while still allowing fast lookups
CREATE UNIQUE INDEX IF NOT EXISTS ux_sessions_sid_asset ON iiot.sessions(session_id, asset_id);
CREATE INDEX IF NOT EXISTS idx_sessions_asset_load_cond ON iiot.sessions(asset_id, load_nm, condition);
CREATE INDEX IF NOT EXISTS idx_sessions_asset_modality ON iiot.sessions(asset_id, modality);

-- --------------------------------------------------------------------
-- RAW TIME-SERIES TABLES (faithful representation of the files)
-- --------------------------------------------------------------------

-- Temperature + motor current (TDMS): Time, TempA, TempB, U, V, W
CREATE TABLE IF NOT EXISTS iiot.current_temp_raw (
  asset_id     INT NOT NULL,
  session_id   BIGINT NOT NULL,
  sample_idx   BIGINT NOT NULL,          -- 0..N-1
  t_s          DOUBLE PRECISION NOT NULL, -- time in seconds since session start

  temp_a_c     DOUBLE PRECISION,
  temp_b_c     DOUBLE PRECISION,
  i_u_a        DOUBLE PRECISION,
  i_v_a        DOUBLE PRECISION,
  i_w_a        DOUBLE PRECISION,

  source       TEXT NOT NULL DEFAULT 'DATASET',

  FOREIGN KEY (session_id, asset_id)
    REFERENCES iiot.sessions(session_id, asset_id) ON DELETE CASCADE,

  PRIMARY KEY (session_id, sample_idx)
);

CREATE INDEX IF NOT EXISTS idx_ctraw_asset_session ON iiot.current_temp_raw(asset_id, session_id);
CREATE INDEX IF NOT EXISTS idx_ctraw_session_time ON iiot.current_temp_raw(session_id, t_s);

-- Vibration (MAT): Time, xA, yA, xB, yB
CREATE TABLE IF NOT EXISTS iiot.vibration_raw (
  asset_id     INT NOT NULL,
  session_id   BIGINT NOT NULL,
  sample_idx   BIGINT NOT NULL,
  t_s          DOUBLE PRECISION NOT NULL, -- seconds since session start

  xA_g         DOUBLE PRECISION,
  yA_g         DOUBLE PRECISION,
  xB_g         DOUBLE PRECISION,
  yB_g         DOUBLE PRECISION,

  source       TEXT NOT NULL DEFAULT 'DATASET',

  FOREIGN KEY (session_id, asset_id)
    REFERENCES iiot.sessions(session_id, asset_id) ON DELETE CASCADE,

  PRIMARY KEY (session_id, sample_idx)
);

CREATE INDEX IF NOT EXISTS idx_vibraw_asset_session ON iiot.vibration_raw(asset_id, session_id);
CREATE INDEX IF NOT EXISTS idx_vibraw_session_time ON iiot.vibration_raw(session_id, t_s);

-- Acoustic (MAT): Time, value
-- (Paper notes: acoustic is only provided under 0 Nm due to brake noise at 2/4 Nm.)
CREATE TABLE IF NOT EXISTS iiot.acoustic_raw (
  asset_id     INT NOT NULL,
  session_id   BIGINT NOT NULL,
  sample_idx   BIGINT NOT NULL,
  t_s          DOUBLE PRECISION NOT NULL, -- seconds since session start

  value_pa     DOUBLE PRECISION,

  source       TEXT NOT NULL DEFAULT 'DATASET',

  FOREIGN KEY (session_id, asset_id)
    REFERENCES iiot.sessions(session_id, asset_id) ON DELETE CASCADE,

  PRIMARY KEY (session_id, sample_idx)
);

CREATE INDEX IF NOT EXISTS idx_acraw_asset_session ON iiot.acoustic_raw(asset_id, session_id);
CREATE INDEX IF NOT EXISTS idx_acraw_session_time ON iiot.acoustic_raw(session_id, t_s);

-- --------------------------------------------------------------------
-- ML-FRIENDLY 1-SECOND TABLES (optional but recommended)
-- These tables support fast training/inference with manageable size.
-- --------------------------------------------------------------------

-- Current/Temp aggregated per second (1 row per second)
-- Matches your training/dashboard expectation (ts + sensor columns).
CREATE TABLE IF NOT EXISTS iiot.current_temp_samples (
  ts          TIMESTAMPTZ NOT NULL,
  asset_id    INT NOT NULL,
  session_id  BIGINT NOT NULL,
  load_nm     INT NOT NULL CHECK (load_nm IN (0,2,4)),
  condition   iiot.condition_t NOT NULL,
  severity    TEXT NOT NULL CHECK (severity ~ '^[0-9]+(mg)?$' OR severity IN ('00','-')),

  temp_a_c    DOUBLE PRECISION,
  temp_b_c    DOUBLE PRECISION,
  i_u_a       DOUBLE PRECISION,
  i_v_a       DOUBLE PRECISION,
  i_w_a       DOUBLE PRECISION,

  source      TEXT NOT NULL DEFAULT 'DATASET',

  FOREIGN KEY (session_id, asset_id)
    REFERENCES iiot.sessions(session_id, asset_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ct_asset_ts ON iiot.current_temp_samples(asset_id, ts);
CREATE INDEX IF NOT EXISTS idx_ct_session_ts ON iiot.current_temp_samples(session_id, ts);

-- Vibration features per second per axis (4 axes)
CREATE TABLE IF NOT EXISTS iiot.vibration_1s_features (
  ts          TIMESTAMPTZ NOT NULL,
  asset_id    INT NOT NULL,
  session_id  BIGINT NOT NULL,
  load_nm     INT NOT NULL CHECK (load_nm IN (0,2,4)),
  condition   iiot.condition_t NOT NULL,
  severity    TEXT NOT NULL CHECK (severity ~ '^[0-9]+(mg)?$' OR severity IN ('00','-')),

  axis        iiot.axis_t NOT NULL,
  rms_g       DOUBLE PRECISION,
  max_g       DOUBLE PRECISION,
  p95_g       DOUBLE PRECISION,
  kurtosis    DOUBLE PRECISION,

  source      TEXT NOT NULL DEFAULT 'DATASET',

  FOREIGN KEY (session_id, asset_id)
    REFERENCES iiot.sessions(session_id, asset_id) ON DELETE CASCADE,

  -- One row per second per axis
  PRIMARY KEY (session_id, ts, axis)
);

CREATE INDEX IF NOT EXISTS idx_vib1s_asset_ts ON iiot.vibration_1s_features(asset_id, ts);
CREATE INDEX IF NOT EXISTS idx_vib1s_session_ts ON iiot.vibration_1s_features(session_id, ts);

-- Acoustic features per second
CREATE TABLE IF NOT EXISTS iiot.acoustic_1s_features (
  ts          TIMESTAMPTZ NOT NULL,
  asset_id    INT NOT NULL,
  session_id  BIGINT NOT NULL,
  load_nm     INT NOT NULL CHECK (load_nm IN (0,2,4)),
  condition   iiot.condition_t NOT NULL,
  severity    TEXT NOT NULL CHECK (severity ~ '^[0-9]+(mg)?$' OR severity IN ('00','-')),

  rms_pa      DOUBLE PRECISION,
  max_pa      DOUBLE PRECISION,
  p95_pa      DOUBLE PRECISION,
  kurtosis    DOUBLE PRECISION,

  source      TEXT NOT NULL DEFAULT 'DATASET',

  FOREIGN KEY (session_id, asset_id)
    REFERENCES iiot.sessions(session_id, asset_id) ON DELETE CASCADE,

  PRIMARY KEY (session_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_ac1s_asset_ts ON iiot.acoustic_1s_features(asset_id, ts);
CREATE INDEX IF NOT EXISTS idx_ac1s_session_ts ON iiot.acoustic_1s_features(session_id, ts);

-- --------------------------------------------------------------------
-- Wide view for vibration (names in lowercase to match downstream code)
-- --------------------------------------------------------------------
CREATE OR REPLACE VIEW iiot.vibration_1s_wide AS
SELECT
  ts, asset_id, session_id, load_nm, condition, severity,
  MAX(CASE WHEN axis='xA' THEN p95_g END) AS vib_xa_p95,
  MAX(CASE WHEN axis='yA' THEN p95_g END) AS vib_ya_p95,
  MAX(CASE WHEN axis='xB' THEN p95_g END) AS vib_xb_p95,
  MAX(CASE WHEN axis='yB' THEN p95_g END) AS vib_yb_p95,
  MAX(CASE WHEN axis='xA' THEN max_g END) AS vib_xa_max,
  MAX(CASE WHEN axis='yA' THEN max_g END) AS vib_ya_max,
  MAX(CASE WHEN axis='xB' THEN max_g END) AS vib_xb_max,
  MAX(CASE WHEN axis='yB' THEN max_g END) AS vib_yb_max
FROM iiot.vibration_1s_features
GROUP BY ts, asset_id, session_id, load_nm, condition, severity;

COMMIT;
