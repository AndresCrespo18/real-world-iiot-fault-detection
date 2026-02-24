-- aligned_ct_vib_views_v3.sql
-- GLOBAL unique run_idx + CT<->VIB pairing per (asset, load, condition, severity, rep_idx).
-- Depends on:
--   - iiot.sessions(modality iiot.modality_t)
--   - iiot.current_temp_1s
--   - iiot.vibration_1s_wide_full

BEGIN;

DROP VIEW IF EXISTS iiot.aligned_ct_vib_summary_v2;
DROP VIEW IF EXISTS iiot.aligned_ct_vib_1s_v2;
DROP VIEW IF EXISTS iiot.ct_sec_v2;
DROP VIEW IF EXISTS iiot.vib_sec_v2;
DROP VIEW IF EXISTS iiot.session_runs_v2;

CREATE OR REPLACE VIEW iiot.session_runs_v2 AS
WITH
ct_sessions AS (
  SELECT
    s.asset_id,
    s.load_nm,
    s.condition::text AS condition,
    s.severity::text  AS severity,
    s.session_id      AS ct_session_id,
    MIN(ct.ts)        AS ct_start_ts,
    MAX(ct.ts)        AS ct_end_ts
  FROM iiot.sessions s
  JOIN iiot.current_temp_1s ct
    ON ct.session_id = s.session_id
  WHERE s.modality = 'current_temp'::iiot.modality_t
  GROUP BY s.asset_id, s.load_nm, s.condition::text, s.severity::text, s.session_id
),
vib_sessions AS (
  SELECT
    s.asset_id,
    s.load_nm,
    s.condition::text AS condition,
    s.severity::text  AS severity,
    s.session_id      AS vib_session_id,
    MIN(v.ts)         AS vib_start_ts,
    MAX(v.ts)         AS vib_end_ts
  FROM iiot.sessions s
  JOIN iiot.vibration_1s_wide_full v
    ON v.session_id = s.session_id
  WHERE s.modality = 'vibration'::iiot.modality_t
  GROUP BY s.asset_id, s.load_nm, s.condition::text, s.severity::text, s.session_id
),
ct_ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY asset_id, load_nm, condition, severity
      ORDER BY ct_start_ts, ct_session_id
    ) AS rep_idx
  FROM ct_sessions
),
vib_ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY asset_id, load_nm, condition, severity
      ORDER BY vib_start_ts, vib_session_id
    ) AS rep_idx
  FROM vib_sessions
),
paired AS (
  SELECT
    c.asset_id,
    c.load_nm,
    c.condition,
    c.severity,
    c.rep_idx,
    c.ct_session_id,
    v.vib_session_id,
    c.ct_start_ts,
    c.ct_end_ts,
    v.vib_start_ts,
    v.vib_end_ts
  FROM ct_ranked c
  JOIN vib_ranked v
    ON v.asset_id  = c.asset_id
   AND v.load_nm   = c.load_nm
   AND v.condition = c.condition
   AND v.severity  = c.severity
   AND v.rep_idx   = c.rep_idx
)
SELECT
  DENSE_RANK() OVER (ORDER BY asset_id, load_nm, condition, severity, rep_idx)::bigint AS run_idx,
  asset_id, load_nm, condition, severity, rep_idx,
  ct_session_id, vib_session_id,
  ct_start_ts, ct_end_ts,
  vib_start_ts, vib_end_ts
FROM paired;

CREATE OR REPLACE VIEW iiot.ct_sec_v2 AS
SELECT
  r.run_idx,
  r.asset_id,
  r.load_nm,
  r.condition,
  r.severity,
  (EXTRACT(EPOCH FROM (ct.ts - r.ct_start_ts)))::int AS sec_idx,
  ct.ts AS ct_ts,
  ct.temp_a_c,
  ct.temp_b_c,
  ct.i_u_a,
  ct.i_v_a,
  ct.i_w_a
FROM iiot.session_runs_v2 r
JOIN iiot.current_temp_1s ct
  ON ct.session_id = r.ct_session_id;

CREATE OR REPLACE VIEW iiot.vib_sec_v2 AS
SELECT
  r.run_idx,
  r.asset_id,
  r.load_nm,
  r.condition,
  r.severity,
  (EXTRACT(EPOCH FROM (v.ts - r.vib_start_ts)))::int AS sec_idx,
  v.ts AS vib_ts,
  v.vib_xa_p95, v.vib_ya_p95, v.vib_xb_p95, v.vib_yb_p95,
  v.vib_xa_max, v.vib_ya_max, v.vib_xb_max, v.vib_yb_max,
  v.vib_xa_rms, v.vib_ya_rms, v.vib_xb_rms, v.vib_yb_rms,
  v.vib_xa_kurt, v.vib_ya_kurt, v.vib_xb_kurt, v.vib_yb_kurt
FROM iiot.session_runs_v2 r
JOIN iiot.vibration_1s_wide_full v
  ON v.session_id = r.vib_session_id;

CREATE OR REPLACE VIEW iiot.aligned_ct_vib_1s_v2 AS
SELECT
  r.asset_id,
  r.load_nm,
  r.condition,
  r.severity,
  r.run_idx,
  c.sec_idx,
  v.vib_ts,
  c.ct_ts,
  v.vib_xa_p95, v.vib_ya_p95, v.vib_xb_p95, v.vib_yb_p95,
  v.vib_xa_max, v.vib_ya_max, v.vib_xb_max, v.vib_yb_max,
  v.vib_xa_rms, v.vib_ya_rms, v.vib_xb_rms, v.vib_yb_rms,
  v.vib_xa_kurt, v.vib_ya_kurt, v.vib_xb_kurt, v.vib_yb_kurt,
  c.temp_a_c, c.temp_b_c,
  c.i_u_a, c.i_v_a, c.i_w_a
FROM iiot.session_runs_v2 r
JOIN iiot.ct_sec_v2  c ON c.run_idx = r.run_idx
JOIN iiot.vib_sec_v2 v ON v.run_idx = r.run_idx AND v.sec_idx = c.sec_idx;

CREATE OR REPLACE VIEW iiot.aligned_ct_vib_summary_v2 AS
WITH
ct_stats AS (
  SELECT run_idx,
         COUNT(DISTINCT sec_idx)::bigint AS ct_seconds,
         MIN(sec_idx)::int AS ct_sec_min,
         MAX(sec_idx)::int AS ct_sec_max
  FROM iiot.ct_sec_v2
  GROUP BY run_idx
),
vib_stats AS (
  SELECT run_idx,
         COUNT(DISTINCT sec_idx)::bigint AS vib_seconds,
         MIN(sec_idx)::int AS vib_sec_min,
         MAX(sec_idx)::int AS vib_sec_max
  FROM iiot.vib_sec_v2
  GROUP BY run_idx
),
join_stats AS (
  SELECT run_idx,
         COUNT(DISTINCT sec_idx)::bigint AS n_seconds_joined,
         MIN(sec_idx)::int AS sec_min,
         MAX(sec_idx)::int AS sec_max
  FROM iiot.aligned_ct_vib_1s_v2
  GROUP BY run_idx
)
SELECT
  r.asset_id,
  r.load_nm,
  r.condition,
  r.severity,
  r.run_idx,
  r.rep_idx,
  r.ct_session_id,
  r.vib_session_id,
  c.ct_seconds,
  v.vib_seconds,
  j.n_seconds_joined,
  j.sec_min,
  j.sec_max,
  (j.sec_max - j.sec_min + 1) AS window_seconds,
  ROUND(100.0 * j.n_seconds_joined::numeric / NULLIF(LEAST(c.ct_seconds, v.vib_seconds),0), 2) AS pct_of_min,
  ROUND(100.0 * j.n_seconds_joined::numeric / NULLIF(GREATEST(c.ct_seconds, v.vib_seconds),0), 2) AS pct_of_max
FROM iiot.session_runs_v2 r
JOIN ct_stats  c ON c.run_idx = r.run_idx
JOIN vib_stats v ON v.run_idx = r.run_idx
JOIN join_stats j ON j.run_idx = r.run_idx;

COMMIT;
