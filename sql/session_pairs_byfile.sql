-- session_pairs_byfile.sql
-- Pair CT and VIB sessions by base filename (without extension), plus load/condition/severity.
-- Example:
--   0Nm_BPFI_03.tdms (CT) <-> 0Nm_BPFI_03.mat (VIB)
-- file_key = filename without folder + without extension.

CREATE OR REPLACE VIEW iiot.session_pairs_byfile AS
WITH
ct AS (
  SELECT
    asset_id,
    load_nm,
    lower(condition::text) AS condition,
    severity::text AS severity,
    session_id AS ct_session_id,
    regexp_replace(
      regexp_replace(lower(source_file), '^.*[\\\/]', ''),  -- strip folders (/ or \)
      '\.[^.]+$', ''                                       -- strip extension
    ) AS file_key
  FROM iiot.sessions
  WHERE modality = 'current_temp'::iiot.modality_t
),
vib AS (
  SELECT
    asset_id,
    load_nm,
    lower(condition::text) AS condition,
    severity::text AS severity,
    session_id AS vib_session_id,
    regexp_replace(
      regexp_replace(lower(source_file), '^.*[\\\/]', ''),
      '\.[^.]+$', ''
    ) AS file_key
  FROM iiot.sessions
  WHERE modality = 'vibration'::iiot.modality_t
)
SELECT
  ct.asset_id,
  ct.load_nm,
  ct.condition,
  ct.severity,
  ct.file_key,
  ct.ct_session_id,
  vib.vib_session_id
FROM ct
JOIN vib
  ON vib.asset_id = ct.asset_id
 AND vib.load_nm  = ct.load_nm
 AND vib.condition = ct.condition
 AND vib.severity  = ct.severity
 AND vib.file_key  = ct.file_key
ORDER BY ct.asset_id, ct.load_nm, ct.condition, ct.severity, ct.file_key;
