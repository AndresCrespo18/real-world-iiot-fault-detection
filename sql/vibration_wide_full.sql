-- vibration_wide_full.sql
-- Extiende el wide de vibración para incluir también RMS y Kurtosis por eje.
-- No rompe tu dashboard: crea una NUEVA vista (recomendada para entrenar).

CREATE OR REPLACE VIEW iiot.vibration_1s_wide_full AS
SELECT
  date_trunc('second', ts) AS ts,
  asset_id,
  session_id,
  load_nm,
  condition,
  severity,

  MAX(CASE WHEN axis='xA'::iiot.axis_t THEN p95_g END) AS vib_xa_p95,
  MAX(CASE WHEN axis='yA'::iiot.axis_t THEN p95_g END) AS vib_ya_p95,
  MAX(CASE WHEN axis='xB'::iiot.axis_t THEN p95_g END) AS vib_xb_p95,
  MAX(CASE WHEN axis='yB'::iiot.axis_t THEN p95_g END) AS vib_yb_p95,

  MAX(CASE WHEN axis='xA'::iiot.axis_t THEN max_g END) AS vib_xa_max,
  MAX(CASE WHEN axis='yA'::iiot.axis_t THEN max_g END) AS vib_ya_max,
  MAX(CASE WHEN axis='xB'::iiot.axis_t THEN max_g END) AS vib_xb_max,
  MAX(CASE WHEN axis='yB'::iiot.axis_t THEN max_g END) AS vib_yb_max,

  MAX(CASE WHEN axis='xA'::iiot.axis_t THEN rms_g END) AS vib_xa_rms,
  MAX(CASE WHEN axis='yA'::iiot.axis_t THEN rms_g END) AS vib_ya_rms,
  MAX(CASE WHEN axis='xB'::iiot.axis_t THEN rms_g END) AS vib_xb_rms,
  MAX(CASE WHEN axis='yB'::iiot.axis_t THEN rms_g END) AS vib_yb_rms,

  MAX(CASE WHEN axis='xA'::iiot.axis_t THEN kurtosis END) AS vib_xa_kurt,
  MAX(CASE WHEN axis='yA'::iiot.axis_t THEN kurtosis END) AS vib_ya_kurt,
  MAX(CASE WHEN axis='xB'::iiot.axis_t THEN kurtosis END) AS vib_xb_kurt,
  MAX(CASE WHEN axis='yB'::iiot.axis_t THEN kurtosis END) AS vib_yb_kurt

FROM iiot.vibration_1s_features
WHERE source='DATASET'
GROUP BY date_trunc('second', ts), asset_id, session_id, load_nm, condition, severity;
