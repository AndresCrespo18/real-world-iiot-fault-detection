-- views_for_training_v4.sql
-- (Opcional) Vistas recomendadas para CT 1Hz y VIB wide full.
-- Si ya las tienes, no pasa nada: CREATE OR REPLACE.

-- CT a 1Hz
CREATE OR REPLACE VIEW iiot.current_temp_1s AS
SELECT
  asset_id,
  session_id,
  load_nm,
  condition,
  severity,
  date_trunc('second', ts) AS ts,
  AVG(temp_a_c) AS temp_a_c,
  AVG(temp_b_c) AS temp_b_c,
  AVG(i_u_a)    AS i_u_a,
  AVG(i_v_a)    AS i_v_a,
  AVG(i_w_a)    AS i_w_a
FROM iiot.current_temp_samples
GROUP BY asset_id, session_id, load_nm, condition, severity, date_trunc('second', ts);
