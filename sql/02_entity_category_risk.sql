-- Fraud history per link entity (merchant / device / email / billing ...) and per category, using only
-- LABEL-MATURE rows: each frame ends {lag} days before the current row, so a label is used only once known.
-- The select list and window list below are expanded once per entity by src/features.py.
CREATE OR REPLACE TABLE feat_risk AS
SELECT
  txn_id,
  {entity_select}
  COALESCE(SUM(is_fraud) OVER crange, 0) AS cat_fraud_known,
  COUNT(*)               OVER crange     AS cat_txn_known
FROM txn
WINDOW
  {entity_windows}
  crange AS (PARTITION BY category ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL {lag} DAY PRECEDING);
