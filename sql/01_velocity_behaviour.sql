-- Card-level velocity, deviation and context features. Every feature for a row
-- uses only that card's PRIOR activity (current row subtracted or excluded).
CREATE OR REPLACE MACRO haversine_km(la1, lo1, la2, lo2) AS
  2 * 6371 * asin(sqrt(
    pow(sin(radians(la2 - la1) / 2), 2) +
    cos(radians(la1)) * cos(radians(la2)) * pow(sin(radians(lo2 - lo1) / 2), 2)));

CREATE OR REPLACE TABLE feat_core AS
WITH w AS (
  SELECT t.*,
    ROW_NUMBER() OVER cardseq                                         AS card_txn_seq,
    COUNT(*)     OVER w1h  - 1                                        AS txn_cnt_1h,
    COALESCE(SUM(amt) OVER w1h, 0)  - amt                             AS amt_sum_1h,
    COUNT(*)     OVER w24h - 1                                        AS txn_cnt_24h,
    COALESCE(SUM(amt) OVER w24h, 0) - amt                             AS amt_sum_24h,
    COUNT(*)     OVER w7d  - 1                                        AS txn_cnt_7d,
    AVG(amt)         OVER prior                                       AS prior_avg_amt,
    STDDEV_SAMP(amt) OVER prior                                       AS prior_std_amt,
    MAX(amt)         OVER prior                                       AS prior_max_amt,
    ROW_NUMBER() OVER (PARTITION BY cc_num, merchant ORDER BY ts, txn_id) AS merch_seq,
    ROW_NUMBER() OVER (PARTITION BY cc_num, category ORDER BY ts, txn_id) AS cat_seq,
    MIN(ts) OVER (PARTITION BY cc_num)                                AS card_first_ts,
    LAG(ts)        OVER cardseq                                       AS prev_ts,
    LAG(merch_lat) OVER cardseq                                       AS prev_lat,
    LAG(merch_lon) OVER cardseq                                       AS prev_lon
  FROM txn t
  WINDOW
    cardseq AS (PARTITION BY cc_num ORDER BY ts, txn_id),
    prior   AS (PARTITION BY cc_num ORDER BY ts, txn_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    w1h     AS (PARTITION BY cc_num ORDER BY ts RANGE BETWEEN INTERVAL 1 HOUR  PRECEDING AND CURRENT ROW),
    w24h    AS (PARTITION BY cc_num ORDER BY ts RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW),
    w7d     AS (PARTITION BY cc_num ORDER BY ts RANGE BETWEEN INTERVAL 7 DAY   PRECEDING AND CURRENT ROW)
)
SELECT
  txn_id,
  amt,
  ln(1 + amt)                                          AS log_amt,
  hour(ts)                                             AS hour,
  dayofweek(ts)                                        AS dow,
  CASE WHEN hour(ts) >= 22 OR hour(ts) <= 3 THEN 1 ELSE 0 END AS is_night,
  category,
  ln(1 + city_pop)                                     AS log_city_pop,
  age,
  haversine_km(lat, lon, merch_lat, merch_lon)         AS dist_home_km,
  txn_cnt_1h, amt_sum_1h, txn_cnt_24h, amt_sum_24h, txn_cnt_7d,
  card_txn_seq - 1                                     AS card_prior_txns,
  COALESCE(prior_avg_amt, amt)                         AS card_prior_avg_amt,
  amt / NULLIF(prior_avg_amt, 0)                       AS amt_ratio_avg,
  CASE WHEN prior_std_amt > 0 THEN (amt - prior_avg_amt) / prior_std_amt ELSE 0 END AS amt_z,
  amt / NULLIF(prior_max_amt, 0)                       AS amt_ratio_max,
  CASE WHEN merch_seq = 1 THEN 1 ELSE 0 END            AS is_first_merchant,
  CASE WHEN cat_seq = 1 THEN 1 ELSE 0 END              AS is_first_category,
  epoch(ts - prev_ts)                                  AS secs_since_prev,
  haversine_km(prev_lat, prev_lon, merch_lat, merch_lon) AS km_since_prev,
  haversine_km(prev_lat, prev_lon, merch_lat, merch_lon)
      / NULLIF(epoch(ts - prev_ts) / 3600.0, 0)        AS implied_kmh,
  date_diff('day', card_first_ts, ts)                  AS card_tenure_days
FROM w;
