"""Run the SQL feature layer in DuckDB and join graph + native features."""
import duckdb
import pandas as pd
from . import config as C
from .graph import graph_features
from .profiles import Profile, SPARKOV

GRAPH_STATS = ("fraud_rate", "cpp_cards", "cpp_share", "is_hot", "exposure")


def graph_feature_names(profile: Profile):
    return [f"g_{p}_{s}" for p in profile.entities for s in GRAPH_STATS]


def build_features(raw: pd.DataFrame, profile: Profile = SPARKOV, with_graph=True) -> pd.DataFrame:
    lag = C.LABEL_LAG_DAYS
    con = duckdb.connect()
    con.register("txn", raw)
    con.execute((C.SQL_DIR / profile.sql_core).read_text())
    ents = list(profile.entities)
    sql = (C.SQL_DIR / "02_entity_category_risk.sql").read_text().format(
        lag=lag,
        entity_select="".join(
            f"COALESCE(SUM(is_fraud) OVER r_{p}, 0) AS {p}_fraud_known, COUNT(*) OVER r_{p} AS {p}_txn_known,\n  " for p in ents),
        entity_windows="".join(
            f"r_{p} AS (PARTITION BY ent_{p} ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL {lag} DAY PRECEDING),\n  "
            for p in ents))
    con.execute(sql)
    f = con.execute("SELECT c.*, r.* EXCLUDE (txn_id) FROM feat_core c JOIN feat_risk r USING (txn_id) ORDER BY txn_id").df()
    con.close()
    base = raw["is_fraud"].mean()
    a = 50.0  # smoothing: pseudo-observations at the portfolio base rate
    f["cat_fraud_rate"] = (f.cat_fraud_known + a * base) / (f.cat_txn_known + a)
    for p in ents:
        f[f"g_{p}_fraud_rate"] = (f[f"{p}_fraud_known"] + a * base) / (f[f"{p}_txn_known"] + a)
        f = f.drop(columns=[f"{p}_fraud_known", f"{p}_txn_known"])
    f = f.drop(columns=["cat_fraud_known", "cat_txn_known"])
    if with_graph:
        for p in ents:
            g = graph_features(raw, f"ent_{p}", profile.cpp_window_days).reset_index(drop=True)
            f = pd.concat([f, g.add_prefix(f"g_{p}_")], axis=1)
    core = ["txn_id", "ts", "cc_num", "merchant", "state", "is_fraud", "trans_num"]
    native = [c for c in profile.native_num + profile.native_cat if c in raw.columns]
    out = raw[core + native].merge(f, on="txn_id")
    for c in profile.native_cat:
        if c in out.columns:
            out[c] = out[c].astype("category")
    out["category"] = out["category"].astype("category")
    return out
