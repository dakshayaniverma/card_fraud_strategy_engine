"""End-to-end: load -> SQL+graph features -> time-split model (+ablation) -> strategy -> artifacts.

    python -m src.pipeline --demo            # synthetic stand-in, pipeline check only
    python -m src.pipeline                   # real Sparkov CSVs in data/raw/
"""
import argparse, json, time
import numpy as np
import pandas as pd
from . import config as C
from .data import load_transactions
from .profiles import PROFILES
from .features import build_features
from .graph import cpp_leaderboard
from . import model as M
from . import strategy as S


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(dataset: str, demo: bool, seeds: int, out):
    prof = PROFILES[dataset]
    out = out or C.ROOT / (prof.demo_artifacts if demo else prof.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    cost = C.CostParams()
    raw = load_transactions(prof.demo_dir if demo else prof.data_dir, prof)
    log(f"{len(raw):,} transactions, fraud rate {raw.is_fraud.mean():.3%}, {raw.cc_num.nunique():,} cards, "
        f"{raw.merchant.nunique():,} merchants, {raw.ts.min().date()} -> {raw.ts.max().date()}")
    df = build_features(raw, prof)
    log("features built (SQL + graph)")

    tr, va, te, v0, t0 = M.time_split(df, C.VALID_START_Q, C.TEST_START_Q, C.LABEL_LAG_DAYS)
    log(f"split: train {tr.sum():,} | valid {va.sum():,} (from {v0.date()}) | test {te.sum():,} (from {t0.date()})")
    y = df.is_fraud.values
    base_cols, graph_cols = M.feature_sets(prof, df.columns)
    all_cols = base_cols + graph_cols
    log(f"{len(base_cols)} base features, {len(graph_cols)} graph features")

    # ---- ablation: with vs without graph features, several seeds -------------------
    abl = {"without_graph": [], "with_graph": []}
    models = {}
    for name, cols in [("without_graph", base_cols), ("with_graph", all_cols)]:
        for s in range(seeds):
            m = M.fit(df.loc[tr, cols], y[tr], df.loc[va, cols], y[va], seed=s)
            sc = m.predict_proba(df.loc[te, cols])[:, 1]
            abl[name].append(M.evaluate_scores(sc, y[te], df.loc[te, "amt"].values))
            models[(name, s)] = m
        log(f"ablation {name}: PR-AUC " + ", ".join(f"{r['pr_auc']:.4f}" for r in abl[name]))

    # ---- final model = with-graph, seed 0, calibrated on validation ------------------
    mdl = models[("with_graph", 0)]
    raw_va = mdl.predict_proba(df.loc[va, all_cols])[:, 1]
    iso = M.calibrate(raw_va, y[va])
    raw_te = mdl.predict_proba(df.loc[te, all_cols])[:, 1]
    p_va, p_te = iso.predict(raw_va), iso.predict(raw_te)

    # ---- segments (cutoff learned on train only) -------------------------------------
    cut = df.loc[tr, "card_prior_avg_amt"].quantile(C.PREMIUM_AMT_QUANTILE)
    prem = ((df.card_tenure_days >= prof.premium_min_tenure_days) & (df.card_prior_avg_amt >= cut)).values
    V, T = df[va].reset_index(drop=True), df[te].reset_index(drop=True)
    V["amt"] = df.loc[va, "amt"].values; T["amt"] = df.loc[te, "amt"].values
    prem_v, prem_t = prem[va], prem[te]
    yv, yt = y[va], y[te]

    # ---- strategies (all tuned on VALIDATION, reported on TEST) ----------------------
    Fv, Gv = S.cost_tables(V.amt.values, prem_v, cost)
    Ft, Gt = S.cost_tables(T.amt.values, prem_t, cost)
    rule = S.tune_rule(V, yv, cost, prem_v)
    thr_flat = S.tune_thresholds(p_va, yv, V.amt.values, prem_v, cost, leeways=(1.0,))
    thr_seg = S.tune_thresholds(p_va, yv, V.amt.values, prem_v, cost)
    strategies = {
        "Approve all": np.zeros(len(T), int),
        f"Current-rule baseline (amt>={rule['amt_min']} & {rule['flag']})": S.rule_actions(T, rule["amt_min"], rule["flag"]),
        "A: model, flat thresholds": S.threshold_actions(p_te, prem_t, thr_flat["decline_thr"], thr_flat["stepup_thr"], 1.0),
        "B: model, segment-aware thresholds": S.threshold_actions(p_te, prem_t, thr_seg["decline_thr"], thr_seg["stepup_thr"], thr_seg["premium_leeway"]),
        "C: model, cost-optimal per transaction": S.cost_optimal_actions(p_te, Ft, Gt),
    }
    results = {k: S.evaluate(a, yt, T.amt.values, Ft, Gt, cost) for k, a in strategies.items()}
    rule_nv = next(v["net_value_per_10k"] for k, v in results.items() if k.startswith("Current-rule"))
    for v in results.values():
        v["net_vs_rule_per_10k"] = v["net_value_per_10k"] - rule_nv
    log("strategy results (test):")
    for k, v in results.items():
        log(f"  {k:55s} fraud$ caught {v['pct_fraud_dollars_caught']:6.1%} | good declines/10k {v['good_declined_per_10k']:6.1f} "
            f"| net/10k ${v['net_value_per_10k']:9.0f}")

    # ---- reason codes on flagged test rows --------------------------------------------
    flagged = strategies["C: model, cost-optimal per transaction"] > 0
    flagged |= p_te >= np.quantile(p_te, 0.98)
    Xte = df.loc[te, all_cols].reset_index(drop=True)
    rc, imp, cols = M.reason_codes(mdl, Xte[flagged])
    keep = ["txn_id", "ts", "cc_num", "merchant", "category", "state", "amt", "is_fraud", "is_night",
            "is_first_merchant", "hour", "txn_cnt_1h", "amt_ratio_avg", "dist_home_km"]
    scored = T[[c for c in keep if c in T.columns]].copy()
    scored["score"] = p_te; scored["premium"] = prem_t
    scored["action_C"] = strategies["C: model, cost-optimal per transaction"]
    scored = scored.join(rc)
    for c in rc.columns:
        scored[c] = scored[c].fillna("")
    scored["category"] = scored["category"].astype(str)
    scored.to_parquet(out / "scored_test.parquet", index=False, compression="zstd")

    # ---- monitoring: PSI of score by month vs validation reference ---------------------
    T["score"] = p_te; T["month"] = T.ts.dt.to_period("M").astype(str)
    psi_rows = [{"month": mth, "n": len(g), "score_psi": S.psi(p_va, g.score.values), "mean_score": g.score.mean(),
                 "fraud_rate": g.is_fraud.mean()} for mth, g in T.groupby("month") if len(g) > 200]
    psi_cols = [c for c in ["amt", "dist_home_km", "txn_cnt_24h", "amt_ratio_avg"] if c in df.columns]
    psi_cols += [c for c in graph_cols if c.endswith("_fraud_rate")][:1]
    feat_psi = {c: S.psi(df.loc[va, c].astype(float).values, df.loc[te, c].astype(float).values) for c in psi_cols}
    pd.DataFrame(psi_rows).to_csv(out / "psi_by_month.csv", index=False)

    # ---- portfolio view (all data) & graph tables ---------------------------------------
    port = df.assign(month=df.ts.dt.to_period("M").astype(str), category=df.category.astype(str))
    port["fraud_amt"] = port.amt * port.is_fraud
    port.groupby("category").agg(txns=("amt", "size"), fraud_txns=("is_fraud", "sum"), fraud_amt=("fraud_amt", "sum"),
                                 spend=("amt", "sum")).reset_index().to_csv(out / "portfolio_category.csv", index=False)
    port.groupby("hour").agg(txns=("amt", "size"), fraud_txns=("is_fraud", "sum"), fraud_amt=("fraud_amt", "sum")
                             ).reset_index().to_csv(out / "portfolio_hour.csv", index=False)
    port.groupby("state").agg(txns=("amt", "size"), fraud_txns=("is_fraud", "sum"), fraud_amt=("fraud_amt", "sum")
                              ).reset_index().to_csv(out / "portfolio_state.csv", index=False)
    port.groupby("month").agg(txns=("amt", "size"), fraud_txns=("is_fraud", "sum"), fraud_amt=("fraud_amt", "sum")
                              ).reset_index().to_csv(out / "portfolio_month.csv", index=False)
    lb = pd.concat([cpp_leaderboard(raw.assign(category=raw.category.astype(str)), f"ent_{p}", prof.cpp_window_days, p)
                    for p in prof.entities], ignore_index=True)
    lb.to_csv(out / "cpp_merchants.csv", index=False)
    pd.DataFrame({"feature": cols, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)\
        .to_csv(out / "shap_global.csv", index=False)

    def agg(rs, k):
        v = [r[k] for r in rs]; return float(np.mean(v)), float(np.std(v))
    summary = {
        "data_source": "DEMO (synthetic, not for citation)" if demo else prof.label, "dataset": prof.name,
        "entities": list(prof.entities), "n_base_features": len(base_cols), "n_graph_features": len(graph_cols),
        "rows": int(len(df)), "fraud_rate": float(y.mean()),
        "split": {"valid_start": str(v0), "test_start": str(t0), "label_lag_days": C.LABEL_LAG_DAYS,
                  "n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_test": int(te.sum()),
                  "test_fraud_rate": float(yt.mean())},
        "ablation": {k: {"pr_auc_mean_std": agg(v, "pr_auc"),
                         "recall_at_1pct": agg([{"x": r["at_1pct_decline"]["recall"]} for r in v], "x"),
                         "fraud_dollars_caught_at_1pct": agg([{"x": r["at_1pct_decline"]["fraud_dollars_caught_pct"]} for r in v], "x"),
                         "runs": v} for k, v in abl.items()},
        "cost_params": cost.to_dict(), "rule": rule, "thresholds_flat": thr_flat, "thresholds_segmented": thr_seg,
        "premium_amt_cutoff": float(cut), "strategies_test": results, "feature_psi": feat_psi,
        "premium_share_test": float(prem_t.mean()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    mdl.save_model(str(out / "model.json"))
    log(f"done -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(PROFILES), default="sparkov")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.dataset, a.demo, a.seeds, (C.ROOT / a.out) if a.out else None)
