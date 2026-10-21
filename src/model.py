"""XGBoost fraud scorer, time-based split, calibration, ablation and TreeSHAP reason codes."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_SPARKOV = ["amt", "log_amt", "hour", "dow", "is_night", "category", "log_city_pop", "age", "dist_home_km",
                "txn_cnt_1h", "amt_sum_1h", "txn_cnt_24h", "amt_sum_24h", "txn_cnt_7d", "card_prior_txns",
                "card_prior_avg_amt", "amt_ratio_avg", "amt_z", "amt_ratio_max", "is_first_merchant",
                "is_first_category", "secs_since_prev", "km_since_prev", "implied_kmh", "card_tenure_days",
                "cat_fraud_rate"]


def feature_sets(profile, columns):
    """(base, graph) feature lists for a dataset profile, restricted to columns that exist."""
    from .features import graph_feature_names
    base = BASE_SPARKOV
    graph = graph_feature_names(profile)
    return [c for c in base if c in columns], [c for c in graph if c in columns]


ENTITY_NOUN = {"merchant": "merchant", "device": "device", "email": "email domain", "billing": "billing address"}


def reason_text(col):
    """Human-readable reason for a feature name (used in reason codes)."""
    if col in REASON_TEXT:
        return REASON_TEXT[col]
    if col.startswith("g_"):
        _, ent, stat = col.split("_", 2)
        noun = ENTITY_NOUN.get(ent, ent)
        return {"fraud_rate": f"This {noun} has a high recent fraud rate",
                "exposure": f"Account previously linked to compromised {noun}(s)"}.get(
            stat, f"This {noun} is a likely point of compromise")
    if col.startswith("C") and col[1:].isdigit():
        return "Unusual account/address linkage count"
    if col.startswith("D") and col[1:].isdigit():
        return "Unusual time since an earlier event"
    if col.startswith("M") and col[1:].isdigit():
        return "Identity/address match check failed"
    if col.startswith("card") or col.startswith("addr"):
        return "Unusual card/billing profile"
    if col.startswith("id_") or col == "DeviceType":
        return "Unusual device/browser profile"
    if "email" in col.lower():
        return "Risky email domain"
    return col


REASON_TEXT = {
    "amt": "Transaction amount is unusual", "log_amt": "Transaction amount is unusual",
    "amt_ratio_avg": "Amount far above this card's usual spend", "amt_z": "Amount far above this card's usual spend",
    "amt_ratio_max": "Largest purchase this card has made", "card_prior_avg_amt": "Card's typical spend level",
    "txn_cnt_1h": "Burst of transactions in the last hour", "amt_sum_1h": "High spend in the last hour",
    "txn_cnt_24h": "High transaction count in 24h", "amt_sum_24h": "High spend in 24h",
    "txn_cnt_7d": "High transaction count in 7 days", "secs_since_prev": "Very short gap since previous transaction",
    "km_since_prev": "Far from the previous transaction", "implied_kmh": "Implausible travel speed",
    "dist_home_km": "Far from cardholder's home", "is_night": "Night-time transaction", "hour": "Unusual hour of day",
    "is_first_merchant": "First time at this merchant/device", "is_first_category": "First time in this category",
    "category": "High-risk merchant category / product", "cat_fraud_rate": "Category has high recent fraud",
    "card_tenure_days": "Newly active card", "card_prior_txns": "Little history on this card",
    "dow": "Unusual day of week", "age": "Cardholder age profile", "log_city_pop": "Cardholder city profile",
    "email_match": "Purchaser and recipient email differ",
}


def time_split(df, valid_q, test_q, lag_days):
    v0 = df.ts.quantile(valid_q); t0 = df.ts.quantile(test_q)
    lag = pd.Timedelta(days=lag_days)
    tr = df.ts < v0 - lag            # drop the label-maturity gap before validation
    va = (df.ts >= v0) & (df.ts < t0 - lag)
    te = df.ts >= t0
    return tr.values, va.values, te.values, v0, t0


def fit(X_tr, y_tr, X_va, y_va, seed=0):
    m = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0, tree_method="hist", enable_categorical=True,
        eval_metric="aucpr", early_stopping_rounds=40, random_state=seed, n_jobs=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def metrics_at_rate(score, y, amt, rate=0.01):
    k = max(int(np.ceil(rate * len(score))), 1)
    top = np.argsort(-score)[:k]
    y = np.asarray(y).astype(bool); amt = np.asarray(amt, float)
    caught = y[top]
    return {"recall": float(caught.sum() / max(y.sum(), 1)),
            "precision": float(caught.mean()),
            "fraud_dollars_caught_pct": float(amt[top][caught].sum() / max(amt[y].sum(), 1e-9))}


def evaluate_scores(score, y, amt, rates=(0.005, 0.01, 0.02)):
    out = {"pr_auc": float(average_precision_score(y, score)), "roc_auc": float(roc_auc_score(y, score)),
           "base_rate": float(np.mean(y))}
    for r in rates:
        out[f"at_{r*100:g}pct_decline"] = metrics_at_rate(score, y, amt, r)
    return out


def calibrate(raw_va, y_va):
    return IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(raw_va, y_va)


def reason_codes(model, X, top=3):
    """Exact TreeSHAP contributions (XGBoost pred_contribs); top positive drivers per row."""
    contrib = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[:, :-1]
    cols = list(X.columns)
    order = np.argsort(-contrib, axis=1)[:, :top]
    out = []
    for i in range(len(X)):
        seen, texts = set(), []
        for j in order[i]:
            if contrib[i, j] <= 0:
                continue
            t = reason_text(cols[j])
            if t not in seen:
                seen.add(t); texts.append(t)
        out.append(texts + [""] * (top - len(texts)))
    return pd.DataFrame(out, columns=[f"reason_{k+1}" for k in range(top)], index=X.index), \
        np.abs(contrib).mean(axis=0), cols
