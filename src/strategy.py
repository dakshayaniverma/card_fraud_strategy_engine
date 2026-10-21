"""Decision economics: approve / step-up / decline, costed in dollars.

Every transaction has, for each action, a cost if it is fraud (F) and a cost if it is good (G):
    approve : F = amount + handling cost                G = 0
    step-up : F = friction + (1-catch)*(amount+handling)   G = friction + abandon*revenue*amount
    decline : F = 0                                        G = revenue*amount + attrition_prob*CLV(segment)
Realised cost uses the true label (y); expected cost uses the model probability (p), so the
cost-optimal policy is simply argmin over actions of p*F + (1-p)*G.
Net value of a strategy = cost of approving everything - cost of the strategy.
"""
import numpy as np
import pandas as pd
from .config import CostParams

APPROVE, STEPUP, DECLINE = 0, 1, 2


def cost_tables(amt, premium, p: CostParams):
    amt = np.asarray(amt, float); prem = np.asarray(premium, bool)
    loss = amt + p.fraud_fixed_cost
    clv = np.where(prem, p.clv_premium, p.clv_standard)
    F = np.vstack([loss, p.stepup_cost + (1 - p.stepup_fraud_catch) * loss, np.zeros_like(amt)])
    G = np.vstack([np.zeros_like(amt), p.stepup_cost + p.stepup_good_abandon * p.revenue_rate * amt,
                   p.revenue_rate * amt + p.attrition_prob * clv])
    return F, G


def cost_optimal_actions(prob, F, G):
    return np.argmin(prob * F + (1 - prob) * G, axis=0)


def threshold_actions(score, premium, decline_thr, stepup_thr, premium_leeway=1.0):
    """Two-threshold policy; premium customers' thresholds are multiplied by `premium_leeway` (>=1)."""
    mult = np.where(premium, premium_leeway, 1.0)
    a = np.zeros(len(score), dtype=int)
    a[score >= stepup_thr * mult] = STEPUP
    a[score >= decline_thr * mult] = DECLINE
    return a


def evaluate(actions, y, amt, F, G, p: CostParams):
    actions = np.asarray(actions); y = np.asarray(y).astype(bool); amt = np.asarray(amt, float)
    n = len(y); idx = np.arange(n)
    realised = np.where(y, F[actions, idx], G[actions, idx])
    base_cost = F[APPROVE][y].sum()
    fraud_dollars = amt[y].sum()
    prevented = amt[y & (actions == DECLINE)].sum() + p.stepup_fraud_catch * amt[y & (actions == STEPUP)].sum()
    per10k = 10_000 / n
    return {
        "n_txns": int(n),
        "fraud_dollars_total": float(fraud_dollars),
        "fraud_dollars_prevented": float(prevented),
        "pct_fraud_dollars_caught": float(prevented / fraud_dollars) if fraud_dollars else 0.0,
        "pct_fraud_txns_flagged": float((y & (actions > 0)).sum() / max(y.sum(), 1)),
        "good_declined_per_10k": float((~y & (actions == DECLINE)).sum() * per10k),
        "good_stepped_up_per_10k": float((~y & (actions == STEPUP)).sum() * per10k),
        "decline_rate_pct": float((actions == DECLINE).mean() * 100),
        "strategy_cost": float(realised.sum()),
        "net_value": float(base_cost - realised.sum()),
        "net_value_per_10k": float((base_cost - realised.sum()) * per10k),
    }


def _grid(score, n=28):
    qs = np.unique(np.quantile(score, np.linspace(0.80, 0.9995, n)))
    return qs


def tune_thresholds(score, y, amt, premium, p: CostParams, leeways=(1.0, 1.5, 2.0, 3.0)):
    """Grid-search (step-up thr, decline thr, premium leeway) maximising net value. Use on VALIDATION."""
    F, G = cost_tables(amt, premium, p)
    best = None
    grid = _grid(score)
    for lw in leeways:
        for s in grid:
            for d in grid[grid >= s]:
                a = threshold_actions(score, premium, d, s, lw)
                nv = evaluate(a, y, amt, F, G, p)["net_value"]
                if best is None or nv > best[0]:
                    best = (nv, float(s), float(d), lw)
    return {"stepup_thr": best[1], "decline_thr": best[2], "premium_leeway": best[3]}


def rule_actions(df: pd.DataFrame, amt_min, flag):
    cond = {"none": np.ones(len(df), bool), "night": df.is_night.values == 1,
            "first_merchant": df.is_first_merchant.values == 1,
            "night_or_first": (df.is_night.values == 1) | (df.is_first_merchant.values == 1)}[flag]
    return np.where((df.amt.values >= amt_min) & cond, DECLINE, APPROVE)


def tune_rule(df: pd.DataFrame, y, p: CostParams, premium):
    """Best single decline rule of the form 'amount >= A AND flag' on validation: a fair 'current rule' baseline."""
    F, G = cost_tables(df.amt.values, premium, p)
    best = None
    for flag in ["none", "night", "first_merchant", "night_or_first"]:
        for A in [50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]:
            nv = evaluate(rule_actions(df, A, flag), y, df.amt.values, F, G, p)["net_value"]
            if best is None or nv > best[0]:
                best = (nv, A, flag)
    return {"amt_min": best[1], "flag": best[2]}


def psi(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))
