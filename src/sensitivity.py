"""Does the strategy still beat the rule baseline when the cost assumptions are wrong?

The policies are FIXED at the default assumptions (what we would have deployed). We then re-score both under a grid of
'true' economics and report the net advantage of the model strategy over the rule per 10K transactions.
    python -m src.sensitivity [--artifacts artifacts]
"""
import argparse, itertools, json
import numpy as np
import pandas as pd
from . import config as C
from . import strategy as S


def main(art):
    summ = json.loads((art / "summary.json").read_text())
    T = pd.read_parquet(art / "scored_test.parquet")
    y, amt, p, prem = T.is_fraud.values, T.amt.values, T.score.values, T.premium.values
    base = C.CostParams()
    F0, G0 = S.cost_tables(amt, prem, base)
    acts = {"C: cost-optimal": S.cost_optimal_actions(p, F0, G0),
            "B: segment thresholds": S.threshold_actions(p, prem, summ["thresholds_segmented"]["decline_thr"],
                                                         summ["thresholds_segmented"]["stepup_thr"],
                                                         summ["thresholds_segmented"]["premium_leeway"]),
            "rule": S.rule_actions(T, summ["rule"]["amt_min"], summ["rule"]["flag"])}
    grid = {"revenue_rate": [0.01, 0.02, 0.04], "attrition_prob": [0.0, 0.05, 0.15],
            "stepup_fraud_catch": [0.5, 0.8, 0.95], "clv_standard": [100.0, 300.0, 900.0]}
    rows = []
    for vals in itertools.product(*grid.values()):
        cp = C.CostParams(**dict(zip(grid, vals)), clv_premium=5 * dict(zip(grid, vals))["clv_standard"])
        F, G = S.cost_tables(amt, prem, cp)
        r = {k: S.evaluate(a, y, amt, F, G, cp)["net_value_per_10k"] for k, a in acts.items()}
        rows.append({**dict(zip(grid, vals)), **{f"net_{k}": v for k, v in r.items()},
                     "C_minus_rule": r["C: cost-optimal"] - r["rule"], "B_minus_rule": r["B: segment thresholds"] - r["rule"]})
    df = pd.DataFrame(rows)
    df.to_csv(art / "sensitivity.csv", index=False)
    for k in ["C_minus_rule", "B_minus_rule"]:
        print(f"{k}: wins in {(df[k] > 0).mean():.0%} of {len(df)} scenarios | min {df[k].min():,.0f} | "
              f"median {df[k].median():,.0f} | max {df[k].max():,.0f}  ($ per 10K txns)")
    worst = df.sort_values("C_minus_rule").head(3)
    print("Worst scenarios for C vs rule:\n", worst[list(grid) + ["C_minus_rule"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--artifacts", default="artifacts")
    main(C.ROOT / ap.parse_args().artifacts)
