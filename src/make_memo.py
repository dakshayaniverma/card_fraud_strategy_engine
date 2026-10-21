"""Fill the executive memo from artifacts/summary.json so no number is ever typed by hand."""
import argparse, json
from . import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--artifacts", default="artifacts", help="artifact folder, e.g. artifacts_demo")
art = C.ROOT / ap.parse_args().artifacts
s = json.loads((art / "summary.json").read_text())
if s["data_source"].startswith("DEMO"):
    raise SystemExit("Refusing to write a memo from demo data. Run the pipeline on the real Sparkov CSVs first.")
st = s["strategies_test"]; ab = s["ablation"]
rule = next(k for k in st if k.startswith("Current-rule"))
best = max((k for k in st if k[0] in "ABC"), key=lambda k: st[k]["net_value_per_10k"])
b, r = st[best], st[rule]
w, wo = ab["with_graph"], ab["without_graph"]
memo = f"""# Recommendation: move card authorisation from a single rule to a cost-based approve / step-up / decline strategy

**To:** Head of Fraud Risk  |  **From:** <your name>  |  **Data:** {s['data_source']}, {s['rows']:,} transactions, fraud rate {s['fraud_rate']:.2%}

## Recommendation
Adopt **{best}**. On a held-out period that starts after all training and tuning data
(test start {s['split']['test_start'][:10]}), it catches **{b['pct_fraud_dollars_caught']:.1%}** of fraud dollars
with **{b['good_declined_per_10k']:.1f}** good transactions declined and {b['good_stepped_up_per_10k']:.0f} stepped up per 10,000,
a net value of **${b['net_value_per_10k']:,.0f} per 10,000 transactions**, which is
**${b['net_vs_rule_per_10k']:,.0f} more** than the best single rule ({rule}), tuned on the same validation data.

## Why this is better
- It prices each decision: a missed fraud costs the amount plus handling; a false decline costs lost revenue plus an
  attrition penalty; a step-up costs friction. Thresholds maximise net dollars, not accuracy.
- Long-tenure, high-spend customers get more leeway ({s['thresholds_segmented']['premium_leeway']}x on thresholds).
- A card-to-merchant network adds common-point-of-compromise signals. Effect on PR-AUC: {wo['pr_auc_mean_std'][0]:.4f} -> {w['pr_auc_mean_std'][0]:.4f}
  (mean over seeds, sd {w['pr_auc_mean_std'][1]:.4f}). *Interpret honestly: state here whether this is meaningful or within noise.*

## Assumptions that drive the result (flex them in the app)
{json.dumps(s['cost_params'], indent=2)}

## Risks and next steps
- Fraudsters adapt: velocity and amount patterns decay; monitor score PSI monthly (app, Monitoring tab) and retrain on drift.
- Labels mature with a {s['split']['label_lag_days']}-day lag; recent fraud is under-counted at scoring time.
- Cost parameters are assumptions; validate revenue rate, attrition and step-up effectiveness with business owners / A-B test.
- Synthetic data (Sparkov) is not real portfolio behaviour: treat the absolute numbers as a method demonstration.
"""
(C.ROOT / "memo").mkdir(exist_ok=True)
(C.ROOT / "memo" / f"executive_memo_{s.get('dataset', 'sparkov')}.md").write_text(memo)
print(memo)
