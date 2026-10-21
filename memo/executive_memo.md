# Recommendation: move card authorisation from a single rule to a cost-based approve / step-up / decline strategy

**To:** Head of Fraud Risk  |  **From:** <your name>  |  **Data:** Sparkov (Kaggle), 1,852,394 transactions, fraud rate 0.52%

## Recommendation
Adopt **C: model, cost-optimal per transaction**. On a held-out period that starts after all training and tuning data
(test start 2020-07-23), it catches **99.2%** of fraud dollars
with **2.1** good transactions declined and 72 stepped up per 10,000,
a net value of **$20,169 per 10,000 transactions**, which is
**$5,728 more** than the best single rule (Current-rule baseline (amt>=250 & night)), tuned on the same validation data.

## Why this is better
- It prices each decision: a missed fraud costs the amount plus handling; a false decline costs lost revenue plus an
  attrition penalty; a step-up costs friction. Thresholds maximise net dollars, not accuracy.
- Long-tenure, high-spend customers get more leeway (3.0x on thresholds).
- A card-to-merchant network adds common-point-of-compromise signals. Effect on PR-AUC: 0.9742 -> 0.9736
  (mean over seeds, sd 0.0002). *Interpret honestly: state here whether this is meaningful or within noise.*

## Assumptions that drive the result (flex them in the app)
{
  "revenue_rate": 0.02,
  "fraud_fixed_cost": 5.0,
  "attrition_prob": 0.05,
  "clv_standard": 300.0,
  "clv_premium": 1500.0,
  "stepup_cost": 0.3,
  "stepup_fraud_catch": 0.8,
  "stepup_good_abandon": 0.03
}

## Risks and next steps
- Fraudsters adapt: velocity and amount patterns decay; monitor score PSI monthly (app, Monitoring tab) and retrain on drift.
- Labels mature with a 7-day lag; recent fraud is under-counted at scoring time.
- Cost parameters are assumptions; validate revenue rate, attrition and step-up effectiveness with business owners / A-B test.
- Synthetic data (Sparkov) is not real portfolio behaviour: treat the absolute numbers as a method demonstration.
