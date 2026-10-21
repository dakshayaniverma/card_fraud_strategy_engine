# Interview prep: Card Fraud Strategy Engine

## 30-second pitch
"I built a decisioning engine that answers one question for every card transaction: approve, step up, or decline, and what
is the net dollar impact. Features come from SQL (velocity, deviation from the card's own behaviour, merchant history), a
gradient-boosted model scores risk on a strict time-based split, and a cost layer turns scores into actions by minimising
expected dollar cost, with more leeway for long-tenure, high-spend customers. On a held-out period it caught 99% of fraud
dollars while declining about 2 good transactions per 10,000, worth roughly $5.7K more per 10K transactions than the best
single rule, and that advantage held in all 81 cost scenarios I tested."

## Likely questions
- **Why not accuracy?** At 0.38% fraud, approving everything is 99.6% accurate and catches nothing. I used PR-AUC, fraud recall and fraud *dollars* caught at a fixed decline rate, and net dollars for decisions.
- **Class imbalance?** No resampling. Boosted trees handle 0.4% positives; I calibrated scores (isotonic, on validation) because the cost layer needs real probabilities, and judged by PR-AUC rather than ROC.
- **Why a time-based split?** Fraud patterns drift and labels arrive late. A random split lets the model see the future. I also leave a 7-day gap before validation so no training label was unknown at that time, and build label-derived features only from labels older than 7 days (unit-tested).
- **How did you set the cost of a false decline?** Lost revenue on that spend (2%) plus churn probability (5%) times customer value ($300 standard, $1,500 premium). They are assumptions, so I stress-tested 81 combinations and the strategy won in all of them; in production these come from experiments or finance.
- **How would fraudsters adapt?** Stay under velocity and amount thresholds, spread across cards and merchants, use cards from fresh compromises, or pass step-up with social engineering/SIM swap (my 80% step-up catch is an assumption). Defences: monitor score PSI, retrain regularly, add network/device signals, and review step-up effectiveness.
- **Your graph features did nothing. Why include them?** The ablation was designed to test it. Sparkov assigns fraud to cards without any merchant-compromise mechanism (every merchant shows ~12% contaminated visitors), so there is nothing to find. I kept the leak-free implementation and reported the null result; with real network data (issuer-side merchant and cardholder views) it is exactly the common-point-of-compromise technique issuers use.
- **Is 0.974 PR-AUC believable?** Not as a real-world number: Sparkov is simulated with strong amount/time/category patterns. It demonstrates the method; the economics and the discipline (split, leakage control, calibration, thresholds on validation) transfer.
- **Why step-up at all?** It is cheaper than a decline when the model is unsure: a small friction cost, most fraudsters fail, most good customers pass. The cost-optimal policy uses it for the uncertain middle (72 good step-ups per 10K vs 2 good declines).
- **What would you do next with Amex data?** Add network-level signals (spend at merchant across all cardholders), a rolling-origin backtest, an A/B test of step-up thresholds, and champion/challenger monitoring.

## Mapping to the job description
| JD line | Where it shows up |
|---|---|
| Economic logic to drive profitable actions | `src/strategy.py`: per-action cost tables, argmin expected cost, net value per 10K |
| Minimise disruption of good spending | Good declines per 10K, step-up friction, premium-customer leeway |
| Closed-loop network (cardholder + merchant) | Card-to-merchant graph, common-point-of-compromise features (honest null on Sparkov) |
| Big data and ML | 1.85M transactions, DuckDB SQL, XGBoost, TreeSHAP reason codes |
| Articulate findings to leadership | `memo/executive_memo.md` plus the simulator tab for live what-ifs |
| Credit + Fraud (the C and F) | Pair with your Unsecured Lending Decision project |
