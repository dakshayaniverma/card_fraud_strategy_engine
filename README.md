# Card Fraud Strategy Engine

> For each card transaction: **approve, step up (OTP), or decline**, and what is the **net dollar impact** of that strategy?

An end-to-end card-fraud decisioning project: SQL feature layer, card-to-merchant graph analysis
(common point of compromise), a time-split gradient-boosted model, a cost-based strategy optimiser, and a
Streamlit app with a live strategy simulator.

## Why it is built this way
| Design choice | Reason |
|---|---|
| Time-based train / validation / test, with a label-maturity gap | Fraud labels arrive late; a random split leaks the future. |
| PR-AUC, recall at a fixed decline rate, **fraud dollars caught** (no accuracy) | At ~0.5% fraud, "approve everything" scores ~99.5% accuracy. |
| Label-derived features (merchant risk, compromise) use only labels older than 7 days | Otherwise they leak. `tests/test_core.py::test_no_label_leakage` proves it. |
| Leave-one-out graph features | A card's own fraud must not make its own later transactions look risky. |
| Thresholds tuned on validation, reported on test | Test is touched once. |
| Baseline = best single rule, *tuned on the same validation data* | A fair "current rule", not a strawman. |
| Cost assumptions are inputs, flexible in the app | They are assumptions, not facts. |

## Layers
1. **SQL features** (DuckDB, `sql/`): velocity (1h / 24h / 7d count and spend), deviation vs the card's own history,
   first-time merchant / category, distance from home, implied travel speed, night-time, lagged merchant & category fraud rate.
2. **Graph** (`src/graph.py`): bipartite card-merchant network. A card is *contaminated* at a merchant if fraud follows
   its visit within 60 days. A merchant is *hot* when >= 3 contaminated cards (and >= 25% of its visitors) are known.
   Features: merchant compromise count/share, hot flag, and the card's exposure to hot merchants it used earlier.
3. **Model** (`src/model.py`): XGBoost, isotonic calibration on validation, ablation with vs without graph features
   (several seeds), exact TreeSHAP reason codes for flagged transactions.
4. **Economics** (`src/strategy.py`): per-action cost tables; strategies A (flat thresholds), B (segment-aware thresholds,
   premium customers get leeway), C (per-transaction cost-optimal); each compared to a tuned rule baseline.
5. **App** (`app/streamlit_app.py`): portfolio, **strategy simulator**, case investigation (reason codes + network), PSI monitoring, model & network.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 1. Get the real data: Kaggle "Credit Card Transactions Fraud Detection Dataset" (kartik2112)
#    Put fraudTrain.csv and fraudTest.csv in data/raw/
python -m src.pipeline            # ~ minutes on 1.8M rows; writes artifacts/
pytest -q                         # 5 tests incl. label-leakage
streamlit run app/streamlit_app.py
python -m src.make_memo           # fills memo/executive_memo.md from real results
```
No Kaggle data yet? `python -m src.pipeline --demo` runs everything on a synthetic stand-in
(`artifacts_demo/`, banner in the app). **Demo numbers are meaningless**: the fraud patterns were written by the
generator, so the model finds them trivially (the same trap as a model trained on a rule you wrote yourself). Never quote them.

## Deploy (Streamlit Community Cloud)
Commit `artifacts/` (about tens of MB: `scored_test.parquet` plus small CSVs), push to GitHub, create an app pointing at
`app/streamlit_app.py`. `requirements.txt` is the lean app-only set; training deps are in `requirements-dev.txt`.

## Results (Sparkov, 1.85M transactions, held-out test: 23 Jul to 31 Dec 2020, 463K txns, 0.38% fraud)
Thresholds and the rule baseline are tuned on a validation window that ends before the test window starts. USD, default cost assumptions.

| Strategy | Fraud $ caught | Good declines / 10K | Good step-ups / 10K | Net value / 10K | vs rule / 10K |
|---|---|---|---|---|---|
| Current-rule baseline (amt >= 250 and night), validation-tuned | 83.0% | 55.6 | 0 | $14,441 | - |
| A: model, flat thresholds | 99.7% | 25.1 | 93 | $19,402 | +$4,960 |
| B: model, segment-aware thresholds | 98.9% | 20.6 | 0 | $19,636 | +$5,195 |
| **C: model, cost-optimal per transaction** | **99.2%** | **2.1** | 72 | **$20,169** | **+$5,728** |

- **Model:** PR-AUC 0.974 on the test period (base rate 0.38%). Sparkov is simulated and fairly easy to separate, so read this as a demonstration of method, not a claim about a real portfolio.
- **Robustness:** policies fixed at the default assumptions, re-scored under 81 different "true" economics (revenue rate, churn, step-up effectiveness, customer value): Strategy C beats the rule in **81 of 81** scenarios (min +$3.4K, median +$5.5K per 10K). Run `python -m src.sensitivity`.
- **Graph / common-point-of-compromise features: no lift on Sparkov** (PR-AUC 0.9742 without vs 0.9736 with, same recall and fraud $ caught at 1% decline rate, 3 seeds). Every merchant has ~12% of its visitors "contaminated", which is what you see when fraud is assigned to cards independently of where they shop. The method is tested (no-leakage unit test) and the null result is reported rather than hidden. Top drivers are 24h spend, amount, category and night-time.
- **Monitoring:** monthly score PSI is ~0.002 throughout the test period (stable, as expected for stationary simulated data).
