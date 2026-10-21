Card Fraud Strategy Engine | Python, SQL (DuckDB), XGBoost, Graph Analytics, Streamlit

- Built an end-to-end card fraud decisioning system on 1.85M transactions: SQL-engineered velocity and behavioural features, time-based train/validation/test split with label-maturity controls, calibrated XGBoost scorer with SHAP reason codes (test PR-AUC 0.97 at 0.38% fraud).
- Designed a cost-based approve / step-up / decline strategy (missed fraud, false-decline revenue and attrition, step-up friction): caught 99% of fraud dollars while declining ~2 good transactions per 10K, +$5.7K net value per 10K transactions over a validation-tuned rule baseline (83% fraud dollars, 56 good declines per 10K); advantage held in 81 of 81 stress-tested cost scenarios.
- Tested card-to-merchant network (common-point-of-compromise) features with a leakage-tested ablation; found no lift on the simulated data and traced it to the data having no merchant-level fraud mechanism.
- Deployed an interactive strategy simulator, case-investigation view and PSI monitoring in Streamlit; wrote an executive recommendation memo.
