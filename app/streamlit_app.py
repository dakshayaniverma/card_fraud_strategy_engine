"""Card Fraud Strategy Engine: portfolio, strategy simulator, case investigation, monitoring."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import strategy as S          # noqa: E402  (pure numpy/pandas, safe on Streamlit Cloud)
from src.config import CostParams      # noqa: E402

st.set_page_config(page_title="Card Fraud Strategy Engine", layout="wide")
RUNS = sorted((d.name for d in ROOT.glob("artifacts*") if (d / "summary.json").exists()),
              key=lambda n: ("demo" in n, n))          # real runs first, demo runs last
if not RUNS:
    st.error("No artifacts found. Run `python -m src.pipeline` (or add `--demo`) first.")
    st.stop()
with st.sidebar:
    run = st.selectbox("Dataset run", RUNS, help="Each folder is one pipeline run on one dataset.")
ART = ROOT / run


@st.cache_data
def load(art_dir: str):
    art = Path(art_dir)
    s = json.loads((art / "summary.json").read_text())
    t = pd.read_parquet(art / "scored_test.parquet")
    t["ts"] = pd.to_datetime(t["ts"])
    tabs = {n: pd.read_csv(art / f"{n}.csv") for n in
            ["portfolio_category", "portfolio_hour", "portfolio_state", "portfolio_month",
             "cpp_merchants", "shap_global", "psi_by_month"]}
    return s, t, tabs


summ, T, tabs = load(str(ART))
DEMO = summ["data_source"].startswith("DEMO")

st.title("Card Fraud Strategy Engine")
st.caption(f"For each card transaction: approve, step up, or decline, and what is the net dollar impact? "
           f"Data: {summ['data_source']}.")
if DEMO:
    st.warning("DEMO DATA: synthetic transactions used only to test the pipeline. "
               "Numbers here mean nothing; do not quote them.")

# ------------------------------- sidebar: economics --------------------------------
d = CostParams()
with st.sidebar:
    st.header("Economic assumptions")
    inr = st.toggle("Show Rs (1 USD = Rs)", value=False)
    fx = st.number_input("FX rate", 60.0, 120.0, 85.0, disabled=not inr)
    cp = CostParams(
        revenue_rate=st.slider("Revenue rate on good spend", 0.005, 0.05, d.revenue_rate, 0.005),
        fraud_fixed_cost=st.slider("Handling cost per missed fraud ($)", 0.0, 50.0, d.fraud_fixed_cost, 1.0),
        attrition_prob=st.slider("P(churn | false decline)", 0.0, 0.3, d.attrition_prob, 0.01),
        clv_standard=st.slider("Value of a standard customer ($)", 50.0, 1000.0, d.clv_standard, 50.0),
        clv_premium=st.slider("Value of a premium customer ($)", 200.0, 5000.0, d.clv_premium, 100.0),
        stepup_cost=st.slider("Cost per step-up ($)", 0.0, 2.0, d.stepup_cost, 0.05),
        stepup_fraud_catch=st.slider("Step-up stops fraud (%)", 0.3, 1.0, d.stepup_fraud_catch, 0.05),
        stepup_good_abandon=st.slider("Good customers abandoning step-up", 0.0, 0.2, d.stepup_good_abandon, 0.01))
    st.caption("Cost assumptions are inputs, not facts. Check the strategy still wins when you flex them.")
cur = "Rs" if inr else "$"
k = fx if inr else 1.0
money = lambda x: f"{cur} {x * k:,.0f}"
md_money = lambda x: money(x).replace("$", "\\$")      # '$' pairs would be rendered as LaTeX inside st.markdown

y, amt, score, prem = T.is_fraud.values, T.amt.values, T.score.values, T.premium.values
F, G = S.cost_tables(amt, prem, cp)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Portfolio", "Strategy simulator", "Case investigation", "Monitoring", "Model & network"])

# ------------------------------------ portfolio -------------------------------------
with tab1:
    m = tabs["portfolio_month"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{m.txns.sum():,}")
    c2.metric("Fraud rate", f"{m.fraud_txns.sum() / m.txns.sum():.2%}")
    c3.metric("Fraud loss", money(m.fraud_amt.sum()))
    c4.metric("Avg fraud ticket", money(m.fraud_amt.sum() / max(m.fraud_txns.sum(), 1)))
    a, b = st.columns(2)
    cat = tabs["portfolio_category"].assign(fraud_rate=lambda x: x.fraud_txns / x.txns).sort_values("fraud_amt")
    a.plotly_chart(px.bar(cat, x="fraud_amt", y="category", orientation="h", title="Fraud loss by category",
                          hover_data=["fraud_rate", "txns"]), width="stretch")
    h = tabs["portfolio_hour"].assign(fraud_rate=lambda x: x.fraud_txns / x.txns)
    b.plotly_chart(px.bar(h, x="hour", y="fraud_rate", title="Fraud rate by hour of day"), width="stretch")
    st_ = tabs["portfolio_state"].assign(fraud_rate=lambda x: x.fraud_txns / x.txns)
    st_ = st_[st_.txns >= 500].sort_values("fraud_rate", ascending=False).head(20)
    a.plotly_chart(px.bar(st_, x="state", y="fraud_rate", title="Fraud rate by state (>=500 txns, top 20)"),
                   width="stretch")
    b.plotly_chart(px.line(m.assign(fraud_rate=m.fraud_txns / m.txns), x="month", y="fraud_rate",
                           title="Fraud rate by month"), width="stretch")

# ------------------------------- strategy simulator ---------------------------------
with tab2:
    st.subheader("Set the policy, see the dollars")
    pct = lambda p: float(np.round(p, 6))
    c1, c2, c3 = st.columns(3)
    sy = summ["thresholds_segmented"]
    dec = c1.select_slider("Decline if P(fraud) >=", options=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                           value=min([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], key=lambda x: abs(x - sy["decline_thr"])))
    stp = c2.select_slider("Step-up if P(fraud) >=", options=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
                           value=min([0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5], key=lambda x: abs(x - sy["stepup_thr"])))
    lw = c3.slider("Premium leeway (x thresholds)", 1.0, 4.0, float(sy["premium_leeway"]), 0.25,
                   help="Long-tenure, high-spend customers need a higher score before we decline or challenge them.")
    stp = min(stp, dec)
    acts = {"Your thresholds": S.threshold_actions(score, prem, dec, stp, lw),
            "Cost-optimal per transaction": S.cost_optimal_actions(score, F, G)}
    rule = summ["rule"]
    acts[f"Rule baseline (amt>={rule['amt_min']} & {rule['flag']})"] = S.rule_actions(T, rule["amt_min"], rule["flag"])
    res = {k_: S.evaluate(a_, y, amt, F, G, cp) for k_, a_ in acts.items()}
    rule_key = next(k_ for k_ in res if k_.startswith("Rule"))
    mine = res["Your thresholds"]
    r0 = res[rule_key]
    st.success(
        f"**Your strategy** catches **{mine['pct_fraud_dollars_caught']:.1%}** of fraud dollars while declining "
        f"**{mine['good_declined_per_10k']:.1f}** good transactions per 10,000, stepping up "
        f"{mine['good_stepped_up_per_10k']:.0f} good ones; net **{md_money(mine['net_value_per_10k'])}** per 10K "
        f"({md_money(mine['net_value_per_10k'] - r0['net_value_per_10k'])} vs the rule baseline).")
    cmp_ = pd.DataFrame(res).T[["pct_fraud_dollars_caught", "good_declined_per_10k", "good_stepped_up_per_10k",
                                "decline_rate_pct", "net_value_per_10k"]]
    cmp_["net_vs_rule_per_10k"] = cmp_.net_value_per_10k - r0["net_value_per_10k"]
    for c_ in ["net_value_per_10k", "net_vs_rule_per_10k"]:
        cmp_[c_] = cmp_[c_] * k
    cmp_.columns = ["Fraud $ caught", "Good declines /10K", "Good step-ups /10K", "Decline rate %",
                    f"Net value /10K ({cur})", f"vs rule /10K ({cur})"]
    st.dataframe(cmp_.style.format({"Fraud $ caught": "{:.1%}", "Good declines /10K": "{:.1f}", "Good step-ups /10K": "{:.0f}",
                                    "Decline rate %": "{:.2f}", cmp_.columns[4]: "{:,.0f}", cmp_.columns[5]: "{:,.0f}"}),
                 width="stretch")
    grid = np.geomspace(0.002, 0.95, 40)
    sweep = pd.DataFrame({"decline_thr": grid, "net_per_10k": [
        S.evaluate(S.threshold_actions(score, prem, g_, min(stp, g_), lw), y, amt, F, G, cp)["net_value_per_10k"] * k
        for g_ in grid]})
    fig = px.line(sweep, x="decline_thr", y="net_per_10k", title=f"Net value per 10K vs decline threshold ({cur})", log_x=True)
    fig.update_xaxes(tickvals=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ticktext=["0.5%", "1%", "2%", "5%", "10%", "20%", "50%"], title="Decline if P(fraud) >=")
    fig.add_vline(x=dec, line_dash="dash")
    st.plotly_chart(fig, width="stretch")
    st.caption("Test period only (strictly after training and validation). Thresholds default to the values tuned on the "
               "validation period.")

# ----------------------------- case investigation -----------------------------------
with tab3:
    flagged = T[(T.action_C > 0) | (T.is_fraud == 1)]
    top = (flagged.groupby("cc_num").agg(max_score=("score", "max"), flagged_txns=("score", "size"),
                                         fraud=("is_fraud", "sum"), amt=("amt", "sum"))
           .sort_values("max_score", ascending=False).head(200))
    card = st.selectbox("Flagged card (ranked by highest score)", top.index,
                        format_func=lambda c: f"{c} | max score {top.loc[c, 'max_score']:.2f} | {top.loc[c, 'flagged_txns']} flagged")
    ct = T[T.cc_num == card].sort_values("ts")
    show = ct[["ts", "merchant", "category", "amt", "score", "action_C", "is_fraud", "reason_1", "reason_2", "reason_3"]].copy()
    show["action_C"] = show.action_C.map({0: "Approve", 1: "Step-up", 2: "Decline"})
    show.columns = ["Time", "Merchant", "Category", "Amount", "Score", "Decision", "Actual fraud", "Reason 1", "Reason 2", "Reason 3"]
    st.dataframe(show.style.format({"Amount": "{:,.2f}", "Score": "{:.3f}"}), width="stretch", height=300, hide_index=True)
    fig = px.scatter(ct, x="ts", y="score", size="amt", color=ct.is_fraud.map({0: "good", 1: "fraud"}),
                     color_discrete_map={"good": "#3b82a0", "fraud": "#d1495b"}, title="Score timeline (bubble = amount)")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Card-merchant network** (card in centre; other cards that shared its merchants; red = card has fraud)")
    lb = tabs["cpp_merchants"]
    primary = (summ.get("entities") or ["merchant"])[0]
    if "entity_type" in lb:
        lb = lb[lb.entity_type == primary]
    hot = set(lb.query("contaminated_cards >= 3 and contaminated_share >= 0.25").merchant)   # same rule as the model features
    ct_net = ct[ct.merchant != "(none)"]
    # focus: merchants behind this card's flagged/fraud rows, hot merchants it used, and merchants shared with fraud cards
    key_m = set(ct_net[(ct_net.action_C > 0) | (ct_net.is_fraud == 1)].merchant) | (set(ct_net.merchant) & hot)
    fraud_cards = T[(T.is_fraud == 1) & (T.cc_num != card) & T.merchant.isin(ct_net.merchant)]
    shared = fraud_cards.groupby("merchant").cc_num.nunique().sort_values(ascending=False)
    ms = list(key_m)[:12] + [m for m in shared.index if m not in key_m][:max(0, 20 - len(key_m))]
    top_cards = fraud_cards[fraud_cards.merchant.isin(ms)].groupby("cc_num").merchant.nunique().sort_values(ascending=False).head(12).index
    sub = pd.concat([ct_net[ct_net.merchant.isin(ms)][["cc_num", "merchant", "is_fraud", "amt"]],
                     fraud_cards[fraud_cards.cc_num.isin(top_cards) & fraud_cards.merchant.isin(ms)][["cc_num", "merchant", "is_fraud", "amt"]]])
    if sub.empty:
        st.info("This account has no link entity to draw (no flagged merchants or shared fraud cards).")
        st.stop()
    st.caption(f"Showing {len(ms)} merchants that matter for this card (its flagged or fraud rows, merchants meeting the compromise rule, "
               f"merchants shared with fraud cards) and up to {len(top_cards)} other fraud cards that used them.")
    g = nx.Graph()
    for (c_, m_), grp in sub.groupby(["cc_num", "merchant"]):
        g.add_node(("c", c_), fraud=int(sub[sub.cc_num == c_].is_fraud.max()))
        g.add_node(("m", m_), hot=m_ in hot)
        g.add_edge(("c", c_), ("m", m_))
    pos = nx.spring_layout(g, seed=3, k=0.6)
    ex, ey = [], []
    for u, v in g.edges:
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#888", width=1), hoverinfo="none", showlegend=False))
    def add(nodes, name, color, sym, size):
        if nodes:
            fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name=name,
                                     marker=dict(color=color, symbol=sym, size=size), text=[str(n[1]) for n in nodes], hoverinfo="text"))
    cards_ = [n for n in g if n[0] == "c"]
    add([n for n in cards_ if n[1] == card], "Selected card", "#ffd166", "star", 20)
    add([n for n in cards_ if n[1] != card and g.nodes[n]["fraud"]], "Other card (fraud)", "#d1495b", "circle", 9)
    add([n for n in cards_ if n[1] != card and not g.nodes[n]["fraud"]], "Other card (clean)", "#3b82a0", "circle", 9)
    add([n for n in g if n[0] == "m" and g.nodes[n]["hot"]], "Merchant (meets compromise rule)", "#e08a1e", "square", 13)
    add([n for n in g if n[0] == "m" and not g.nodes[n]["hot"]], "Merchant", "#9aa5b1", "square", 10)
    fig.update_layout(height=520, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

# --------------------------------- monitoring ---------------------------------------
with tab4:
    p_ = tabs["psi_by_month"]
    fig = px.bar(p_, x="month", y="score_psi", title="Score PSI vs validation period (>0.10 watch, >0.25 act)")
    fig.add_hline(y=0.10, line_dash="dot"); fig.add_hline(y=0.25, line_dash="dash")
    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(px.line(p_, x="month", y=["mean_score", "fraud_rate"], title="Mean score vs realised fraud rate"),
                    width="stretch")
    fp = pd.Series(summ["feature_psi"], name="PSI").rename_axis("feature").reset_index()
    st.dataframe(fp.style.format({"PSI": "{:.3f}"}), width="stretch")

# --------------------------------- model & network ----------------------------------
with tab5:
    ab = summ["ablation"]
    st.subheader("Do graph features add value? (test period, mean +/- sd over seeds)")
    rows = []
    for k_, lab in [("without_graph", "Without graph features"), ("with_graph", "With graph features")]:
        v = ab[k_]
        rows.append({"Model": lab, "PR-AUC": f"{v['pr_auc_mean_std'][0]:.4f} +/- {v['pr_auc_mean_std'][1]:.4f}",
                     "Fraud recall @1% decline rate": f"{v['recall_at_1pct'][0]:.1%}",
                     "Fraud $ caught @1% decline rate": f"{v['fraud_dollars_caught_at_1pct'][0]:.1%}"})
    st.table(pd.DataFrame(rows))
    st.caption(f"Test fraud base rate {summ['split']['test_fraud_rate']:.2%}. Accuracy is deliberately not reported: "
               "approving everything would score ~ 1 - base rate.")
    a, b = st.columns(2)
    sh = tabs["shap_global"].head(15).sort_values("mean_abs_shap")
    a.plotly_chart(px.bar(sh, x="mean_abs_shap", y="feature", orientation="h", title="Top drivers (mean |SHAP|, flagged txns)"),
                   width="stretch")
    b.markdown("**Common-point-of-compromise leaderboard** (full sample, exploratory)")
    lb_all = tabs["cpp_merchants"]
    if "entity_type" in lb_all:
        kind = b.selectbox("Link entity", list(lb_all.entity_type.unique()))
        lb_all = lb_all[lb_all.entity_type == kind]
    b.dataframe(lb_all.head(15), width="stretch", height=420)
