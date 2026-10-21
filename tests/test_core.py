import numpy as np
import pandas as pd
import pytest
from src import config as C, strategy as S
from src.features import build_features
from src.generate_demo_data import generate


def test_cost_optimal_matches_hand_calc():
    cp = C.CostParams()
    amt = np.array([100.0]); prem = np.array([False])
    F, G = S.cost_tables(amt, prem, cp)
    # p = 0.5 on a $100 txn: approve costs 0.5*105, decline costs 0.5*(2 + 15)
    assert S.cost_optimal_actions(np.array([0.5]), F, G)[0] == S.DECLINE
    assert S.cost_optimal_actions(np.array([0.0001]), F, G)[0] == S.APPROVE


def test_premium_gets_more_leeway():
    cp = C.CostParams()
    amt = np.full(2, 100.0); prem = np.array([False, True])
    F, G = S.cost_tables(amt, prem, cp)
    # at a score where standard customers are declined, premium customers should not be (higher attrition cost)
    a = S.cost_optimal_actions(np.full(2, 0.2), F, G)
    assert a[1] <= a[0]


def test_evaluate_approve_all_is_zero_value():
    cp = C.CostParams(); y = np.array([1, 0, 0, 1]); amt = np.array([50., 20, 10, 300])
    F, G = S.cost_tables(amt, np.zeros(4, bool), cp)
    r = S.evaluate(np.zeros(4, int), y, amt, F, G, cp)
    assert r["net_value"] == pytest.approx(0) and r["pct_fraud_dollars_caught"] == 0


def test_psi_zero_for_identical():
    x = np.random.default_rng(0).normal(size=5000)
    assert S.psi(x, x) < 1e-6


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    from src.data import load_transactions
    d = tmp_path_factory.mktemp("demo")
    generate(d, n_cards=150, n_merch=120, seed=3)
    return load_transactions(d)


def test_no_label_leakage(small):
    """Flipping labels AFTER time T must not change features of rows up to T + label lag."""
    T = small.ts.quantile(0.6)
    alt = small.copy()
    late = alt.ts > T
    alt.loc[late, "is_fraud"] = 1 - alt.loc[late, "is_fraud"]
    a, b = build_features(small), build_features(alt)
    # base rate (global smoothing prior) differs by construction; compare via cols not using it
    ok = (small.ts <= T + pd.Timedelta(days=C.LABEL_LAG_DAYS)).values
    for col in ["g_merchant_cpp_cards", "g_merchant_is_hot", "g_merchant_exposure", "txn_cnt_1h", "amt_ratio_avg"]:
        np.testing.assert_allclose(a.loc[ok, col].values.astype(float), b.loc[ok, col].values.astype(float), equal_nan=True, err_msg=col)
