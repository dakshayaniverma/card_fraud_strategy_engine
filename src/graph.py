"""Account-to-entity network and common-point-of-compromise (CPP) analysis.

Bipartite graph: account (card) -- link entity (a merchant). An account is *contaminated* at an entity when it used that entity and then had a confirmed
fraud within `win_days`. An entity becomes *hot* once CPP_MIN_CARDS contaminated accounts are KNOWN (fraud time
+ label lag) and they are at least CPP_MIN_SHARE of the accounts that used it. All features are point-in-time and
leave-one-out for the account being scored, so an account's own fraud never makes its own later rows look risky.
Rows whose entity is missing (NaN) get zeros.
"""
import numpy as np
import pandas as pd
import networkx as nx
from . import config as C

LAG = pd.Timedelta(days=C.LABEL_LAG_DAYS)
K = C.CPP_MIN_CARDS
S = C.CPP_MIN_SHARE
FAR = pd.Timestamp("2262-01-01")


def build_pairs(df: pd.DataFrame, ent: str, win_days: int) -> pd.DataFrame:
    """One row per (account, entity): first use, and when contamination becomes known."""
    win = pd.Timedelta(days=win_days)
    first = df.groupby(["cc_num", ent], sort=False).ts.min().rename("visit").reset_index()   # NaN entities dropped
    fraud_ts = df.loc[df.is_fraud == 1].groupby("cc_num").ts.apply(lambda s: np.sort(s.values))
    mats = np.full(len(first), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    visits = first.visit.values
    for card, idx in first.groupby("cc_num").indices.items():
        if card not in fraud_ts.index:
            continue
        f = fraud_ts[card]
        pos = np.searchsorted(f, visits[idx], side="right")   # first fraud strictly after the visit
        ok = pos < len(f)
        nxt = np.where(ok, f[np.minimum(pos, len(f) - 1)], np.datetime64("NaT", "ns"))
        good = ok & (nxt - visits[idx] <= win.to_timedelta64())
        mats[idx[good]] = nxt[good] + LAG.to_timedelta64()
    first = first.rename(columns={ent: "entity"})
    first["mature"] = pd.to_datetime(mats)
    return first


def _entity_tables(pairs):
    return {m: np.sort(g.mature.values) for m, g in pairs[pairs.mature.notna()].groupby("entity")}


def _hot_date(mats, visits):
    """First time an entity satisfies the CPP rule, given sorted maturity and visit times."""
    far = np.datetime64(FAR, "ns")
    if len(mats) < K:
        return far
    j = np.arange(1, len(mats) + 1)
    den = np.searchsorted(visits, mats - LAG.to_timedelta64(), side="right")
    ok = (j >= K) & (j >= S * np.maximum(den, 1))
    return mats[np.argmax(ok)] if ok.any() else far


def graph_features(df: pd.DataFrame, ent: str, win_days: int) -> pd.DataFrame:
    """cpp_cards, cpp_share, is_hot, exposure aligned to df's index (zeros where the entity is missing)."""
    zeros = pd.DataFrame({"cpp_cards": 0, "cpp_share": 0.0, "is_hot": np.int8(0), "exposure": 0}, index=df.index)
    sub = df[df[ent].notna()]
    if sub.empty:
        return zeros
    far = np.datetime64(FAR, "ns")
    pairs = build_pairs(sub, ent, win_days)
    mat_by_e = _entity_tables(pairs)
    vis_by_e = {m: np.sort(g.visit.values) for m, g in pairs.groupby("entity")}
    ts = sub.ts.values
    own_mat = sub[["cc_num", ent]].rename(columns={ent: "entity"}).merge(
        pairs, on=["cc_num", "entity"], how="left").mature.values

    cpp = np.zeros(len(sub), dtype=np.int32); share = np.zeros(len(sub))
    for m, idx in sub.groupby(ent, sort=False).indices.items():
        arr = mat_by_e.get(m)
        if arr is None:
            continue
        n = np.searchsorted(arr, ts[idx], side="right") - (own_mat[idx] <= ts[idx])   # NaT -> False
        d = np.searchsorted(vis_by_e[m], ts[idx] - LAG.to_timedelta64(), side="right")
        cpp[idx] = n; share[idx] = n / np.maximum(d, 1)
    out = pd.DataFrame({"cpp_cards": cpp, "cpp_share": share,
                        "is_hot": ((cpp >= K) & (share >= S)).astype(np.int8)}, index=sub.index)

    rows = []
    pairs_by_e = dict(tuple(pairs.groupby("entity")))
    for m, arr in mat_by_e.items():
        vis = vis_by_e[m]
        base = _hot_date(arr, vis)
        if base == far:
            continue
        pm = pairs_by_e[m]
        for card, visit, mat in zip(pm.cc_num.values, pm.visit.values, pm.mature.values):
            hot = base
            if not np.isnat(mat) and mat <= base:      # this account helped make it hot: recompute without it
                hot = _hot_date(np.delete(arr, np.searchsorted(arr, mat)), vis)
            if hot < far:
                rows.append((card, max(visit, hot)))
    exposure = np.zeros(len(sub), dtype=np.int32)
    if rows:
        cand = pd.DataFrame(rows, columns=["cc_num", "key"])
        keys = {c: np.sort(g.key.values) for c, g in cand.groupby("cc_num")}
        cards = sub.cc_num.values
        for card, idx in sub.groupby("cc_num", sort=False).indices.items():
            k = keys.get(card)
            if k is not None:
                exposure[idx] = np.searchsorted(k, ts[idx], side="left")
    out["exposure"] = exposure
    return out.reindex(df.index).fillna(zeros).astype({"cpp_cards": "int32", "is_hot": "int8", "exposure": "int32"})


def cpp_leaderboard(df: pd.DataFrame, ent: str, win_days: int, entity_type: str, top=25) -> pd.DataFrame:
    """Entities ranked by contaminated accounts (full-sample, exploratory: NOT point-in-time)."""
    sub = df[df[ent].notna()]
    pairs = build_pairs(sub, ent, win_days)
    cont = pairs[pairs.mature.notna()].groupby("entity").agg(
        contaminated_cards=("cc_num", "nunique"), first_known=("mature", "min"))
    stats = sub.groupby(ent).agg(cards=("cc_num", "nunique"), txns=("txn_id", "size"),
                                 category=("category", "first"), fraud_txns_at_merchant=("is_fraud", "sum"))
    stats.index.name = "entity"
    cont = cont.join(stats)
    cont["contaminated_share"] = cont.contaminated_cards / cont.cards
    cont = cont.sort_values("contaminated_cards", ascending=False).head(top).reset_index()
    cont.insert(0, "entity_type", entity_type)
    return cont.rename(columns={"entity": "merchant"})


def card_merchant_graph(sub: pd.DataFrame, hot_merchants: set) -> nx.Graph:
    g = nx.Graph()
    for (c, m), grp in sub.groupby(["cc_num", "merchant"]):
        g.add_node(("c", c), kind="card", fraud=int(sub.loc[sub.cc_num == c, "is_fraud"].max()))
        g.add_node(("m", m), kind="merchant", hot=m in hot_merchants)
        g.add_edge(("c", c), ("m", m), n=len(grp), amt=float(grp.amt.sum()))
    return g
