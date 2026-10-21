"""Synthetic stand-in with the Sparkov schema, ONLY for testing the pipeline end to end.

It injects three fraud mechanisms (card takeover bursts, night high-value, and
common-point-of-compromise merchants) so every layer has something to find.
Because *we* wrote the fraud patterns, any metric computed on this data says nothing
about real performance. Never quote demo numbers; run on the real Sparkov data.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CATS = {  # category: (median ticket, online?)
    "grocery_pos": (65, 0), "gas_transport": (55, 0), "home": (60, 0), "shopping_pos": (50, 0),
    "kids_pets": (45, 0), "personal_care": (40, 0), "health_fitness": (45, 0), "food_dining": (30, 0),
    "entertainment": (40, 0), "misc_pos": (40, 0), "travel": (120, 0),
    "shopping_net": (70, 1), "misc_net": (55, 1), "grocery_net": (50, 1)}


def generate(out_dir: Path, n_cards=900, n_merch=450, seed=7):
    rng = np.random.default_rng(seed)
    cats = list(CATS)
    med = np.array([CATS[c][0] for c in cats]); online = np.array([CATS[c][1] for c in cats])

    # merchants
    m_cat = rng.integers(0, len(cats), n_merch)
    m_lat = rng.uniform(26, 48, n_merch); m_lon = rng.uniform(-123, -70, n_merch)
    m_name = np.array([f"Merchant_{i:04d}" for i in range(n_merch)])
    online_ids = np.where(online[m_cat] == 1)[0]
    pos_ids = np.where(online[m_cat] == 0)[0]

    # cards
    c_lat = rng.uniform(27, 47, n_cards); c_lon = rng.uniform(-122, -71, n_cards)
    c_num = rng.integers(10**15, 10**16 - 1, n_cards)
    c_start = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 120, n_cards), "D")
    c_rate = rng.uniform(0.15, 0.6, n_cards)
    c_scale = rng.lognormal(0, 0.35, n_cards)
    c_dob = pd.Timestamp("1950-01-01") + pd.to_timedelta(rng.integers(0, 365 * 50, n_cards), "D")
    # each card shops at ~25 nearby merchants
    d2 = (c_lat[:, None] - m_lat[None, pos_ids]) ** 2 + (c_lon[:, None] - m_lon[None, pos_ids]) ** 2
    near = pos_ids[np.argsort(d2, axis=1)[:, :25]]

    end = pd.Timestamp("2020-12-31")
    rows = []
    hour_p = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 7, 7, 7, 8, 8, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2], float)
    hour_p /= hour_p.sum()

    def add(ci, ts, mi, amt, fraud):
        rows.append((ci, ts, mi, amt, fraud))

    # legitimate traffic
    for ci in range(n_cards):
        days = (end - c_start[ci]).days
        n = rng.poisson(c_rate[ci] * days)
        d = rng.integers(0, days, n)
        h = rng.choice(24, n, p=hour_p)
        ts = c_start[ci] + pd.to_timedelta(d, "D") + pd.to_timedelta(h * 3600 + rng.integers(0, 3600, n), "s")
        use_online = rng.random(n) < 0.12
        mi = np.where(use_online, rng.choice(online_ids, n), rng.choice(near[ci], n))
        amt = np.round(rng.lognormal(np.log(med[m_cat[mi]]), 0.8) * c_scale[ci], 2).clip(1, 3000)
        for t, m, a in zip(ts, mi, amt):
            add(ci, t, m, a, 0)

    # CPP: compromised merchants, cards that shop there in the window get defrauded later
    comp = rng.choice(pos_ids, 10, replace=False)
    for m in comp:
        w0 = pd.Timestamp("2019-04-01") + pd.to_timedelta(int(rng.integers(0, 500)), "D")
        victims = rng.choice(n_cards, 70, replace=False)
        for ci in victims:
            t0 = w0 + pd.to_timedelta(int(rng.integers(0, 45)), "D") + pd.to_timedelta(int(rng.integers(8, 20)), "h")
            if t0 < c_start[ci] or t0 > end - pd.Timedelta(days=60):
                continue
            add(ci, t0, m, round(float(rng.lognormal(3.6, 0.5)), 2), 0)
            if rng.random() < 0.55:
                f0 = t0 + pd.to_timedelta(int(rng.integers(8, 40)), "D")
                for k in range(int(rng.integers(1, 4))):
                    fm = rng.choice(online_ids)
                    add(ci, f0 + pd.to_timedelta(int(rng.integers(0, 4000)), "s"), fm,
                        round(float(rng.uniform(80, 1400)), 2), 1)

    # takeover bursts: small card-testing then a large night purchase, away from home
    for ci in rng.choice(n_cards, int(0.45 * n_cards), replace=False):
        for _ in range(int(rng.integers(1, 3))):
            day = c_start[ci] + pd.to_timedelta(int(rng.integers(30, (end - c_start[ci]).days)), "D")
            t0 = day + pd.to_timedelta(int(rng.choice([22, 23, 0, 1, 2, 3])), "h")
            for k in range(int(rng.integers(2, 6))):
                big = k >= 1
                add(ci, t0 + pd.to_timedelta(int(k * rng.integers(60, 900)), "s"), rng.choice(online_ids),
                    round(float(rng.uniform(250, 1400) if big else rng.uniform(1, 12)), 2), 1)

    r = pd.DataFrame(rows, columns=["ci", "ts", "mi", "amt", "is_fraud"])
    r = r[r.ts <= end + pd.Timedelta(days=1)].sort_values("ts").reset_index(drop=True)
    online_m = online[m_cat[r.mi.values]] == 1
    # online merchants have no meaningful location: put the "merchant" near a random spot
    mlat = np.where(online_m, c_lat[r.ci.values] + rng.normal(0, 6, len(r)), m_lat[r.mi.values] + rng.normal(0, .01, len(r)))
    mlon = np.where(online_m, c_lon[r.ci.values] + rng.normal(0, 8, len(r)), m_lon[r.mi.values] + rng.normal(0, .01, len(r)))
    df = pd.DataFrame({
        "trans_date_trans_time": r.ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": c_num[r.ci.values],
        "merchant": "fraud_" + m_name[r.mi.values],
        "category": np.array(cats)[m_cat[r.mi.values]],
        "amt": r.amt.values,
        "first": "A", "last": "B", "gender": np.where(r.ci.values % 2, "F", "M"),
        "street": "1 Main St", "city": "Town", "state": "NA", "zip": 10000,
        "lat": c_lat[r.ci.values], "long": c_lon[r.ci.values],
        "city_pop": (1000 * (1 + r.ci.values % 90)),
        "job": "x", "dob": c_dob[r.ci.values].strftime("%Y-%m-%d"),
        "trans_num": [f"t{i:08d}" for i in range(len(r))],
        "unix_time": r.ts.astype("int64") // 10**9,
        "merch_lat": mlat, "merch_long": mlon, "is_fraud": r.is_fraud.values})
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "demo_transactions.csv")
    print(f"demo data: {len(df):,} rows, fraud rate {df.is_fraud.mean():.3%} -> {out_dir}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    generate(Path(ap.parse_args().out))
