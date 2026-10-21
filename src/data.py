"""Load the Sparkov transactions into one normalised, time-ordered DataFrame.

Core columns (both datasets): txn_id, ts, cc_num (integer account id), merchant (display name of the
primary link entity), category, amt, state, trans_num, is_fraud, plus ent_<prefix> link-entity columns
(NaN = unknown) that drive the graph layer.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .profiles import Profile, SPARKOV

SPARKOV_KEEP = ["txn_id", "ts", "cc_num", "merchant", "ent_merchant", "category", "amt", "lat", "lon", "merch_lat",
                "merch_lon", "city_pop", "state", "gender", "age", "trans_num", "is_fraud"]


def load_sparkov(data_dir: Path) -> pd.DataFrame:
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSVs in {data_dir}. Download fraudTrain.csv and fraudTest.csv from Kaggle "
            "('Credit Card Transactions Fraud Detection Dataset') into data/raw/, or run with --demo.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.rename(columns={"trans_date_trans_time": "ts", "long": "lon", "merch_long": "merch_lon"})
    df["ts"] = pd.to_datetime(df["ts"])
    df["merchant"] = df["merchant"].str.replace("^fraud_", "", regex=True)
    df["ent_merchant"] = df["merchant"]
    df["cc_num"] = df["cc_num"].astype("int64")
    df["age"] = df["ts"].dt.year - pd.to_datetime(df["dob"]).dt.year
    df = df.drop_duplicates("trans_num").sort_values(["ts", "trans_num"]).reset_index(drop=True)
    df["txn_id"] = range(len(df))
    return df[SPARKOV_KEEP]


def load_transactions(data_dir: Path, profile: Profile = SPARKOV) -> pd.DataFrame:
    return load_sparkov(data_dir)
