"""Central configuration: paths, label-maturity assumptions and cost parameters."""
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DEMO_DIR = ROOT / "data" / "demo"
SQL_DIR = ROOT / "sql"
ARTIFACTS = ROOT / "artifacts"

# A fraud label is only "known" this many days after the fraudulent transaction
# (chargeback / customer report delay). Every label-derived feature is built only
# from labels older than this, and train stops this many days before validation.
LABEL_LAG_DAYS = 7

# Common point of compromise (CPP)
CPP_WINDOW_DAYS = 60   # card is "contaminated" if fraud follows a visit within this window
CPP_MIN_CARDS = 3      # merchant is "hot" once this many contaminated cards are known ...
CPP_MIN_SHARE = 0.25   # ... AND they are at least this share of the cards that visited (tune on TRAIN only)

# Time-based split (quantiles of the timeline)
VALID_START_Q = 0.60
TEST_START_Q = 0.75

# Segmentation
PREMIUM_MIN_TENURE_DAYS = 180
PREMIUM_AMT_QUANTILE = 0.75


@dataclass
class CostParams:
    """Economic assumptions (USD). These are assumptions, not facts: the app lets you flex them."""
    revenue_rate: float = 0.02         # net revenue earned on a good transaction's spend
    fraud_fixed_cost: float = 5.0      # handling / chargeback cost per missed fraud transaction
    attrition_prob: float = 0.05       # P(customer churns | falsely declined)
    clv_standard: float = 300.0        # value lost if a standard customer churns
    clv_premium: float = 1500.0        # value lost if a premium customer churns
    stepup_cost: float = 0.30          # OTP / verification cost + friction per step-up
    stepup_fraud_catch: float = 0.80   # share of fraudulent step-ups that fail verification
    stepup_good_abandon: float = 0.03  # share of good step-ups abandoned (lost spend)

    def to_dict(self):
        return asdict(self)
