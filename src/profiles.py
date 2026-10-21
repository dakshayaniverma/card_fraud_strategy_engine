"""Dataset profiles: everything that differs is configured here."""
from dataclasses import dataclass, field
from pathlib import Path
from . import config as C

@dataclass(frozen=True)
class Profile:
    name: str
    label: str
    data_dir: Path
    demo_dir: Path
    sql_core: str
    entities: dict                 # prefix -> human label; raw column is f"ent_{prefix}"
    cpp_window_days: int
    premium_min_tenure_days: int
    native_num: tuple = ()
    native_cat: tuple = ()
    artifacts: str = "artifacts"
    demo_artifacts: str = "artifacts_demo"


SPARKOV = Profile("sparkov", "Sparkov (Kaggle)", C.RAW_DIR, C.DEMO_DIR, "01_velocity_behaviour.sql",
                  {"merchant": "merchant"}, C.CPP_WINDOW_DAYS, C.PREMIUM_MIN_TENURE_DAYS)
PROFILES = {"sparkov": SPARKOV}
