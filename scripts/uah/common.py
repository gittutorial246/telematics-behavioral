"""Shared configuration for the UAH-DriveSet pipeline.

Every evaluation script imports the feature list and model factory from here so
that a change in one place cannot silently disagree with another. The values are
the ones reported in the paper.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

WINDOWS = Path("data/processed/uah/window_features.parquet")
PROCESSED = Path("data/processed/uah")
RESULTS = Path("results/uah")
FIGURES = Path("figures/uah")
SEED = 42

# Window geometry (Section 3.2). Declared once because the feature builder and
# the timing benchmark must agree: each active vehicle emits one scored window
# every STEP_S seconds, not every WINDOW_S. A stale copy of STEP_S in the
# benchmark is what made every fleet-fill figure twice its correct value.
WINDOW_S = 5.0      # streaming feature window length, seconds
STEP_S = 2.5        # sliding step, seconds (50 percent overlap)

# Twelve window-level features (Section 5.1).
FEATURES = [
    "speed_mean", "speed_std", "speed_max",
    "lon_accel_mean", "lon_accel_std", "lon_accel_absmax",
    "lat_accel_mean", "lat_accel_std", "lat_accel_absmax",
    "harsh_lon_rate", "harsh_lat_rate", "jerk_std",
]

# Primary classifier (Section 5.2). n_jobs defaults to -1 for evaluation, where
# only wall-clock speed is affected; the timing benchmark overrides it to 1 so
# that its measurements are genuinely single-core.
RF_PARAMS = dict(n_estimators=300, max_depth=12, min_samples_leaf=5,
                 class_weight="balanced", random_state=SEED)


def make_rf(n_jobs=-1):
    return RandomForestClassifier(n_jobs=n_jobs, **RF_PARAMS)


def load_windows():
    """Return the window table, the feature matrix, and the label vector.

    Driver and trip groups are columns of the returned frame ("driver",
    "trip_id"); the evaluation scripts take them from there.
    """
    df = pd.read_parquet(WINDOWS)
    return df, df[FEATURES].values, df["behavior"].values


def ensure_dirs():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
