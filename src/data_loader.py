"""
data_loader.py
Centralised data loading and preprocessing for the California Housing project.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_COL = "MedHouseVal"

_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "california_housing.csv")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """Load the raw California Housing CSV and return a DataFrame."""
    return pd.read_csv(os.path.abspath(_CSV_PATH))


def get_features_and_target(df: pd.DataFrame):
    """Split a DataFrame into feature matrix X and target series y.

    Returns
    -------
    X : pd.DataFrame  — shape (n, 8)
    y : pd.Series     — shape (n,)
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def get_scaled_data(X: pd.DataFrame, scaler: StandardScaler = None):
    """Apply StandardScaler to X.

    If *scaler* is None, a new scaler is fitted on X.
    If *scaler* is provided, it is used to transform X (no re-fitting).

    Returns
    -------
    X_scaled : pd.DataFrame  — scaled features, preserves column names
    scaler   : StandardScaler
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled_values = scaler.fit_transform(X)
    else:
        X_scaled_values = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled_values, columns=FEATURE_COLS, index=X.index)
    return X_scaled, scaler
