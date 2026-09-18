"""
train.py
Train a LinearRegression model on the California Housing dataset,
evaluate it, and persist the model + scaler to disk.

Usage:
    python src/train.py
"""

import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Make sure src/ is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_features_and_target, get_scaled_data, load_raw_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def train():
    # 1. Load data
    print("Loading data...")
    df = load_raw_data()
    print(f"  Dataset shape: {df.shape}")

    # 2. Split features / target
    X, y = get_features_and_target(df)

    # 3. Train / test split  (80 / 20, reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    # 4. Scale features — fit only on train set
    X_train_scaled, scaler = get_scaled_data(X_train)
    X_test_scaled, _ = get_scaled_data(X_test, scaler=scaler)

    # 5. Train model
    print("Training LinearRegression...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Test Set Metrics ---")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}  (×$100k = ${rmse * 100_000:,.0f})")
    print(f"  MAE  : {mae:.4f}  (×$100k = ${mae * 100_000:,.0f})")

    # Train metrics (for reference)
    y_train_pred = model.predict(X_train_scaled)
    r2_train = r2_score(y_train, y_train_pred)
    print(f"  Train R²: {r2_train:.4f}")

    # 7. Persist
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved  : {os.path.abspath(MODEL_PATH)}")
    print(f"Scaler saved : {os.path.abspath(SCALER_PATH)}")


if __name__ == "__main__":
    train()
