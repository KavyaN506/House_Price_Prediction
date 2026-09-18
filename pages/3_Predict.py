"""
pages/3_Predict.py — Interactive House Price Prediction Form
Sliders for all 8 features -> scaled -> LinearRegression -> price in dollars.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import FEATURE_COLS, load_raw_data

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset stats...")
def get_stats():
    df = load_raw_data()
    return df[FEATURE_COLS].agg(["min", "max", "mean"]).to_dict()


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return joblib.load(os.path.join("model", "model.pkl"))


@st.cache_resource(show_spinner="Loading scaler...")
def get_scaler():
    return joblib.load(os.path.join("model", "scaler.pkl"))


stats = get_stats()
model = get_model()
scaler = get_scaler()

# ---------------------------------------------------------------------------
# Human-readable labels for sliders
# ---------------------------------------------------------------------------

FEATURE_META = {
    "MedInc":     {"label": "Median Income (10k$)",        "step": 0.1},
    "HouseAge":   {"label": "Median House Age (years)",    "step": 1.0},
    "AveRooms":   {"label": "Average Rooms per Household", "step": 0.1},
    "AveBedrms":  {"label": "Average Bedrooms per Household", "step": 0.1},
    "Population": {"label": "Block Group Population",      "step": 1.0},
    "AveOccup":   {"label": "Average Household Occupants", "step": 0.1},
    "Latitude":   {"label": "Latitude",                    "step": 0.01},
    "Longitude":  {"label": "Longitude",                   "step": 0.01},
}

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🔮 Predict House Price")
st.markdown(
    "Adjust the sliders to match the characteristics of a block group and click "
    "**Predict Price** to get an estimated median house value."
)
st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

user_inputs = {}

with st.form("prediction_form"):
    st.subheader("Block Group Features")
    col_a, col_b = st.columns(2)

    for i, col in enumerate(FEATURE_COLS):
        meta = FEATURE_META[col]
        col_min = float(stats[col]["min"])
        col_max = float(stats[col]["max"])
        col_mean = float(stats[col]["mean"])
        step = meta["step"]

        target_col = col_a if i % 2 == 0 else col_b
        user_inputs[col] = target_col.slider(
            label=meta["label"],
            min_value=round(col_min, 2),
            max_value=round(col_max, 2),
            value=round(col_mean, 2),
            step=step,
            format="%.2f",
        )

    predict_btn = st.form_submit_button("Predict Price", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if predict_btn:
    input_df = pd.DataFrame([user_inputs], columns=FEATURE_COLS)
    input_scaled = scaler.transform(input_df)
    predicted_value = model.predict(input_scaled)[0]
    predicted_dollars = predicted_value * 100_000

    st.divider()
    st.subheader("Prediction Result")

    if predicted_dollars > 0:
        st.success(f"### Predicted Median House Value: **${predicted_dollars:,.0f}**")
    else:
        st.warning(f"### Predicted Median House Value: **${predicted_dollars:,.0f}**")

    st.markdown(
        "_This estimate is based on a Linear Regression model. "
        "Actual market values may differ._"
    )

    # Show input summary
    st.subheader("Your Input Values")
    display_df = pd.DataFrame(
        {
            "Feature": [FEATURE_META[c]["label"] for c in FEATURE_COLS],
            "Value": [round(user_inputs[c], 4) for c in FEATURE_COLS],
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
