"""
app.py — California Housing Price Prediction
Streamlit home page: dataset overview and summary statistics.
"""

import os

import joblib
import streamlit as st

from src.data_loader import load_raw_data

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CA Housing Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join("model", "model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")


@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    return load_raw_data()


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading scaler...")
def get_scaler():
    return joblib.load(SCALER_PATH)


# Pre-load so all pages can call the same cached functions
df = get_data()
model = get_model()
scaler = get_scaler()

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------

st.title("🏡 California Housing Price Prediction")
st.markdown(
    """
    This app uses a **Linear Regression** model trained on the
    [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
    to predict median house values across California block groups.

    Use the sidebar to navigate between pages:
    | Page | Description |
    |---|---|
    | **EDA** | Feature distributions, correlation heatmap, and a California geo map |
    | **Metrics** | Model performance — R², RMSE, MAE, actual vs. predicted, residuals |
    | **Predict** | Enter feature values and get an instant price prediction |
    """
)

st.divider()

# --- Dataset snapshot ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Features", "8")
col3.metric("Target", "Median House Value")

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Summary Statistics")
st.dataframe(df.describe().T.style.format("{:.3f}"), use_container_width=True)

st.subheader("Target Distribution")
import plotly.express as px

fig = px.histogram(
    df,
    x="MedHouseVal",
    nbins=60,
    labels={"MedHouseVal": "Median House Value ($100k)"},
    title="Distribution of Median House Values",
    color_discrete_sequence=["#3b82d4"],
)
fig.update_layout(bargap=0.05, plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)
