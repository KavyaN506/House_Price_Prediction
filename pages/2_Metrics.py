"""
pages/2_Metrics.py — Model Performance Metrics
- KPI cards: R², RMSE, MAE
- Actual vs. Predicted scatter plot
- Residuals distribution
- Feature coefficients bar chart
"""

import os

import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_loader import FEATURE_COLS, get_features_and_target, get_scaled_data, load_raw_data

st.set_page_config(page_title="Model Metrics", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def get_data():
    return load_raw_data()


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return joblib.load(os.path.join("model", "model.pkl"))


@st.cache_resource(show_spinner="Loading scaler...")
def get_scaler():
    return joblib.load(os.path.join("model", "scaler.pkl"))


@st.cache_data(show_spinner="Computing predictions...")
def get_test_predictions():
    df = get_data()
    model = get_model()
    scaler = get_scaler()
    X, y = get_features_and_target(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled, _ = get_scaled_data(X_test, scaler=scaler)
    y_pred = model.predict(X_test_scaled)
    return y_test.values, y_pred


df = get_data()
model = get_model()
y_test, y_pred = get_test_predictions()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("📈 Model Performance Metrics")
st.markdown("Linear Regression evaluated on the held-out 20% test set (`random_state=42`).")
st.divider()

# ---------------------------------------------------------------------------
# 1. KPI cards
# ---------------------------------------------------------------------------

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.4f}", help="Proportion of variance explained (1.0 = perfect)")
col2.metric("RMSE", f"${rmse * 100_000:,.0f}", help="Root Mean Squared Error in dollars")
col3.metric("MAE", f"${mae * 100_000:,.0f}", help="Mean Absolute Error in dollars")

st.divider()

# ---------------------------------------------------------------------------
# 2. Actual vs. Predicted
# ---------------------------------------------------------------------------

st.subheader("Actual vs. Predicted House Values")

scatter_fig = go.Figure()
scatter_fig.add_trace(go.Scatter(
    x=y_test,
    y=y_pred,
    mode="markers",
    marker=dict(color="#3b82d4", opacity=0.35, size=4),
    name="Predictions",
))
# Perfect-prediction diagonal
diag_min = float(min(y_test.min(), y_pred.min()))
diag_max = float(max(y_test.max(), y_pred.max()))
scatter_fig.add_trace(go.Scatter(
    x=[diag_min, diag_max],
    y=[diag_min, diag_max],
    mode="lines",
    line=dict(color="red", dash="dash", width=2),
    name="Perfect Prediction",
))
scatter_fig.update_layout(
    xaxis_title="Actual Value ($100k)",
    yaxis_title="Predicted Value ($100k)",
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(x=0.02, y=0.95),
    margin=dict(t=20),
)
scatter_fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
scatter_fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
st.plotly_chart(scatter_fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 3. Residuals distribution
# ---------------------------------------------------------------------------

st.subheader("Residuals Distribution")
st.markdown("Residuals = Predicted − Actual. A well-calibrated model shows residuals centred near zero.")

residuals = y_pred - y_test
res_fig = px.histogram(
    x=residuals,
    nbins=80,
    labels={"x": "Residual ($100k)"},
    color_discrete_sequence=["#7c5cd8"],
    title="Distribution of Residuals",
)
res_fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero", annotation_position="top right")
res_fig.update_layout(bargap=0.05, plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
st.plotly_chart(res_fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 4. Feature Coefficients
# ---------------------------------------------------------------------------

st.subheader("Feature Coefficients")
st.markdown(
    "The magnitude and sign of each coefficient shows how each standardised feature "
    "influences the predicted house value."
)

coef_data = sorted(
    zip(FEATURE_COLS, model.coef_),
    key=lambda x: abs(x[1]),
    reverse=True,
)
features_sorted = [c[0] for c in coef_data]
coefs_sorted = [c[1] for c in coef_data]
colors = ["#3b82d4" if c >= 0 else "#ef4444" for c in coefs_sorted]

coef_fig = go.Figure(go.Bar(
    x=coefs_sorted,
    y=features_sorted,
    orientation="h",
    marker_color=colors,
))
coef_fig.update_layout(
    xaxis_title="Coefficient (on standardised features)",
    yaxis_title="",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=20),
)
coef_fig.add_vline(x=0, line_color="#57606a", line_width=1)
coef_fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
st.plotly_chart(coef_fig, use_container_width=True)

st.caption("Blue bars = positive effect on price | Red bars = negative effect on price")
