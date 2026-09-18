"""
pages/1_EDA.py — Exploratory Data Analysis
- Feature distribution (histogram + KDE)
- Correlation heatmap
- California geo scatter map
"""

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st

from src.data_loader import FEATURE_COLS, TARGET_COL, load_raw_data

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    return load_raw_data()


df = get_data()
ALL_COLS = FEATURE_COLS + [TARGET_COL]

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("📊 Exploratory Data Analysis")
st.markdown("Explore distributions, feature relationships, and the geographic spread of the California Housing dataset.")
st.divider()

# ---------------------------------------------------------------------------
# 1. Feature Distribution
# ---------------------------------------------------------------------------

st.subheader("Feature Distribution")

selected_col = st.selectbox(
    "Select a feature or the target to plot:",
    options=ALL_COLS,
    index=0,
)

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df[selected_col], kde=True, bins=50, color="#3b82d4", ax=ax)
ax.set_xlabel(selected_col, fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"Distribution of {selected_col}", fontsize=14)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

# ---------------------------------------------------------------------------
# 2. Correlation Heatmap
# ---------------------------------------------------------------------------

st.subheader("Correlation Heatmap")

corr = df[ALL_COLS].corr()

fig2, ax2 = plt.subplots(figsize=(10, 7))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax2,
    cbar_kws={"shrink": 0.8},
)
ax2.set_title("Pearson Correlation Matrix", fontsize=14)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.caption(
    "Strong positive correlation between **MedInc** and **MedHouseVal** is expected — "
    "median income is the single strongest predictor of house value."
)

st.divider()

# ---------------------------------------------------------------------------
# 3. California Geo Map
# ---------------------------------------------------------------------------

st.subheader("Geographic Distribution — California")

st.markdown(
    "Each point is a census block group. **Colour** = Median House Value, **Size** = Population."
)

# Sample for rendering speed — use full dataset (20k points renders fine with Plotly)
map_fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color="MedHouseVal",
    size="Population",
    color_continuous_scale="Viridis",
    size_max=15,
    zoom=5,
    center={"lat": 37.5, "lon": -119.5},
    mapbox_style="open-street-map",
    labels={"MedHouseVal": "Med. House Value ($100k)", "Population": "Population"},
    title="Median House Value by Location",
    hover_data={"Latitude": True, "Longitude": True, "MedInc": True},
)
map_fig.update_layout(
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    coloraxis_colorbar={"title": "Value ($100k)"},
)
st.plotly_chart(map_fig, use_container_width=True)

st.caption(
    "Coastal areas (San Francisco Bay Area, Los Angeles) show the highest median house values."
)
