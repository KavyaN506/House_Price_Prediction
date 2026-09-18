# California Housing Price Prediction — Project Plan

## Top-Level Overview

Build a Python-only house price prediction app using the **California Housing dataset** (`california_housing.csv` already present in the workspace).

- **Backend**: A trained `LinearRegression` model (scikit-learn), persisted to disk as a `.pkl` file via `joblib`.
- **Frontend**: A **Streamlit** multi-page app with three sections — EDA (Exploratory Data Analysis), Model Metrics, and an interactive Prediction Form.
- **Data**: The raw CSV is loaded and preprocessed with `pandas`. No external API or database is required.
- **Target column**: `MedHouseVal` (Median House Value, in $100,000s).

---

## Architecture Overview

```
california_housing.csv
        │
        ▼
src/data_loader.py      ← loads & preprocesses the CSV
        │
        ▼
src/train.py            ← trains LinearRegression, saves model + scaler
        │
  model/model.pkl
  model/scaler.pkl
        │
        ▼
app.py  (Streamlit)
  ├── pages/1_EDA.py          ← distributions, correlation heatmap, geo map
  ├── pages/2_Metrics.py      ← R², RMSE, MAE, actual-vs-predicted plot
  └── pages/3_Predict.py      ← user input form → live price prediction
```

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffolding

**Intent**  
Set up the folder structure, `requirements.txt`, and a top-level `README.md` so that every subsequent task has a stable home.

**Expected Outcomes**
- Folder structure: `src/`, `model/`, `pages/`, `app.py` (entry), `requirements.txt`, `README.md`.
- `requirements.txt` pins: `streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `joblib`.

**Todo List**
1. Create `src/` and `model/` and `pages/` directories (empty `__init__.py` where needed).
2. Write `requirements.txt` with all required libraries.
3. Write `README.md` with project description, setup steps (`pip install -r requirements.txt`, `python src/train.py`, `streamlit run app.py`).

**Relevant Context**
- Workspace root already contains `california_housing.csv`.

**Status**: `[x] done`

---

### Sub-Task 2 — Data Loading & Preprocessing Module

**Intent**  
Centralise all data loading and preprocessing into `src/data_loader.py` so both the training script and the Streamlit app share one source of truth.

**Expected Outcomes**
- `load_raw_data()` → returns a `pd.DataFrame` from the CSV.
- `get_features_and_target(df)` → returns `X` (8 feature columns) and `y` (`MedHouseVal`).
- `get_scaled_data(X, scaler=None)` → fits a `StandardScaler` if none supplied, returns `(X_scaled, scaler)`.
- All column names are defined as a constant `FEATURE_COLS`.

**Todo List**
1. Create `src/data_loader.py`.
2. Implement `load_raw_data()` — reads `california_housing.csv` relative to the project root.
3. Implement `get_features_and_target(df)`.
4. Implement `get_scaled_data(X, scaler=None)` using `StandardScaler`.
5. Export `FEATURE_COLS` list.

**Relevant Context**
- CSV columns: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`, `MedHouseVal`.
- No missing values expected (sklearn's built-in version is clean; CSV is the same data).

**Status**: `[x] done`

---

### Sub-Task 3 — Model Training Script

**Intent**  
Train a `LinearRegression` model, evaluate it on a test split, save the model and scaler to `model/`, and print key metrics so the developer can verify training succeeded.

**Expected Outcomes**
- Running `python src/train.py` produces `model/model.pkl` and `model/scaler.pkl`.
- Console output shows Train R², Test R², RMSE, MAE.
- An 80/20 stratified split is used with `random_state=42` for reproducibility.

**Todo List**
1. Create `src/train.py`.
2. Call `data_loader` functions to load and preprocess data.
3. Split into train/test with `train_test_split`.
4. Fit `StandardScaler` on train set; transform both sets.
5. Fit `LinearRegression`.
6. Compute and print R², RMSE, MAE on test set.
7. Save model and scaler using `joblib.dump` to `model/model.pkl` and `model/scaler.pkl`.

**Relevant Context**
- Uses `src/data_loader.py` from Sub-Task 2.
- `model/` directory created in Sub-Task 1.

**Status**: `[x] done`

---

### Sub-Task 4 — Streamlit App Entry Point

**Intent**  
Create the top-level `app.py` that acts as the Streamlit home/landing page, loads shared resources (model, scaler, dataframe) into `st.session_state`, and provides navigation context.

**Expected Outcomes**
- `streamlit run app.py` launches without errors (after training).
- Home page shows project title, a short description, and dataset summary statistics.
- Model and scaler are loaded once and cached via `@st.cache_resource`.
- Raw dataframe is loaded once and cached via `@st.cache_data`.

**Todo List**
1. Create `app.py`.
2. Cache-load the raw dataframe using `src/data_loader.load_raw_data`.
3. Cache-load `model/model.pkl` and `model/scaler.pkl` via `joblib.load`.
4. Render home page: title, subtitle, dataset shape, `st.dataframe` preview, basic `describe()` table.

**Relevant Context**
- Streamlit multi-page apps place extra pages in `pages/` — Streamlit discovers them automatically.
- Cache decorators prevent re-loading on every interaction.

**Status**: `[x] done`

---

### Sub-Task 5 — EDA Page

**Intent**  
Give the user an exploratory data analysis page (`pages/1_EDA.py`) with rich visuals: feature distributions, a correlation heatmap, and a geographic scatter map of California.

**Expected Outcomes**
- **Distribution section**: histogram + KDE for each of the 8 features and the target using `seaborn`/`matplotlib`.
- **Correlation heatmap**: annotated heatmap of all 9 columns using `seaborn`.
- **Geo map**: Plotly `scatter_mapbox` (or `scatter_geo`) coloured by `MedHouseVal`, sized by `Population`, using `Latitude`/`Longitude`.

**Todo List**
1. Create `pages/1_EDA.py`.
2. Load data from `app.py`'s cached loader or call `load_raw_data()` directly with `@st.cache_data`.
3. Add sidebar selector so user can pick which feature's distribution to view.
4. Render histogram with KDE for the selected feature.
5. Render full correlation heatmap.
6. Render Plotly geo scatter map with colour scale for house value.

**Relevant Context**
- `Latitude` and `Longitude` columns are already in the dataset — no geocoding needed.
- Plotly `scatter_mapbox` requires a Mapbox token OR use `open-street-map` style (no token).

**Status**: `[x] done`

---

### Sub-Task 6 — Model Metrics Page

**Intent**  
Display model performance in `pages/2_Metrics.py` so users can understand how well the Linear Regression model generalises.

**Expected Outcomes**
- KPI cards showing: Test R², RMSE, MAE.
- Actual vs. Predicted scatter plot with a perfect-prediction diagonal line.
- Residuals distribution plot (histogram of `y_pred - y_test`).
- Feature coefficients bar chart showing each feature's linear weight.

**Todo List**
1. Create `pages/2_Metrics.py`.
2. Load model, scaler, and data; recreate the same 80/20 test split (`random_state=42`).
3. Compute predictions on the test set.
4. Display R², RMSE, MAE using `st.metric`.
5. Plot actual vs. predicted with Plotly (include diagonal reference line).
6. Plot residuals histogram with Plotly.
7. Plot feature coefficients as a horizontal bar chart using `model.coef_` and `FEATURE_COLS`.

**Relevant Context**
- Must use the identical `random_state=42` and split ratio as `src/train.py` to ensure the metrics match.
- `FEATURE_COLS` from `src/data_loader.py`.

**Status**: `[x] done`

---

### Sub-Task 7 — Prediction Page

**Intent**  
Provide an interactive form in `pages/3_Predict.py` where a user enters housing feature values and gets an instant predicted median house value.

**Expected Outcomes**
- Sliders/number inputs for all 8 features with sensible min/max/default values derived from the dataset.
- "Predict Price" button triggers inference.
- Predicted price displayed in a large `st.success` callout in dollars (multiply by $100,000).
- Input vector shown as a mini-table for transparency.

**Todo List**
1. Create `pages/3_Predict.py`.
2. Load model and scaler from `model/`.
3. Derive min/max/mean for each feature from the dataset to set slider bounds.
4. Render `st.slider` or `st.number_input` for each of the 8 features.
5. On button click: assemble input array → scale → predict → display result.
6. Show the input feature table below the result.

**Relevant Context**
- Scaler must be applied before prediction (same scaler saved in `model/scaler.pkl`).
- Price in the dataset is in units of $100,000 — multiply by 100,000 for display.

**Status**: `[x] done`

---

## Implementation Order

Sub-Task 1 → Sub-Task 2 → Sub-Task 3 → Sub-Task 4 → Sub-Task 5 → Sub-Task 6 → Sub-Task 7

Each sub-task is independently reviewable. After Sub-Task 3 completes, `python src/train.py` must succeed before Streamlit pages are built.
