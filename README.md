# California Housing Price Prediction

A Streamlit web application that predicts California median house values using a Linear Regression model trained on the California Housing dataset.

---

## Project Structure

```
.
├── california_housing.csv   # Raw dataset
├── requirements.txt         # Python dependencies
├── app.py                   # Streamlit home page
├── pages/
│   ├── 1_EDA.py             # Exploratory Data Analysis
│   ├── 2_Metrics.py         # Model performance metrics
│   └── 3_Predict.py         # Interactive prediction form
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading & preprocessing
│   └── train.py             # Model training script
└── model/
    ├── model.pkl            # Saved LinearRegression model
    └── scaler.pkl           # Saved StandardScaler
```

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

This reads `california_housing.csv`, trains a Linear Regression model, and saves `model/model.pkl` and `model/scaler.pkl`.

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## App Pages

| Page | Description |
|---|---|
| **Home** | Dataset preview and summary statistics |
| **EDA** | Feature distributions, correlation heatmap, California geo map |
| **Metrics** | R², RMSE, MAE, actual vs. predicted, residuals, feature coefficients |
| **Predict** | Interactive sliders to get a live house price prediction |

---

## Dataset

The California Housing dataset contains 20,640 records with the following features:

| Feature | Description |
|---|---|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| `MedHouseVal` | **Target** — median house value (in $100,000s) |

---

## Model

- **Algorithm**: Linear Regression (scikit-learn)
- **Preprocessing**: StandardScaler on all 8 features
- **Split**: 80% train / 20% test, `random_state=42`
