# 🌟 Monthly Revenue Forecasting — Time Series SARIMA Model (Python)

This project performs **time series forecasting** on monthly U.S. retail sales data using a **SARIMA** model.  
The goal is to analyze long-term trends, seasonality patterns, and generate a **12-month future forecast** of retail revenue.

The pipeline includes:

- Data cleaning & preprocessing  
- Exploratory time-series analysis  
- STL decomposition  
- Stationarity testing (ADF)  
- SARIMA hyperparameter search (AIC-based)  
- Train/Test evaluation  
- Final model training & 12-month forecast  
- Saving & loading a trained model artifact  
- Exporting forecast visualizations  

This project is built end-to-end in **Python**, following real-world best practices for forecasting workflows.

---

## 📂 Project Structure

```text
monthly-revenue-forecasting/
│
├── data/
│   ├── raw/                → Original CSV dataset
│   └── processed/          → Cleaned & transformed time-series
│
├── notebooks/
│   └── 01_monthly_revenue_forecasting.ipynb
│
├── visuals/
│   └── monthly_retail_sarima_forecast.png
│
├── models/
│   └── sarima_retail_total.pkl   → Trained SARIMA model artifact
│
└── README.md

Dataset Description

The dataset contains monthly U.S. retail sales with multiple business categories.
For this project, we extract:

➡️ Retail Sales, Total

because it provides:

Complete data from 1992–2024

Strong trend + seasonality (ideal for SARIMA)

Reliable signal for long-term forecasting

Key columns:

Column	Description
month	Monthly timestamp (YYYY-MM)
kind_of_business	Retail category
value	Total monthly sales (Millions USD)
🚀 Project Workflow
1. Load & Clean Data

Import raw CSV from data/raw/

Filter to "Retail sales, total"

Convert month to datetime

Enforce monthly frequency:

ts = df_total.set_index("month").asfreq("MS")

2. Exploratory Time-Series Analysis

Visualize raw trend

Inspect seasonal cycles

Identify structural breaks (e.g., COVID-19 shock)

3. STL Decomposition (Trend + Seasonality + Residual)

We use STL to decompose the series into:

Long-term trend

Yearly seasonality (12-month cycle)

Residual (noise / irregular shocks)

This helps us visually confirm strong seasonal structure and trend.

4. Stationarity Check (ADF Test)

We perform an Augmented Dickey–Fuller test.

Result:

ADF p-value ≈ 0.99

This indicates the series is non-stationary, mainly due to trend and seasonality.

Therefore, we apply:

First-order differencing: d = 1

Seasonal differencing with 12-month period: D = 1

5. Differencing
ts_diff = ts.diff().dropna()
ts_diff_seasonal = ts_diff.diff(12).dropna()


After differencing:

The series fluctuates around zero

Seasonality and trend are much weaker

The series is suitable for SARIMA modeling

6. SARIMA Hyperparameter Search (AIC Grid Search)

We perform an AIC-based grid search over:

p, q ∈ {0, 1, 2}

d = 1

P, Q ∈ {0, 1, 2}

D = 1

Seasonal period s = 12

The best model (lowest AIC):

order = (1, 1, 2)
seasonal_order = (2, 1, 2, 12)

7. Train/Test Split

Reserve the last 24 months as a test set

Train SARIMA on the remaining historical data

We evaluate forecast accuracy using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MAPE (Mean Absolute Percentage Error)

8. Forecast Evaluation (Test Period)

We compare:

Actual test values vs

SARIMA forecast for the same period

This is visualized in the notebook to visually assess how well the model tracks real data.

9. Final Model Training (Full Dataset)

After validating the configuration, we retrain SARIMA on the full time series to leverage all available historical data.

10. Future Forecast (12 Months)

We generate a 12-month ahead forecast using the final model, with confidence intervals.

The resulting figure is saved to:

visuals/monthly_retail_sarima_forecast.png


This plot is the main output visualization of the project.

🤖 Model Artifact

The trained SARIMA model is saved as a reusable artifact:

models/sarima_retail_total.pkl


This allows us (or other users) to load the model and generate new forecasts without retraining.

Example usage:

import joblib
import pandas as pd

# Load processed time series
ts = pd.read_csv("data/processed/monthly_revenue_retail_total.csv",
                 parse_dates=["month"],
                 index_col="month")
ts = ts.asfreq("MS")

# Load trained SARIMA model
model = joblib.load("models/sarima_retail_total.pkl")

# Generate a 12-month forecast
n_future = 12
forecast_obj = model.get_forecast(steps=n_future)
forecast_mean = forecast_obj.predicted_mean
forecast_ci = forecast_obj.conf_int()

📈 Results Summary

Clear upward long-term trend in U.S. retail spending

Strong annual seasonality (12-month patterns)

SARIMA (1, 1, 2) × (2, 1, 2, 12) captures the dynamics well

Provides a realistic 12-month forward forecast of total retail sales

Ready to be integrated into reporting, dashboards, or decision-making workflows

Final forecast chart:

visuals/monthly_retail_sarima_forecast.png

💡 Business Value

This forecasting pipeline helps organizations:

Plan future revenue and budgets

Manage inventory and supply more efficiently

Anticipate periods of high or low demand

Understand structural breaks (e.g., crisis periods)

Design data-driven sales and marketing strategies

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Statsmodels (STL decomposition, SARIMA)

Joblib (model persistence)

Jupyter Notebook

🧠 Possible Future Improvements

Add exogenous variables (SARIMAX): macroeconomic indicators, holidays, inflation, etc.

Compare SARIMA with Prophet, ETS, or XGBoost/LightGBM models

Automate model retraining and backtesting

Deploy the model via FastAPI or Flask

Build a simple BI/dashboard using Streamlit or similar tools

🎉 Project Status

✅ Completed – ready as a portfolio project.
The repository contains:

A reproducible end-to-end forecasting pipeline

Clean project structure

Trained SARIMA model artifact

Visual outputs for communication and reporting.
