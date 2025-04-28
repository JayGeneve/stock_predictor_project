import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Pseudo Data Generation ---
np.random.seed(42)
n_points = 100

# Ticker Data
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
selected_ticker = st.sidebar.selectbox("Ticker", tickers)
actual_prices = np.random.randint(80, 180, n_points) + np.cumsum(np.random.normal(0, 2, n_points))
predictions = actual_prices + np.random.normal(0, 5, n_points)
dates = pd.to_datetime(pd.date_range(start="2024-01-01", periods=n_points))
df_prices = pd.DataFrame({"Date": dates, "Actual": actual_prices, "Predictions": predictions})

# S&P 500 Data (for comparison)
sp500_actual = np.random.randint(1, 5, n_points) + np.cumsum(np.random.normal(0, 0.1, n_points))
sp500_predictions = sp500_actual + np.random.normal(0, 0.2, n_points)
df_sp500 = pd.DataFrame({"Date": dates, "S&P 500 Actual": sp500_actual, "S&P 500 Predictions": sp500_predictions})

# Model Type
model_types = ["LSTM", "GRU", "Transformer"]
selected_model = st.sidebar.selectbox("Model Type", model_types)

# Fold Type (CV)
fold_types = ["Fold1", "Fold2", "Fold3"]
selected_fold = st.sidebar.selectbox("Fold Type (CV)", fold_types)

# Metrics Data
metrics_data = {
    "Metrics": ["MSE", "MAE", "R²", "MAPE"],
    "Value": [8.392, 2.222, 0.941, 0.014],
}
df_metrics = pd.DataFrame(metrics_data)

# --- Dashboard Layout ---
st.title("Neural Network Dashboard - Asset Pricing")

# Row 1: Predictions vs Actual
st.subheader("Predictions vs Actual")
fig_prices = px.line(df_prices, x="Date", y=["Predictions", "Actual"],
                     labels={"value": "Price", "Date": "Date"})
st.plotly_chart(fig_prices, use_container_width=True)

# Row 2: Predictions vs Actual vs S&P 500
st.subheader("Predictions vs Actual vs S&P 500")
df_comparison = df_prices.merge(df_sp500, on="Date")
fig_comparison = px.line(df_comparison, x="Date", y=["Predictions", "Actual", "S&P 500 Actual"],
                         labels={"value": "Value", "Date": "Date"})
fig_comparison.update_layout(legend_title_text="Legend")
st.plotly_chart(fig_comparison, use_container_width=True)

# Row 3: Metrics Table
st.subheader("Metrics")
st.dataframe(df_metrics, hide_index=True)
