import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Pseudo Data Generation ---
np.random.seed(42)
n_points = 100
dates = pd.to_datetime(pd.date_range(start="2024-01-01", periods=n_points))

# Function to generate pseudo-predictions based on selections
def generate_predictions(actual_prices, selected_ticker, selected_model, selected_fold):
    noise_level = 0
    if selected_model == "LSTM":
        noise_level += 2
    elif selected_model == "GRU":
        noise_level += 4
    elif selected_model == "Transformer":
        noise_level += 1

    if selected_fold == "Rolling":
        noise_level += 1
    elif selected_fold == "Expanding":
        noise_level += 3

    if selected_ticker == "AAPL":
        noise_level += 1
    elif selected_ticker == "GOOGL":
        noise_level += 3
    elif selected_ticker == "MSFT":
        noise_level += 2
    elif selected_ticker == "AMZN":
        noise_level += 4

    return actual_prices + np.random.normal(0, 0.5 * noise_level, n_points)

# Ticker Data
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
selected_ticker = st.sidebar.selectbox("Ticker", tickers)
actual_prices_aapl = np.random.randint(80, 180, n_points) + np.cumsum(np.random.normal(0, 2, n_points))
actual_prices_googl = np.random.randint(1500, 2500, n_points) + np.cumsum(np.random.normal(0, 5, n_points))
actual_prices_msft = np.random.randint(200, 350, n_points) + np.cumsum(np.random.normal(0, 3, n_points))
actual_prices_amzn = np.random.randint(2500, 4000, n_points) + np.cumsum(np.random.normal(0, 7, n_points))

actual_prices_dict = {
    "AAPL": actual_prices_aapl,
    "GOOGL": actual_prices_googl,
    "MSFT": actual_prices_msft,
    "AMZN": actual_prices_amzn,
}
actual_prices = actual_prices_dict[selected_ticker]
predictions = generate_predictions(actual_prices, selected_ticker, selected_model, selected_fold)
df_prices = pd.DataFrame({"Date": dates, "Actual": actual_prices, "Predictions": predictions})

# S&P 500 Data (for comparison)
sp500_actual = np.random.randint(1, 5, n_points) + np.cumsum(np.random.normal(0, 0.1, n_points))
sp500_predictions_lstm_rolling = sp500_actual + np.random.normal(0, 0.15, n_points)
sp500_predictions_lstm_expanding = sp500_actual + np.random.normal(0, 0.25, n_points)
sp500_predictions_gru_rolling = sp500_actual + np.random.normal(0, 0.20, n_points)
sp500_predictions_gru_expanding = sp500_actual + np.random.normal(0, 0.30, n_points)
sp500_predictions_transformer_rolling = sp500_actual + np.random.normal(0, 0.12, n_points)
sp500_predictions_transformer_expanding = sp500_actual + np.random.normal(0, 0.22, n_points)

sp500_predictions_dict = {
    ("LSTM", "Rolling"): sp500_predictions_lstm_rolling,
    ("LSTM", "Expanding"): sp500_predictions_lstm_expanding,
    ("GRU", "Rolling"): sp500_predictions_gru_rolling,
    ("GRU", "Expanding"): sp500_predictions_gru_expanding,
    ("Transformer", "Rolling"): sp500_predictions_transformer_rolling,
    ("Transformer", "Expanding"): sp500_predictions_transformer_expanding,
}
sp500_predictions = sp500_predictions_dict[(selected_model, selected_fold)]
df_sp500 = pd.DataFrame({"Date": dates, "S&P 500 Actual": sp500_actual, "S&P 500 Predictions": sp500_predictions})

# Model Type
model_types = ["LSTM", "GRU", "Transformer"]
selected_model = st.sidebar.selectbox("Model Type", model_types)

# Fold Type (CV)
fold_types = ["Rolling", "Expanding"]
selected_fold = st.sidebar.selectbox("Fold Type (CV)", fold_types)

# Metrics Data (dependent on selections for demonstration)
def generate_metrics(selected_ticker, selected_model, selected_fold):
    base_mse = 5
    base_mae = 1
    base_r2 = 0.9
    base_mape = 0.01

    if selected_ticker == "GOOGL":
        base_mse *= 1.2
        base_mae *= 1.1
        base_r2 -= 0.02
        base_mape *= 1.05
    elif selected_ticker == "MSFT":
        base_mse *= 0.9
        base_mae *= 0.95
        base_r2 += 0.01
        base_mape *= 0.98
    elif selected_ticker == "AMZN":
        base_mse *= 1.5
        base_mae *= 1.3
        base_r2 -= 0.05
        base_mape *= 1.1

    if selected_model == "GRU":
        base_mse *= 1.1
        base_mae *= 1.05
        base_r2 -= 0.01
        base_mape *= 1.02
    elif selected_model == "Transformer":
        base_mse *= 0.8
        base_mae *= 0.9
        base_r2 += 0.03
        base_mape *= 0.95

    if selected_fold == "Expanding":
        base_mse *= 1.05
        base_mae *= 1.02
        base_r2 -= 0.005
        base_mape *= 1.01

    metrics_data = {
        "Metrics": ["MSE", "MAE", "R²", "MAPE"],
        "Value": [base_mse + np.random.normal(0, 0.5),
                  base_mae + np.random.normal(0, 0.1),
                  max(0, min(1, base_r2 + np.random.normal(0, 0.01))),
                  max(0, base_mape + np.random.normal(0, 0.001))],
    }
    return pd.DataFrame(metrics_data)

df_metrics = generate_metrics(selected_ticker, selected_model, selected_fold)

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
fig_comparison = px.line(df_comparison, x="Date", y=["Predictions", "Actual", "S&P 500 Actual", "S&P 500 Predictions"],
                        labels={"value": "Value", "Date": "Date"})
fig_comparison.update_layout(legend_title_text="Legend")
st.plotly_chart(fig_comparison, use_container_width=True)

# Row 3: Metrics Table
st.subheader("Metrics")
st.dataframe(df_metrics, hide_index=True)
