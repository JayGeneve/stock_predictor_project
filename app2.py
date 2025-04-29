import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Pseudo Data Generation ---
np.random.seed(42)
n_points = 100
dates = pd.to_datetime(pd.date_range(start="2024-01-01", periods=n_points))

# Model Type
model_types = ["Lasso", "Ridge", "RandomForest", "HistGradientBoostingRegressor"]
selected_model = st.sidebar.selectbox("Model Type", model_types)

# Layers Data
layers_options = ["1", "2", "3", "4"]
selected_layers = st.sidebar.selectbox("Layers", layers_options)

# Fold Type (CV)
fold_types = ["Rolling", "Expanding"]
selected_fold = st.sidebar.selectbox("Fold Type (CV)", fold_types)

# Function to generate pseudo-predictions based on selections
def generate_predictions(actual_prices, selected_layers, selected_model, selected_fold):
    noise_level = 0
    if selected_model == "Lasso":
        noise_level += 1
    elif selected_model == "Ridge":
        noise_level += 2
    elif selected_model == "RandomForest":
        noise_level += 3
    elif selected_model == "HistGradientBoostingRegressor":
        noise_level += 4

    if selected_fold == "Rolling":
        noise_level += 1
    elif selected_fold == "Expanding":
        noise_level += 3

    if selected_layers == "1":
        noise_level += 1
    elif selected_layers == "2":
        noise_level += 2
    elif selected_layers == "3":
        noise_level += 3
    elif selected_layers == "4":
        noise_level += 4

    return actual_prices + np.random.normal(0, 0.5 * noise_level, n_points)

# Dummy actual prices (we don't have ticker data anymore in the same way)
actual_prices = np.random.randint(50, 200, n_points) + np.cumsum(np.random.normal(0, 1.5, n_points))

predictions = generate_predictions(actual_prices, selected_layers, selected_model, selected_fold)
df_prices = pd.DataFrame({"Date": dates, "Actual": actual_prices, "Predictions": predictions})

# S&P 500 Data (for comparison) - Adjusted for new model types
sp500_actual = np.random.randint(1, 5, n_points) + np.cumsum(np.random.normal(0, 0.1, n_points))
sp500_predictions_lasso_rolling = sp500_actual + np.random.normal(0, 0.10, n_points)
sp500_predictions_lasso_expanding = sp500_actual + np.random.normal(0, 0.18, n_points)
sp500_predictions_ridge_rolling = sp500_actual + np.random.normal(0, 0.12, n_points)
sp500_predictions_ridge_expanding = sp500_actual + np.random.normal(0, 0.20, n_points)
sp500_predictions_rf_rolling = sp500_actual + np.random.normal(0, 0.15, n_points)
sp500_predictions_rf_expanding = sp500_actual + np.random.normal(0, 0.25, n_points)
sp500_predictions_hgb_rolling = sp500_actual + np.random.normal(0, 0.18, n_points)
sp500_predictions_hgb_expanding = sp500_actual + np.random.normal(0, 0.28, n_points)

sp500_predictions_dict = {
    ("Lasso", "Rolling"): sp500_predictions_lasso_rolling,
    ("Lasso", "Expanding"): sp500_predictions_lasso_expanding,
    ("Ridge", "Rolling"): sp500_predictions_ridge_rolling,
    ("Ridge", "Expanding"): sp500_predictions_ridge_expanding,
    ("RandomForest", "Rolling"): sp500_predictions_rf_rolling,
    ("RandomForest", "Expanding"): sp500_predictions_rf_expanding,
    ("HistGradientBoostingRegressor", "Rolling"): sp500_predictions_hgb_rolling,
    ("HistGradientBoostingRegressor", "Expanding"): sp500_predictions_hgb_expanding,
}
sp500_predictions = sp500_predictions_dict[(selected_model, selected_fold)]
df_sp500 = pd.DataFrame({"Date": dates, "S&P 500 Actual": sp500_actual, "S&P 500 Predictions": sp500_predictions})

# Metrics Data (dependent on selections for demonstration)
def generate_metrics(selected_layers, selected_model, selected_fold):
    base_mse = 5
    base_mae = 1
    base_r2 = 0.9
    base_mape = 0.01

    if selected_model == "Ridge":
        base_mse *= 1.1
        base_mae *= 1.05
        base_r2 -= 0.01
        base_mape *= 1.02
    elif selected_model == "RandomForest":
        base_mse *= 0.8
        base_mae *= 0.9
        base_r2 += 0.03
        base_mape *= 0.95
    elif selected_model == "HistGradientBoostingRegressor":
        base_mse *= 0.7
        base_mae *= 0.85
        base_r2 += 0.05
        base_mape *= 0.9

    if selected_fold == "Expanding":
        base_mse *= 1.05
        base_mae *= 1.02
        base_r2 -= 0.005
        base_mape *= 1.01

    if selected_layers == "2":
        base_mse *= 1.02
        base_mae *= 1.01
        base_r2 -= 0.002
        base_mape *= 1.005
    elif selected_layers == "3":
        base_mse *= 1.04
        base_mae *= 1.02
        base_r2 -= 0.004
        base_mape *= 1.01
    elif selected_layers == "4":
        base_mse *= 1.06
        base_mae *= 1.03
        base_r2 -= 0.006
        base_mape *= 1.015

    metrics_data = {
        "Metrics": ["MSE", "MAE", "R²", "MAPE"],
        "Value": [base_mse + np.random.normal(0, 0.5),
                  base_mae + np.random.normal(0, 0.1),
                  max(0, min(1, base_r2 + np.random.normal(0, 0.01))),
                  max(0, base_mape + np.random.normal(0, 0.001))],
    }
    return pd.DataFrame(metrics_data)

df_metrics = generate_metrics(selected_layers, selected_model, selected_fold)

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
