import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from lightgbm import LGBMClassifier

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.label_generator import LabelGenerator
from src.visualizer import Visualizer

# ────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────

st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📊 Stock Return Direction Predictor")

# ────────────────────────────────
# GLOSSARY: Technical Indicators
# ────────────────────────────────
indicator_info = {
    "SMA_10": "Simple Moving Average over 10 periods. Smooths price to identify trends.",
    "RSI_14": "Relative Strength Index over 14 periods. Identifies overbought/oversold conditions.",
    "Volatility": "Rolling standard deviation of returns. Measures price variability.",
    "Return": "Daily percentage return. Measures daily price change.",
    "MACD_12_26_9": "MACD Line: Difference between 12 and 26 EMA.",
    "MACDh_12_26_9": "MACD Histogram: Difference between MACD and Signal line.",
    "MACDs_12_26_9": "MACD Signal: 9-period EMA of MACD.",
    "BBL_5_2.0": "Bollinger Band Lower. Shows lower volatility threshold.",
    "BBM_5_2.0": "Bollinger Band Middle. Simple moving average line.",
    "BBU_5_2.0": "Bollinger Band Upper. Shows upper volatility threshold.",
    "STOCHRSIk_14_14_3_3": "Stochastic RSI %K. Measures RSI momentum.",
    "STOCHRSId_14_14_3_3": "Stochastic RSI %D. Signal line for %K.",
    "MOM_10": "Momentum over 10 periods. Measures rate of price change.",
    "OBV": "On-Balance Volume. Measures volume flow to detect accumulation/distribution."
}

with st.expander("ℹ️ Indicator Glossary"):
    for name, explanation in indicator_info.items():
        st.markdown(f"**{name}**: {explanation}")

# ────────────────────────────────
# USER INPUTS
# ────────────────────────────────
ticker = st.text_input("Enter a stock ticker symbol (e.g., AAPL, MSFT)", value="AAPL")
model_choice = st.selectbox("Select a model to use:", [
    "Neural Network",
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "LSTM"
])
predict_button = st.button("Run Prediction")

if predict_button:
    # ────────────────────────────────
    # STEP 1: Load + preprocess data
    # ────────────────────────────────
    loader = DataLoader(ticker)
    df = loader.load()

    fe = FeatureEngineer(df)
    df = fe.add_indicators()

    lg = LabelGenerator(df)
    df = lg.create_labels()

    # ────────────────────────────────
    # STEP 2: Feature selection
    # ────────────────────────────────
    features = [
        'SMA_10', 'RSI_14', 'Volatility', 'Return',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0',
        'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3',
        'MOM_10', 'OBV'
    ]

    df_model = df.copy()
    X = df_model[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ────────────────────────────────
    # STEP 3: Model selection & prediction
    # ────────────────────────────────
    if model_choice == "Neural Network":
        model = load_model("models/model.h5")
        y_probs = model.predict(X_scaled).flatten()

    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, df_model['Target'])
        y_probs = model.predict_proba(X_scaled)[:, 1]

    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, df_model['Target'])
        y_probs = model.predict_proba(X_scaled)[:, 1]

    elif model_choice == "XGBoost":
        model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_scaled, df_model['Target'])
        y_probs = model.predict_proba(X_scaled)[:, 1]

    elif model_choice == "LightGBM":
        model = LGBMClassifier(n_estimators=100, learning_rate=0.05)
        model.fit(X_scaled, df_model['Target'])
        y_probs = model.predict_proba(X_scaled)[:, 1]

    elif model_choice == "LSTM":
        time_steps = 10
        X_seq = []
        y_seq = []
        for i in range(time_steps, len(X_scaled)):
            X_seq.append(X_scaled[i - time_steps:i])
            y_seq.append(df_model['Target'].values[i])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        model = Sequential([
            LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2])),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=2)])

        y_probs = model.predict(X_seq).flatten()
        df_model = df_model.iloc[-len(y_probs):]  # Align df with predictions

    df_model["Prediction"] = (y_probs > 0.5).astype(int)

    # ────────────────────────────────
    # STEP 4: Display predictions
    # ────────────────────────────────
    st.subheader(f"📌 Predictions using: {model_choice}")
    st.dataframe(df_model[['Close', 'Prediction']].tail(10))

    # ────────────────────────────────
    # STEP 5: Visualizations
    # ────────────────────────────────
    visual = Visualizer(df_model, features, df_model['Target'], y_probs, model, X_scaled)

    with st.expander("📉 Confusion Matrix"):
        st.pyplot(visual.plot_confusion_matrix())

    with st.expander("📈 ROC Curve"):
        st.pyplot(visual.plot_roc_curve())

    with st.expander("📊 Probability Distribution"):
        st.pyplot(visual.plot_probability_distribution())

    with st.expander("🔍 Feature Correlation"):
        st.pyplot(visual.plot_feature_correlation())

    with st.expander("💹 Price with Prediction Signals"):
        st.pyplot(visual.plot_price_with_predictions())
