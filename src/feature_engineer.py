import pandas_ta as ta

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def add_indicators(self):
        # Existing indicators
        self.df.ta.sma(length=10, append=True)
        self.df.ta.rsi(length=14, append=True)
        self.df['Volatility'] = self.df['Close'].rolling(window=10).std()
        self.df['Return'] = self.df['Close'].pct_change()

        # ✅ New indicators
        self.df.ta.macd(append=True)
        self.df.ta.bbands(append=True)
        self.df.ta.stochrsi(append=True)
        self.df.ta.mom(append=True)
        self.df.ta.obv(append=True, volume='Volume')

        # Drop rows with any NaNs in used columns
        self.df = self.df.dropna()

        return self.df
