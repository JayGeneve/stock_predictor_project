import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, ticker, start="2015-01-01", end="2023-12-31"):
        self.ticker = ticker
        self.start = start
        self.end = end

    def load(self):
        df = yf.download(self.ticker, start=self.start, end=self.end, auto_adjust=False)

        # Flatten column names if they're MultiIndex (e.g. ('Close', 'AAPL') → 'Close')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]  # take just the name, not the ticker suffix

        df = df.dropna()
        return df




 