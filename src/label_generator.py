class LabelGenerator:
    def __init__(self, df, horizon=5):
        self.df = df.copy()
        self.horizon = horizon

    def create_labels(self):
        self.df['Future_Return'] = self.df['Close'].shift(-self.horizon) / self.df['Close'] - 1
        self.df['Target'] = (self.df['Future_Return'] > 0).astype(int)
        return self.df.dropna()
