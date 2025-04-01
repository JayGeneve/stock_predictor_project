import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

class Visualizer:
    def __init__(self, df, features, y_test, y_probs, model=None, X_test=None):
        self.df = df
        self.features = features
        self.y_test = y_test
        self.y_probs = y_probs
        self.model = model
        self.X_test = X_test

    def plot_confusion_matrix(self):
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            self.y_test,
            (self.y_probs > 0.5).astype(int),
            display_labels=["Down", "Up"],
            cmap="Blues",
            ax=ax
        )
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        return fig

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_probability_distribution(self):
        fig, ax = plt.subplots()
        sns.histplot(self.y_probs, bins=25, kde=True, ax=ax)
        ax.set_title("Distribution of Predicted Probabilities")
        ax.set_xlabel("Probability of Up (Class 1)")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        return fig

    def plot_feature_correlation(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = self.df[self.features + ['Target']].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        return fig

    def plot_price_with_predictions(self):
        fig, ax = plt.subplots(figsize=(12, 5))
        df_test = self.df.iloc[-len(self.y_test):].copy()
        df_test["Predicted Class"] = (self.y_probs > 0.5).astype(int)

        ax.plot(df_test.index, df_test['Close'], label='Stock Price', alpha=0.7)

        up_signals = df_test[df_test["Predicted Class"] == 1]
        down_signals = df_test[df_test["Predicted Class"] == 0]

        ax.scatter(up_signals.index, up_signals['Close'], marker="^", color="green", label="Up Signal")
        ax.scatter(down_signals.index, down_signals['Close'], marker="v", color="red", label="Down Signal")

        ax.set_title("Stock Price with Predicted Return Directions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.tight_layout()
        return fig
