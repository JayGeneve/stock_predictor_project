from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

class BaselineModels:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run_logistic_regression(self):
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_test)[:, 1]
        preds = model.predict(self.X_test)

        report = classification_report(self.y_test, preds)
        roc = roc_auc_score(self.y_test, probs)
        print("\n--- Logistic Regression ---")
        print(report)
        print(f"ROC AUC: {roc:.4f}")

    def run_random_forest(self):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_test)[:, 1]
        preds = model.predict(self.X_test)

        report = classification_report(self.y_test, preds)
        roc = roc_auc_score(self.y_test, probs)
        print("\n--- Random Forest ---")
        print(report)
        print(f"ROC AUC: {roc:.4f}")
