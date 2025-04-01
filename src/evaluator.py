from sklearn.metrics import classification_report, roc_auc_score

class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        preds = self.model.predict(self.X_test).flatten()
        pred_labels = (preds > 0.5).astype(int)
        report = classification_report(self.y_test, pred_labels)
        roc = roc_auc_score(self.y_test, preds)
        return report, roc
