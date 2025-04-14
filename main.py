from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.label_generator import LabelGenerator
from src.model_builder import ModelBuilder
from src.evaluator import Evaluator
from src.baseline_models import BaselineModels
from src.visualizer import Visualizer


def main():
    # Step 1: Load stock data
    loader = DataLoader('AAPL')
    df = loader.load()
    print("Raw data shape:", df.shape)
    print("Column names:", df.columns.tolist())

    # Step 2: Add technical indicators
    fe = FeatureEngineer(df)
    df = fe.add_indicators()
    print("After feature engineering:", df.shape)

    # Step 3: Create return-based labels
    lg = LabelGenerator(df)
    df = lg.create_labels()
    print("After labeling:", df.shape)

    # Step 4: Select features
    features = [
        'SMA_10', 'RSI_14', 'Volatility', 'Return',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0',
        'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3',
        'MOM_10', 'OBV'
    ]

    # Step 5: Train Neural Network
    builder = ModelBuilder(df, features)
    builder.preprocess()
    model = builder.build_and_train()
    builder.save()

    # Step 6: Evaluate Neural Network
    evaluator = Evaluator(model, builder.X_test, builder.y_test)
    report, roc = evaluator.evaluate()
    print("\n--- Neural Network ---")
    print(report)
    print(f"ROC AUC: {roc:.4f}")

    # Step 7: Visualizations
    y_probs = model.predict(builder.X_test).flatten()
    visuals = Visualizer(df, features, builder.y_test, y_probs, model, builder.X_test)

    visuals.plot_confusion_matrix()
    visuals.plot_roc_curve()
    visuals.plot_probability_distribution()
    visuals.plot_feature_correlation()
    visuals.plot_price_with_predictions()
    visuals.plot_predictions_vs_actual()

    # Step 8: Compare with Baseline Models
    baseline = BaselineModels(builder.X_train, builder.X_test, builder.y_train, builder.y_test)
    baseline.run_logistic_regression()
    baseline.run_random_forest()


if __name__ == "__main__":
    main()
