from src.data_loader import load_data
from src.data_validation import validate_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.train import train_models
from src.evaluate import evaluate_models
from src.explainability import explain_model


def main():
    print("\n" + "=" * 60)
    print("SUPPLY CHAIN DISRUPTION AND FAILURE PREDICTOR")
    print("=" * 60)

    print("\nSTEP 1: Loading data")
    df = load_data()

    print("\nSTEP 2: Validating data")
    validate_data(df)

    print("\nSTEP 3: Checking preprocessing")
    cleaned_df = clean_data(df)
    print("Cleaned dataset shape:", cleaned_df.shape)

    print("\nSTEP 4: Checking feature engineering")
    engineered_df = engineer_features(cleaned_df)
    print("Engineered dataset shape:", engineered_df.shape)

    print("\nSTEP 5: Training models")
    train_models()

    print("\nSTEP 6: Evaluating models")
    evaluate_models()

    print("\nSTEP 7: Explaining model predictions")
    explain_model()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()