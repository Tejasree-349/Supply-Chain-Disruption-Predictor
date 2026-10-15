import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from src.config import (
    MODEL_DIR,
    REPORT_DIR,
    PLOT_DIR,
    TARGET_COLUMN,
)


def explain_model():
    model_path = MODEL_DIR / "random_forest.joblib"
    test_data_path = REPORT_DIR / "test_data.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            "Random Forest model not found. Train the models first."
        )

    if not test_data_path.exists():
        raise FileNotFoundError(
            "Test data not found. Train the models first."
        )

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)

    y_test = test_df[TARGET_COLUMN].astype(int)
    X_test = test_df.drop(columns=[TARGET_COLUMN])

    # Use a subset for faster permutation importance.
    sample_size = min(2000, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42,
    )

    y_sample = y_test.loc[X_sample.index]

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="f1",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values(
        by="Importance_Mean",
        ascending=False,
    )

    importance_df.to_csv(
        REPORT_DIR / "permutation_importance.csv",
        index=False,
    )

    top_features = importance_df.head(20).sort_values(
        by="Importance_Mean",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["Feature"],
        top_features["Importance_Mean"],
        xerr=top_features["Importance_Std"],
    )

    plt.title("Top 20 Features by Permutation Importance")
    plt.xlabel("Decrease in F1-score after shuffling")
    plt.ylabel("Feature")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "permutation_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Feature importance analysis completed.")
    print(importance_df.head(20).to_string(index=False))

    return importance_df


if __name__ == "__main__":
    explain_model()