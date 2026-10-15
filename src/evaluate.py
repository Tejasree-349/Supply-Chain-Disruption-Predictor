import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.config import (
    MODEL_DIR,
    REPORT_DIR,
    PLOT_DIR,
    TARGET_COLUMN,
)


def evaluate_models():
    test_data_path = REPORT_DIR / "test_data.csv"

    if not test_data_path.exists():
        raise FileNotFoundError(
            "Test data not found. Run python -m src.train first."
        )

    test_df = pd.read_csv(test_data_path)

    y_test = test_df[TARGET_COLUMN].astype(int)
    X_test = test_df.drop(columns=[TARGET_COLUMN])

    results = []

    model_paths = list(MODEL_DIR.glob("*.joblib"))

    if not model_paths:
        raise FileNotFoundError(
            "No trained models found. Run python -m src.train first."
        )

    for model_path in model_paths:
        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model": model_path.stem,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(
                y_test, probabilities
            ),
        }

        results.append(metrics)

        print(f"\n{'=' * 60}")
        print("MODEL:", model_path.stem)
        print("=" * 60)

        print(classification_report(
            y_test,
            predictions,
            zero_division=0,
        ))

        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                y_test,
                predictions,
            )
        )

        display.plot()
        plt.title(f"Confusion Matrix - {model_path.stem}")
        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / f"{model_path.stem}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "model_evaluation.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print(results_df.round(4).to_string(index=False))

    print(
        "\nSaved evaluation report:",
        REPORT_DIR / "model_evaluation.csv",
    )

    return results_df


if __name__ == "__main__":
    evaluate_models()