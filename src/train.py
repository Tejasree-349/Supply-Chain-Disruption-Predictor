import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.config import (
    TARGET_COLUMN,
    MODEL_DIR,
    REPORT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features


def build_preprocessor(X):
    numerical_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, numerical_columns),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def train_models():
    df = clean_data(load_data())
    df = engineer_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    if X.empty:
        raise ValueError("No features available for training.")

    # Initial stratified holdout.
    # For a final time-sensitive model, replace this with a
    # chronological split after verifying date semantics.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    preprocessor = build_preprocessor(X_train)

    saved_models = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        print(f"\nTraining {model_name}...")

        pipeline.fit(X_train, y_train)

        model_path = MODEL_DIR / f"{model_name}.joblib"

        joblib.dump(pipeline, model_path)

        saved_models[model_name] = pipeline

        print(f"Saved model: {model_path}")

    test_data_path = REPORT_DIR / "test_data.csv"

    test_output = X_test.copy()
    test_output[TARGET_COLUMN] = y_test

    test_output.to_csv(test_data_path, index=False)

    with open(
        REPORT_DIR / "feature_columns.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(X.columns.tolist(), file, indent=2)

    print("\nTraining completed.")
    print("Test data saved to:", test_data_path)

    return saved_models


if __name__ == "__main__":
    train_models()