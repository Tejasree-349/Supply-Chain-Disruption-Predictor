import pandas as pd

from src.config import TARGET_COLUMN


def validate_data(df):

    """
    Validate the structure and target
    of the supply chain dataset.
    """

    if df.empty:

        raise ValueError(
            "The dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found."
        )

    # Check missing target values
    if df[TARGET_COLUMN].isnull().any():

        raise ValueError(
            "The target column contains missing values."
        )

    # Check target values
    target_values = set(
        df[TARGET_COLUMN].unique()
    )

    if not target_values.issubset({0, 1}):

        raise ValueError(
            f"Unexpected target values: {target_values}"
        )

    print("DATA VALIDATION SUCCESSFUL")

    print("\nDataset shape:")
    print(df.shape)

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nMissing values:")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    print("\nTarget distribution:")
    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nTarget percentage:")
    print(
        (
            df[TARGET_COLUMN]
            .value_counts(normalize=True)
            * 100
        ).sort_index().round(2)
    )

    return True


if __name__ == "__main__":

    from src.data_loader import load_data

    df = load_data()

    validate_data(df)