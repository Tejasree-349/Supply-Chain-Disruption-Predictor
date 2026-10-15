import pandas as pd

from src.config import TARGET_COLUMN, LEAKAGE_COLUMNS


def clean_data(df):
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    df = df.dropna(subset=[TARGET_COLUMN])


    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="raise"
    )

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    df = df.drop_duplicates().reset_index(drop=True)

    columns_to_drop = [
        column
        for column in LEAKAGE_COLUMNS
        if column != TARGET_COLUMN and column in df.columns
    ]

    df = df.drop(columns=columns_to_drop)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was removed "
            "during preprocessing."
        )

    return df


if __name__ == "__main__":
    from src.data_loader import load_data

    df = load_data()
    cleaned_df = clean_data(df)

    print("DATA PREPROCESSING SUCCESSFUL")

    print("\nOriginal shape:", df.shape)
    print("Cleaned shape:", cleaned_df.shape)

    print("\nRemaining columns:")
    print(cleaned_df.columns.tolist())

    print("\nTarget distribution:")
    print(
        cleaned_df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nTarget percentages:")
    print(
        (
            cleaned_df[TARGET_COLUMN]
            .value_counts(normalize=True)
            .sort_index() * 100
        ).round(2)
    )