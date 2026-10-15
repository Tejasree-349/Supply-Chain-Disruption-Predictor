import pandas as pd

from src.config import RAW_DATA_PATH


def load_data():

    """
    Load the raw supply chain dataset.
    """

    if not RAW_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}\n"
            "Place DataCoSupplyChainDataset.csv "
            "inside data/raw/."
        )

    df = pd.read_csv(
        RAW_DATA_PATH,
        encoding="latin-1",
        low_memory=False
    )

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    return df


if __name__ == "__main__":

    df = load_data()

    print("\nDATASET LOADED SUCCESSFULLY")

    print("\nDataset shape:")
    print(df.shape)

    print("\nNumber of rows:")
    print(df.shape[0])

    print("\nNumber of columns:")
    print(df.shape[1])

    print("\nColumn names:")

    for column in df.columns:
        print(column)

    print("\nFirst five records:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    print("\nMissing values:")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )