import pandas as pd

from src.config import TARGET_COLUMN


IDENTIFIER_COLUMNS = [
    "Order_Id",
    "Order_Item_Id",
    "Customer_Id",
    "Customer_Email",
    "Customer_Password",
    "Customer_Fname",
    "Customer_Lname",
    "Product_Card_Id",
    "Product_Category_Id",
    "Product_Image",
    "Product_Description",
    "Order_Zipcode",
    "Customer_Zipcode",
]

DATE_COLUMNS = [
    "order_date_(DateOrders)",
    "shipping_date_(DateOrders)",
]


def engineer_features(df):
    df = df.copy()

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    order_date_column = "order_date_(DateOrders)"

    if order_date_column in df.columns:
        df["Order_Month"] = df[order_date_column].dt.month
        df["Order_Day"] = df[order_date_column].dt.day
        df["Order_DayOfWeek"] = (
            df[order_date_column].dt.dayofweek
        )


    existing_date_columns = [
        column for column in DATE_COLUMNS
        if column in df.columns
    ]

    df = df.drop(
        columns=existing_date_columns,
        errors="ignore",
    )

    existing_identifiers = [
        column for column in IDENTIFIER_COLUMNS
        if column in df.columns
    ]

    df = df.drop(
        columns=existing_identifiers,
        errors="ignore",
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column missing after feature engineering.")

    return df


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.preprocessing import clean_data

    df = clean_data(load_data())
    df = engineer_features(df)

    print("Feature engineering completed.")
    print("Shape:", df.shape)
    print("\nFeatures:")
    print(df.columns.tolist())