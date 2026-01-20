import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed2")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_transactions(filename):
    df = pd.read_csv(RAW_DIR / filename)
    df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_daily_demand(df):
    """
    Aggregate transactional data into daily demand per store & product.
    """
    daily = (
        df[df["event_type"] == "sale"]
        .groupby(["store_id", "product_id", "date"])
        .agg(
            daily_demand=("quantity", "sum"),
            price=("price", "mean"),
            holiday_flag=("holiday_flag", "max"),
            weather=("weather", "first"),
        )
        .reset_index()
    )

    return daily


def add_temporal_features(df):
    """
    Add lagged, rolling, and calendar features (NO leakage).
    """
    df = df.sort_values(["store_id", "product_id", "date"])

    # Weekly lag
    df["quantity_lag_7"] = (
        df.groupby(["store_id", "product_id"])["daily_demand"]
        .shift(7)
    )

    # Rolling mean over previous 7 days
    df["quantity_rolling_mean_7"] = (
        df.groupby(["store_id", "product_id"])["daily_demand"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    return df


def add_product_velocity(df):
    """
    Expanding mean demand per product (safe, past-only).
    """
    df["product_velocity"] = (
        df.groupby("product_id")["daily_demand"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def preprocess_year(filename):
    df = load_transactions(filename)
    daily = aggregate_daily_demand(df)
    daily = add_temporal_features(daily)
    daily = add_product_velocity(daily)

    # Remove rows with insufficient history
    daily = daily.dropna().reset_index(drop=True)

    return daily


def main():
    print("Starting preprocessing...")

    train_df = preprocess_year("transactions_3stores_2023_fullyear.csv")
    val_df = preprocess_year("transactions_3stores_2024_fullyear.csv")

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)

    print("Preprocessing complete.")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")


if __name__ == "__main__":
    main()
