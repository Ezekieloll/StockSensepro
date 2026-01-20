import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 1. Filter only sales
    df = df[df["event_type"] == "sale"].copy()
    df = df[df["quantity"] > 0]

    # 2. Convert date
    df["date"] = pd.to_datetime(df["date"])

    # 3. Aggregate to daily demand
    daily_df = (
        df.groupby(["store_id", "product_id", "date"])
        .agg(
            daily_demand=("quantity", "sum"),
            price=("price", "mean"),
            holiday_flag=("holiday_flag", "max"),
            weather=("weather", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    return daily_df


def main():
    # 2023 → training
    train_df = load_and_preprocess(
        RAW_DIR / "transactions_3stores_2023_fullyear.csv"
    )
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)

    # 2024 → validation
    val_df = load_and_preprocess(
        RAW_DIR / "transactions_3stores_2024_fullyear.csv"
    )
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)

    print("Preprocessing complete.")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")


if __name__ == "__main__":
    main()
