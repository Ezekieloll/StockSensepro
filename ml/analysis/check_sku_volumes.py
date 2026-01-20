import pandas as pd

CSV_PATH = "data/processed2/train.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    # Group by SKU (product_id)
    sku_stats = (
        df.groupby("product_id")
        .agg(
            avg_daily_demand=("daily_demand", "mean"),
            total_demand=("daily_demand", "sum"),
            non_zero_days=("daily_demand", lambda x: (x > 0).sum()),
            total_days=("daily_demand", "count"),
        )
        .reset_index()
    )

    # Sparsity ratio
    sku_stats["sparsity_ratio"] = (
        1 - sku_stats["non_zero_days"] / sku_stats["total_days"]
    )

    # Sort by average demand (descending)
    sku_stats = sku_stats.sort_values(
        "avg_daily_demand", ascending=False
    )

    # Save for later analysis
    sku_stats.to_csv("analysis/sku_volume_stats.csv", index=False)

    print("✅ Saved: analysis/sku_volume_stats.csv")
    print("\nTop 10 high-volume SKUs:")
    print(sku_stats.head(10))

    print("\nBottom 10 low-volume SKUs:")
    print(sku_stats.tail(10))


if __name__ == "__main__":
    main()
