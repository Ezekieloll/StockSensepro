"""
Generate SKU volume statistics from live database.
Analyzes demand patterns across all products.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.database import SessionLocal
from app.models.transaction import DailyDemand
from sqlalchemy import func
import csv


def main():
    db = SessionLocal()
    
    try:
        # Query all SKU demand data from database
        print("📊 Querying database for SKU volume statistics...")
        
        results = db.query(
            DailyDemand.product_id,
            func.avg(DailyDemand.total_quantity).label('avg_daily_demand'),
            func.sum(DailyDemand.total_quantity).label('total_demand'),
            func.count(func.nullif(DailyDemand.total_quantity, 0)).label('non_zero_days'),
            func.count(DailyDemand.total_quantity).label('total_days')
        ).group_by(
            DailyDemand.product_id
        ).all()
        
        # Process results
        sku_stats = []
        for row in results:
            sparsity_ratio = 1 - (row.non_zero_days / row.total_days) if row.total_days > 0 else 1.0
            sku_stats.append({
                'product_id': row.product_id,
                'avg_daily_demand': float(row.avg_daily_demand) if row.avg_daily_demand else 0.0,
                'total_demand': float(row.total_demand) if row.total_demand else 0.0,
                'non_zero_days': row.non_zero_days,
                'total_days': row.total_days,
                'sparsity_ratio': sparsity_ratio
            })
        
        # Sort by average demand (descending)
        sku_stats.sort(key=lambda x: x['avg_daily_demand'], reverse=True)
        
        # Save to CSV
        output_path = Path(__file__).parent / "results" / "sku_volume_stats.csv"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['product_id', 'avg_daily_demand', 'total_demand', 'non_zero_days', 'total_days', 'sparsity_ratio']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sku_stats)
        
        print(f"✅ Saved: {output_path}")
        print(f"📦 Total SKUs analyzed: {len(sku_stats)}")
        
        print("\n🔝 Top 10 high-volume SKUs:")
        for i, sku in enumerate(sku_stats[:10], 1):
            print(f"{i:2d}. {sku['product_id']:15s} - Avg: {sku['avg_daily_demand']:7.2f} units/day, Total: {int(sku['total_demand']):6d} units")
        
        print("\n🔻 Bottom 10 low-volume SKUs:")
        for i, sku in enumerate(sku_stats[-10:], 1):
            print(f"{i:2d}. {sku['product_id']:15s} - Avg: {sku['avg_daily_demand']:7.2f} units/day, Total: {int(sku['total_demand']):6d} units")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
