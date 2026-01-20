"""
Historical Data Service

Loads and caches real transaction data from CSV files
to show actual historical demand patterns.
"""

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class HistoricalDataService:
    """
    Service for loading and querying historical demand data
    from transaction CSV files.
    """
    
    def __init__(self):
        self.daily_demand: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Structure: daily_demand[store_id][product_id][date_str] = quantity
        
        self.date_range = {"min": None, "max": None}
        self.loaded = False
        
        self._load_data()
    
    def _find_data_files(self) -> List[Path]:
        """Find transaction CSV files."""
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "ml" / "data" / "raw",
            Path("../ml/data/raw"),
            Path("../../ml/data/raw"),
        ]
        
        data_dir = None
        for p in possible_paths:
            if p.exists():
                data_dir = p
                break
        
        if not data_dir:
            return []
        
        files = []
        for pattern in ["transactions_*.csv"]:
            files.extend(data_dir.glob(pattern))
        
        return files
    
    def _load_data(self):
        """Load transaction data from CSV files."""
        files = self._find_data_files()
        
        if not files:
            print("⚠️ No transaction files found")
            return
        
        total_records = 0
        
        for file_path in files:
            print(f"📂 Loading {file_path.name}...")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        # Only count sales
                        if row.get('event_type') != 'sale':
                            continue
                        
                        store_id = row.get('store_id', 'S1')
                        product_id = row.get('product_id', '')
                        
                        # Parse date
                        date_str = row.get('date', '')
                        if ' ' in date_str:
                            date_str = date_str.split(' ')[0]
                        
                        try:
                            parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        except:
                            continue
                        
                        quantity = float(row.get('quantity', 1))
                        
                        # Initialize nested dicts
                        if store_id not in self.daily_demand:
                            self.daily_demand[store_id] = {}
                        if product_id not in self.daily_demand[store_id]:
                            self.daily_demand[store_id][product_id] = {}
                        
                        # Aggregate daily demand
                        date_key = parsed_date.isoformat()
                        if date_key not in self.daily_demand[store_id][product_id]:
                            self.daily_demand[store_id][product_id][date_key] = 0
                        
                        self.daily_demand[store_id][product_id][date_key] += quantity
                        
                        # Track date range
                        if self.date_range["min"] is None or parsed_date < self.date_range["min"]:
                            self.date_range["min"] = parsed_date
                        if self.date_range["max"] is None or parsed_date > self.date_range["max"]:
                            self.date_range["max"] = parsed_date
                        
                        total_records += 1
                
            except Exception as e:
                print(f"⚠️ Error loading {file_path.name}: {e}")
        
        self.loaded = True
        
        # Count unique products
        all_products = set()
        for store in self.daily_demand.values():
            all_products.update(store.keys())
        
        print(f"✅ Loaded {total_records:,} sales records")
        print(f"   Stores: {list(self.daily_demand.keys())}")
        print(f"   Products: {len(all_products)}")
        print(f"   Date range: {self.date_range['min']} to {self.date_range['max']}")
    
    def get_historical_demand(
        self,
        product_id: str,
        store_id: str = "S1",
        days: int = 30,
        end_date: Optional[date] = None
    ) -> List[Dict]:
        """
        Get historical daily demand for a product.
        
        Args:
            product_id: SKU ID
            store_id: Store ID
            days: Number of days of history
            end_date: End date (defaults to max date in data)
        
        Returns:
            List of {date, actual, forecast, is_forecast} dicts
        """
        if not self.loaded:
            return []
        
        # Use last available date as end date
        if end_date is None:
            end_date = self.date_range["max"]
        
        if end_date is None:
            return []
        
        # Get data for this store/product
        if store_id not in self.daily_demand:
            store_id = list(self.daily_demand.keys())[0] if self.daily_demand else "S1"
        
        product_data = self.daily_demand.get(store_id, {}).get(product_id, {})
        
        result = []
        
        for i in range(days, 0, -1):
            day = end_date - timedelta(days=i)
            date_key = day.isoformat()
            
            demand = product_data.get(date_key, 0)
            
            result.append({
                "date": date_key,
                "actual": round(demand, 1),
                "forecast": None,
                "is_forecast": False
            })
        
        return result
    
    def get_available_dates(self) -> Dict:
        """Get the date range of available data."""
        return {
            "min_date": self.date_range["min"].isoformat() if self.date_range["min"] else None,
            "max_date": self.date_range["max"].isoformat() if self.date_range["max"] else None
        }
    
    def get_product_stats(self, product_id: str, store_id: str = "S1") -> Dict:
        """Get statistics for a product."""
        product_data = self.daily_demand.get(store_id, {}).get(product_id, {})
        
        if not product_data:
            return {"avg_demand": 0, "max_demand": 0, "min_demand": 0, "data_points": 0}
        
        values = list(product_data.values())
        
        return {
            "avg_demand": round(sum(values) / len(values), 1) if values else 0,
            "max_demand": round(max(values), 1) if values else 0,
            "min_demand": round(min(values), 1) if values else 0,
            "data_points": len(values)
        }


# Singleton instance
_historical_service = None
# Need to import timedelta
from datetime import timedelta


def get_historical_service() -> HistoricalDataService:
    """Get or create the historical data service."""
    global _historical_service
    if _historical_service is None:
        _historical_service = HistoricalDataService()
    return _historical_service
