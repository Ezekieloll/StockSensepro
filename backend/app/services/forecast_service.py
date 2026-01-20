"""
Forecast Service - Calls ML Microservice

This service acts as a bridge between the backend API
and the ML inference microservice.
"""

import httpx
from datetime import date, timedelta
from typing import List, Dict, Optional
import asyncio


# ML Service URL
ML_SERVICE_URL = "http://localhost:8001"


class ForecastService:
    """
    Service for getting demand forecasts.
    Calls the ML microservice for predictions.
    """
    
    def __init__(self):
        self.ml_service_available = False
        self._check_ml_service()
    
    def _check_ml_service(self):
        """Check if ML service is available."""
        try:
            import httpx
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{ML_SERVICE_URL}/health")
                if response.status_code == 200:
                    data = response.json()
                    self.ml_service_available = True
                    print(f"✅ Connected to ML service (model_loaded: {data.get('model_loaded', False)})")
                    return
        except Exception as e:
            pass
        
        self.ml_service_available = False
        print("⚠️ ML service not available, using fallback mock data")
    
    async def get_forecasts_async(
        self,
        store_id: str = None,
        sku: str = None,
        days_ahead: int = 7
    ) -> List[Dict]:
        """Get forecasts asynchronously from ML service."""
        try:
            params = {"days_ahead": days_ahead}
            if store_id:
                params["store_id"] = store_id
            if sku:
                params["sku"] = sku
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{ML_SERVICE_URL}/predict", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    self.ml_service_available = True
                    return self._transform_ml_response(data.get("forecasts", []))
        except Exception as e:
            print(f"⚠️ ML service error: {e}")
            self.ml_service_available = False
        
        # Fallback to mock data
        return self._get_fallback_forecasts(store_id, sku, days_ahead)
    
    def get_forecasts(
        self,
        store_id: str = None,
        sku: str = None,
        days_ahead: int = 7
    ) -> List[Dict]:
        """Get forecasts synchronously."""
        try:
            params = {"days_ahead": days_ahead}
            if store_id:
                params["store_id"] = store_id
            if sku:
                params["sku"] = sku
            
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{ML_SERVICE_URL}/predict", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    self.ml_service_available = True
                    return self._transform_ml_response(data.get("forecasts", []))
        except Exception as e:
            print(f"⚠️ ML service error: {e}")
            self.ml_service_available = False
        
        # Fallback to mock data
        return self._get_fallback_forecasts(store_id, sku, days_ahead)
    
    def _transform_ml_response(self, forecasts: List[Dict]) -> List[Dict]:
        """Transform ML service response to include stock status."""
        result = []
        for f in forecasts:
            # Simulate current stock
            hash_val = hash(f"{f['sku']}{f['store_id']}") % 100
            current_stock = int(50 + hash_val * 1.5)
            
            # Calculate stock status
            predicted = f.get("predicted_demand", 10)
            days_of_stock = current_stock / predicted if predicted > 0 else 999
            
            if days_of_stock < 2:
                status = "critical"
            elif days_of_stock < 5:
                status = "low"
            else:
                status = "ok"
            
            result.append({
                "sku": f["sku"],
                "product_name": f["product_name"],
                "category": f["category"],
                "store_id": f["store_id"],
                "date": f["date"],
                "predicted_demand": f["predicted_demand"],
                "current_stock": current_stock,
                "stock_status": status,
                "confidence": f.get("confidence", 0.85)
            })
        
        return result
    
    def _get_fallback_forecasts(
        self,
        store_id: str = None,
        sku: str = None,
        days_ahead: int = 7
    ) -> List[Dict]:
        """Generate fallback mock forecasts."""
        forecasts = []
        today = date.today()
        
        products = [
            ("SKU_BKDY001", "White Bread (500g)", "BKDY", 18.5),
            ("SKU_BKDY005", "Milk (1L)", "BKDY", 22.3),
            ("SKU_FRPR001", "Apples (1kg)", "FRPR", 15.7),
            ("SKU_FRPR002", "Bananas (1 Dozen)", "FRPR", 28.4),
            ("SKU_GROC001", "Rice (5kg Bag)", "GROC", 8.9),
            ("SKU_BEVG001", "Cola (2L)", "BEVG", 25.6),
            ("SKU_SNCK001", "Potato Chips (200g)", "SNCK", 21.2),
            ("SKU_FRZN001", "Frozen Pizza", "FRZN", 9.4),
            ("SKU_MEAT001", "Chicken Breast (1kg)", "MEAT", 11.2),
            ("SKU_PRSN001", "Shampoo (400ml)", "PRSN", 6.3),
        ]
        
        stores = ["S1", "S2", "S3"] if store_id is None else [store_id]
        
        for store in stores:
            store_mult = {"S1": 1.0, "S2": 0.85, "S3": 0.95}.get(store, 1.0)
            
            for sku_id, name, category, base in products:
                if sku and sku != sku_id:
                    continue
                
                hash_val = hash(f"{sku_id}{store}") % 100
                current_stock = int(50 + hash_val * 1.5)
                
                for day in range(days_ahead):
                    forecast_date = today + timedelta(days=day+1)
                    dow = forecast_date.weekday()
                    dow_factor = 1.2 if dow >= 5 else 1.0
                    noise = 0.85 + (hash(f"{sku_id}{day}{store}") % 30) / 100
                    
                    predicted = round(base * store_mult * dow_factor * noise, 1)
                    
                    days_of_stock = current_stock / predicted if predicted > 0 else 999
                    if days_of_stock < 2:
                        status = "critical"
                    elif days_of_stock < 5:
                        status = "low"
                    else:
                        status = "ok"
                    
                    forecasts.append({
                        "sku": sku_id,
                        "product_name": name,
                        "category": category,
                        "store_id": store,
                        "date": forecast_date.isoformat(),
                        "predicted_demand": predicted,
                        "current_stock": current_stock,
                        "stock_status": status,
                        "confidence": 0.5  # Low confidence for mock data
                    })
                    
                    current_stock = max(0, current_stock - int(predicted * 0.3))
        
        return forecasts


# Global service instance
_forecast_service = None


def get_forecast_service() -> ForecastService:
    """Get or create the forecast service singleton."""
    global _forecast_service
    if _forecast_service is None:
        _forecast_service = ForecastService()
    return _forecast_service
