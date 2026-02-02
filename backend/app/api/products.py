from fastapi import APIRouter, HTTPException
import pandas as pd
import os
from pathlib import Path

router = APIRouter()

# Path to the categories_products CSV
CSV_PATH = Path(__file__).parent.parent.parent.parent / "ml" / "data" / "raw" / "categories_products.csv"

@router.get("/products/catalog")
async def get_product_catalog():
    """Get the complete product catalog with SKU IDs and product names"""
    try:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=404, detail="Product catalog file not found")
        
        df = pd.read_csv(CSV_PATH)
        
        # Convert to list of dictionaries
        products = df.to_dict('records')
        
        # Create a lookup dictionary for quick access
        sku_lookup = {
            row['SKU_ID']: {
                'category': row['Category_Code'],
                'name': row['Product_Name']
            }
            for row in products
        }
        
        return {
            "products": products,
            "sku_lookup": sku_lookup,
            "total_products": len(products)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading product catalog: {str(e)}")

@router.get("/products/{sku_id}")
async def get_product_details(sku_id: str):
    """Get details for a specific product by SKU ID"""
    try:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=404, detail="Product catalog file not found")
        
        df = pd.read_csv(CSV_PATH)
        product = df[df['SKU_ID'] == sku_id]
        
        if product.empty:
            raise HTTPException(status_code=404, detail=f"Product {sku_id} not found")
        
        return product.to_dict('records')[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading product: {str(e)}")
