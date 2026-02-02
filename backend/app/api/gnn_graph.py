"""
GNN Graph API - Product Relationships

Provides endpoints to query product relationships from the GNN graph.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict
from pathlib import Path
import json

from app.database import get_db
from app.models.transaction import DailyDemand

router = APIRouter(prefix="/gnn", tags=["GNN Graph"])

# Path to GNN graph data
ML_DIR = Path(__file__).parent.parent.parent.parent / "ml"
GRAPH_PATH = ML_DIR / "gnn" / "product_graph.json"


@router.get("/product-categories")
async def get_product_categories(db: Session = Depends(get_db)):
    """
    Get all unique product categories from the database.
    
    Returns mapping of category codes to product lists.
    """
    # Query unique product_id and product_category combinations
    results = db.query(
        DailyDemand.product_id,
        DailyDemand.product_category
    ).distinct().all()
    
    # Group by category
    categories = {}
    for row in results:
        category = row.product_category or "Unknown"
        if category not in categories:
            categories[category] = []
        if row.product_id not in categories[category]:
            categories[category].append(row.product_id)
    
    return categories


@router.get("/product-relationships")
async def get_product_relationships():
    """
    Get product relationship data from GNN graph.
    
    Returns edge list with weights showing which products influence each other.
    """
    if not GRAPH_PATH.exists():
        # Return mock data if graph file doesn't exist
        return {
            "message": "GNN graph not built yet",
            "categories": {
                "FRPR": ["Milk", "Yogurt", "Cheese"],
                "BKDY": ["Bread", "Bagels", "Muffins"],
                "BEVR": ["Juice", "Soda", "Water"]
            },
            "edges": [
                {"source": "SKU_001", "target": "SKU_015", "weight": 0.65},
                {"source": "SKU_001", "target": "SKU_103", "weight": 0.82}
            ]
        }
    
    try:
        with open(GRAPH_PATH, 'r') as f:
            graph_data = json.load(f)
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading graph: {str(e)}")


@router.get("/category-summary")
async def get_category_summary():
    """
    Get simplified category-based product groups for AI context.
    
    Returns human-readable category descriptions.
    """
    category_info = {
        "AUTO": {"name": "Automotive", "keywords": ["auto", "car", "vehicle"]},
        "BABC": {"name": "Baby Care", "keywords": ["baby", "infant", "diaper"]},
        "BAGL": {"name": "Bagels", "keywords": ["bagel", "bakery"]},
        "BEDM": {"name": "Bedding & Mattress", "keywords": ["bed", "mattress", "bedding"]},
        "BEVG": {"name": "Beverages", "keywords": ["drink", "beverage", "juice", "soda"]},
        "BKDY": {"name": "Bakery", "keywords": ["bread", "bakery", "baked"]},
        "BOOK": {"name": "Books", "keywords": ["book", "reading", "literature"]},
        "CLNS": {"name": "Cleaning Supplies", "keywords": ["clean", "detergent", "soap"]},
        "CLOT": {"name": "Clothing", "keywords": ["clothes", "apparel", "shirt", "pants"]},
        "ELEC": {"name": "Electronics", "keywords": ["electronics", "tech", "gadget"]},
        "FRPR": {"name": "Fresh Produce & Dairy", "keywords": ["dairy", "milk", "eggs", "fresh"]},
        "FRZN": {"name": "Frozen Foods", "keywords": ["frozen", "ice cream"]},
        "FTRW": {"name": "Footwear", "keywords": ["footwear", "shoes", "sneakers", "boots"]},
        "FURH": {"name": "Furniture", "keywords": ["furniture", "chair", "table", "sofa", "desk", "bed"]},
        "GROC": {"name": "Groceries", "keywords": ["grocery", "food", "canned"]},
        "JWCH": {"name": "Jewelry & Watches", "keywords": ["jewelry", "watch", "accessories"]},
        "KICH": {"name": "Kitchenware", "keywords": ["kitchen", "cookware", "utensils"]},
        "MEAT": {"name": "Meat & Seafood", "keywords": ["meat", "beef", "chicken", "fish"]},
        "PETC": {"name": "Pet Care", "keywords": ["pet", "dog", "cat", "animal"]},
        "PRSN": {"name": "Personal Care", "keywords": ["personal", "hygiene", "toiletries"]},
        "SNCK": {"name": "Snacks", "keywords": ["snack", "chips", "crackers"]},
        "SPRT": {"name": "Sports & Outdoor", "keywords": ["sports", "outdoor", "fitness"]},
        "STOF": {"name": "Stationery & Office", "keywords": ["office", "stationery", "paper"]},
        "TOYG": {"name": "Toys & Games", "keywords": ["toy", "game", "play"]}
    }
    
    return category_info
