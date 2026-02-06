"""
What-If Scenario Simulation API

Provides endpoints for demand scenario simulations based on historical data.
Supports both predefined scenarios and AI-powered custom scenario analysis.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import json
import sys
from pathlib import Path

from app.database import get_db
from app.models.transaction import DailyDemand
from app.services.gnn_propagation import get_gnn_propagator

# Import category names from category_relationships
ML_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "ml" / "config"
if str(ML_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(ML_CONFIG_DIR))

try:
    from category_relationships import CATEGORY_NAMES
    USE_CATEGORY_NAMES = True
except ImportError:
    USE_CATEGORY_NAMES = False
    print("⚠️ Could not import CATEGORY_NAMES, using fallback")
    # Fallback category names (corrected from original)
    CATEGORY_NAMES = {
        "AUTO": "Automotive", "BABC": "Baby Care", "BAGL": "Bags & Luggage",
        "BEDM": "Bedding & Mattress", "BEVG": "Beverages", "BKDY": "Bakery & Dairy",
        "BOOK": "Books & Media", "CLNS": "Cleaning Supplies", "CLOT": "Clothing",
        "ELEC": "Electronics", "FRPR": "Fresh Produce (Fruits & Vegetables)", "FRZN": "Frozen Foods",
        "FTRW": "Footwear", "FURH": "Furniture", "GROC": "Grocery (Staples & Grains)",
        "JWCH": "Jewelry & Watches", "KICH": "Kitchen Appliances", "MEAT": "Meat & Seafood",
        "PETC": "Pet Care", "PRSN": "Personal Care", "SNCK": "Snacks",
        "SPRT": "Sports Equipment", "STOF": "Stationery & Office", "TOYG": "Toys & Games"
    }

router = APIRouter(prefix="/simulations", tags=["Simulations"])


class ProductImpact(BaseModel):
    """Product-level impact from GNN propagation"""
    multiplier: float
    name: str


class SimulationScenario(BaseModel):
    """Scenario configuration for what-if simulation"""
    scenario: str
    demand_multiplier: Optional[float] = None  # If None, AI will determine it
    weather_impact: Optional[str] = None
    holiday_effect: Optional[bool] = False
    custom_description: Optional[str] = None  # Free text for AI analysis


class SimulationResult(BaseModel):
    """Result of a scenario simulation"""
    scenario: str
    demand: int
    risk: str
    confidence: int
    description: str
    ai_reasoning: Optional[str] = None  # AI explanation of the multiplier choice
    affected_categories: Optional[List[str]] = None  # Which categories are affected
    category_impacts: Optional[Dict[str, float]] = None  # Category-specific multipliers
    affected_products: Optional[Dict[str, ProductImpact]] = None  # SKU -> impact with name from GNN propagation


def analyze_scenario_with_ai(scenario_text: str, baseline_demand: float, db) -> Dict:
    """
    Use LLM to analyze a custom scenario and determine appropriate demand multiplier.
    AI has knowledge of product categories, relationships, and GNN graph structure.
    
    Returns:
        {
            "multiplier": float,
            "reasoning": str,
            "confidence": int,
            "affected_categories": list[str],  # Which product categories are affected
            "category_impacts": dict,  # Category-specific multipliers
            "affected_products": dict  # SKU-level impacts from graph propagation
        }
    """
    # Use CATEGORY_NAMES imported from category_relationships
    category_context = CATEGORY_NAMES
    
    try:
        import requests
        
        # Call local Ollama API with category context
        prompt = f"""You are analyzing a business scenario for a retail store with these product categories:
{json.dumps(category_context, indent=2)}

Scenario: "{scenario_text}"
Current baseline demand: {baseline_demand:.0f} units/day (total across all categories)

Analyze which SPECIFIC CATEGORIES will be affected and by how much.

CRITICAL RULE - READ THIS FIRST:
=================================
When prices INCREASE → people buy LESS → multiplier MUST be < 1.0
When prices DECREASE → people buy MORE → multiplier MUST be > 1.0

THIS IS THE MOST IMPORTANT RULE. NEVER VIOLATE IT.

Examples:
- "Rice prices up" → Rice demand DOWN → {{"GROC": 0.85}}
- "Rice prices down" → Rice demand UP → {{"GROC": 1.3}}
- "Electronics sale" → Electronics demand UP → {{"ELEC": 1.5}}
- "Beef expensive" → Beef demand DOWN → {{"MEAT": 0.7}}

MULTIPLIER MEANING:
- Multiplier > 1.0 = DEMAND INCREASES (e.g., 1.3 = +30% demand)
- Multiplier < 1.0 = DEMAND DECREASES (e.g., 0.8 = -20% demand)
- Multiplier = 1.0 = NO CHANGE

KEY SCENARIOS:
1. PRICE INCREASES → Demand goes DOWN (multiplier < 1.0)
   - "Rice expensive" → {{"GROC": 0.85}}
   - "Furniture expensive" → {{"FURH": 0.6}}
   
2. SALES/PROMOTIONS → Demand goes UP (multiplier > 1.0)
   - "Electronics sale" → {{"ELEC": 1.8}}
   
3. WEATHER EVENTS → Essentials UP, Others DOWN
   - "Snowstorm" → {{"GROC": 1.5, "FRPR": 1.6, "FURH": 0.7}}
   
4. RECESSION → Luxuries DOWN hard, Essentials stable
   - "Economic crisis" → {{"FURH": 0.5, "JWCH": 0.4, "GROC": 1.0}}
   
5. HOLIDAYS → Food + Gifts UP
   - "Christmas" → {{"BKDY": 2.0, "TOYG": 2.2, "MEAT": 1.9}}
   
6. COMPETITOR CLOSURE → Everything UP
   - "Competitor closed" → ALL categories 1.3-1.5

Respond ONLY with JSON (no markdown):
{{
    "affected_categories": ["FRPR", "BKDY"],
    "category_impacts": {{
        "FRPR": 1.3,
        "BKDY": 1.5
    }},
    "overall_multiplier": 1.2,
    "reasoning": "<one sentence why these categories and multipliers>",
    "confidence": 75
}}

If ALL categories affected equally, use empty affected_categories list.
"""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=15
        )
        
        if response.status_code == 200:
            ai_response = response.json()
            result_text = ai_response.get("response", "{}")
            
            # Parse JSON response
            result = json.loads(result_text)
            
            # Validate and sanitize
            multiplier = float(result.get("overall_multiplier", 1.0))
            multiplier = max(0.3, min(3.0, multiplier))
            
            return {
                "multiplier": multiplier,
                "reasoning": result.get("reasoning", "AI analysis completed"),
                "confidence": int(result.get("confidence", 75)),
                "affected_categories": result.get("affected_categories", []),
                "category_impacts": result.get("category_impacts", {})
            }
        
    except Exception as e:
        print(f"AI analysis failed: {e}")
    
    # Fallback: category-aware keyword matching
    scenario_lower = scenario_text.lower()
    
    # Detect affected categories based on keywords
    affected_categories = []
    category_impacts = {}
    
    # Furniture price changes
    if any(word in scenario_lower for word in ["chair", "table", "sofa", "couch", "desk", "furniture", "bed", "mattress"]) and "price" in scenario_lower:
        affected_categories = ["FTRW", "FURH", "BEDM"]
        category_impacts = {"FTRW": 0.8, "FURH": 0.82, "BEDM": 0.85}
        reasoning = "Furniture price increase reduces demand for furniture categories"
        overall_mult = 0.89
        
    # Specific food items (rice, pasta, etc.) - only affect GROC
    elif any(word in scenario_lower for word in ["rice", "pasta", "noodle", "cereal", "grain"]) and "price" in scenario_lower:
        affected_categories = ["GROC"]
        category_impacts = {"GROC": 0.85}
        reasoning = "Specific grocery item price increase reduces demand for that category"
        overall_mult = 0.93
        
    # General food/grocery price increases - affects multiple food categories
    elif any(word in scenario_lower for word in ["grocery", "groceries", "food"]) and any(word in scenario_lower for word in ["price", "expensive", "inflation"]):
        affected_categories = ["FRPR", "BKDY", "BEVG", "GROC", "MEAT", "FRZN", "SNCK"]
        category_impacts = {"FRPR": 0.85, "BKDY": 0.88, "BEVG": 0.9, "GROC": 0.87, "MEAT": 0.82, "FRZN": 0.9, "SNCK": 0.92}
        reasoning = "Broad grocery price increase reduces demand across food categories"
        overall_mult = 0.88
        
    # Weather events - primarily food essentials
    elif any(word in scenario_lower for word in ["storm", "weather", "snow", "rain", "hurricane", "disaster"]):
        affected_categories = ["FRPR", "BKDY", "BEVG", "GROC", "MEAT", "CLNS"]
        category_impacts = {"FRPR": 1.5, "BKDY": 1.6, "BEVG": 1.3, "GROC": 1.4, "MEAT": 1.3, "CLNS": 1.2}
        reasoning = "Weather event increases panic buying for essentials"
        overall_mult = 1.4
        
    # Holidays - broad food impact + gifts
    elif any(word in scenario_lower for word in ["holiday", "christmas", "thanksgiving", "festival", "celebration"]):
        affected_categories = ["FRPR", "BKDY", "BEVG", "SNCK", "MEAT", "GROC", "TOYG", "JWCH", "CLOT"]
        category_impacts = {
            "FRPR": 1.8, "BKDY": 2.0, "BEVG": 1.6, "SNCK": 1.7, "MEAT": 1.9,
            "GROC": 1.7, "TOYG": 2.2, "JWCH": 1.5, "CLOT": 1.4
        }
        reasoning = "Holiday season surge across food and gift categories"
        overall_mult = 1.7
        
    # Economic recession - non-essentials hit hardest
    elif any(word in scenario_lower for word in ["recession", "crisis", "downturn", "economy", "unemployment"]):
        affected_categories = ["FTRW", "ELEC", "JWCH", "CLOT", "TOYG", "SPRT", "FURH"]
        category_impacts = {"FTRW": 0.5, "ELEC": 0.6, "JWCH": 0.4, "CLOT": 0.7, "TOYG": 0.6, "SPRT": 0.55, "FURH": 0.5}
        reasoning = "Economic downturn reduces non-essential purchases"
        overall_mult = 0.75
        
    # Baby boom or families moving in
    elif any(word in scenario_lower for word in ["baby", "families", "housing development", "newborns"]):
        affected_categories = ["BABC", "FRPR", "GROC", "CLNS", "PRSN"]
        category_impacts = {"BABC": 2.0, "FRPR": 1.3, "GROC": 1.2, "CLNS": 1.3, "PRSN": 1.2}
        reasoning = "New families increase demand for baby and household essentials"
        overall_mult = 1.3
        
    # Pet adoption trend
    elif any(word in scenario_lower for word in ["pet", "dog", "cat", "adoption"]):
        affected_categories = ["PETC"]
        category_impacts = {"PETC": 1.8}
        reasoning = "Pet adoption trend boosts pet care products"
        overall_mult = 1.1
        
    # School season
    elif any(word in scenario_lower for word in ["school", "back to school", "semester", "students"]):
        affected_categories = ["STOF", "BOOK", "CLOT", "SNCK"]
        category_impacts = {"STOF": 2.5, "BOOK": 1.8, "CLOT": 1.4, "SNCK": 1.3}
        reasoning = "Back-to-school season drives stationery and book sales"
        overall_mult = 1.3
        
    # Competitor closure - general increase
    elif any(word in scenario_lower for word in ["competitor", "closed", "shutdown", "out of business"]):
        affected_categories = []  # All categories
        category_impacts = {}
        reasoning = "Competitor closure increases demand across all categories"
        overall_mult = 1.5
        
    else:
        affected_categories = []
        category_impacts = {}
        reasoning = "Moderate impact across categories"
        overall_mult = 1.2
    
    return {
        "multiplier": overall_mult,
        "reasoning": reasoning,
        "confidence": 70,
        "affected_categories": affected_categories,
        "category_impacts": category_impacts
    }


def apply_graph_propagation(db, category_impacts: Dict[str, float]) -> Dict[str, float]:
    """
    Apply GNN graph propagation to translate category impacts to product-level impacts.
    
    Args:
        db: Database session
        category_impacts: Dict of category -> multiplier (e.g., {"FRPR": 1.5, "GROC": 1.3})
    
    Returns:
        Dict of SKU -> impact multiplier with graph-based propagation
    """
    try:
        # Get GNN propagator singleton
        propagator = get_gnn_propagator()
        
        if not propagator.graph_loaded:
            print("⚠️ GNN graph not loaded, skipping product-level propagation")
            return {}
        
        # Find directly affected SKUs based on categories
        affected_skus = []
        for category, multiplier in category_impacts.items():
            skus_in_category = propagator.find_skus_by_category(category)
            affected_skus.extend(skus_in_category)
        
        if not affected_skus:
            return {}
        
        # Calculate average multiplier for directly affected products
        avg_multiplier = sum(category_impacts.values()) / len(category_impacts)
        
        # Propagate impact through GNN graph (2 hops with 0.5 decay)
        product_impacts = propagator.propagate_impact(
            affected_skus=affected_skus,
            direct_multiplier=avg_multiplier,
            propagation_depth=2,
            decay_factor=0.5
        )
        
        # Apply category-specific multipliers to directly affected products
        for category, multiplier in category_impacts.items():
            skus = propagator.find_skus_by_category(category)
            for sku in skus:
                if sku in product_impacts:
                    product_impacts[sku] = multiplier  # Override with category-specific multiplier
        
        # Convert to dict with product names
        product_impacts_with_names = {}
        for sku, mult in product_impacts.items():
            product_impacts_with_names[sku] = {
                "multiplier": mult,
                "name": propagator.get_product_name(sku)
            }
        
        return product_impacts_with_names
        
    except Exception as e:
        print(f"Graph propagation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def calculate_baseline_demand(db: Session, days: int = 30) -> float:
    """Calculate baseline demand from recent historical data (total daily demand across all products)"""
    cutoff_date = datetime.now().date() - timedelta(days=days)
    
    # Get total demand per day, then average across days
    result = db.query(
        DailyDemand.date,
        func.sum(DailyDemand.total_quantity).label('daily_total')
    ).filter(
        DailyDemand.date >= cutoff_date
    ).group_by(DailyDemand.date).all()
    
    if result:
        # Calculate average daily total across all products
        avg_daily_demand = sum(row.daily_total for row in result) / len(result)
        return float(avg_daily_demand)
    
    # Fallback to all-time average if recent data not available
    result = db.query(
        DailyDemand.date,
        func.sum(DailyDemand.total_quantity).label('daily_total')
    ).group_by(DailyDemand.date).all()
    
    if result:
        avg_daily_demand = sum(row.daily_total for row in result) / len(result)
        return float(avg_daily_demand)
    
    return 1000.0  # Ultimate fallback


def calculate_risk_level(multiplier: float) -> str:
    """Determine risk level based on demand multiplier"""
    if multiplier >= 2.0:
        return "high"
    elif multiplier >= 1.3:
        return "medium"
    elif multiplier <= 0.7:
        return "medium"
    else:
        return "low"


def calculate_confidence(scenario_type: str) -> int:
    """Estimate confidence level for scenario prediction"""
    confidence_map = {
        "baseline": 95,
        "demand_spike": 85,
        "holiday": 80,
        "weather": 82,
        "demand_drop": 88,
        "combined": 75
    }
    return confidence_map.get(scenario_type, 85)


@router.post("/run", response_model=List[SimulationResult])
async def run_simulation(
    scenarios: Optional[List[SimulationScenario]] = None,
    db: Session = Depends(get_db)
):
    """
    Run what-if scenario simulations based on historical data.
    
    If no scenarios provided, returns default set of common scenarios.
    """
    
    # Get baseline demand from historical data
    baseline_demand = calculate_baseline_demand(db, days=30)
    
    # Default scenarios if none provided
    if not scenarios:
        scenarios = [
            SimulationScenario(
                scenario="Baseline",
                demand_multiplier=1.0,
                description="Current trend projection"
            ),
            SimulationScenario(
                scenario="Demand Spike +50%",
                demand_multiplier=1.5,
                description="Sudden demand increase"
            ),
            SimulationScenario(
                scenario="Holiday Season",
                demand_multiplier=1.8,
                holiday_effect=True,
                description="Holiday shopping surge"
            ),
            SimulationScenario(
                scenario="Weather Shock",
                demand_multiplier=1.2,
                weather_impact="storm",
                description="Adverse weather impact"
            ),
        ]
    
    results = []
    
    for scenario in scenarios:
        # Determine multiplier: use AI if custom_description provided, otherwise use explicit value
        ai_result = None
        if scenario.custom_description:
            # AI-powered analysis with graph awareness
            ai_result = analyze_scenario_with_ai(scenario.custom_description, baseline_demand, db)
            multiplier = ai_result["multiplier"]
            ai_reasoning = ai_result["reasoning"]
            confidence = ai_result["confidence"]
        elif scenario.demand_multiplier is not None:
            # Explicitly provided multiplier
            multiplier = scenario.demand_multiplier
            ai_reasoning = None
            
            # Determine scenario type for confidence
            scenario_type = "baseline"
            if "spike" in scenario.scenario.lower():
                scenario_type = "demand_spike"
            elif "holiday" in scenario.scenario.lower():
                scenario_type = "holiday"
            elif "weather" in scenario.scenario.lower():
                scenario_type = "weather"
            elif "drop" in scenario.scenario.lower():
                scenario_type = "demand_drop"
            
            confidence = calculate_confidence(scenario_type)
        else:
            # Default baseline
            multiplier = 1.0
            ai_reasoning = None
            confidence = 95
        
        # Add weather impact
        if scenario.weather_impact:
            multiplier *= 1.15
        
        # Add holiday boost
        if scenario.holiday_effect:
            multiplier *= 1.15
        
        # Apply GNN graph propagation if we have category impacts
        affected_products = {}
        if ai_result and ai_result.get("category_impacts"):
            affected_products = apply_graph_propagation(db, ai_result["category_impacts"])
        
        projected_demand = int(baseline_demand * multiplier)
        risk = calculate_risk_level(multiplier)
        
        results.append(SimulationResult(
            scenario=scenario.scenario,
            demand=projected_demand,
            risk=risk,
            confidence=confidence,
            description=scenario.custom_description or getattr(scenario, 'description', scenario.scenario),
            ai_reasoning=ai_reasoning,
            affected_categories=ai_result.get("affected_categories") if ai_result else None,
            category_impacts=ai_result.get("category_impacts") if ai_result else None,
            affected_products=affected_products if affected_products else None
        ))
    
    return results


@router.get("/baseline", response_model=Dict[str, float])
async def get_baseline_metrics(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get baseline metrics for simulation reference.
    
    Returns average daily demand and trends from recent history.
    """
    cutoff_date = datetime.now().date() - timedelta(days=days)
    
    # Use same calculation as calculate_baseline_demand for consistency
    baseline_demand = calculate_baseline_demand(db, days)
    
    # Get daily totals for min/max calculation
    daily_totals = db.query(
        DailyDemand.date,
        func.sum(DailyDemand.total_quantity).label('daily_total')
    ).filter(
        DailyDemand.date >= cutoff_date
    ).group_by(DailyDemand.date).all()
    
    if not daily_totals:
        raise HTTPException(status_code=404, detail="No historical data available")
    
    totals = [row.daily_total for row in daily_totals]
    
    return {
        "avg_demand": baseline_demand,
        "min_demand": float(min(totals)),
        "max_demand": float(max(totals)),
        "data_points": len(daily_totals),
        "days_analyzed": days
    }


@router.post("/custom", response_model=SimulationResult)
async def run_custom_scenario(
    scenario_text: str,
    db: Session = Depends(get_db)
):
    """
    Run AI-powered custom scenario simulation with GNN graph propagation.
    
    Analyst provides natural language scenario description,
    AI determines appropriate demand multiplier and affected categories,
    GNN propagates impacts through product relationships.
    
    Examples:
    - "Major snowstorm forecast for next week"
    - "Competitor in the area just closed permanently"
    - "Economic recession predicted by analysts"
    - "New housing development opening nearby with 500 families"
    """
    baseline_demand = calculate_baseline_demand(db, days=30)
    
    # Let AI analyze the scenario
    ai_result = analyze_scenario_with_ai(scenario_text, baseline_demand, db)
    
    multiplier = ai_result["multiplier"]
    projected_demand = int(baseline_demand * multiplier)
    risk = calculate_risk_level(multiplier)
    
    # Apply GNN graph propagation if we have category impacts
    affected_products = {}
    if ai_result.get("category_impacts"):
        affected_products = apply_graph_propagation(db, ai_result["category_impacts"])
    
    return SimulationResult(
        scenario=scenario_text[:50] + "..." if len(scenario_text) > 50 else scenario_text,
        demand=projected_demand,
        risk=risk,
        confidence=ai_result["confidence"],
        affected_categories=ai_result.get("affected_categories"),
        category_impacts=ai_result.get("category_impacts"),
        affected_products=affected_products if affected_products else None,
        description=scenario_text,
        ai_reasoning=ai_result["reasoning"]
    )

