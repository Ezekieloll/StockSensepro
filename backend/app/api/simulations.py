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
    The LLM reasons from economic first principles — no hardcoded scenario rules.

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
    # Category information for AI context (all 24 categories)
    category_context = {
        "AUTO": "Automotive", "BABC": "Baby Care", "BAGL": "Bagels",
        "BEDM": "Bedding & Mattress", "BEVG": "Beverages", "BKDY": "Bakery",
        "BOOK": "Books", "CLNS": "Cleaning Supplies", "CLOT": "Clothing",
        "ELEC": "Electronics", "FRPR": "Fresh Produce & Dairy", "FRZN": "Frozen Foods",
        "FTRW": "Footwear", "FURH": "Furniture", "GROC": "Groceries",
        "JWCH": "Jewelry & Watches", "KICH": "Kitchenware", "MEAT": "Meat & Seafood",
        "PETC": "Pet Care", "PRSN": "Personal Care", "SNCK": "Snacks",
        "SPRT": "Sports & Outdoor", "STOF": "Stationery & Office", "TOYG": "Toys & Games"
    }
    
    try:
        import requests

        prompt = f"""You are an expert retail economist. A store manager has described a scenario and you must predict how it will affect demand across product categories.

STORE CATEGORIES (use only these exact codes):
{json.dumps(category_map, indent=2)}

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
            timeout=30
        )

        if response.status_code == 200:
            ai_response = response.json()
            result_text = ai_response.get("response", "{}")
            result = json.loads(result_text)

            # Validate category codes — reject any code not in our known set
            valid_codes = set(category_map.keys())
            raw_impacts = result.get("category_impacts", {})
            category_impacts = {
                k: float(v)
                for k, v in raw_impacts.items()
                if k in valid_codes and isinstance(v, (int, float))
            }
            affected_categories = [c for c in result.get("affected_categories", []) if c in valid_codes]

            multiplier = float(result.get("overall_multiplier", 1.0))
            multiplier = max(0.1, min(5.0, multiplier))

            return {
                "multiplier": multiplier,
                "reasoning": result.get("reasoning", "AI analysis completed"),
                "confidence": int(result.get("confidence", 70)),
                "affected_categories": affected_categories,
                "category_impacts": category_impacts,
            }

    except Exception as e:
        print(f"AI analysis failed: {e}")

    # ── Fallback when Ollama is unavailable ──────────────────────────────────
    # Uses semantic name matching against category descriptions + economic
    # direction signals.  No hardcoded multiplier values per scenario type.
    return _semantic_fallback(scenario_text, category_map)


def _economic_direction(scenario_lower: str) -> float:
    """
    Return a base direction signal from the scenario text using economic keywords.
    Positive signal → demand increase; negative → demand decrease.
    Magnitude is derived from intensity words, not scenario type.
    """
    increase_words = [
        # weather
        "sale", "discount", "promo", "promotion", "free", "cheap", "shortage",
        "panic", "storm", "snow", "rain", "hurricane", "disaster", "flood",
        # competitor / market
        "lockdown", "closed", "competitor", "shutdown",
        # seasonal / cultural
        "festival", "holiday", "christmas", "thanksgiving", "eid", "diwali",
        "new year", "celebration", "event",
        # demand drivers
        "boom", "surge", "spike", "opening", "launch", "new",
        # demographic
        "born", "baby", "adoption", "families", "population", "housing", "development",
        # payday / stimulus
        "payday", "bonus", "salary", "stimulus", "rebate", "handout", "cash transfer",
        # sports / entertainment
        "super bowl", "world cup", "match", "concert", "fair", "game",
        # school
        "school", "semester", "university", "back to school",
    ]
    decrease_words = [
        "tax", "tariff", "expensive", "hike", "surcharge", "inflation", "price up",
        "increase", "cost", "recession", "crisis", "downturn", "unemployment",
        "bankrupt", "penalty", "fine", "regulation", "ban", "embargo",
        "layoff", "retrenchment", "job loss", "interest rate",
    ]

    inc_score = sum(1 for w in increase_words if w in scenario_lower)
    dec_score = sum(1 for w in decrease_words if w in scenario_lower)

    # Intensity amplifiers
    intensity = 1.0
    if any(w in scenario_lower for w in ["100%", "double", "triple", "massive", "extreme", "huge", "severe"]):
        intensity = 2.0
    elif any(w in scenario_lower for w in ["50%", "significant", "major", "heavy", "large"]):
        intensity = 1.5
    elif any(w in scenario_lower for w in ["10%", "slight", "minor", "small", "little"]):
        intensity = 0.6

    net = inc_score - dec_score
    if net > 0:
        # Demand increase: baseline 1.0 + intensity-scaled lift
        return min(1.0 + 0.25 * intensity * net, 3.5)
    elif net < 0:
        # Demand decrease: baseline 1.0 - intensity-scaled drop
        return max(1.0 - 0.18 * intensity * abs(net), 0.15)
    return 1.1  # neutral lean


def _semantic_fallback(scenario_text: str, category_map: Dict[str, str]) -> Dict:
    """
    Fallback when the LLM is unavailable.
    Matches scenario keywords against category descriptions without any
    hardcoded scenario→category→multiplier rules.
    """
    scenario_lower = scenario_text.lower()
    base_direction = _economic_direction(scenario_lower)

    # Build a token set from the scenario
    import re
    tokens = set(re.findall(r"[a-z]+", scenario_lower))

    # For each category, score relevance by how many of its description words appear in the scenario
    category_scores: Dict[str, float] = {}
    for code, description in category_map.items():
        desc_tokens = set(re.findall(r"[a-z]+", description.lower()))
        # Remove very common words
        stop = {"and", "the", "of", "in", "for", "a", "an", "or", "with"}
        desc_tokens -= stop
        overlap = tokens & desc_tokens
        if overlap:
            # Score = fraction of description tokens matched, weighted by token count
            score = len(overlap) / max(len(desc_tokens), 1)
            category_scores[code] = score

    # Keep only categories with meaningful overlap
    threshold = 0.10
    matched = {k: v for k, v in category_scores.items() if v >= threshold}

    if not matched:
        # No specific categories identified — treat as broad market event
        return {
            "multiplier": base_direction,
            "reasoning": f"No specific product category identified. Applying a broad market signal based on scenario context.",
            "confidence": 40,
            "affected_categories": [],
            "category_impacts": {},
        }

    # Assign per-category multipliers: categories with higher relevance scores
    # get a multiplier closer to the base_direction; lower scores get a dampened effect
    max_score = max(matched.values())
    category_impacts: Dict[str, float] = {}
    for code, score in matched.items():
        dampening = score / max_score  # 0..1 — most-relevant category gets full effect
        if base_direction >= 1.0:
            mult = 1.0 + (base_direction - 1.0) * dampening
        else:
            mult = 1.0 - (1.0 - base_direction) * dampening
        mult = round(max(0.1, min(5.0, mult)), 2)
        category_impacts[code] = mult

    # Overall multiplier: weighted average
    total_weight = sum(matched.values())
    overall = sum(category_impacts[c] * matched[c] for c in matched) / total_weight
    overall = round(max(0.1, min(5.0, overall)), 3)

    return {
        "multiplier": overall,
        "reasoning": (
            f"AI model unavailable. Semantic fallback matched categories from scenario text "
            f"and applied economic direction (direction signal: {base_direction:.2f}). "
            f"Matched on: {', '.join(matched.keys())}."
        ),
        "confidence": 45,
        "affected_categories": list(matched.keys()),
        "category_impacts": category_impacts,
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

