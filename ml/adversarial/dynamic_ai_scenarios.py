"""
Dynamic AI Scenario Generator
Uses Qwen LLM to analyze actual database patterns and generate intelligent scenarios
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import json

# Add paths
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from sqlalchemy import select, func
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import DailyDemand, Inventory, Transaction

# Add ML directory for LLM
ml_dir = Path(__file__).parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from llm.ollama_client import OllamaClient


def analyze_database_patterns(db: Session) -> Dict:
    """
    Analyze actual database to find patterns for AI scenario generation
    """
    print("🔍 Analyzing your database patterns...")
    
    # Get date ranges
    thirty_days_ago = datetime.now().date() - timedelta(days=30)
    sixty_days_ago = datetime.now().date() - timedelta(days=60)
    
    # 1. Demand Volatility Analysis
    volatility_stmt = select(
        DailyDemand.product_id,
        DailyDemand.store_id,
        func.avg(DailyDemand.total_quantity).label('avg_demand'),
        func.stddev(DailyDemand.total_quantity).label('stddev_demand'),
        func.max(DailyDemand.total_quantity).label('max_demand'),
        func.min(DailyDemand.total_quantity).label('min_demand'),
        func.count(DailyDemand.id).label('days_count')
    ).where(
        DailyDemand.date >= thirty_days_ago
    ).group_by(
        DailyDemand.product_id,
        DailyDemand.store_id
    ).having(
        func.count(DailyDemand.id) > 10  # At least 10 days of data
    )
    
    volatility_results = db.execute(volatility_stmt).all()
    
    # Calculate metrics
    high_volatility_products = []
    for result in volatility_results:
        if result.stddev_demand and result.avg_demand:
            coefficient_of_variation = result.stddev_demand / result.avg_demand
            spike_ratio = result.max_demand / result.avg_demand if result.avg_demand > 0 else 0
            
            if coefficient_of_variation > 0.5 or spike_ratio > 3.0:  # High volatility
                high_volatility_products.append({
                    'sku': result.product_id,
                    'store': result.store_id,
                    'avg_demand': float(result.avg_demand),
                    'max_spike': float(result.max_demand / result.avg_demand if result.avg_demand > 0 else 0),
                    'volatility': float(coefficient_of_variation)
                })
    
    # 2. Inventory Risk Analysis
    inventory_stmt = select(
        Inventory.sku,
        Inventory.store_id,
        Inventory.quantity
    )
    inventory_results = db.execute(inventory_stmt).all()
    
    # Calculate average demand for inventory comparison
    demand_avg_stmt = select(
        DailyDemand.product_id,
        DailyDemand.store_id,
        func.avg(DailyDemand.total_quantity).label('avg_demand')
    ).where(
        DailyDemand.date >= thirty_days_ago
    ).group_by(
        DailyDemand.product_id,
        DailyDemand.store_id
    )
    
    demand_avg_results = db.execute(demand_avg_stmt).all()
    demand_lookup = {(r.product_id, r.store_id): r.avg_demand for r in demand_avg_results}
    
    low_stock_items = []
    for inv in inventory_results:
        avg_demand = demand_lookup.get((inv.sku, inv.store_id), 1.0)
        days_of_cover = inv.quantity / avg_demand if avg_demand > 0 else 999
        
        if days_of_cover < 7:  # Less than a week
            low_stock_items.append({
                'sku': inv.sku,
                'store': inv.store_id,
                'current_stock': float(inv.quantity),
                'avg_daily_demand': float(avg_demand),
                'days_of_cover': float(days_of_cover)
            })
    
    # 3. Category Analysis
    category_stmt = select(
        DailyDemand.product_category,
        func.sum(DailyDemand.total_quantity).label('total_sales'),
        func.count(func.distinct(DailyDemand.product_id)).label('sku_count')
    ).where(
        DailyDemand.date >= thirty_days_ago,
        DailyDemand.product_category.isnot(None)
    ).group_by(
        DailyDemand.product_category
    )
    
    category_results = db.execute(category_stmt).all()
    
    top_categories = sorted(
        [{'category': r.product_category, 'total_sales': float(r.total_sales), 'sku_count': r.sku_count} 
         for r in category_results],
        key=lambda x: x['total_sales'],
        reverse=True
    )[:5]
    
    # 4. Overall Statistics
    total_skus = len(set(inv.sku for inv in inventory_results))
    total_stores = len(set(inv.store_id for inv in inventory_results))
    
    return {
        'total_skus': total_skus,
        'total_stores': total_stores,
        'high_volatility_products': high_volatility_products[:10],  # Top 10
        'low_stock_items': low_stock_items[:10],  # Top 10 most critical
        'top_categories': top_categories,
        'analysis_period_days': 30
    }


def generate_ai_scenarios(db: Session = None) -> List[Dict]:
    """
    Use AI to generate scenarios based on actual database analysis
    """
    should_close = False
    if db is None:
        db = next(get_db())
        should_close = True
    
    try:
        # Analyze database
        patterns = analyze_database_patterns(db)
        
        # Initialize LLM
        client = OllamaClient(model_name="qwen2.5:14b")
        
        if not client.is_available():
            print("⚠️ Qwen not available, using fallback hardcoded scenarios")
            from adversarial.ai_scenario_generator import list_all_scenarios
            return [
                {
                    'id': s.id,
                    'name': s.name,
                    'description': s.description,
                    'demand_multiplier': s.demand_multiplier,
                    'duration_days': s.duration_days,
                    'probability': s.probability,
                    'strategies': s.strategies,
                    'priority_level': s.priority_level
                }
                for s in list_all_scenarios()
            ]
        
        print("🤖 Asking Qwen to analyze your data and generate scenarios...")
        
        # Create detailed prompt with actual data
        system_prompt = """You are an expert supply chain analyst. Analyze the provided retail data and generate realistic adversarial scenarios.

You MUST respond with ONLY a valid JSON array of exactly 5 scenarios. Each scenario must have this structure:
[
  {
    "id": "snake_case_id",
    "name": "Short Name",
    "description": "2-3 sentence description",
    "demand_multiplier": <number 0.5-20.0>,
    "duration_days": <number 1-180>,
    "probability": <number 0.0-1.0>,
    "strategies": ["strategy 1", "strategy 2", "strategy 3", "strategy 4"],
    "priority_level": "critical|high|medium|low",
    "reasoning": "Why this scenario is relevant to THIS data"
  }
]

Base scenarios on the ACTUAL patterns in the data provided."""

        analysis_prompt = f"""Analyze this retail business data and generate 5 realistic adversarial scenarios:

DATABASE ANALYSIS:
- Total SKUs: {patterns['total_skus']}
- Total Stores: {patterns['total_stores']}
- Analysis Period: Last {patterns['analysis_period_days']} days

HIGH VOLATILITY PRODUCTS (showing demand spikes):
{json.dumps(patterns['high_volatility_products'][:5], indent=2)}

LOW STOCK ITEMS (critical inventory):
{json.dumps(patterns['low_stock_items'][:5], indent=2)}

TOP SELLING CATEGORIES:
{json.dumps(patterns['top_categories'], indent=2)}

Based on THIS SPECIFIC DATA, generate 5 adversarial scenarios that:
1. Reflect the actual volatility patterns (use max_spike values from data!)
2. Address the low stock issues you see
3. Consider the top categories
4. Include realistic probabilities based on retail industry
5. Provide actionable strategies for THIS business

Generate scenarios with varying severity: 1 critical, 2 high, 1 medium, 1 low.
Use demand multipliers that match the actual spike ratios you see in the data!"""

        # Get AI response
        response = client.generate(
            analysis_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=3000
        )
        
        print(f"📝 AI Response received ({len(response)} chars)")
        
        # Parse JSON response
        try:
            # Extract JSON array from response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                scenarios = json.loads(json_str)
                
                print(f"✅ Generated {len(scenarios)} AI-powered scenarios")
                return scenarios
            else:
                print("⚠️ Could not find JSON array in response, using fallback")
                raise ValueError("No JSON array found")
        
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            print(f"Response preview: {response[:500]}...")
            # Fallback to hardcoded
            from adversarial.ai_scenario_generator import list_all_scenarios
            return [
                {
                    'id': s.id,
                    'name': s.name,
                    'description': s.description,
                    'demand_multiplier': s.demand_multiplier,
                    'duration_days': s.duration_days,
                    'probability': s.probability,
                    'strategies': s.strategies,
                    'priority_level': s.priority_level
                }
                for s in list_all_scenarios()
            ]
    
    finally:
        if should_close:
            db.close()


if __name__ == "__main__":
    print("🚀 Testing AI Scenario Generation")
    scenarios = generate_ai_scenarios()
    
    print("\n" + "="*80)
    print("GENERATED SCENARIOS:")
    print("="*80)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Multiplier: {scenario['demand_multiplier']}×")
        print(f"   Probability: {scenario['probability']*100:.0f}%")
        print(f"   Priority: {scenario['priority_level']}")
        if 'reasoning' in scenario:
            print(f"   Reasoning: {scenario['reasoning']}")
