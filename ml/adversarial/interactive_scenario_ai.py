"""
Interactive AI Scenario Analyst
Natural language interface for "what-if" scenario analysis
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import json

backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from sqlalchemy import select, func
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import DailyDemand, Inventory, Transaction

ml_dir = Path(__file__).parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from llm.ollama_client import OllamaClient


def get_store_context(store_id: str, db: Session) -> Dict:
    """Get detailed context about a specific store"""
    thirty_days_ago = datetime.now().date() - timedelta(days=30)
    
    # Get store inventory
    inv_stmt = select(Inventory).where(Inventory.store_id == store_id)
    inventory = db.execute(inv_stmt).scalars().all()
    
    # Get store demand patterns
    demand_stmt = select(
        DailyDemand.product_id,
        DailyDemand.product_category,
        func.avg(DailyDemand.total_quantity).label('avg_daily_demand'),
        func.sum(DailyDemand.total_quantity).label('total_sales')
    ).where(
        DailyDemand.store_id == store_id,
        DailyDemand.date >= thirty_days_ago
    ).group_by(
        DailyDemand.product_id,
        DailyDemand.product_category
    )
    
    demand_results = db.execute(demand_stmt).all()
    
    return {
        'store_id': store_id,
        'total_inventory_value': sum(inv.quantity for inv in inventory),
        'unique_skus': len(inventory),
        'top_products': sorted(
            [{'sku': r.product_id, 'category': r.product_category, 'avg_daily': float(r.avg_daily_demand), 'total': float(r.total_sales)} 
             for r in demand_results],
            key=lambda x: x['total'],
            reverse=True
        ),
        'inventory_items': [{'sku': inv.sku, 'quantity': float(inv.quantity)} for inv in inventory]
    }


def get_all_stores_overview(db: Session) -> Dict:
    """Get overview of all stores for comparison"""
    stores = db.execute(select(Inventory.store_id).distinct()).scalars().all()
    
    store_data = {}
    for store in stores:
        store_data[store] = get_store_context(store, db)
    
    return store_data


def analyze_user_scenario(user_input: str, db: Session) -> Dict:
    """
    Analyze user's natural language scenario and calculate impact
    
    Examples:
    - "Tomorrow there will be a lockdown"
    - "Additional tax on groceries from tomorrow"
    - "Competitor store closing next week in S1 area"
    - "Weather forecast shows snowstorm for next 3 days"
    """
    
    client = OllamaClient(model_name="qwen2.5:7b")
    
    if not client.is_available():
        return {
            'error': 'Qwen LLM not available. Please install Ollama and run: ollama pull qwen2.5:7b',
            'fallback': True
        }
    
    # Get all stores data
    stores_data = get_all_stores_overview(db)
    
    # Create system prompt - PLAIN TEXT OUTPUT
    system_prompt = """You are an expert retail supply chain analyst analyzing business scenarios.

Write a detailed analysis in PLAIN ENGLISH explaining:
1. What the scenario means for the business
2. Which stores/products are affected
3. Specific actions to take

Be conversational and clear. Use actual store IDs and SKU IDs from the data provided."""

    # Create analysis prompt
    analysis_prompt = f"""USER SCENARIO: "{user_input}"

CURRENT BUSINESS DATA:
{json.dumps(stores_data, indent=2)}

Write a detailed plain-text analysis covering:
- Summary of the scenario impact
- Which stores are affected (S1, S2, S3)
- Estimated demand multiplier (e.g., 1.5× increase)
- Duration (how many days)
- Which SKUs are at risk (use actual SKU IDs from data)
- Specific actions for each affected store
- Overall strategic recommendations

Write naturally - this will be read by a store manager."""

    # Get AI analysis
    response = client.generate(
        analysis_prompt,
        system_prompt=system_prompt,
        temperature=0.7,
        max_tokens=2000
    )
    
    # Return as plain text analysis
    return {
        'scenario_summary': response,
        'raw_response': response,
        'timestamp': datetime.now().isoformat(),
        'format': 'plain_text'
    }


def calculate_precise_impact(analysis: Dict, db: Session) -> Dict:
    """
    Calculate precise inventory impact based on AI analysis
    """
    
    results = []
    
    for store_impact in analysis.get('store_specific_impacts', []):
        store_id = store_impact['store_id']
        
        # Get affected SKUs
        for sku_id in store_impact.get('expected_stockout_skus', []):
            # Get current inventory
            inv = db.execute(
                select(Inventory).where(
                    Inventory.store_id == store_id,
                    Inventory.sku == sku_id
                )
            ).scalar_one_or_none()
            
            if not inv:
                continue
            
            # Get average demand
            thirty_days_ago = datetime.now().date() - timedelta(days=30)
            avg_demand_result = db.execute(
                select(func.avg(DailyDemand.total_quantity)).where(
                    DailyDemand.store_id == store_id,
                    DailyDemand.product_id == sku_id,
                    DailyDemand.date >= thirty_days_ago
                )
            ).scalar()
            
            avg_demand = avg_demand_result or 5.0
            
            # Apply scenario multiplier
            multiplier = analysis['demand_impact']['multiplier']
            scenario_demand = avg_demand * multiplier
            duration = analysis['duration_days']
            
            # Calculate
            total_demand = scenario_demand * duration
            shortage = max(0, total_demand - inv.quantity)
            days_until_stockout = inv.quantity / scenario_demand if scenario_demand > 0 else 999
            
            results.append({
                'store_id': store_id,
                'sku': sku_id,
                'current_inventory': float(inv.quantity),
                'normal_daily_demand': float(avg_demand),
                'scenario_daily_demand': float(scenario_demand),
                'days_until_stockout': float(days_until_stockout),
                'total_shortage': float(shortage),
                'action_needed': 'URGENT' if days_until_stockout < 2 else 'MODERATE' if days_until_stockout < 5 else 'MONITOR'
            })
    
    return {
        'scenario': analysis.get('scenario_summary'),
        'detailed_impacts': results,
        'summary': {
            'total_affected_skus': len(results),
            'urgent_actions_needed': len([r for r in results if r['action_needed'] == 'URGENT']),
            'total_potential_shortage': sum(r['total_shortage'] for r in results)
        }
    }


if __name__ == "__main__":
    print("🤖 Interactive AI Scenario Analyst - Test")
    
    db = next(get_db())
    
    # Test scenarios
    test_scenarios = [
        "Tomorrow there will be a lockdown due to health emergency",
        "Weather forecast shows heavy snowstorm for next 3 days",
        "News: 15% tax increase on all grocery items from next week"
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario}")
        print('='*80)
        
        analysis = analyze_user_scenario(scenario, db)
        
        if 'error' in analysis:
            print(f"❌ Error: {analysis['error']}")
        else:
            print(f"\n📊 Analysis:")
            print(f"Affected Stores: {', '.join(analysis['affected_stores'])}")
            print(f"Demand Impact: {analysis['demand_impact']['multiplier']}×")
            print(f"Duration: {analysis['duration_days']} days")
            print(f"\nStore-Specific Impacts:")
            for impact in analysis['store_specific_impacts']:
                print(f"\n  {impact['store_id']}: {impact['impact_level'].upper()}")
                print(f"  At-risk SKUs: {', '.join(impact['expected_stockout_skus'][:5])}")
            
            # Calculate precise impact
            precise = calculate_precise_impact(analysis, db)
            print(f"\n💡 Precise Impact:")
            print(f"Total SKUs Affected: {precise['summary']['total_affected_skus']}")
            print(f"Urgent Actions: {precise['summary']['urgent_actions_needed']}")
