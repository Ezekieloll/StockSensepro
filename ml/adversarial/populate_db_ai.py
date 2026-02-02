"""
AI-Powered Adversarial Testing with Multiple Scenario Analysis
Run intelligent stress tests with realistic scenarios and strategic recommendations.
"""

import sys
from pathlib import Path

# Add parent directory to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

import torch
from sqlalchemy import select, delete, func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AdversarialRisk, Inventory, DailyDemand

from adversarial.inventory_risk import InventoryRiskEvaluator
from adversarial.ai_scenario_generator import AIScenarioGenerator, Scenario


def run_ai_adversarial_testing(
    selected_scenarios: list[str] = None,
    db: Session = None
) -> dict:
    """
    Run AI-powered adversarial testing with intelligent scenarios.
    
    Args:
        selected_scenarios: List of scenario IDs to test (e.g., ['holiday_rush', 'weather_emergency'])
                          If None, tests all scenarios
        db: Database session (optional, creates one if not provided)
    
    Returns:
        Dict with results summary and recommendations
    """
    
    # Initialize
    should_close_db = False
    if db is None:
        db = next(get_db())
        should_close_db = True
    
    try:
        risk_eval = InventoryRiskEvaluator()
        scenario_gen = AIScenarioGenerator()
        
        # Get scenarios to test
        all_scenarios = scenario_gen.scenarios_library
        if selected_scenarios:
            scenarios_to_test = [s for s in all_scenarios if s.id in selected_scenarios]
        else:
            scenarios_to_test = all_scenarios
        
        print(f"🤖 AI Adversarial Testing - Analyzing {len(scenarios_to_test)} scenarios...")
        
        # Clear existing adversarial risk data
        db.execute(delete(AdversarialRisk))
        db.commit()
        
        # Get all unique SKU-store combinations from inventory
        inventory_stmt = select(Inventory.sku, Inventory.store_id, Inventory.quantity).distinct()
        inventory_records = db.execute(inventory_stmt).all()
        
        # Get unique SKUs and stores
        all_skus = list(set(inv.sku for inv in inventory_records))
        all_stores = list(set(inv.store_id for inv in inventory_records))
        
        # Build inventory lookup
        inventory_lookup = {(inv.sku, inv.store_id): inv.quantity for inv in inventory_records}
        
        # Build demand cache from actual transaction data (daily_demand table)
        print("📊 Building demand baseline from historical transaction data...")
        demand_cache = {}
        
        # Calculate average daily demand per SKU per store from last 30 days
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.now().date() - timedelta(days=30)
        
        demand_stmt = select(
            DailyDemand.product_id,
            DailyDemand.store_id,
            func.avg(DailyDemand.total_quantity).label('avg_daily_demand')
        ).where(
            DailyDemand.date >= thirty_days_ago
        ).group_by(
            DailyDemand.product_id,
            DailyDemand.store_id
        )
        
        demand_results = db.execute(demand_stmt).all()
        
        for result in demand_results:
            demand_cache[(result.product_id, result.store_id)] = result.avg_daily_demand
        
        print(f"   Found {len(demand_cache)} SKU-store combinations with historical demand data")
        
        # Results tracking
        results_by_scenario = {}
        all_risk_records = []
        
        # Test each scenario
        for scenario in scenarios_to_test:
            print(f"\n🎯 Testing: {scenario.name}")
            print(f"   Description: {scenario.description}")
            print(f"   Demand Multiplier: {scenario.demand_multiplier}×")
            print(f"   Duration: {scenario.duration_days} days")
            print(f"   Probability: {scenario.probability * 100:.0f}%")
            
            scenario_risks = []
            
            # Test each SKU × Store combination
            for sku in all_skus:
                for store_id in all_stores:
                    # Check if scenario affects this SKU category
                    # Note: We don't have category info, so we'll apply to all for now
                    # TODO: Add category filtering when SKU table is available
                    
                    # Get baseline demand (DailyDemand uses product_id which matches sku)
                    baseline = demand_cache.get((sku, store_id), 5.0)
                    
                    # Calculate worst-case for this scenario
                    worst_case = baseline * scenario.demand_multiplier
                    
                    # Get current inventory
                    inventory = inventory_lookup.get((sku, store_id), 0)
                    
                    # Evaluate risk
                    risk = risk_eval.evaluate(
                        baseline_demand=baseline,
                        worst_case_demand=worst_case,
                        inventory_level=inventory
                    )
                    
                    # Create risk record
                    risk_record = AdversarialRisk(
                        sku=sku,
                        sku_id=sku,
                        store_id=store_id,
                        scenario_name=scenario.name,
                        scenario_id=scenario.id,
                        baseline_demand=baseline,
                        worst_case_demand=worst_case,
                        current_inventory=inventory,
                        stockout=bool(risk["stockout"]),
                        severity=float(risk["severity"]),
                        days_of_cover=float(risk["days_of_cover"]),
                        risk_score=float(risk["risk_score"]),
                        probability=scenario.probability,
                        strategies="|".join(scenario.strategies),  # Store as pipe-separated
                        priority_level=scenario.priority_level
                    )
                    
                    scenario_risks.append(risk_record)
                    all_risk_records.append(risk_record)
            
            # Calculate scenario summary
            stockouts = sum(1 for r in scenario_risks if r.stockout)
            avg_risk = sum(r.risk_score for r in scenario_risks) / len(scenario_risks) if scenario_risks else 0
            
            results_by_scenario[scenario.id] = {
                "name": scenario.name,
                "records_tested": len(scenario_risks),
                "stockout_count": stockouts,
                "stockout_rate": stockouts / len(scenario_risks) if scenario_risks else 0,
                "avg_risk_score": avg_risk,
                "probability": scenario.probability,
                "strategies": scenario.strategies
            }
            
            print(f"   ⚠️  Stockout Risk: {stockouts}/{len(scenario_risks)} SKU-store combinations")
        
        # Save all records to database
        print(f"\n💾 Saving {len(all_risk_records)} risk assessments to database...")
        db.bulk_save_objects(all_risk_records)
        db.commit()
        
        # Generate summary report
        print("\n" + "="*80)
        print("📊 AI ADVERSARIAL TESTING SUMMARY")
        print("="*80)
        
        for scenario_id, results in results_by_scenario.items():
            print(f"\n{results['name']}:")
            print(f"  Probability: {results['probability']*100:.0f}%")
            print(f"  Stockout Rate: {results['stockout_rate']*100:.1f}%")
            print(f"  Avg Risk Score: {results['avg_risk_score']:.3f}")
            print(f"  Top Strategies:")
            for i, strategy in enumerate(results['strategies'][:3], 1):
                print(f"    {i}. {strategy}")
        
        print("\n" + "="*80)
        
        # Find most critical scenario
        critical_scenario = max(
            results_by_scenario.items(),
            key=lambda x: x[1]['stockout_rate'] * x[1]['probability']
        )
        
        print(f"\n🚨 MOST CRITICAL SCENARIO: {critical_scenario[1]['name']}")
        print(f"   Combined Risk: {critical_scenario[1]['stockout_rate'] * critical_scenario[1]['probability']:.2%}")
        print("\n✅ AI Adversarial Testing Complete!")
        
        return {
            "status": "success",
            "scenarios_tested": len(scenarios_to_test),
            "total_records": len(all_risk_records),
            "results_by_scenario": results_by_scenario,
            "most_critical_scenario": critical_scenario[1]
        }
    
    finally:
        if should_close_db:
            db.close()


if __name__ == "__main__":
    # Command-line usage: python -m adversarial.populate_db_ai
    # Or with specific scenarios: python -m adversarial.populate_db_ai holiday_rush weather_emergency
    
    import sys
    
    if len(sys.argv) > 1:
        # Test specific scenarios
        selected = sys.argv[1:]
        print(f"Testing selected scenarios: {selected}")
        results = run_ai_adversarial_testing(selected_scenarios=selected)
    else:
        # Test all scenarios
        results = run_ai_adversarial_testing()
    
    print(f"\n📈 Results: {results['total_records']} risk assessments generated")
