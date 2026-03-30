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


def _normalize_category(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _expand_affected_categories(
    affected_categories: list[str],
    top_db_categories: list[str],
) -> set[str]:
    """Map scenario category labels to normalized DB category tokens."""
    expanded: set[str] = set()

    alias_map = {
        "freshproduce": {"freshproduce", "produce", "fresh", "fruits", "vegetables", "veg"},
        "bakery": {"bakery", "bread"},
        "beverages": {"beverages", "beverage", "drinks", "drink"},
        "dairy": {"dairy", "milk", "eggs"},
        "cannedgoods": {"cannedgoods", "canned", "pantry"},
    }

    normalized_top = [_normalize_category(cat) for cat in top_db_categories if cat]

    for category in affected_categories:
        normalized = _normalize_category(category)

        if normalized in {"all", "random"}:
            return {"__all__"}

        if normalized == "promotionalcategories":
            # Use top selling categories as promo-sensitive proxies.
            expanded.update(normalized_top[:3])
            continue

        if normalized == "specificaffectedcategory":
            # Use the strongest observed category signal as the affected target.
            if normalized_top:
                expanded.add(normalized_top[0])
            continue

        if normalized in alias_map:
            expanded.update(alias_map[normalized])
        elif normalized:
            expanded.add(normalized)

    return expanded


def _category_matches(
    sku_category: str | None,
    affected_categories: list[str],
    top_db_categories: list[str],
) -> bool:
    normalized_sku_category = _normalize_category(sku_category)
    expanded = _expand_affected_categories(affected_categories, top_db_categories)

    if "__all__" in expanded:
        return True

    if not normalized_sku_category:
        return False

    if normalized_sku_category in expanded:
        return True

    # Fallback partial match for close labels like "freshproduce" vs "produce".
    return any(
        token and (token in normalized_sku_category or normalized_sku_category in token)
        for token in expanded
    )


def run_ai_adversarial_testing(
    selected_scenarios: list[str] = None,
    category_scoped: bool = True,
    custom_scenarios: list[dict] | None = None,
    db: Session = None
) -> dict:
    """
    Run AI-powered adversarial testing with intelligent scenarios.
    
    Args:
        selected_scenarios: List of scenario IDs to test (e.g., ['holiday_rush', 'weather_emergency'])
                          If None, tests all scenarios
        category_scoped: If True, apply scenarios only to matching categories.
                         If False, apply scenarios broadly to all inventory SKU-store pairs.
        custom_scenarios: Optional list of user-defined scenario objects.
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
        all_scenarios = list(scenario_gen.scenarios_library)
        scenario_by_id = {scenario.id: scenario for scenario in all_scenarios}

        if custom_scenarios:
            for item in custom_scenarios:
                try:
                    scenario_obj = Scenario(
                        id=str(item.get("id", "")).strip(),
                        name=str(item.get("name", "Custom Scenario")).strip(),
                        description=str(item.get("description", "")).strip(),
                        demand_multiplier=float(item.get("demand_multiplier", 1.0)),
                        duration_days=int(item.get("duration_days", 1)),
                        affected_categories=list(item.get("affected_categories") or ["All"]),
                        probability=float(item.get("probability", 0.5)),
                        strategies=list(item.get("strategies") or []),
                        priority_level=str(item.get("priority_level", "medium")).strip().lower(),
                    )

                    if scenario_obj.id:
                        # Same ID means "edit/override" of existing scenario.
                        scenario_by_id[scenario_obj.id] = scenario_obj
                except Exception:
                    # Skip malformed custom scenarios without breaking the full run.
                    continue

        all_scenarios = list(scenario_by_id.values())

        if selected_scenarios:
            scenarios_to_test = [s for s in all_scenarios if s.id in selected_scenarios]
            if not scenarios_to_test:
                # Guard against stale/invalid selected IDs from the client.
                print("⚠️ Selected scenario IDs did not match available scenarios; using all scenarios instead")
                scenarios_to_test = all_scenarios
        else:
            scenarios_to_test = all_scenarios
        
        print(f"🤖 AI Adversarial Testing - Analyzing {len(scenarios_to_test)} scenarios...")
        print(f"   Scope mode: {'strict-category' if category_scoped else 'broad'}")
        
        # Clear existing adversarial risk data
        db.execute(delete(AdversarialRisk))
        db.commit()
        
        # Get all SKU-store combinations that actually exist in inventory.
        inventory_stmt = select(Inventory.sku, Inventory.store_id, Inventory.quantity).distinct()
        inventory_records = db.execute(inventory_stmt).all()
        
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

        # Build category lookup per SKU-store from historical data.
        category_stmt = select(
            DailyDemand.product_id,
            DailyDemand.store_id,
            DailyDemand.product_category,
            func.count(DailyDemand.id).label("row_count"),
        ).where(
            DailyDemand.date >= thirty_days_ago,
            DailyDemand.product_category.isnot(None),
        ).group_by(
            DailyDemand.product_id,
            DailyDemand.store_id,
            DailyDemand.product_category,
        )

        category_results = db.execute(category_stmt).all()

        category_lookup: dict[tuple[str, str], str] = {}
        sku_fallback_category: dict[str, str] = {}
        pair_best_count: dict[tuple[str, str], int] = {}
        sku_best_count: dict[str, int] = {}
        category_totals: dict[str, float] = {}

        for row in category_results:
            pair_key = (row.product_id, row.store_id)
            row_count = int(row.row_count or 0)
            category = row.product_category

            if row_count > pair_best_count.get(pair_key, -1):
                pair_best_count[pair_key] = row_count
                category_lookup[pair_key] = category

            if row_count > sku_best_count.get(row.product_id, -1):
                sku_best_count[row.product_id] = row_count
                sku_fallback_category[row.product_id] = category

            category_totals[category] = category_totals.get(category, 0.0) + row_count

        top_db_categories = [
            cat for cat, _ in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        ]
        
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
            
            # Test real inventory pairs; optionally scope by scenario categories.
            for inv in inventory_records:
                sku = inv.sku
                store_id = inv.store_id
                inventory = inv.quantity

                category = category_lookup.get((sku, store_id)) or sku_fallback_category.get(sku)
                if category_scoped and not _category_matches(category, scenario.affected_categories, top_db_categories):
                    continue

                # Get baseline demand (DailyDemand uses product_id which matches sku)
                baseline = demand_cache.get((sku, store_id), 5.0)

                # Calculate worst-case for this scenario
                worst_case = baseline * scenario.demand_multiplier

                # Evaluate risk
                risk = risk_eval.evaluate(
                    baseline_demand=baseline,
                    worst_case_demand=worst_case,
                    inventory_level=inventory,
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
                    priority_level=scenario.priority_level,
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
        
        if not results_by_scenario:
            print("\n⚠️ No scenario results were produced")
            print("\n✅ AI Adversarial Testing Complete!")
            return {
                "status": "success",
                "scope_mode": "strict" if category_scoped else "broad",
                "scenarios_tested": 0,
                "total_records": len(all_risk_records),
                "results_by_scenario": {},
                "most_critical_scenario": {},
            }

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
            "scope_mode": "strict" if category_scoped else "broad",
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
