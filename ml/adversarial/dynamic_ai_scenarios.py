"""
Dynamic AI Scenario Generator
Uses Qwen LLM to analyze actual database patterns and generate intelligent scenarios
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import json
import re

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


SKU_CODE_PATTERN = re.compile(r"(?<!SKU_)\b([A-Za-z]{4}\d{3})\b")


def _normalize_sku_code_refs(text: str) -> str:
    """Convert shorthand SKU references like FRPR002 to SKU_FRPR002."""
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        return f"SKU_{match.group(1).upper()}"

    return SKU_CODE_PATTERN.sub(_replace, text)


def _build_standard_scenarios_fallback() -> List[Dict]:
    """Return schema-safe standard scenarios for API responses."""
    from adversarial.ai_scenario_generator import list_all_scenarios

    return [
        {
            'id': s.id,
            'name': s.name,
            'description': s.description,
            'demand_multiplier': s.demand_multiplier,
            'duration_days': s.duration_days,
            'affected_categories': s.affected_categories,
            'probability': s.probability,
            'strategies': s.strategies,
            'priority_level': s.priority_level,
        }
        for s in list_all_scenarios()
    ]


def _normalize_ai_scenarios(raw_scenarios: List[Dict]) -> List[Dict]:
    """Normalize AI-generated scenarios to match API schema contract."""
    normalized: List[Dict] = []

    for idx, scenario in enumerate(raw_scenarios):
        if not isinstance(scenario, dict):
            continue

        scenario_id = str(scenario.get('id') or f'ai_scenario_{idx + 1}').strip()
        name = _normalize_sku_code_refs(str(scenario.get('name') or 'AI Scenario').strip())
        description = _normalize_sku_code_refs(
            str(scenario.get('description') or 'AI-generated demand shock scenario.').strip()
        )

        try:
            demand_multiplier = float(scenario.get('demand_multiplier', 1.5))
        except (TypeError, ValueError):
            demand_multiplier = 1.5

        try:
            duration_days = int(scenario.get('duration_days', 7))
        except (TypeError, ValueError):
            duration_days = 7

        try:
            probability = float(scenario.get('probability', 0.5))
        except (TypeError, ValueError):
            probability = 0.5

        raw_categories = scenario.get('affected_categories')
        if isinstance(raw_categories, list):
            affected_categories = [str(item).strip() for item in raw_categories if str(item).strip()]
        elif isinstance(raw_categories, str) and raw_categories.strip():
            affected_categories = [item.strip() for item in raw_categories.split(',') if item.strip()]
        else:
            affected_categories = ['All']

        raw_strategies = scenario.get('strategies')
        if isinstance(raw_strategies, list):
            strategies = [str(item).strip() for item in raw_strategies if str(item).strip()]
        elif isinstance(raw_strategies, str) and raw_strategies.strip():
            strategies = [item.strip() for item in raw_strategies.split('|') if item.strip()]
        else:
            strategies = ['Review inventory and supplier capacity']

        priority_level = str(scenario.get('priority_level') or 'medium').strip().lower()
        if priority_level not in {'critical', 'high', 'medium', 'low'}:
            priority_level = 'medium'

        normalized.append(
            {
                'id': scenario_id,
                'name': name,
                'description': description,
                'demand_multiplier': demand_multiplier,
                'duration_days': duration_days,
                'affected_categories': affected_categories,
                'probability': probability,
                'strategies': strategies,
                'priority_level': priority_level,
            }
        )

    return normalized


def _extract_ai_scenarios_from_response(response: str) -> List[Dict]:
    """Extract scenario list from common LLM output formats."""
    if not response:
        return []

    text = response.strip()
    lowered = text.lower()
    if lowered.startswith("error:"):
        return []

    candidates: List[str] = [text]

    # Common fenced format: ```json ... ```
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    # Heuristic slices for first array/object in freeform text
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start >= 0 and array_end > array_start:
        candidates.append(text[array_start:array_end + 1].strip())

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(text[object_start:object_end + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            for key in ("scenarios", "data", "result", "items"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return value

            # Accept a single scenario object and wrap it.
            if any(field in parsed for field in ("name", "demand_multiplier", "duration_days")):
                return [parsed]

    return []


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
        
        # Initialize LLM (prefer 7b for lower latency and better local reliability)
        client = OllamaClient(model_name="qwen2.5:7b")
        
        if not client.is_available():
            print("⚠️ Qwen not available, using fallback hardcoded scenarios")
            return _build_standard_scenarios_fallback()
        
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

Base scenarios on the ACTUAL patterns in the data provided.
Use exact SKU IDs from the data (for example SKU_FRPR002), not shorthand codes like FRPR002.
Preserve the SKU_ prefix in scenario names and descriptions."""

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
            max_tokens=2400
        )

        # If primary request timed out/errored, retry once with a smaller token budget.
        if isinstance(response, str) and response.lower().startswith("error"):
            print(f"⚠️ Primary model response failed ({client.model_name}), retrying with reduced token budget")
            fallback_client = OllamaClient(model_name="qwen2.5:7b")
            response = fallback_client.generate(
                analysis_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1600,
            )
        
        print(f"📝 AI Response received ({len(response)} chars)")
        
        raw_scenarios = _extract_ai_scenarios_from_response(response)

        if not raw_scenarios:
            print("⚠️ Could not parse AI response into scenarios, using fallback")
            print(f"Response preview: {response[:500]}...")
            return _build_standard_scenarios_fallback()

        scenarios = _normalize_ai_scenarios(raw_scenarios)
        if not scenarios:
            print("⚠️ AI scenarios normalized to empty set, using fallback")
            return _build_standard_scenarios_fallback()

        scenarios = scenarios[:5]

        print(f"✅ Generated {len(scenarios)} AI-powered scenarios")
        return scenarios
    
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
