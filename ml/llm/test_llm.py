"""
Test Local LLM Integration
"""

import sys
from pathlib import Path

# Add ml directory to path
ml_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ml_dir))

from llm.ollama_client import OllamaClient


def test_basic_chat():
    """Test basic chat functionality"""
    print("\n🧪 Test 1: Basic Chat")
    print("-" * 80)
    
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama is not running. Please start Ollama first.")
        print("   Run: ollama serve")
        return False
    
    response = client.chat(
        "What are the main risks in inventory management?",
        system_prompt="You are a supply chain expert. Be concise.",
        reset_history=True
    )
    
    print(f"Response: {response}\n")
    return True


def test_scenario_generation():
    """Test adversarial scenario generation"""
    print("\n🧪 Test 2: Scenario Generation")
    print("-" * 80)
    
    client = OllamaClient()
    
    scenario = client.generate_adversarial_scenario(
        sku="SKU001",
        category="Fresh Produce",
        avg_demand=25.5,
        current_inventory=150,
        historical_context="This is milk, highly perishable, high demand on weekends"
    )
    
    print("Generated Scenario:")
    import json
    print(json.dumps(scenario, indent=2))
    return True


def test_data_analysis():
    """Test database data analysis"""
    print("\n🧪 Test 3: Data Analysis")
    print("-" * 80)
    
    client = OllamaClient()
    
    data_context = """
Recent Forecasts:
- SKU001 @ S1: 45 units (Type: baseline)
- SKU001 @ S1: 180 units (Type: worst_case)
- SKU002 @ S1: 30 units (Type: baseline)

Current Inventory:
- SKU001: 120 units
- SKU002: 85 units
"""
    
    response = client.analyze_database_data(
        "Which SKU is at higher risk of stockout?",
        data_context
    )
    
    print(f"Analysis: {response}\n")
    return True


def main():
    print("\n🤖 StockSense Local LLM Integration Test")
    print("=" * 80)
    
    tests = [
        ("Basic Chat", test_basic_chat),
        ("Scenario Generation", test_scenario_generation),
        ("Data Analysis", test_data_analysis),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 80)
    print("📊 Test Results:")
    print("=" * 80)
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! LLM integration is working.")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()
