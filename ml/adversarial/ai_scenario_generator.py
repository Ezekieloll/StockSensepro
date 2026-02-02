"""
AI-Powered Adversarial Scenario Generator
Analyzes historical data to create realistic stress test scenarios with strategic recommendations.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Scenario:
    """Represents a stress test scenario"""
    id: str
    name: str
    description: str
    demand_multiplier: float  # How much demand increases
    duration_days: int  # How long the scenario lasts
    affected_categories: List[str]  # Which product categories are affected
    probability: float  # Likelihood (0-1)
    
    # Strategic recommendations
    strategies: List[str]
    priority_level: str  # "critical", "high", "medium", "low"


class AIScenarioGenerator:
    """
    Generates intelligent adversarial scenarios based on:
    - Historical demand patterns
    - Seasonal trends
    - Product category behavior
    - Real-world events
    """
    
    def __init__(self):
        self.scenarios_library = self._build_scenarios_library()
    
    def _build_scenarios_library(self) -> List[Scenario]:
        """Pre-built scenarios based on common retail disruptions"""
        return [
            # Scenario 1: Holiday Rush
            Scenario(
                id="holiday_rush",
                name="Holiday Shopping Rush",
                description="Major holiday (Christmas, Thanksgiving) causing 3-5× demand spike in specific categories",
                demand_multiplier=4.0,
                duration_days=14,
                affected_categories=["Fresh Produce", "Bakery", "Beverages", "Dairy"],
                probability=0.90,  # Almost certain annually
                strategies=[
                    "Increase safety stock by 200% for affected categories 2 weeks before",
                    "Negotiate expedited delivery with suppliers",
                    "Pre-authorize overtime for warehouse staff",
                    "Set up temporary storage if needed"
                ],
                priority_level="critical"
            ),
            
            # Scenario 2: Supply Chain Disruption
            Scenario(
                id="supply_disruption",
                name="Supply Chain Disruption",
                description="Supplier failure or logistics delay causing inability to restock, normal demand continues",
                demand_multiplier=1.0,  # Demand stays same, supply drops
                duration_days=7,
                affected_categories=["All"],
                probability=0.60,
                strategies=[
                    "Maintain 30-day safety stock for critical SKUs",
                    "Establish backup suppliers for top 20% products",
                    "Implement automated alerts when stock < 14 days",
                    "Create emergency procurement protocols"
                ],
                priority_level="high"
            ),
            
            # Scenario 3: Viral Product Trend
            Scenario(
                id="viral_trend",
                name="Viral Social Media Trend",
                description="Specific product goes viral (TikTok, Instagram), causing 10-20× spike for 1-2 SKUs",
                demand_multiplier=15.0,
                duration_days=5,
                affected_categories=["Random"],  # Unpredictable
                probability=0.30,
                strategies=[
                    "Monitor social media trends for your products",
                    "Keep flexible cash reserves for emergency orders",
                    "Establish rapid-response supplier agreements",
                    "Consider dynamic pricing during extreme demand"
                ],
                priority_level="medium"
            ),
            
            # Scenario 4: Weather Event
            Scenario(
                id="weather_emergency",
                name="Severe Weather Event",
                description="Hurricane, snowstorm, or extreme heat causing panic buying (2-3× spike)",
                demand_multiplier=2.5,
                duration_days=3,
                affected_categories=["Fresh Produce", "Beverages", "Canned Goods"],
                probability=0.40,
                strategies=[
                    "Monitor weather forecasts 7 days ahead",
                    "Pre-position inventory before storms",
                    "Communicate stock levels to customers proactively",
                    "Implement purchase limits on high-demand items"
                ],
                priority_level="high"
            ),
            
            # Scenario 5: Competitor Closure
            Scenario(
                id="competitor_closure",
                name="Competitor Store Closure",
                description="Nearby competitor closes, customers shift to your stores (sustained 30-50% increase)",
                demand_multiplier=1.4,
                duration_days=90,  # Long-term
                affected_categories=["All"],
                probability=0.20,
                strategies=[
                    "Gradually increase baseline stock levels by 40%",
                    "Expand shelf space and storage capacity",
                    "Renegotiate supplier contracts for higher volumes",
                    "Hire additional staff for sustained higher traffic"
                ],
                priority_level="medium"
            ),
            
            # Scenario 6: Product Recall
            Scenario(
                id="product_recall",
                name="Competitor Product Recall",
                description="Major brand recall drives customers to your alternative products (5-8× spike on substitutes)",
                demand_multiplier=6.5,
                duration_days=21,
                affected_categories=["Specific affected category"],
                probability=0.25,
                strategies=[
                    "Maintain relationships with alternative suppliers",
                    "Monitor FDA/USDA recall announcements",
                    "Stock competing brands as backup",
                    "Quick-pivot procurement when recalls announced"
                ],
                priority_level="high"
            ),
            
            # Scenario 7: Economic Boom
            Scenario(
                id="economic_boom",
                name="Local Economic Boom",
                description="New employer, housing development increases local population 20-30%",
                demand_multiplier=1.25,
                duration_days=180,  # 6 months
                affected_categories=["All"],
                probability=0.15,
                strategies=[
                    "Analyze demographic data quarterly",
                    "Partner with local development projects",
                    "Gradually expand inventory capacity",
                    "Consider opening new store locations"
                ],
                priority_level="low"
            ),
            
            # Scenario 8: Promotional Event
            Scenario(
                id="promo_event",
                name="Major Promotional Campaign",
                description="Deep discount promotion (BOGO, 50% off) causing 4-6× temporary spike",
                demand_multiplier=5.0,
                duration_days=7,
                affected_categories=["Promotional categories"],
                probability=0.80,  # Planned events
                strategies=[
                    "Plan promotions 30+ days in advance",
                    "Pre-order promotional inventory",
                    "Coordinate with marketing on realistic stock levels",
                    "Set max quantities per customer if needed"
                ],
                priority_level="medium"
            ),
        ]
    
    def generate_scenarios_for_sku(
        self, 
        sku: str,
        category: str,
        historical_demand: float,
        current_inventory: int,
        store_id: str
    ) -> List[Tuple[Scenario, float]]:
        """
        Generate relevant scenarios for a specific SKU with calculated worst-case demand.
        
        Returns:
            List of (Scenario, worst_case_demand) tuples
        """
        relevant_scenarios = []
        
        for scenario in self.scenarios_library:
            # Check if this scenario affects this category
            if scenario.affected_categories == ["All"] or \
               category in scenario.affected_categories or \
               scenario.affected_categories == ["Random"]:
                
                # Calculate worst-case demand for this scenario
                worst_case_demand = historical_demand * scenario.demand_multiplier
                
                relevant_scenarios.append((scenario, worst_case_demand))
        
        # Sort by probability (most likely first)
        relevant_scenarios.sort(key=lambda x: x[0].probability, reverse=True)
        
        return relevant_scenarios
    
    def analyze_vulnerability(
        self,
        sku: str,
        category: str,
        current_inventory: int,
        avg_daily_demand: float,
        store_id: str
    ) -> Dict:
        """
        Analyze SKU vulnerability across all scenarios and provide recommendations.
        
        Returns:
            Dict with vulnerability analysis and strategic recommendations
        """
        scenarios = self.generate_scenarios_for_sku(
            sku, category, avg_daily_demand, current_inventory, store_id
        )
        
        # Calculate days of cover for each scenario
        vulnerabilities = []
        for scenario, worst_case_demand in scenarios:
            days_of_cover = current_inventory / worst_case_demand if worst_case_demand > 0 else 999
            stockout_risk = days_of_cover < scenario.duration_days
            
            vulnerabilities.append({
                "scenario": scenario.name,
                "probability": scenario.probability,
                "worst_case_demand_daily": worst_case_demand,
                "days_of_cover": days_of_cover,
                "scenario_duration": scenario.duration_days,
                "stockout_risk": stockout_risk,
                "strategies": scenario.strategies,
                "priority": scenario.priority_level
            })
        
        # Get the most critical vulnerability
        critical_vulns = [v for v in vulnerabilities if v["stockout_risk"]]
        
        return {
            "sku": sku,
            "store": store_id,
            "current_inventory": current_inventory,
            "avg_daily_demand": avg_daily_demand,
            "scenarios_analyzed": len(vulnerabilities),
            "critical_vulnerabilities": len(critical_vulns),
            "vulnerabilities": vulnerabilities,
            "overall_risk_score": self._calculate_overall_risk(vulnerabilities),
            "top_recommendations": self._get_top_recommendations(vulnerabilities)
        }
    
    def _calculate_overall_risk(self, vulnerabilities: List[Dict]) -> float:
        """
        Calculate weighted risk score based on probability and stockout risk.
        Returns 0-1 (1 = highest risk)
        """
        if not vulnerabilities:
            return 0.0
        
        weighted_risk = sum(
            v["probability"] * (1.0 if v["stockout_risk"] else 0.0)
            for v in vulnerabilities
        )
        
        max_possible_risk = sum(v["probability"] for v in vulnerabilities)
        
        return weighted_risk / max_possible_risk if max_possible_risk > 0 else 0.0
    
    def _get_top_recommendations(self, vulnerabilities: List[Dict], top_n: int = 4) -> List[str]:
        """Get the most important strategic recommendations"""
        # Prioritize by: high probability + stockout risk + critical priority
        critical_scenarios = [
            v for v in vulnerabilities 
            if v["stockout_risk"] and v["probability"] > 0.3
        ]
        
        if not critical_scenarios:
            # If no critical risks, get general best practices
            critical_scenarios = sorted(
                vulnerabilities, 
                key=lambda x: x["probability"], 
                reverse=True
            )[:2]
        
        # Collect unique strategies
        recommendations = []
        seen = set()
        
        for vuln in critical_scenarios:
            for strategy in vuln["strategies"]:
                if strategy not in seen:
                    recommendations.append(strategy)
                    seen.add(strategy)
                if len(recommendations) >= top_n:
                    return recommendations
        
        return recommendations


# Convenience function for quick scenario testing
def get_scenario_by_id(scenario_id: str) -> Scenario:
    """Get a specific scenario by ID"""
    generator = AIScenarioGenerator()
    for scenario in generator.scenarios_library:
        if scenario.id == scenario_id:
            return scenario
    raise ValueError(f"Scenario '{scenario_id}' not found")


def list_all_scenarios() -> List[Scenario]:
    """List all available scenarios"""
    generator = AIScenarioGenerator()
    return generator.scenarios_library
