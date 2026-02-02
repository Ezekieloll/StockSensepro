"""
Pydantic schemas for AI scenario requests and responses
"""

from pydantic import BaseModel
from typing import List, Optional


class ScenarioInfo(BaseModel):
    """Information about an available adversarial scenario"""
    id: str
    name: str
    description: str
    demand_multiplier: float
    duration_days: int
    affected_categories: List[str]
    probability: float
    strategies: List[str]
    priority_level: str


class RunAITestRequest(BaseModel):
    """Request to run AI adversarial testing"""
    scenario_ids: Optional[List[str]] = None  # If None, runs all scenarios
    

class ScenarioResult(BaseModel):
    """Results for a single scenario"""
    name: str
    records_tested: int
    stockout_count: int
    stockout_rate: float
    avg_risk_score: float
    probability: float
    strategies: List[str]


class AITestResponse(BaseModel):
    """Response from AI adversarial testing"""
    status: str
    scenarios_tested: int
    total_records: int
    results_by_scenario: dict  # scenario_id -> ScenarioResult
    most_critical_scenario: dict
