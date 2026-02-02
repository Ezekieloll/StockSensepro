from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import subprocess
import sys
from pathlib import Path
from typing import List

from app.database import get_db
from app.models.adversarial_risk import AdversarialRisk
from app.api.schemas.adversarial import ScenarioInfo, RunAITestRequest, AITestResponse

router = APIRouter(prefix="/adversarial", tags=["Adversarial"])

# Add ML directory to path for imports
ml_dir = Path(__file__).parent.parent.parent.parent / "ml"
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))


@router.get("/scenarios", response_model=List[ScenarioInfo])
def get_available_scenarios(use_ai: bool = False, db: Session = Depends(get_db)):
    """
    Get all available adversarial scenarios.
    
    - use_ai=false: Returns hardcoded industry-standard scenarios (fast)
    - use_ai=true: Uses Qwen LLM to analyze YOUR database and generate custom scenarios (slow, requires Ollama)
    
    Returns scenario details, probabilities, and recommended strategies.
    """
    try:
        if use_ai:
            # Use AI to generate scenarios based on actual data
            from adversarial.dynamic_ai_scenarios import generate_ai_scenarios
            
            ai_scenarios = generate_ai_scenarios(db=db)
            
            return [ScenarioInfo(**scenario) for scenario in ai_scenarios]
        else:
            # Use hardcoded scenarios
            from adversarial.ai_scenario_generator import AIScenarioGenerator
            
            generator = AIScenarioGenerator()
            scenarios = []
            
            for scenario in generator.scenarios_library:
                scenarios.append(ScenarioInfo(
                    id=scenario.id,
                    name=scenario.name,
                    description=scenario.description,
                    demand_multiplier=scenario.demand_multiplier,
                    duration_days=scenario.duration_days,
                    affected_categories=scenario.affected_categories,
                    probability=scenario.probability,
                    strategies=scenario.strategies,
                    priority_level=scenario.priority_level
                ))
            
            return scenarios
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load scenarios: {str(e)}"
        )


@router.post("/run-ai-test", response_model=AITestResponse)
def run_ai_adversarial_test(
    request: RunAITestRequest,
    db: Session = Depends(get_db)
):
    """
    Run AI-powered adversarial testing with intelligent scenarios.
    
    - If scenario_ids is empty/null, runs all scenarios
    - Returns detailed results for each scenario tested
    - Includes strategic recommendations
    """
    try:
        from adversarial.populate_db_ai import run_ai_adversarial_testing
        
        # Run AI testing
        results = run_ai_adversarial_testing(
            selected_scenarios=request.scenario_ids,
            db=db
        )
        
        return AITestResponse(**results)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI adversarial testing failed: {str(e)}"
        )

@router.get("/")
def get_adversarial_risk(
    sku: str | None = None,
    store_id: str | None = None,
    scenario_id: str | None = None,
    high_risk_only: bool = False,
    db: Session = Depends(get_db),
):
    """
    Get adversarial risk assessments with filtering options.
    
    - Filter by SKU, store, or scenario
    - high_risk_only: Only return stockout risks
    - Returns all matching risk records ordered by risk_score
    """
    query = db.query(AdversarialRisk)

    if sku:
        query = query.filter(AdversarialRisk.sku == sku)
    if store_id:
        query = query.filter(AdversarialRisk.store_id == store_id)
    if scenario_id:
        query = query.filter(AdversarialRisk.scenario_id == scenario_id)
    if high_risk_only:
        query = query.filter(AdversarialRisk.stockout == True)

    return query.order_by(AdversarialRisk.risk_score.desc()).all()


@router.post("/run-test")
def trigger_adversarial_test():
    """
    Trigger adversarial testing script to recalculate risk scores.
    Runs the populate_db.py script in the ml/adversarial directory.
    """
    try:
        # Get the ml directory path
        backend_dir = Path(__file__).parent.parent.parent
        ml_dir = backend_dir.parent / "ml"
        
        if not ml_dir.exists():
            raise HTTPException(
                status_code=500, 
                detail=f"ML directory not found at {ml_dir}"
            )
        
        # Run the adversarial populate_db.py script
        result = subprocess.run(
            ["python", "-m", "adversarial.populate_db"],
            cwd=ml_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Script failed: {result.stderr}"
            )
        
        return {
            "status": "success",
            "message": "Adversarial testing completed successfully",
            "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout  # Last 500 chars
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Adversarial test timed out after 5 minutes"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run adversarial test: {str(e)}"
        )
