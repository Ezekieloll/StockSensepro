from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
import subprocess
import sys
import os
from pathlib import Path
from typing import List

from app.database import get_db
from app.models.adversarial_risk import AdversarialRisk
from app.api.schemas.adversarial import ScenarioInfo, RunAITestRequest, AITestResponse, ScenarioActivityRequest
from app.services.auth_utils import get_current_user
from app.services.audit_utils import write_audit_log

router = APIRouter(prefix="/adversarial", tags=["Adversarial"])

# Add ML directory to path for imports
ml_dir = Path(__file__).parent.parent.parent.parent / "ml"
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))


@router.get("/scenarios", response_model=List[ScenarioInfo])
def get_available_scenarios(
    use_ai: bool = False,
    db: Session = Depends(get_db),
):
    """
    Get all available adversarial scenarios.
    
    - use_ai=false: Returns hardcoded industry-standard scenarios (fast)
    - use_ai=true: Uses Qwen LLM to analyze YOUR database and generate custom scenarios (slow, requires Ollama)
    
    Returns scenario details, probabilities, and recommended strategies.
    """
    try:
        if use_ai:
            # Use AI to generate scenarios based on actual data.
            # If AI output is malformed or unavailable, degrade gracefully to standard scenarios.
            try:
                from adversarial.dynamic_ai_scenarios import generate_ai_scenarios
                ai_scenarios = generate_ai_scenarios(db=db)
                return [ScenarioInfo(**scenario) for scenario in ai_scenarios]
            except Exception as ai_error:
                print(f"⚠️ AI scenario generation failed, falling back to standard scenarios: {ai_error}")

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
    run_request: RunAITestRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
    request: Request = None,
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
            selected_scenarios=run_request.scenario_ids,
            category_scoped=run_request.category_scoped,
            custom_scenarios=[
                item.model_dump() if hasattr(item, "model_dump") else item.dict()
                for item in (run_request.custom_scenarios or [])
            ],
            db=db
        )

        write_audit_log(
            db,
            action="AI_ADVERSARIAL_TEST_COMPLETED",
            user_id=int(user.get("sub")) if user.get("sub") else None,
            entity="adversarial_test:run-ai-test",
            details={
                "status": "success",
                "scenarios_tested": results.get("scenarios_tested"),
                "scope_mode": results.get("scope_mode"),
            },
            ip_address=request.client.host if request and request.client else None,
        )
        db.commit()
        
        return AITestResponse(**results)
    
    except Exception as e:
        write_audit_log(
            db,
            action="AI_ADVERSARIAL_TEST_FAILED",
            user_id=int(user.get("sub")) if user.get("sub") else None,
            entity="adversarial_test:run-ai-test",
            details={"error": str(e)},
            ip_address=request.client.host if request and request.client else None,
        )
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"AI adversarial testing failed: {str(e)}"
        )


@router.post("/scenario-activity")
def log_scenario_activity(
    payload: ScenarioActivityRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Persist custom scenario add/edit/delete actions for recent activity timeline."""
    action_map = {
        "created": "CUSTOM_SCENARIO_CREATED",
        "updated": "CUSTOM_SCENARIO_UPDATED",
        "deleted": "CUSTOM_SCENARIO_DELETED",
    }

    normalized_action = (payload.action or "").strip().lower()
    if normalized_action not in action_map:
        raise HTTPException(status_code=400, detail="Invalid scenario activity action")

    write_audit_log(
        db,
        action=action_map[normalized_action],
        user_id=int(user.get("sub")) if user.get("sub") else None,
        entity=f"scenario:{payload.scenario_id}",
        details={
            "scenario_id": payload.scenario_id,
            "scenario_name": payload.scenario_name,
            **(payload.details or {}),
        },
        ip_address=request.client.host if request and request.client else None,
    )
    db.commit()

    return {"status": "success"}

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
def trigger_adversarial_test(
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
    request: Request = None,
):
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
        
        # Set environment to handle UTF-8 output (emojis)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Run the adversarial populate_db.py script
        result = subprocess.run(
            [sys.executable, "-m", "adversarial.populate_db"],
            cwd=ml_dir,
            capture_output=True,
            text=True,
            env=env,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            write_audit_log(
                db,
                action="ADVERSARIAL_TEST_FAILED",
                user_id=int(user.get("sub")) if user.get("sub") else None,
                entity="adversarial_test:run-test",
                details={"error": result.stderr or "Unknown error"},
                ip_address=request.client.host if request and request.client else None,
            )
            db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Script failed: {result.stderr or 'Unknown error'}"
            )
        
        # Handle stdout safely (could be None)
        stdout = result.stdout or ""
        output_text = stdout[-500:] if len(stdout) > 500 else stdout

        write_audit_log(
            db,
            action="ADVERSARIAL_TEST_COMPLETED",
            user_id=int(user.get("sub")) if user.get("sub") else None,
            entity="adversarial_test:run-test",
            details={"status": "success"},
            ip_address=request.client.host if request and request.client else None,
        )
        db.commit()
        
        return {
            "status": "success",
            "message": "Adversarial testing completed successfully",
            "output": output_text
        }
        
    except subprocess.TimeoutExpired:
        write_audit_log(
            db,
            action="ADVERSARIAL_TEST_TIMEOUT",
            user_id=int(user.get("sub")) if user.get("sub") else None,
            entity="adversarial_test:run-test",
            details={"error": "timed out after 5 minutes"},
            ip_address=request.client.host if request and request.client else None,
        )
        db.commit()
        raise HTTPException(
            status_code=500,
            detail="Adversarial test timed out after 5 minutes"
        )
    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        # Include more error details
        import traceback
        error_detail = f"Failed to run adversarial test: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )
