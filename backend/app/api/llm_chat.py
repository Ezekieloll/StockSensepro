"""
LLM Chat API Endpoint
Provides chatbot functionality using local Ollama LLM
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path

from app.database import get_db
from app.models import Forecast, Inventory, Transaction, AdversarialRisk

# Add ML directory to path
ml_dir = Path(__file__).parent.parent.parent.parent / "ml"
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

router = APIRouter(prefix="/llm", tags=["LLM"])


class ChatRequest(BaseModel):
    message: str
    store_id: Optional[str] = None
    sku: Optional[str] = None
    context: Optional[str] = None  # "forecasts", "inventory", "risks", "general"


class ChatResponse(BaseModel):
    response: str
    context_used: Optional[str] = None
    model: str


class ScenarioGenerationRequest(BaseModel):
    sku: str
    store_id: str
    custom_prompt: Optional[str] = None


@router.post("/analyze-scenario")
async def analyze_interactive_scenario(
    request: dict,
    db: Session = Depends(get_db)
):
    """
    Analyze user's natural language scenario and calculate business impact.
    
    Example inputs:
    - "Tomorrow there will be a lockdown"
    - "News says 20% tax on groceries next week"
    - "Competitor closing in S1 area"
    - "Snowstorm forecast for 3 days"
    
    Returns detailed analysis with store-specific impacts and recommendations.
    """
    try:
        from adversarial.interactive_scenario_ai import analyze_user_scenario, calculate_precise_impact
        
        user_input = request.get('scenario', '')
        
        if not user_input:
            raise HTTPException(status_code=400, detail="Scenario text required")
        
        # Get AI analysis
        analysis = analyze_user_scenario(user_input, db)
        
        if 'error' in analysis:
            return {
                'status': 'error',
                'message': analysis['error'],
                'fallback': analysis.get('fallback', False)
            }
        
        # Normalize the analysis structure (AI might return nested 'analysis' key)
        if 'analysis' in analysis and isinstance(analysis.get('analysis'), dict):
            # Double-nested: {analysis: {analysis: {...}}}
            ai_result = analysis['analysis']
        else:
            ai_result = analysis
        
        # Calculate precise impact
        precise_impact = calculate_precise_impact(ai_result, db)
        
        return {
            'status': 'success',
            'analysis': ai_result,
            'precise_impact': precise_impact
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scenario analysis failed: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_with_llm(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Chat with local LLM with database context
    
    - Automatically pulls relevant data based on context
    - Provides intelligent insights about inventory, forecasts, risks
    - Can answer general supply chain questions
    """
    try:
        from llm.ollama_client import OllamaClient
        
        client = OllamaClient()
        
        # Check if Ollama is available
        if not client.is_available():
            raise HTTPException(
                status_code=503,
                detail="LLM service not available. Make sure Ollama is running."
            )
        
        # Build context from database if needed
        data_context = ""
        
        if request.context == "forecasts" or "forecast" in request.message.lower():
            # Get forecast data
            stmt = select(Forecast)
            if request.store_id:
                stmt = stmt.where(Forecast.store_id == request.store_id)
            if request.sku:
                stmt = stmt.where(Forecast.sku_id == request.sku)
            
            forecasts = db.execute(stmt.limit(10)).scalars().all()
            
            if forecasts:
                data_context += "Recent Forecasts:\n"
                for fc in forecasts:
                    data_context += f"- SKU {fc.sku_id} @ {fc.store_id}: {fc.quantity} units (Type: {fc.forecast_type})\n"
        
        if request.context == "risks" or "risk" in request.message.lower():
            # Get adversarial risk data
            stmt = select(AdversarialRisk).where(AdversarialRisk.stockout == True)
            if request.store_id:
                stmt = stmt.where(AdversarialRisk.store_id == request.store_id)
            if request.sku:
                stmt = stmt.where(AdversarialRisk.sku == request.sku)
            
            risks = db.execute(stmt.order_by(AdversarialRisk.risk_score.desc()).limit(10)).scalars().all()
            
            if risks:
                data_context += "\nHigh-Risk SKUs:\n"
                for risk in risks:
                    data_context += f"- SKU {risk.sku} @ {risk.store_id}: Risk Score {risk.risk_score:.3f}, "
                    data_context += f"Inventory {risk.current_inventory}, Worst-case demand {risk.worst_case_demand}\n"
                    if risk.scenario_name:
                        data_context += f"  Scenario: {risk.scenario_name}\n"
        
        if request.context == "inventory" or "inventory" in request.message.lower():
            # Get inventory summary
            stmt = select(
                func.count(Inventory.sku_id).label('total_skus'),
                func.sum(Inventory.quantity).label('total_units')
            )
            if request.store_id:
                stmt = stmt.where(Inventory.store_id == request.store_id)
            
            result = db.execute(stmt).first()
            if result:
                data_context += f"\nInventory Summary:\n"
                data_context += f"- Total SKUs: {result.total_skus}\n"
                data_context += f"- Total Units: {result.total_units}\n"
        
        # Analyze with LLM
        if data_context:
            response = client.analyze_database_data(request.message, data_context)
        else:
            # General chat without specific data
            system_prompt = """You are a supply chain and inventory management assistant for StockSense. 
You help users understand their inventory data, forecasts, and supply chain risks. 
Provide clear, actionable advice."""
            response = client.generate(request.message, system_prompt=system_prompt)
        
        return ChatResponse(
            response=response,
            context_used=data_context if data_context else None,
            model=client.model_name
        )
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="LLM client not available. Make sure Ollama is installed."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


@router.post("/generate-scenario")
async def generate_custom_scenario(request: ScenarioGenerationRequest, db: Session = Depends(get_db)):
    """
    Generate a custom adversarial scenario using LLM
    
    - Analyzes the specific SKU's historical data
    - Generates realistic scenario with demand multiplier and strategies
    - Can include custom prompts for specific scenario types
    """
    try:
        from llm.ollama_client import OllamaClient
        
        client = OllamaClient()
        
        if not client.is_available():
            raise HTTPException(
                status_code=503,
                detail="LLM service not available"
            )
        
        # Get SKU data
        stmt = select(Forecast).where(
            Forecast.sku_id == request.sku,
            Forecast.store_id == request.store_id,
            Forecast.forecast_type == "baseline"
        ).limit(1)
        
        forecast = db.execute(stmt).scalar_one_or_none()
        
        if not forecast:
            raise HTTPException(
                status_code=404,
                detail=f"No forecast data found for SKU {request.sku} at store {request.store_id}"
            )
        
        # Get inventory
        inv_stmt = select(Inventory).where(
            Inventory.sku_id == request.sku,
            Inventory.store_id == request.store_id
        )
        inventory = db.execute(inv_stmt).scalar_one_or_none()
        current_inv = inventory.quantity if inventory else 0
        
        # Generate scenario
        scenario = client.generate_adversarial_scenario(
            sku=request.sku,
            category="Unknown",  # TODO: Get from SKU table
            avg_demand=forecast.quantity,
            current_inventory=current_inv,
            historical_context=request.custom_prompt
        )
        
        return scenario
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scenario generation failed: {str(e)}"
        )


@router.get("/status")
async def llm_status():
    """Check if LLM service is available"""
    try:
        from llm.ollama_client import OllamaClient
        
        client = OllamaClient()
        available = client.is_available()
        
        return {
            "available": available,
            "model": client.model_name if available else None,
            "base_url": client.base_url
        }
    except:
        return {
            "available": False,
            "error": "Ollama client not installed"
        }
