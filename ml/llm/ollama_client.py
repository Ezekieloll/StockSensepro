"""
Ollama Client for StockSense
Provides interface to local LLM for database queries, scenario generation, and chatbot
"""

import subprocess
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime


class OllamaClient:
    """Client for interacting with Ollama local LLM"""
    
    def __init__(self, model_name: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_history: List[Dict[str, str]] = []
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """
        Generate a response from the local LLM
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Creativity (0.0-1.0, higher = more creative)
            max_tokens: Max response length
            stream: Whether to stream response
        
        Returns:
            Generated text
        """
        try:
            # Build request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            # Make request to Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        reset_history: bool = False
    ) -> str:
        """
        Chat with the model (maintains conversation history)
        
        Args:
            message: User message
            system_prompt: System instructions (only used if reset_history=True)
            reset_history: Clear conversation history
        
        Returns:
            Model response
        """
        if reset_history:
            self.conversation_history = []
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            # Build chat request
            payload = {
                "model": self.model_name,
                "messages": self.conversation_history,
                "stream": False
            }
            
            if system_prompt and len(self.conversation_history) == 1:
                payload["messages"].insert(0, {"role": "system", "content": system_prompt})
            
            # Make request
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result.get("message", {}).get("content", "")
                
                # Add assistant response to history
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                return assistant_message
            else:
                return f"Error: {response.status_code}"
        
        except Exception as e:
            return f"Error in chat: {str(e)}"
    
    def generate_adversarial_scenario(
        self,
        sku: str,
        category: str,
        avg_demand: float,
        current_inventory: int,
        historical_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a custom adversarial scenario using LLM
        
        Args:
            sku: SKU identifier
            category: Product category
            avg_demand: Average daily demand
            current_inventory: Current inventory level
            historical_context: Optional historical data context
        
        Returns:
            Dict with scenario details
        """
        system_prompt = """You are an expert supply chain risk analyst. Your job is to generate realistic adversarial scenarios for inventory stress testing.

You must respond with ONLY a valid JSON object with this exact structure:
{
  "scenario_name": "Short descriptive name",
  "description": "2-3 sentence description of the scenario",
  "demand_multiplier": <number between 0.5 and 20.0>,
  "duration_days": <number of days the scenario lasts>,
  "probability": <number between 0.0 and 1.0>,
  "strategies": ["strategy 1", "strategy 2", "strategy 3"],
  "priority_level": "critical|high|medium|low",
  "reasoning": "Brief explanation of why this scenario is relevant"
}"""

        prompt = f"""Generate a realistic adversarial scenario for this product:

SKU: {sku}
Category: {category}
Average Daily Demand: {avg_demand} units
Current Inventory: {current_inventory} units
Days of Cover: {current_inventory / avg_demand if avg_demand > 0 else 0:.1f} days

{f"Historical Context: {historical_context}" if historical_context else ""}

Consider:
- Seasonal patterns for {category}
- Realistic demand spikes (not just worst-case)
- Supply chain vulnerabilities
- Local market conditions
- Product-specific risks

Generate ONE realistic scenario with actionable strategies."""

        response = self.generate(prompt, system_prompt=system_prompt, temperature=0.8)
        
        # Try to parse JSON response
        try:
            # Extract JSON from response (in case model added extra text)
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                scenario = json.loads(json_str)
                scenario["generated_at"] = datetime.now().isoformat()
                scenario["model"] = self.model_name
                return scenario
            else:
                return {"error": "Failed to extract JSON from response", "raw_response": response}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {str(e)}", "raw_response": response}
    
    def analyze_database_data(self, query: str, data_context: str) -> str:
        """
        Analyze database data and provide insights
        
        Args:
            query: User question about the data
            data_context: String representation of relevant database data
        
        Returns:
            Analysis and recommendations
        """
        system_prompt = """You are a data analyst specializing in inventory management and supply chain optimization. 
Analyze the provided database data and answer questions with:
- Clear insights
- Actionable recommendations
- Quantitative support when possible
- Risk assessments where relevant"""

        prompt = f"""Database Data:
{data_context}

User Question: {query}

Provide a concise, actionable answer:"""

        return self.generate(prompt, system_prompt=system_prompt, temperature=0.5)
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []


# Convenience functions
def quick_generate(prompt: str, model: str = "qwen2.5:14b") -> str:
    """Quick one-off generation"""
    client = OllamaClient(model_name=model)
    return client.generate(prompt)


def quick_chat(message: str, model: str = "qwen2.5:14b") -> str:
    """Quick chat without history"""
    client = OllamaClient(model_name=model)
    return client.chat(message, reset_history=True)
