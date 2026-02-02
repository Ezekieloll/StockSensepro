"""
Local LLM integration for StockSense
Uses Ollama to run Qwen 2.5 or other models locally
"""

from .ollama_client import OllamaClient, quick_generate, quick_chat

__all__ = ['OllamaClient', 'quick_generate', 'quick_chat']
