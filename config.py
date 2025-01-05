"""Configuration for the book generation system"""
from typing import Dict
import os

def get_config() -> Dict:
    """Get the configuration for the agents"""
    
    # Basic config for local LLM
    config_list = [{
        "model": "hf.co/bartowski/LongWriter-llama3.1-8b-GGUF:IQ4_XS",  # Using Ollama model
        'api_key': os.getenv("API_KEY"),
        "base_url": "https://localhost:11434/v1",
        "price": [0,0]
    }]

    # Common configuration for all agents
    agent_config = {
        "temperature": 0.9,
        "config_list": config_list,
        "cache_seed": 41
    }
    
    return agent_config