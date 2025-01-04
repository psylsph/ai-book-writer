"""Configuration for the book generation system"""
from typing import Dict


def get_config() -> Dict:
    """Get the configuration for the agents"""
    
    config_list = [
    {
        "model": "mistral-16384",
        "api_type": "ollama",
        "num_predict": -1,
        "repeat_penalty": 1.1,
        "seed": 42,
        "stream": False,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.8
    }
]

    # Common configuration for all agents
    agent_config = {
        "config_list": config_list,
        "api_type": "ollama",
        "model": "mistral-16384",
        "timeout": 600,
        "cache_seed": None
    }
    
    return agent_config