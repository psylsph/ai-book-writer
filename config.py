"""Configuration for the book generation system"""
from typing import Dict

def get_config(local_url: str = "http://localhost:11434/v1") -> Dict:
    """Get the configuration for the agents"""
    
    # Basic config for local LLM
    config_list = [{
        'model': 'smallthinker-8192:latest',
        'base_url': local_url,
        'api_key': "not-needed",
        'price': [0,0],
        'max_tokens': 8192
    }]

    # Common configuration for all agents
    agent_config = {
        "seed": 42,
        "temperature": 0.7,
        "config_list": config_list,
        "timeout": 1200,
        "cache_seed": None
    }
    
    return agent_config