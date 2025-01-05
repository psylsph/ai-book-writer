"""Configuration for the book generation system"""
import os
from typing import Dict, List

def get_config(local_url: str = "http://localhost:11434/v1") -> Dict:
    """Get the configuration for the agents"""
    
    # Basic config for local LLM
    config_list = [{
        'model': 'hf.co/bartowski/LongWriter-llama3.1-8b-GGUF:IQ4_XS',
        'base_url': local_url,
        'api_key': "not-needed",
        'price': [0,0]
    }]

    # Common configuration for all agents
    agent_config = {
        "seed": 42,
        "temperature": 0.7,
        "config_list": config_list,
        "timeout": 600,
        "cache_seed": None
    }
    
    return agent_config