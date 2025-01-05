"""Configuration for the book generation system"""
from typing import Dict
import os

def get_config(local_url: str = "https://api.deepseek.com/v1") -> Dict:
    """Get the configuration for the agents"""
    
    # Basic config for local LLM
    config_list = [{
        'model': 'deepseek-chat',
        'base_url': local_url,
        'api_key': "na",
        "price" : [0.014, 0.28]
    }]

    # Common configuration for all agents
    agent_config = {
        "seed": 42,
        "temperature": 0.9,
        "config_list": config_list,
        "timeout": 600,
        "cache_seed": 41
    }
    
    return agent_config