"""Configuration for the book generation system"""
from dotenv import load_dotenv
from typing import Dict
import os
load_dotenv()

def get_config(local_url: str = "https://api.deepseek.com/v1") -> Dict:
    """Get the configuration for the agents"""
    
    # Basic config for local LLM
    config_list = [{
        'model': 'deepseek-chat',
        'base_url': local_url,
        'api_key': os.getenv("API_KEY"),
        "price" : [0.014, 0.28]
    }]

    # Common configuration for all agents
    agent_config = {
        "seed": 42,
        "temperature": 0.9,
        "config_list": config_list,
        "timeout": 600,
        "cache_seed": None
    }
    
    return agent_config