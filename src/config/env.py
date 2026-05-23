"""Environment variable loader for API keys and configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_env_variables() -> None:
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

def get_scopus_api_key() -> Optional[str]:
    """Get Scopus API key from environment."""
    load_env_variables()
    return os.getenv("SCOPUS_API_KEY")

def get_scopus_insttoken() -> Optional[str]:
    """Get Scopus institutional token from environment."""
    load_env_variables()
    return os.getenv("instoken")

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment."""
    load_env_variables()
    return os.getenv("OPENAI_API_KEY")

def get_serpapi_key() -> Optional[str]:
    """Get SerpAPI key from environment."""
    load_env_variables()
    return os.getenv("SERPAPI_KEY")

def verify_api_keys() -> dict:
    """Verify which API keys are available."""
    load_env_variables()
    
    keys = {
        "SCOPUS_API_KEY": bool(os.getenv("SCOPUS_API_KEY")),
        "instoken": bool(os.getenv("instoken")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "SERPAPI_KEY": bool(os.getenv("SERPAPI_KEY")),
    }
    
    return keys
