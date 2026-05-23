"""Shared pytest configuration for the test suite.

Loads the repo `.env` at session start so any test reading environment
variables finds them regardless of collection order.
"""

from src.config.env import load_env_variables

load_env_variables()
