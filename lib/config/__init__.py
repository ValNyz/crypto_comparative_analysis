# =============================================================================
# FILE: lib/config/__init__.py
# =============================================================================
"""Configuration module."""

from .base import Config
from .loader import load_config, load_yaml

__all__ = ["Config", "load_config", "load_yaml"]
