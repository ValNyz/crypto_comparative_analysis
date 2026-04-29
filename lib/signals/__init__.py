# =============================================================================
# FILE: lib/signals/__init__.py
# =============================================================================
"""Signals module - signal configurations."""

from .base import SignalConfig
from .registry import get_signal_configs

__all__ = ["SignalConfig", "get_signal_configs"]
