# =============================================================================
# FILE: lib/signals/__init__.py
# =============================================================================
"""Signals module - signal configurations."""

from .base import SignalConfig
from .registry import (
    get_signal_configs,
    get_funding_signals,
    get_technical_signals,
    get_advanced_signals,
    get_combo_signals,
)

__all__ = [
    "SignalConfig",
    "get_signal_configs",
    "get_funding_signals",
    "get_technical_signals",
    "get_advanced_signals",
    "get_combo_signals",
]
