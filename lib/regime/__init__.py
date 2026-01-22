# =============================================================================
# FILE: lib/regime/__init__.py
# =============================================================================
"""Regime detection module."""

from .detector import (
    detect_regime_v3,
    calculate_volatility_score,
    calculate_trend_score,
)
from .classifier import classify_regime, get_regime_confidence

__all__ = [
    "detect_regime_v3",
    "calculate_volatility_score",
    "calculate_trend_score",
    "classify_regime",
    "get_regime_confidence",
]
