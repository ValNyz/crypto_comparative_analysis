# =============================================================================
# FILE: lib/backtest/__init__.py
# =============================================================================
"""Backtest module."""

from .runner import BacktestRunner
from .parser import parse_freqtrade_output
from .rolling import (
    RollingConfig,
    RollingWindow,
    generate_windows,
    aggregate_window_results,
    calculate_consistency,
    get_window_details,
    run_rolling_backtest,
)

__all__ = [
    "BacktestRunner",
    "parse_freqtrade_output",
    "RollingConfig",
    "RollingWindow",
    "generate_windows",
    "aggregate_window_results",
    "calculate_consistency",
    "get_window_details",
    "run_rolling_backtest",
]
