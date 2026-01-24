# =============================================================================
# FILE: lib/report/sections/__init__.py
# =============================================================================
"""Report sections module."""

from .global_metrics import print_global_metrics
from .regime import (
    print_regime_distribution,
    print_regime_performance,
    print_signal_regime_matrix,
)
from .exit_analysis import print_exit_analysis
from .temporal import print_temporal_analysis
from .rankings import print_top_by_sharpe, print_polyvalent_signals
from .recommendations import print_recommendations
from .coin import (
    print_per_coin_summary,
    print_coin_comparison_matrix,
    print_consistent_performers,
    get_per_coin_stats,
)

__all__ = [
    "print_global_metrics",
    "print_regime_distribution",
    "print_regime_performance",
    "print_signal_regime_matrix",
    "print_exit_analysis",
    "print_temporal_analysis",
    "print_top_by_sharpe",
    "print_polyvalent_signals",
    "print_recommendations",
    "print_per_coin_summary",
    "print_coin_comparison_matrix",
    "print_consistent_performers",
    "get_per_coin_stats",
]
