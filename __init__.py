"""
Freqtrade Comparative Backtest Library V3
==========================================

Architecture modulaire pour l'analyse comparative de stratégies de trading.
"""

from .config import Config, load_config
from .signals import SignalConfig, get_signal_configs
from .exits import ExitConfig, get_exit_config, get_all_exit_configs
from .backtest import BacktestRunner, parse_freqtrade_output
from .generation import StrategyGenerator
from .report import ReportGenerator
from .data import discover_pairs, expand_pair_patterns

__version__ = "3.0.0"
__all__ = [
    "Config",
    "load_config",
    "SignalConfig",
    "get_signal_configs",
    "ExitConfig",
    "get_exit_config",
    "get_all_exit_configs",
    "BacktestRunner",
    "parse_freqtrade_output",
    "StrategyGenerator",
    "ReportGenerator",
    "discover_pairs",
    "expand_pair_patterns",
]
