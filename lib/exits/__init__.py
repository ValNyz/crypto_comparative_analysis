# =============================================================================
# FILE: lib/exits/__init__.py
# =============================================================================
"""Exit methods module."""

from .base import ExitConfig
from .registry import get_exit_config, get_all_exit_configs, load_exits_from_yaml

__all__ = [
    "ExitConfig",
    "get_exit_config",
    "get_all_exit_configs",
    "load_exits_from_yaml",
]
