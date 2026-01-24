# =============================================================================
# FILE: lib/report/__init__.py
# =============================================================================
"""Report generation module."""

from .base import ReportGenerator
from .rolling import RollingReportGenerator
from .formatters import (
    format_table_row,
    format_bar,
    format_percent,
    format_profit,
    format_sharpe,
    abbrev_regime,
    print_header,
    print_section,
    print_separator,
)

__all__ = [
    "ReportGenerator",
    "RollingReportGenerator",
    # Formatters
    "format_table_row",
    "format_bar",
    "format_percent",
    "format_profit",
    "format_sharpe",
    "abbrev_regime",
    "print_header",
    "print_section",
    "print_separator",
]
