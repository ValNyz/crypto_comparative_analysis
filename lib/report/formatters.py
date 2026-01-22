# =============================================================================
# FILE: lib/report/formatters.py
# =============================================================================
"""Formatting helpers for report output."""

from typing import List


def format_table_row(columns: List[str], widths: List[int], sep: str = "│") -> str:
    """Format a table row with fixed column widths."""
    parts = []
    for col, width in zip(columns, widths):
        parts.append(f"{str(col):<{width}}")
    return f"  {sep} ".join(parts)


def format_bar(value: float, max_value: float, width: int = 30, char: str = "█") -> str:
    """Create a horizontal bar chart."""
    if max_value == 0:
        return ""
    ratio = min(value / max_value, 1.0)
    filled = int(ratio * width)
    return char * filled


def format_percent(value: float, width: int = 6, precision: int = 1) -> str:
    """Format a percentage value."""
    return f"{value:>{width}.{precision}f}%"


def format_profit(value: float, width: int = 7, precision: int = 1) -> str:
    """Format a profit value with sign."""
    return f"{value:>+{width}.{precision}f}"


def format_sharpe(value: float, width: int = 7, precision: int = 2) -> str:
    """Format a Sharpe ratio."""
    return f"{value:>+{width}.{precision}f}"


def abbrev_regime(regime: str) -> str:
    """Get abbreviated regime name."""
    abbrevs = {
        "bull": "Bu",
        "bear": "Be",
        "range": "Ra",
        "volatile": "Vo",
        "quiet": "Qu",
    }
    return abbrevs.get(regime, regime[:2].capitalize())


def print_header(title: str, char: str = "=", width: int = 120):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def print_section(title: str, char: str = "-", width: int = 110):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_separator(char: str = "-", width: int = 90):
    """Print a separator line."""
    print("  " + char * width)
