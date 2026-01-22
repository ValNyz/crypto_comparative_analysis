# =============================================================================
# FILE: lib/utils/helpers.py
# =============================================================================
"""General helper functions."""

import fnmatch
from typing import List, Dict, Optional


def get_allowed_regimes(
    signal_name: str, regime_mappings: Optional[Dict[str, List[str]]] = None
) -> List[str]:
    """
    Get allowed regimes for a signal based on pattern matching.

    Args:
        signal_name: Name of the signal
        regime_mappings: Dict of pattern -> list of regimes

    Returns:
        List of allowed regime names
    """
    if regime_mappings is None:
        # Import here to avoid circular imports
        from ..config.loader import _default_regime_mappings

        regime_mappings = _default_regime_mappings()

    for pattern, regimes in regime_mappings.items():
        if fnmatch.fnmatch(signal_name, pattern):
            return regimes

    # Default: all regimes allowed
    return ["bull", "bear", "range", "volatile"]


def sanitize_class_name(name: str) -> str:
    """
    Convert a signal name to a valid Python class name.

    Args:
        name: Original name with potential special characters

    Returns:
        Valid Python identifier
    """
    # Replace dots and dashes with underscores
    sanitized = name.replace(".", "_").replace("-", "_")

    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = f"S_{sanitized}"

    return sanitized


def format_percent(value: float, width: int = 6, precision: int = 1) -> str:
    """Format a percentage value."""
    return f"{value:>{width}.{precision}f}%"


def format_profit(value: float, width: int = 7, precision: int = 1) -> str:
    """Format a profit value with sign."""
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


def chunk_list(lst: List, size: int) -> List[List]:
    """Split a list into chunks of given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]
