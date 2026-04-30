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


def short_pair(pair: str) -> str:
    """Compact display form: 'BTC/USDC:USDC' → 'BTC'.

    Use only for printing — keep the canonical form everywhere else
    (freqtrade configs, cache keys, filenames).
    """
    return pair.split("/")[0] if "/" in pair else pair


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


def chunk_list(lst: List, size: int) -> List[List]:
    """Split a list into chunks of given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]
