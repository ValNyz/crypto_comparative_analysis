# =============================================================================
# FILE: lib/data/discovery.py
# =============================================================================
"""Automatic pair discovery from data files."""

import fnmatch
from pathlib import Path
from typing import List, Set


def discover_pairs(data_dir: str, timeframe: str = "1h") -> List[str]:
    """
    Discover available trading pairs from data directory.

    Args:
        data_dir: Path to data directory
        timeframe: Timeframe to look for

    Returns:
        Sorted list of pair names in Freqtrade format (e.g., "BTC/USDC:USDC")
    """
    data_path = Path(data_dir)
    pairs: Set[str] = set()

    # Patterns to search for
    patterns = [f"*-{timeframe}-futures.feather", f"*-{timeframe}.feather"]

    for pattern in patterns:
        for filepath in data_path.glob(pattern):
            pair = _extract_pair_from_filename(filepath.stem)
            if pair:
                pairs.add(pair)

    # Sort with priority pairs first
    sorted_pairs = sorted(pairs)
    priority = ["BTC/USDC:USDC", "ETH/USDC:USDC", "SOL/USDC:USDC"]

    result = [p for p in priority if p in sorted_pairs]
    for p in result:
        sorted_pairs.remove(p)

    return result + sorted_pairs


def _extract_pair_from_filename(stem: str) -> str | None:
    """
    Extract pair name from filename stem.

    Args:
        stem: Filename without extension (e.g., "BTC_USDC_USDC-1h-futures")

    Returns:
        Pair name in Freqtrade format or None
    """
    parts = stem.split("-")
    if len(parts) < 2:
        return None

    underscores = parts[0].split("_")

    if len(underscores) == 3:
        # Format: BTC_USDC_USDC -> BTC/USDC:USDC
        return f"{underscores[0]}/{underscores[1]}:{underscores[2]}"
    elif len(underscores) == 2:
        # Format: BTC_USDC -> BTC/USDC:USDC
        return f"{underscores[0]}/{underscores[1]}:{underscores[1]}"

    return None


def expand_pair_patterns(
    patterns: List[str], data_dir: str, timeframe: str = "1h"
) -> List[str]:
    """
    Expand wildcard patterns to actual pair names.

    Args:
        patterns: List of patterns (can include wildcards like "*/USDC:*")
        data_dir: Path to data directory
        timeframe: Timeframe for discovery

    Returns:
        List of matching pair names
    """
    discovered = discover_pairs(data_dir, timeframe)

    if not discovered:
        return []

    result: Set[str] = set()

    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            # Wildcard pattern
            matched = fnmatch.filter(discovered, pattern)
            if matched:
                result.update(matched)
                print(f"  🔍 Pattern '{pattern}' → {len(matched)} paires")
        elif pattern in discovered:
            # Exact match
            result.add(pattern)

    return sorted(result)


def list_available_pairs(data_dir: str, timeframe: str = "1h") -> None:
    """Print all available pairs."""
    pairs = discover_pairs(data_dir, timeframe)
    print(f"\n📊 {len(pairs)} paires disponibles:\n")
    for pair in pairs:
        print(f"  {pair}")


def get_pair_data_files(pair: str, data_dir: str, timeframe: str = "1h") -> dict:
    """
    Get all data files for a specific pair.

    Args:
        pair: Pair name (e.g., "BTC/USDC:USDC")
        data_dir: Path to data directory
        timeframe: Timeframe

    Returns:
        Dict with file paths for different data types
    """
    data_path = Path(data_dir)

    # Convert pair format to filename format
    parts = pair.replace("/", "_").replace(":", "_")

    files = {
        "ohlcv": None,
        "ohlcv_futures": None,
        "funding_rate": None,
    }

    # Search for files
    for filepath in data_path.glob(f"{parts}*"):
        stem = filepath.stem.lower()
        if "funding_rate" in stem:
            files["funding_rate"] = filepath
        elif "futures" in stem:
            files["ohlcv_futures"] = filepath
        else:
            files["ohlcv"] = filepath

    return files
