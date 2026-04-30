# =============================================================================
# FILE: lib/null_pool/pool_builder.py
# =============================================================================
"""Parquet cache for null-pool trade DataFrames.

Cache key encodes every dimension that affects the pool's distribution:
(pair, timeframe, exit_config, stoploss, roi, timerange, seed). Two
strategies sharing all these dims share the same pool — a single freqtrade
run amortizes the cost across all signals testing those dimensions.
"""

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _hash_dim(value: Any) -> str:
    """Short stable hash for non-trivial dims (ROI dicts, etc.)."""
    s = json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value)
    return hashlib.md5(s.encode()).hexdigest()[:6]


def compute_cache_key(
    pair: str,
    timeframe: str,
    exit_config: str,
    stoploss: float,
    roi: Dict[str, float],
    timerange: str,
    seed: int,
    direction: str = "both",
) -> str:
    """Deterministic filename-safe cache key.

    `direction` ∈ {"long", "short", "both"} selects the random-entry pool —
    a long-only signal needs a long-only pool to avoid biasing the null
    distribution with short trades (and vice versa). Encoded as suffix
    `_dirL`/`_dirS`/`_dirB` so direction-mismatched pools get distinct files.

    Example: 'null_BTC_USDC_USDC_1h_tr_2_1_sl5_roi-a1b2c3_20240101-20250101_seed42_dirL'
    """
    safe_pair = pair.replace("/", "_").replace(":", "_")
    sl_str = f"sl{abs(int(round(stoploss * 100)))}"
    roi_str = f"roi-{_hash_dim(roi)}"
    dir_str = {"long": "L", "short": "S", "both": "B"}.get(direction, "B")
    return (
        f"null_{safe_pair}_{timeframe}_{exit_config}_"
        f"{sl_str}_{roi_str}_{timerange}_seed{seed}_dir{dir_str}"
    )


def load_pool(cache_key: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """Return the cached trade DataFrame, or None on miss / read error."""
    fp = cache_dir / f"{cache_key}.parquet"
    if not fp.exists():
        return None
    try:
        return pd.read_parquet(fp)
    except Exception:
        return None


def save_pool(df: pd.DataFrame, cache_key: str, cache_dir: Path) -> Path:
    """Persist trade DataFrame to parquet (creates cache_dir if needed)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"{cache_key}.parquet"
    df.to_parquet(fp)
    return fp


def extract_trades_from_zip(
    zip_path: Path, class_name: str, log=None
) -> Optional[pd.DataFrame]:
    """Parse the inner backtest JSON from a freqtrade export zip.

    Returns a DataFrame with bootstrap-relevant columns (profit_ratio,
    open_date, close_date, is_short, exit_reason). Returns None on parse
    error or if the strategy/trades aren't present.

    `log` is an optional callable(msg) for failure diagnostics — pass the
    runner's `_phase1_log` to coordinate with the tqdm bar so messages
    don't break the rendered bar. When None, falls back to print().
    """
    def _log(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg)

    try:
        with zipfile.ZipFile(zip_path) as z:
            inner_name = next(
                (
                    n
                    for n in z.namelist()
                    if n.endswith(".json") and "_config" not in n
                ),
                None,
            )
            if not inner_name:
                _log(f"    extract: no inner JSON in {zip_path.name}")
                return None
            data = json.loads(z.read(inner_name))
        strategies = data.get("strategy", {}) or {}
        if class_name not in strategies:
            _log(
                f"    extract: class '{class_name}' not in {zip_path.name}; "
                f"available: {list(strategies.keys())[:5]}"
            )
            return None
        strat = strategies[class_name]
        trades = strat.get("trades", [])
        if not trades:
            tot = strat.get("total_trades", "?")
            _log(
                f"    extract: trades=[] for {class_name} (total_trades={tot}) "
                f"in {zip_path.name} — freqtrade ran but produced no per-trade data"
            )
            return None
    except Exception as e:
        _log(f"    extract: exception on {zip_path.name}: {type(e).__name__}: {e}")
        return None

    df = pd.DataFrame(trades)
    keep = ["profit_ratio", "open_date", "close_date", "is_short", "exit_reason"]
    available = [c for c in keep if c in df.columns]
    if "profit_ratio" not in available:
        # Without profit_ratio there's nothing to bootstrap on.
        _log(
            f"    extract: no profit_ratio col for {class_name}; "
            f"got {list(df.columns)[:8]}"
        )
        return None
    out = pd.DataFrame(df[available]).copy()
    # Normalize dates for time-aware sorting in the bootstrap layer.
    if "open_date" in out.columns:
        out["open_date"] = pd.to_datetime(out["open_date"], utc=True, errors="coerce")
    return out
