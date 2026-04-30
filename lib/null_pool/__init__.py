# =============================================================================
# FILE: lib/null_pool/__init__.py
# =============================================================================
"""Null-pool comparison: random-entry baselines for empirical p-values.

Pipeline:
  1. Build pool   — Phase 1 of BacktestRunner runs random_baseline strategies
                    (one freqtrade run per (pair, TF, exit, sl, roi, timerange)
                    cell) and persists the trade list as parquet. Cache key
                    includes seed for reproducibility.
  2. Bootstrap    — For each tested signal producing K trades with PnL X%,
                    sample K trades from the matching pool 1000× via
                    stationary block bootstrap (preserves local autocorr).
                    p-value = fraction of bootstrap iterations >= X%.
  3. FDR adjust   — Benjamini-Hochberg correction across all p-values to
                    control false discovery rate when many signals are tested.
"""

from .pool_builder import (
    compute_cache_key,
    load_pool,
    save_pool,
    extract_trades_from_zip,
)
from .bootstrap import (
    pvalue_vs_null,
    pvalue_vs_null_mixed,
    bh_adjusted_pvalues,
    storey_q_values,
    estimate_pi0,
)

__all__ = [
    "compute_cache_key",
    "load_pool",
    "save_pool",
    "extract_trades_from_zip",
    "pvalue_vs_null",
    "pvalue_vs_null_mixed",
    "bh_adjusted_pvalues",
    "storey_q_values",
    "estimate_pi0",
]
