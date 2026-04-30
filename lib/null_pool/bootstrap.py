# =============================================================================
# FILE: lib/null_pool/bootstrap.py
# =============================================================================
"""Empirical p-value via stationary block bootstrap.

Port of framework/null_pool.py:pvalue_vs_null adapted to consume parquet
trade pools (built by pool_builder.py) instead of the framework's
internal evaluator output.

The bootstrap RECONSTRUCTS synthetic K-trade strategies from the trade
pool — we don't run K random freqtrade backtests. With a pool of N>>K
independent random trades, sampling K from the pool 1000 times gives us
1000 hypothetical random K-trade strategies, each with a final equity
return. Compared to the observed strategy's return → empirical p-value.

Stationary block bootstrap (Politis & Romano 1994): samples contiguous
blocks of geometric length (mean = `mean_block_len`) instead of i.i.d.
draws. This preserves local autocorrelation in the trade sequence (vol
clusters, regime persistence) → tighter null distribution → fewer false
positives than i.i.d. bootstrap.
"""

import numpy as np


def _stationary_block_bootstrap_matrix(
    pnls: np.ndarray,
    n_bootstrap: int,
    n_trades: int,
    mean_block_len: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized stationary block bootstrap.

    Returns shape (n_bootstrap, n_trades). Algorithm:
      - At each position i, decide "reboot" with prob p_end=1/mean_block_len.
      - If reboot: position = uniform in [0, n_src).
      - Else: position = previous + 1 (mod n_src).

    The for-loop runs n_trades times (typically 50-300) and replaces a
    Python double-loop O(n_bootstrap × n_trades) with vectorized numpy ops.
    """
    n_src = len(pnls)
    if n_src == 0:
        return np.empty((n_bootstrap, 0), dtype=pnls.dtype)
    p_end = 1.0 / max(mean_block_len, 1.0)

    reboots = rng.random((n_bootstrap, n_trades)) < p_end
    reboots[:, 0] = True
    random_pos = rng.integers(0, n_src, size=(n_bootstrap, n_trades))

    indices = np.empty((n_bootstrap, n_trades), dtype=np.int64)
    current = random_pos[:, 0].copy()
    indices[:, 0] = current
    for i in range(1, n_trades):
        current = np.where(
            reboots[:, i], random_pos[:, i], (current + 1) % n_src
        )
        indices[:, i] = current

    return pnls[indices]


def pvalue_vs_null(
    pool_pnls: np.ndarray,
    observed_return_pct: float,
    n_trades: int,
    capital_pct_per_trade: float = 0.10,
    n_bootstrap: int = 1000,
    seed: int = 42,
    fee_pct: float = 0.0,
    mean_block_len: float = 5.0,
) -> float:
    """One-sided empirical p-value: P(null_return >= observed_return).

    Args:
      pool_pnls: per-trade returns as fractions (freqtrade's profit_ratio).
      observed_return_pct: observed strategy total return in PERCENT (e.g. 12.3).
      n_trades: trade count of the observed strategy.
      capital_pct_per_trade: equity fraction per trade (matches the framework
        default 0.10; tune to your real allocation if comparing apples-to-apples
        is critical).
      n_bootstrap: number of bootstrap iterations (default 1000).
      seed: RNG seed for reproducibility.
      fee_pct: per-trade fee fraction (already baked into profit_ratio when
        freqtrade is configured with `fee` — leave at 0 to avoid double-counting).
      mean_block_len: average block length for stationary block bootstrap.
        5 is a good default for hourly/daily trade sequences with mild local
        autocorrelation; raise to 10-20 for highly clustered series.

    Returns:
      p-value in (0, 1]. Returns 1.0 (no significance) if the pool is too
      small to bootstrap n_trades samples reliably.
    """
    pool = np.asarray(pool_pnls, dtype=float)
    pool = pool[~np.isnan(pool)]
    if len(pool) < max(10, n_trades) or n_trades <= 0:
        return 1.0

    rng = np.random.default_rng(seed)
    samples = _stationary_block_bootstrap_matrix(
        pool, n_bootstrap, n_trades, mean_block_len, rng
    )

    # Compound per-trade returns at fixed capital fraction → final return per
    # bootstrap iteration. Each row of `samples` represents one synthetic
    # K-trade random strategy.
    eq = np.cumprod(1.0 + (samples - fee_pct) * capital_pct_per_trade, axis=1)
    null_rets = (eq[:, -1] - 1.0) * 100.0

    # +1 / +1 smoothing (avoids p=0 when no bootstrap exceeds observed)
    return float((np.sum(null_rets >= observed_return_pct) + 1) / (n_bootstrap + 1))


def pvalue_vs_null_mixed(
    long_pool_pnls: np.ndarray,
    short_pool_pnls: np.ndarray,
    observed_return_pct: float,
    n_long: int,
    n_short: int,
    capital_pct_per_trade: float = 1.0,
    n_bootstrap: int = 1000,
    seed: int = 42,
    fee_pct: float = 0.0,
    mean_block_len: float = 5.0,
) -> float:
    """One-sided p-value for a direction='both' signal.

    Builds each synthetic strategy by sampling `n_long` trades from the long
    pool AND `n_short` trades from the short pool — mirroring the observed
    signal's L/S split. This avoids comparing a 60L/20S signal to a null
    where 50% of the trades are short (which would be a different distribution
    than the signal actually faces).

    Bootstrap samples are concatenated then cumprod'd, so the order
    long-then-short is fixed. With `mean_block_len > 1` and the stationary
    block sampler, internal autocorrelation within each pool is preserved;
    the L→S transition is artificial but happens once per iteration → minimal
    impact on the bootstrap variance.
    """
    long_pool = np.asarray(long_pool_pnls, dtype=float)
    short_pool = np.asarray(short_pool_pnls, dtype=float)
    long_pool = long_pool[~np.isnan(long_pool)]
    short_pool = short_pool[~np.isnan(short_pool)]
    n_total = n_long + n_short
    if n_total <= 0:
        return 1.0
    if n_long > 0 and len(long_pool) < max(10, n_long):
        return 1.0
    if n_short > 0 and len(short_pool) < max(10, n_short):
        return 1.0

    rng = np.random.default_rng(seed)
    parts = []
    if n_long > 0:
        parts.append(_stationary_block_bootstrap_matrix(
            long_pool, n_bootstrap, n_long, mean_block_len, rng,
        ))
    if n_short > 0:
        parts.append(_stationary_block_bootstrap_matrix(
            short_pool, n_bootstrap, n_short, mean_block_len, rng,
        ))
    samples = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    eq = np.cumprod(1.0 + (samples - fee_pct) * capital_pct_per_trade, axis=1)
    null_rets = (eq[:, -1] - 1.0) * 100.0

    return float((np.sum(null_rets >= observed_return_pct) + 1) / (n_bootstrap + 1))


def bh_adjusted_pvalues(pvalues) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values.

    Use this when testing many signals at once: raw p-values multiply false
    discoveries linearly in N. BH controls expected FDR at the chosen alpha
    instead of family-wise error (more powerful than Bonferroni for
    multi-signal screens).
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted_sorted = ranked * n / np.arange(1, n + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    out = np.empty(n)
    out[order] = adjusted_sorted
    return out
