# =============================================================================
# FILE: lib/config/base.py
# =============================================================================
"""Base configuration dataclass."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class Config:
    """Configuration principale pour le backtest comparatif."""

    # Paths
    freqtrade_path: str = "."
    user_data: str = "./user_data"
    config_file: str = "./user_data/config.json"
    data_dir: str = "./data/hyperliquid"
    strategies_dir: str = "./user_data/strategies/generated_v3"
    configs_dir: str = "./configs"
    signals: str = "./configs/signals.yaml"
    exits: str = "./configs/exits.yaml"
    signal_conditions: str = "./config/signal_conditions.yaml"
    regime_mappings: str = "./configs/regime_mappings.yaml"

    # Backtest parameters
    timerange: str = "20250101-20250930"
    timeframe_detail: Optional[str] = (
        None  # e.g. "1m" for high-fidelity exit fills; None disables
    )
    min_trades: int = 0
    min_sharpe: float = -2
    min_months: int = 2
    max_workers: int = 6

    # Wallet/Position
    dry_run_wallet: float = 1000
    # Fixed stake per trade (was "unlimited"). 100 USDC matches the
    # null-pool baseline so per-trade profit_ratio is directly comparable
    # across observed and pool runs. With wallet=1000 + max_open_trades=1,
    # this exposes 10% of capital per trade.
    stake_amount: Union[float, str] = 100
    max_open_trades: int = 1

    # Regime detection parameters
    regime_lookback: int = 72
    regime_adx_threshold: int = 20
    regime_adx_strong: int = 25
    regime_atr_volatile: float = 0.75
    regime_atr_low: float = 0.25

    # Feature flags
    enable_regime_filter: bool = False
    debug: bool = False
    # Reuse already-computed backtest results (parsed from
    # user_data/backtest_results/*.zip) instead of running freqtrade for
    # strategies whose class_name + timeframe + timerange match a previous
    # export. ON by default so re-running a partially-completed grid is
    # fast; pass --refresh to force a full re-run from freqtrade.
    use_cache: bool = True

    # === Null-pool comparison (always-on) ===
    # Phase 1 builds a "random entries" trade pool per
    # (pair, tf, exit_config, sl, roi, direction) cell using lib/null_pool.
    # Each tested signal then gets an empirical p-value via stationary block
    # bootstrap against the matching pool. See lib/null_pool/__init__.py.
    # `--refresh` does NOT invalidate pools (they're expensive); use
    # `--refresh-null-pool` explicitly to rebuild.
    enable_null_pool: bool = True
    null_pool_seed: int = 42
    null_pool_target_trades: int = 1000  # entries emitted per pool run
    null_pool_n_bootstrap: int = 1000
    # Equity fraction per trade in bootstrap. 0.10 matches stake=100 +
    # wallet=1000 (10% of capital per trade) so the bootstrap simulates
    # the same compounding as observed strategies.
    null_pool_capital_pct: float = 0.10
    null_pool_block_len: float = 5.0  # mean block length for stationary bootstrap
    refresh_null_pool: bool = False  # force re-build of parquet pool cache

    def __post_init__(self):
        """Ensure directories exist."""
        Path(self.strategies_dir).mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        """Directory for analysis results."""
        path = Path(self.user_data) / "analysis_results"
        path.mkdir(exist_ok=True)
        return path

    @property
    def null_pool_cache_dir(self) -> Path:
        """Parquet cache directory for null pools."""
        path = Path(self.user_data) / "null_pool_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path
