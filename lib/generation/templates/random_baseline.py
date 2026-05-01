# =============================================================================
# FILE: lib/generation/templates/random_baseline.py
# =============================================================================
"""Null-pool baseline strategy template.

Emits N seeded random long/short entries over the timerange, using the
SAME exit logic / SL / ROI as the strategy under test. The resulting
freqtrade trade list IS the null pool: bootstrap-sampled in
lib/null_pool/ to compute empirical p-values for observed strategy
returns.

Design notes:
- Per-pair seed: mixing the pair name into NULL_POOL_SEED gives each pair
  its own deterministic random pattern. Without this, BTC/ETH/HYPE would
  all enter at the same row indices → spurious cross-coin correlation.
- Indicators: only INDICATORS_CORE is injected (rsi_14, bb_pos, funding_zscore
  via funding template would be different — random_baseline runs against
  spot exits only, no funding-aware exits). This keeps the pool builder
  fast while still supporting rsi/bb/etc. exit conditions.
- Adaptive entry probability: target ~NULL_POOL_TARGET_TRADES entries
  regardless of timeframe length. p = min(0.5, target / (n - warmup)).
- No regime detection: the pool represents "what would random entries
  return", not "what would random entries return inside an allowed regime".
  Keep it bias-free.
"""

STRATEGY_TEMPLATE_RANDOM_BASELINE = '''
"""Auto-generated null pool: {name} seed={seed}"""
import hashlib
from freqtrade.strategy import IStrategy
import talib
import pandas as pd
from pandas import DataFrame
import numpy as np

class {class_name}(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '{timeframe}'
    can_short = True
    minimal_roi = {roi}
    stoploss = {stoploss}
    trailing_stop = {trailing_stop}
    trailing_stop_positive = {trailing_stop_positive}
    trailing_stop_positive_offset = {trailing_stop_positive_offset}
    trailing_only_offset_is_reached = {trailing_only_offset_is_reached}
    startup_candle_count = 100

    NULL_POOL_SEED = {seed}
    NULL_POOL_TARGET_TRADES = {target_trades}
    NULL_POOL_DIRECTION = '{direction}'  # "long" | "short" | "both"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
{indicators_block}
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Per-pair deterministic RNG so each pair has its own random pattern
        # (same SEED across pairs would correlate entries by row index).
        pair = metadata.get("pair", "")
        seed_int = int(hashlib.md5(f"{{self.NULL_POOL_SEED}}_{{pair}}_{{self.NULL_POOL_DIRECTION}}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)
        n = len(dataframe)
        warmup = self.startup_candle_count
        eligible = max(n - warmup, 1)
        # Adaptive entry probability — converges to target trade count.
        # Capped at 0.5 so we never trigger more than half the bars.
        p = min(0.5, self.NULL_POOL_TARGET_TRADES / eligible)
        trigger = rng.uniform(0.0, 1.0, n) < p
        # Skip warmup so indicators referenced by exit_logic are valid
        if n > warmup:
            trigger[:warmup] = False

        if self.NULL_POOL_DIRECTION == 'long':
            dataframe['enter_long']  = trigger.astype(int)
            dataframe['enter_short'] = 0
        elif self.NULL_POOL_DIRECTION == 'short':
            dataframe['enter_long']  = 0
            dataframe['enter_short'] = trigger.astype(int)
        else:  # both — 50/50 split (kept for backwards compat / spot-check)
            side = rng.uniform(0.0, 1.0, n) < 0.5
            dataframe['enter_long']  = (trigger &  side).astype(int)
            dataframe['enter_short'] = (trigger & ~side).astype(int)

        # Tag for L/S parser bucketing (regex requires adjacent
        # `(long|short)_(bull|bear|range|volatile)`). `_volatile` is a
        # placeholder regime suffix.
        if 'enter_tag' not in dataframe.columns:
            dataframe['enter_tag'] = ''
        dataframe.loc[dataframe['enter_long']  == 1, 'enter_tag'] = 'random_long_volatile'
        dataframe.loc[dataframe['enter_short'] == 1, 'enter_tag'] = 'random_short_volatile'
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
{exit_logic}
        return dataframe

{custom_exit_method}
{partial_exit_method}

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: str, side: str, **kwargs) -> float:
        return 1.0
'''
