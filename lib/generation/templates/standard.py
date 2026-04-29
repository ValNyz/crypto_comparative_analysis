# =============================================================================
# FILE: lib/generation/templates/standard.py
# =============================================================================
"""Standard strategy template for technical signals."""

STRATEGY_TEMPLATE_STANDARD = '''
"""Auto-generated V3: {name}"""
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
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    startup_candle_count = 250

    REGIME_LOOKBACK = {regime_lookback}
    REGIME_ADX_THRESHOLD = {regime_adx_threshold}
    REGIME_ADX_STRONG = {regime_adx_strong}
    REGIME_ATR_VOLATILE = {regime_atr_volatile}
    REGIME_ATR_LOW = {regime_atr_low}
    ALLOWED_REGIMES = {allowed_regimes}
    ENABLE_FILTER = {enable_filter}
    USE_ATR_FILTER = {use_atr_filter}
    ATR_MIN_PERCENTILE = {atr_min}
    ATR_MAX_PERCENTILE = {atr_max}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
{indicators_block}
        if self.USE_ATR_FILTER:
            _atr = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
            dataframe['atr_pct'] = _atr.rolling(72).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
            )
        return self._detect_regime_v3(dataframe)

{regime_detection_block}

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        regime = dataframe['regime']
        if self.ENABLE_FILTER:
            regime_ok = regime.isin(self.ALLOWED_REGIMES)
        else:
            regime_ok = True
        if self.USE_ATR_FILTER:
            _atr_ok = (dataframe['atr_pct'] >= self.ATR_MIN_PERCENTILE) & (dataframe['atr_pct'] <= self.ATR_MAX_PERCENTILE)
            regime_ok = regime_ok & _atr_ok
{entry_logic}
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
{exit_logic}
        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: str, side: str, **kwargs) -> float:
        return 1.0
'''
