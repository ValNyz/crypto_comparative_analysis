# =============================================================================
# FILE: lib/generation/templates/standard.py
# =============================================================================
"""Standard strategy template for technical signals."""

STRATEGY_TEMPLATE_STANDARD = '''
"""Auto-generated V3: {name}"""
from pathlib import Path
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

    DATA_DIR = Path("{data_dir}/futures")

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

    # === P1bis macro filters (default disabled) ===
    USE_FNG_FILTER = {use_fng_filter}
    FNG_FEAR = {fng_fear}
    FNG_GREED = {fng_greed}
    FNG_CONSEC_DAYS = {fng_consec_days}
    USE_VIX_FILTER = {use_vix_filter}
    VIX_MAX_LONG = {vix_max_long}
    VIX_MIN_SHORT = {vix_min_short}
    USE_DXY_FILTER = {use_dxy_filter}
    DXY_SLOPE_MAX_LONG = {dxy_slope_max_long}
    DXY_SLOPE_MIN_SHORT = {dxy_slope_min_short}
    USE_ETF_FLOW_FILTER = {use_etf_flow_filter}
    ETF_REF = '{etf_ref}'
    ETF_INFLOW_MIN_LONG = {etf_inflow_min_long}
    ETF_OUTFLOW_MAX_SHORT = {etf_outflow_max_short}
    USE_FUNDING_SPREAD_FILTER = {use_funding_spread_filter}
    SPREAD_REF = '{spread_ref}'
    SPREAD_Z_THRESHOLD = {spread_z_threshold}
    USE_BTC_REGIME_FILTER = {use_btc_regime_filter}
    BTC_REGIME_ALLOWED = {btc_regime_allowed}
    USE_VOLUME_ZSCORE_FILTER = {use_volume_zscore_filter}
    VOLUME_ZSCORE_MIN = {volume_zscore_min}
    USE_BBW_SQUEEZE_FILTER = {use_bbw_squeeze_filter}
    BBW_PCT_MAX = {bbw_pct_max}

{external_loaders_block}
{cross_coin_block}

    def get_coin(self, pair: str) -> str:
        return pair.split("/")[0]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        coin = self.get_coin(metadata["pair"])
{indicators_block}
        # Cross-coin merges (no-op when block is empty)
        if hasattr(self, "merge_btc_regime"):
            dataframe = self.merge_btc_regime(dataframe)
        if hasattr(self, "merge_ratio_coin"):
            dataframe = self.merge_ratio_coin(dataframe, "BTC", own_coin=coin)
            dataframe = self.merge_ratio_coin(dataframe, "ETH", own_coin=coin)
        # === P1bis: external macro merges (each is no-op when its filter is disabled) ===
        if self.USE_FNG_FILTER and hasattr(self, "merge_external_fng"):
            dataframe = self.merge_external_fng(dataframe)
        if self.USE_VIX_FILTER and hasattr(self, "merge_external_vix"):
            dataframe = self.merge_external_vix(dataframe)
        if self.USE_DXY_FILTER and hasattr(self, "merge_external_dxy"):
            dataframe = self.merge_external_dxy(dataframe)
        if self.USE_ETF_FLOW_FILTER and hasattr(self, "merge_external_etf_flow"):
            dataframe = self.merge_external_etf_flow(dataframe, self.ETF_REF)
        if self.USE_FUNDING_SPREAD_FILTER and hasattr(self, "merge_external_funding_spread"):
            dataframe = self.merge_external_funding_spread(dataframe, self.SPREAD_REF)
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

        # === P1bis: macro filter post-step (zeros enter_long/short for rows that don't pass) ===
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0
        if 'enter_short' not in dataframe.columns:
            dataframe['enter_short'] = 0

        if self.USE_FNG_FILTER:
            _fng_lo = (dataframe['fng_value'] <= self.FNG_FEAR).rolling(self.FNG_CONSEC_DAYS, min_periods=self.FNG_CONSEC_DAYS).min().fillna(0).astype(bool)
            _fng_hi = (dataframe['fng_value'] >= self.FNG_GREED).rolling(self.FNG_CONSEC_DAYS, min_periods=self.FNG_CONSEC_DAYS).min().fillna(0).astype(bool)
            fng_ok_long, fng_ok_short = _fng_lo, _fng_hi
        else:
            fng_ok_long = fng_ok_short = pd.Series(True, index=dataframe.index)
        if self.USE_VIX_FILTER:
            vix_ok_long = dataframe['vix_close'] < self.VIX_MAX_LONG
            vix_ok_short = dataframe['vix_close'] > self.VIX_MIN_SHORT
        else:
            vix_ok_long = vix_ok_short = pd.Series(True, index=dataframe.index)
        if self.USE_DXY_FILTER:
            dxy_ok_long = dataframe['dxy_slope10'] < self.DXY_SLOPE_MAX_LONG
            dxy_ok_short = dataframe['dxy_slope10'] > self.DXY_SLOPE_MIN_SHORT
        else:
            dxy_ok_long = dxy_ok_short = pd.Series(True, index=dataframe.index)
        if self.USE_ETF_FLOW_FILTER:
            etf_ok_long = dataframe['etf_flow_usd_m'] >= self.ETF_INFLOW_MIN_LONG
            etf_ok_short = dataframe['etf_flow_usd_m'] <= -self.ETF_OUTFLOW_MAX_SHORT
        else:
            etf_ok_long = etf_ok_short = pd.Series(True, index=dataframe.index)
        if self.USE_FUNDING_SPREAD_FILTER:
            spread_ok_long = dataframe['funding_spread_zscore'] < -self.SPREAD_Z_THRESHOLD
            spread_ok_short = dataframe['funding_spread_zscore'] > self.SPREAD_Z_THRESHOLD
        else:
            spread_ok_long = spread_ok_short = pd.Series(True, index=dataframe.index)
        if self.USE_BTC_REGIME_FILTER:
            btc_regime_ok = dataframe['btc_regime'].isin(self.BTC_REGIME_ALLOWED)
        else:
            btc_regime_ok = pd.Series(True, index=dataframe.index)
        if self.USE_VOLUME_ZSCORE_FILTER:
            vz_ok = dataframe['volume_zscore'] > self.VOLUME_ZSCORE_MIN
        else:
            vz_ok = pd.Series(True, index=dataframe.index)
        if self.USE_BBW_SQUEEZE_FILTER:
            bbw_ok = dataframe['bbw_pct'] < self.BBW_PCT_MAX
        else:
            bbw_ok = pd.Series(True, index=dataframe.index)

        long_pass  = fng_ok_long  & vix_ok_long  & dxy_ok_long  & etf_ok_long  & spread_ok_long  & btc_regime_ok & vz_ok & bbw_ok
        short_pass = fng_ok_short & vix_ok_short & dxy_ok_short & etf_ok_short & spread_ok_short & btc_regime_ok & vz_ok & bbw_ok
        dataframe.loc[~long_pass,  'enter_long']  = 0
        dataframe.loc[~short_pass, 'enter_short'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
{exit_logic}
        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: str, side: str, **kwargs) -> float:
        return 1.0
'''
