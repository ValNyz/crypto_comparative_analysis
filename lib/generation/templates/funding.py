# =============================================================================
# FILE: lib/generation/templates/funding.py
# =============================================================================
"""Funding contrarian strategy template."""

STRATEGY_TEMPLATE_FUNDING = '''
"""Auto-generated FundingContrarian V3: {name}"""
import logging
import pandas as pd
from pandas import DataFrame
import numpy as np
from pathlib import Path
from freqtrade.strategy import IStrategy
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib

logger = logging.getLogger(__name__)

class {class_name}(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '{timeframe}'
    can_short = True
    minimal_roi = {roi}
    stoploss = {stoploss}
    trailing_stop = False
    startup_candle_count = 250

    ZSCORE_THRESHOLD = {zscore_threshold}
    FUNDING_LOOKBACK_HOURS = {funding_lookback}
    # Multi-lookback funding z-score. Empty list = single-lookback (current behavior).
    EXTRA_LOOKBACKS = {extra_lookbacks_list}
    LOOKBACK_COMBINE = '{lookback_combine}'
    FUNDING_DELAY_MINUTES = 5
    USE_RSI_FILTER = {use_rsi_filter}
    RSI_MIN = {rsi_min}
    RSI_MAX = {rsi_max}
    USE_VOLUME_FILTER = {use_volume_filter}
    VOLUME_MIN_RATIO = {volume_min}
    VOLUME_MAX_RATIO = {volume_max}
    USE_ATR_FILTER = {use_atr_filter}
    ATR_MIN_PERCENTILE = {atr_min}
    ATR_MAX_PERCENTILE = {atr_max}
    USE_TREND_FILTER = {use_antitrend_filter}
    TREND_ADX_MAX = {adx_max}
    USE_EMA_CONTRATREND = {use_ema_contra_filter}
    USE_BB_FILTER = {use_bb_filter}
    BB_LONG_MAX = {bb_long_max}
    BB_SHORT_MIN = {bb_short_min}
    USE_STOCH_FILTER = {use_stoch_filter}
    STOCH_LONG_MAX = {stoch_long_max}
    STOCH_SHORT_MIN = {stoch_short_min}
    USE_STOCH_CROSS = {use_stoch_cross_filter}
    USE_MACD_FILTER = {use_macd_filter}
    USE_CANDLE_FILTER = {use_candle_filter}
    USE_ENGULFING_FILTER = {use_engulfing_filter}

    REGIME_LOOKBACK = {regime_lookback}
    REGIME_ADX_THRESHOLD = {regime_adx_threshold}
    REGIME_ADX_STRONG = {regime_adx_strong}
    REGIME_ATR_VOLATILE = {regime_atr_volatile}
    REGIME_ATR_LOW = {regime_atr_low}
    ALLOWED_REGIMES = {allowed_regimes}
    ENABLE_FILTER = {enable_filter}

    DATA_DIR = Path("{data_dir}/futures")

    def get_coin(self, pair: str) -> str:
        return pair.split("/")[0]

    def _detect_funding_timeframe_hours(self, df: pd.DataFrame) -> int:
        if len(df) < 2:
            return 1
        time_diffs = df.sort_values("date")["date"].diff().dropna()
        if len(time_diffs) == 0:
            return 1
        median_diff = time_diffs.median().total_seconds() / 3600
        if median_diff <= 2: return 1
        elif median_diff <= 5: return 4
        elif median_diff <= 12: return 8
        return int(round(median_diff))

    def load_funding_as_ohlcv(self, coin: str, reference_df: pd.DataFrame) -> pd.DataFrame:
        for pattern in [f"{{coin}}_USDC_USDC-1h-funding_rate.feather",
                        f"{{coin}}_USDC_USDC-8h-funding_rate.feather",
                        f"{{coin}}_USDC-1h-funding_rate.feather"]:
            path = self.DATA_DIR / pattern
            if path.exists():
                try:
                    df = pd.read_feather(path)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    funding_tf = self._detect_funding_timeframe_hours(df)
                    ref_tz = reference_df["date"].dt.tz
                    if ref_tz is not None:
                        if df["date"].dt.tz is None:
                            df["date"] = df["date"].dt.tz_localize("UTC").dt.tz_convert(ref_tz)
                        else:
                            df["date"] = df["date"].dt.tz_convert(ref_tz)
                    else:
                        if df["date"].dt.tz is not None:
                            df["date"] = df["date"].dt.tz_localize(None)
                    result = pd.DataFrame()
                    result["date"] = df["date"]
                    result["funding_rate"] = df["open"].astype(float)
                    lookback = max(1, self.FUNDING_LOOKBACK_HOURS // funding_tf)
                    min_periods = max(20, lookback // 4)
                    result["funding_mean"] = result["funding_rate"].rolling(lookback, min_periods=min_periods).mean()
                    result["funding_std"] = result["funding_rate"].rolling(lookback, min_periods=min_periods).std()
                    result["funding_zscore"] = (result["funding_rate"] - result["funding_mean"]) / (result["funding_std"] + 1e-10)
                    # Multi-lookback extras
                    for _lb_hours in self.EXTRA_LOOKBACKS:
                        _lb_bars = max(1, _lb_hours // funding_tf)
                        _min = max(20, _lb_bars // 4)
                        _m = result["funding_rate"].rolling(_lb_bars, min_periods=_min).mean()
                        _s = result["funding_rate"].rolling(_lb_bars, min_periods=_min).std()
                        result[f"funding_zscore_lb{{_lb_hours}}"] = (result["funding_rate"] - _m) / (_s + 1e-10)
                    return result.sort_values("date").drop_duplicates(subset=["date"])
                except Exception as e:
                    logger.error(f"Error loading funding: {{e}}")
        return pd.DataFrame()

    def _merge_funding_data(self, dataframe: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
        extra_cols = [f"funding_zscore_lb{{lb}}" for lb in self.EXTRA_LOOKBACKS]
        if funding_df.empty:
            dataframe["funding_rate"] = 0.0
            dataframe["funding_zscore"] = 0.0
            for c in extra_cols:
                dataframe[c] = 0.0
            return dataframe
        extra_cols_present = [c for c in extra_cols if c in funding_df.columns]
        keep = ["date", "funding_rate", "funding_zscore"] + extra_cols_present
        funding = funding_df[keep].copy()
        funding["date_available"] = funding["date"] + pd.Timedelta(minutes=self.FUNDING_DELAY_MINUTES)
        dataframe = dataframe.sort_values("date").reset_index(drop=True)
        merge_cols = ["date_available", "funding_rate", "funding_zscore"] + extra_cols_present
        merged = pd.merge_asof(dataframe, funding[merge_cols],
                               left_on="date", right_on="date_available", direction="backward")
        merged["funding_rate"] = merged["funding_rate"].fillna(0.0)
        merged["funding_zscore"] = merged["funding_zscore"].fillna(0.0)
        for c in extra_cols_present:
            merged[c] = merged[c].fillna(0.0)
        return merged.drop(columns=["date_available"], errors="ignore")

{regime_detection_block}

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        coin = self.get_coin(metadata["pair"])
        funding_df = self.load_funding_as_ohlcv(coin, dataframe)
        dataframe = self._merge_funding_data(dataframe, funding_df)
        dataframe["rsi_14"] = qtpylib.rsi(dataframe["close"], window=14).fillna(50)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = bb_upper, bb_middle, bb_lower
        dataframe['bb_pos'] = (dataframe['close'] - bb_middle) / ((bb_upper - bb_lower) / 2 + 1e-10)
        dataframe['ema_8'] = talib.EMA(dataframe['close'], timeperiod=8)
        dataframe['ema_21'] = talib.EMA(dataframe['close'], timeperiod=21)
        dataframe['ema_50'] = talib.EMA(dataframe['close'], timeperiod=50)
        dataframe['adx'] = talib.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['di_plus'] = talib.PLUS_DI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['di_minus'] = talib.MINUS_DI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(dataframe['close'])
        dataframe['macd'], dataframe['macd_signal'], dataframe['macd_hist'] = macd, macd_signal, macd_hist
        dataframe['stoch_k'], dataframe['stoch_d'] = talib.STOCH(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'].rolling(72).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
        )
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / (dataframe['volume_sma'] + 1e-10)
        return self._detect_regime_v3(dataframe)

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        threshold = self.ZSCORE_THRESHOLD
        regime = dataframe['regime']
        # Multi-lookback reduction: compute direction-specific z-scores.
        # When EXTRA_LOOKBACKS is empty, zscore_long == zscore_short == primary (single-lookback behavior).
        _z_cols = ["funding_zscore"] + [f"funding_zscore_lb{{lb}}" for lb in self.EXTRA_LOOKBACKS if f"funding_zscore_lb{{lb}}" in dataframe.columns]
        if len(_z_cols) == 1:
            zscore_long = dataframe[_z_cols[0]]
            zscore_short = dataframe[_z_cols[0]]
        elif self.LOOKBACK_COMBINE == "all":
            # AND: long requires every z_i <= -threshold → max(z_i) <= -threshold
            #      short requires every z_i >= +threshold → min(z_i) >= +threshold
            zscore_long = dataframe[_z_cols].max(axis=1)
            zscore_short = dataframe[_z_cols].min(axis=1)
        elif self.LOOKBACK_COMBINE == "mean":
            _m = dataframe[_z_cols].mean(axis=1)
            zscore_long = _m
            zscore_short = _m
        elif self.LOOKBACK_COMBINE == "max_abs":
            _vals = dataframe[_z_cols].values
            _argmax = np.abs(_vals).argmax(axis=1)
            _signed = _vals[np.arange(_vals.shape[0]), _argmax]
            zscore_long = pd.Series(_signed, index=dataframe.index)
            zscore_short = zscore_long
        else:
            zscore_long = dataframe[_z_cols[0]]
            zscore_short = dataframe[_z_cols[0]]
        # Backward-compat alias for downstream code that expects a single `zscore`
        zscore = zscore_long
        rsi_ok = ((dataframe["rsi_14"] >= self.RSI_MIN) & (dataframe["rsi_14"] <= self.RSI_MAX)) if self.USE_RSI_FILTER else True
        volumef_ok = ((dataframe['volume_ratio'] >= self.VOLUME_MIN_RATIO) & (dataframe['volume_ratio'] <= self.VOLUME_MAX_RATIO)) if self.USE_VOLUME_FILTER else True
        atr_ok = ((dataframe['atr_pct'] >= self.ATR_MIN_PERCENTILE) & (dataframe['atr_pct'] <= self.ATR_MAX_PERCENTILE)) if self.USE_ATR_FILTER else True
        if self.USE_TREND_FILTER:
            trend_ok_long = ~((dataframe['adx'] > self.TREND_ADX_MAX) &
                              (dataframe['di_minus'] > dataframe['di_plus']))
            trend_ok_short = ~((dataframe['adx'] > self.TREND_ADX_MAX) &
                               (dataframe['di_plus'] > dataframe['di_minus']))
        else:
            trend_ok_long = True
            trend_ok_short = True
        if self.USE_EMA_CONTRATREND:
            ema_ok_long = dataframe['close'] < dataframe['ema_21']
            ema_ok_short = dataframe['close'] > dataframe['ema_21']
        else:
            ema_ok_long = True
            ema_ok_short = True
        if self.USE_BB_FILTER:
            bb_ok_long = dataframe['bb_pos'] <= self.BB_LONG_MAX
            bb_ok_short = dataframe['bb_pos'] >= self.BB_SHORT_MIN
        else:
            bb_ok_long = True
            bb_ok_short = True
        if self.USE_STOCH_FILTER:
            stoch_ok_long = dataframe['stoch_k'] <= self.STOCH_LONG_MAX
            stoch_ok_short = dataframe['stoch_k'] >= self.STOCH_SHORT_MIN
        else:
            stoch_ok_long = True
            stoch_ok_short = True
        if self.USE_STOCH_CROSS:
            stoch_cross_long = (dataframe['stoch_k'] > dataframe['stoch_d']) & \\
                               (dataframe['stoch_k'].shift(1) <= dataframe['stoch_d'].shift(1))
            stoch_cross_short = (dataframe['stoch_k'] < dataframe['stoch_d']) & \\
                                (dataframe['stoch_k'].shift(1) >= dataframe['stoch_d'].shift(1))
        else:
            stoch_cross_long = True
            stoch_cross_short = True
        if self.USE_MACD_FILTER:
            macd_ok_long = dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)
            macd_ok_short = dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)
        else:
            macd_ok_long = True
            macd_ok_short = True
        if self.USE_CANDLE_FILTER:
            candle_ok_long = dataframe['close'] > dataframe['open']
            candle_ok_short = dataframe['close'] < dataframe['open']
        else:
            candle_ok_long = True
            candle_ok_short = True
        if self.USE_ENGULFING_FILTER:
            engulf_ok_long = (dataframe['close'] > dataframe['open']) & \\
                             (dataframe['close'].shift(1) < dataframe['open'].shift(1)) & \\
                             (dataframe['close'] > dataframe['open'].shift(1)) & \\
                             (dataframe['open'] < dataframe['close'].shift(1))
            engulf_ok_short = (dataframe['close'] < dataframe['open']) & \\
                             (dataframe['close'].shift(1) > dataframe['open'].shift(1)) & \\
                             (dataframe['close'] < dataframe['open'].shift(1)) & \\
                             (dataframe['open'] > dataframe['close'].shift(1))
        else:
            engulf_ok_long = True
            engulf_ok_short = True
        volume_ok = dataframe["volume"] > 0
        regime_ok = regime.isin(self.ALLOWED_REGIMES) if self.ENABLE_FILTER else True

        for direction, cond, col in {direction_loop}:
            full_cond = cond & rsi_ok & volumef_ok & atr_ok & volume_ok & regime_ok
            if direction == "long":
                full_cond = full_cond & trend_ok_long & ema_ok_long & bb_ok_long & stoch_ok_long & stoch_cross_long & macd_ok_long & candle_ok_long & engulf_ok_long
            elif direction == "short":
                full_cond = full_cond & trend_ok_short & ema_ok_short & bb_ok_short & stoch_ok_short & stoch_cross_short & macd_ok_short & candle_ok_short & engulf_ok_short

            for reg in ['bull', 'bear', 'range', 'volatile']:
                mask = full_cond & (regime == reg)
                dataframe.loc[mask, col] = 1
                # Capture |z| and atr% at entry time for dynamic-ROI policies.
                # Use median across entries in this bar to get one scalar per tag.
                _zabs = float(abs(zscore.loc[mask].median())) if mask.any() else 0.0
                _atrp = float(
                    (dataframe.loc[mask, "atr"] / dataframe.loc[mask, "close"]).median()
                ) if mask.any() else 0.0
                # Tag layout: direction and regime must be CONTIGUOUS at the end
                # because the framework's regex parser (lib/backtest/parser.py:180)
                # matches `(long|short)_(bull|bear|range|volatile)` as adjacent
                # tokens. z/atr suffixes therefore go BEFORE the direction_regime
                # pair. Token-scanning parsers for z/atr remain order-agnostic.
                dataframe.loc[mask, "enter_tag"] = (
                    f"funding_z{{_zabs:.2f}}_atr{{_atrp:.4f}}_{{direction}}_{{reg}}"
                )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
{exit_logic}
        return dataframe

{custom_exit_method}

{partial_exit_method}

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return 1.0
'''
