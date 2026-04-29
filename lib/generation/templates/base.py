# =============================================================================
# FILE: lib/generation/templates/base.py
# =============================================================================
"""Common template blocks shared across strategies."""

INDICATORS_BLOCK = """
        dataframe['rsi_14'] = talib.RSI(dataframe['close'], timeperiod=14)
        dataframe['rsi_21'] = talib.RSI(dataframe['close'], timeperiod=21)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = bb_upper, bb_middle, bb_lower
        dataframe['bb_pos'] = (dataframe['close'] - bb_middle) / ((bb_upper - bb_lower) / 2 + 1e-10)
        dataframe['ema_8'] = talib.EMA(dataframe['close'], timeperiod=8)
        dataframe['ema_21'] = talib.EMA(dataframe['close'], timeperiod=21)
        dataframe['ema_50'] = talib.EMA(dataframe['close'], timeperiod=50)
        dataframe['adx'] = talib.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['di_plus'] = talib.PLUS_DI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['di_minus'] = talib.MINUS_DI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        stoch_k, stoch_d = talib.STOCH(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['stoch_k'], dataframe['stoch_d'] = stoch_k, stoch_d
        macd, macd_signal, macd_hist = talib.MACD(dataframe['close'])
        dataframe['macd'], dataframe['macd_signal'], dataframe['macd_hist'] = macd, macd_signal, macd_hist
        dataframe['is_green'] = dataframe['close'] > dataframe['open']
        dataframe['is_red'] = dataframe['close'] < dataframe['open']
        direction = np.sign(dataframe['close'] - dataframe['open'])
        dataframe['consec_red'] = (direction == -1).astype(int).groupby((direction != -1).cumsum()).cumsum()
        dataframe['consec_green'] = (direction == 1).astype(int).groupby((direction != 1).cumsum()).cumsum()
        mean, std = dataframe['close'].rolling(50).mean(), dataframe['close'].rolling(50).std()
        dataframe['zscore'] = (dataframe['close'] - mean) / (std + 1e-10)
        dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['willr'] = talib.WILLR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['cci'] = talib.CCI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=20)
        kc_middle = talib.EMA(dataframe['close'], timeperiod=20)
        kc_range = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=20)
        dataframe['kc_upper'] = kc_middle + 1.5 * kc_range
        dataframe['kc_lower'] = kc_middle - 1.5 * kc_range
        dataframe['donchian_high'] = dataframe['high'].rolling(20).max()
        dataframe['donchian_low'] = dataframe['low'].rolling(20).min()
        dataframe['obv'] = talib.OBV(dataframe['close'], dataframe['volume'])
        dataframe['vwap'] = (dataframe['volume'] * (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3).cumsum() / dataframe['volume'].cumsum()
        dataframe['vwap_std'] = dataframe['close'].rolling(20).std()
        dataframe['roc'] = talib.ROC(dataframe['close'], timeperiod=12)
        dataframe['mfi'] = talib.MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'], timeperiod=14)
"""

REGIME_DETECTION_BLOCK_V4EMA = '''
    def _detect_regime_v3(self, dataframe: DataFrame) -> DataFrame:
        """EMA-alignment regime classifier (v4ema).

        bull  = close > ema_50 > ema_200 AND atr_percentile < 0.7
        bear  = close < ema_50 < ema_200 AND atr_percentile < 0.7
        volatile = atr_percentile >= 0.7
        range = else
        """
        lb = self.REGIME_LOOKBACK

        if 'ema_200' not in dataframe.columns:
            dataframe['ema_200'] = talib.EMA(dataframe['close'], timeperiod=200)

        atr_norm = dataframe['atr'] / dataframe['close']
        dataframe['atr_percentile'] = atr_norm.rolling(lb).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        bull_aligned = (dataframe['close'] > dataframe['ema_50']) & (dataframe['ema_50'] > dataframe['ema_200'])
        bear_aligned = (dataframe['close'] < dataframe['ema_50']) & (dataframe['ema_50'] < dataframe['ema_200'])

        conditions = [
            bull_aligned & (dataframe['atr_percentile'] < 0.7),
            bear_aligned & (dataframe['atr_percentile'] < 0.7),
            dataframe['atr_percentile'] >= 0.7,
        ]
        dataframe['regime'] = np.select(
            conditions, ['bull', 'bear', 'volatile'], default='range'
        )
        dataframe['regime_confidence'] = 0.5
        return dataframe
'''

REGIME_DETECTION_BLOCK = '''
    def _detect_regime_v3(self, dataframe: DataFrame) -> DataFrame:
        """
        Régime multi-facteur amélioré:
        - Trend: ADX + DI + EMA alignment
        - Volatility: ATR percentile + BB width
        - Momentum: RSI regime + MACD histogram
        """
        lb = self.REGIME_LOOKBACK

        # === VOLATILITY SCORE ===
        dataframe['atr_pct'] = dataframe['atr'].rolling(lb).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False)

        bb_width = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        dataframe['bb_width_pct'] = bb_width.rolling(lb).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False)

        dataframe['vol_score'] = (dataframe['atr_pct'] + dataframe['bb_width_pct']) / 2

        # === TREND SCORE ===
        adx_score = np.clip(dataframe['adx'] / 40, 0, 1)

        ema_bull = ((dataframe['ema_8'] > dataframe['ema_21']) &
                    (dataframe['ema_21'] > dataframe['ema_50'])).astype(float)
        ema_bear = ((dataframe['ema_8'] < dataframe['ema_21']) &
                    (dataframe['ema_21'] < dataframe['ema_50'])).astype(float)

        dataframe['trend_strength'] = adx_score * (ema_bull - ema_bear + 1) / 2

        # === MOMENTUM REGIME ===
        rsi_bull = (dataframe['rsi_14'] > 50).astype(float)
        macd_bull = (dataframe['macd_hist'] > 0).astype(float)
        macd_rising = (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)).astype(float)

        dataframe['momentum_score'] = (rsi_bull + macd_bull + macd_rising) / 3

        # === CLASSIFICATION FINALE ===
        is_high_vol = dataframe['vol_score'] > 0.6
        is_low_vol = dataframe['vol_score'] < 0.3
        is_trending = dataframe['adx'] > self.REGIME_ADX_THRESHOLD

        is_strong_bull = (
            is_trending &
            (dataframe['di_plus'] > dataframe['di_minus']) &
            (dataframe['momentum_score'] > 0.6)
        )
        is_strong_bear = (
            is_trending &
            (dataframe['di_minus'] > dataframe['di_plus']) &
            (dataframe['momentum_score'] < 0.4)
        )
        is_volatile = is_high_vol & ~is_strong_bull & ~is_strong_bear
        is_quiet_range = is_low_vol & ~is_trending

        conditions = [is_strong_bull, is_strong_bear, is_volatile, is_quiet_range]
        choices = ['bull', 'bear', 'volatile', 'quiet']
        dataframe['regime'] = np.select(conditions, choices, default='range')

        dataframe['regime_confidence'] = np.where(
            is_strong_bull | is_strong_bear,
            dataframe['adx'] / 40,
            np.where(is_volatile, dataframe['vol_score'], 0.5)
        )

        return dataframe
'''


REGIME_DETECTION_BLOCK_V4EMA_SLOPE = '''
    def _detect_regime_v3(self, dataframe: DataFrame) -> DataFrame:
        """EMA-alignment + EMA50-slope sub-classifier.

        Base: v4ema (bull/bear/range/volatile).
        Sub-state from slope = (ema_50[i] - ema_50[i-10]) / ema_50[i]
          accel: slope > +0.003
          decay: slope < -0.003
          cons:  otherwise
        Volatile regime is not sub-classified; regime_full stays 'volatile'.
        """
        lb = self.REGIME_LOOKBACK
        if 'ema_200' not in dataframe.columns:
            dataframe['ema_200'] = talib.EMA(dataframe['close'], timeperiod=200)
        atr_norm = dataframe['atr'] / dataframe['close']
        dataframe['atr_percentile'] = atr_norm.rolling(lb).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        bull_aligned = (dataframe['close'] > dataframe['ema_50']) & (dataframe['ema_50'] > dataframe['ema_200'])
        bear_aligned = (dataframe['close'] < dataframe['ema_50']) & (dataframe['ema_50'] < dataframe['ema_200'])
        conditions = [
            bull_aligned & (dataframe['atr_percentile'] < 0.7),
            bear_aligned & (dataframe['atr_percentile'] < 0.7),
            dataframe['atr_percentile'] >= 0.7,
        ]
        dataframe['regime'] = np.select(conditions, ['bull', 'bear', 'volatile'], default='range')
        slope = (dataframe['ema_50'] - dataframe['ema_50'].shift(10)) / dataframe['ema_50']
        sub_conditions = [slope > 0.003, slope < -0.003]
        dataframe['regime_sub'] = np.select(sub_conditions, ['accel', 'decay'], default='cons')
        dataframe['regime_full'] = np.where(
            dataframe['regime'] == 'volatile',
            'volatile',
            dataframe['regime'].astype(str) + '_' + dataframe['regime_sub'].astype(str)
        )
        dataframe['regime'] = dataframe['regime_full']
        dataframe['regime_confidence'] = 0.5
        return dataframe
'''


REGIME_DETECTION_BLOCK_V4EMA_ADX = '''
    def _detect_regime_v3(self, dataframe: DataFrame) -> DataFrame:
        """EMA-alignment + ADX-level sub-classifier.

        Base: v4ema (bull/bear/range/volatile).
        Sub-state from adx:
          strong: adx >= 25
          weak:   adx < 20
          mid:    otherwise
        Volatile regime is not sub-classified; regime_full stays 'volatile'.
        """
        lb = self.REGIME_LOOKBACK
        if 'ema_200' not in dataframe.columns:
            dataframe['ema_200'] = talib.EMA(dataframe['close'], timeperiod=200)
        atr_norm = dataframe['atr'] / dataframe['close']
        dataframe['atr_percentile'] = atr_norm.rolling(lb).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        bull_aligned = (dataframe['close'] > dataframe['ema_50']) & (dataframe['ema_50'] > dataframe['ema_200'])
        bear_aligned = (dataframe['close'] < dataframe['ema_50']) & (dataframe['ema_50'] < dataframe['ema_200'])
        conditions = [
            bull_aligned & (dataframe['atr_percentile'] < 0.7),
            bear_aligned & (dataframe['atr_percentile'] < 0.7),
            dataframe['atr_percentile'] >= 0.7,
        ]
        dataframe['regime'] = np.select(conditions, ['bull', 'bear', 'volatile'], default='range')
        sub_conditions = [dataframe['adx'] >= 25, dataframe['adx'] < 20]
        dataframe['regime_sub'] = np.select(sub_conditions, ['strong', 'weak'], default='mid')
        dataframe['regime_full'] = np.where(
            dataframe['regime'] == 'volatile',
            'volatile',
            dataframe['regime'].astype(str) + '_' + dataframe['regime_sub'].astype(str)
        )
        dataframe['regime'] = dataframe['regime_full']
        dataframe['regime_confidence'] = 0.5
        return dataframe
'''


REGIME_DETECTION_BLOCK_V4EMA_ATR = '''
    def _detect_regime_v3(self, dataframe: DataFrame) -> DataFrame:
        """EMA-alignment + ATR-percentile sub-classifier.

        Base: v4ema (bull/bear/range/volatile).
        Sub-state from atr_pct (72-bar rank, 0-1):
          expand:   atr_pct > 0.66
          compress: atr_pct < 0.33
          mid:      otherwise
        Volatile regime is not sub-classified; regime_full stays 'volatile'.
        """
        lb = self.REGIME_LOOKBACK
        if 'ema_200' not in dataframe.columns:
            dataframe['ema_200'] = talib.EMA(dataframe['close'], timeperiod=200)
        atr_norm = dataframe['atr'] / dataframe['close']
        dataframe['atr_percentile'] = atr_norm.rolling(lb).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        bull_aligned = (dataframe['close'] > dataframe['ema_50']) & (dataframe['ema_50'] > dataframe['ema_200'])
        bear_aligned = (dataframe['close'] < dataframe['ema_50']) & (dataframe['ema_50'] < dataframe['ema_200'])
        conditions = [
            bull_aligned & (dataframe['atr_percentile'] < 0.7),
            bear_aligned & (dataframe['atr_percentile'] < 0.7),
            dataframe['atr_percentile'] >= 0.7,
        ]
        dataframe['regime'] = np.select(conditions, ['bull', 'bear', 'volatile'], default='range')
        _atr_pct72 = dataframe['atr'].rolling(72).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        sub_conditions = [_atr_pct72 > 0.66, _atr_pct72 < 0.33]
        dataframe['regime_sub'] = np.select(sub_conditions, ['expand', 'compress'], default='mid')
        dataframe['regime_full'] = np.where(
            dataframe['regime'] == 'volatile',
            'volatile',
            dataframe['regime'].astype(str) + '_' + dataframe['regime_sub'].astype(str)
        )
        dataframe['regime'] = dataframe['regime_full']
        dataframe['regime_confidence'] = 0.5
        return dataframe
'''


REGIME_DETECTION_BLOCK_V4EMA_COMBO = '''
    def _detect_regime_v3(self, dataframe: DataFrame) -> DataFrame:
        """EMA-alignment + (slope x adx) 2x2 sub-classifier.

        Base: v4ema (bull/bear/range/volatile).
        Sub-state from slope (positive/negative) x adx (>=22/<22):
          accel_strong:  slope > 0 AND adx >= 22
          accel_weak:    slope > 0 AND adx <  22
          cons_strong:   slope <= 0 AND adx >= 22
          cons_weak:     slope <= 0 AND adx <  22
        Volatile regime is not sub-classified; regime_full stays 'volatile'.
        """
        lb = self.REGIME_LOOKBACK
        if 'ema_200' not in dataframe.columns:
            dataframe['ema_200'] = talib.EMA(dataframe['close'], timeperiod=200)
        atr_norm = dataframe['atr'] / dataframe['close']
        dataframe['atr_percentile'] = atr_norm.rolling(lb).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        bull_aligned = (dataframe['close'] > dataframe['ema_50']) & (dataframe['ema_50'] > dataframe['ema_200'])
        bear_aligned = (dataframe['close'] < dataframe['ema_50']) & (dataframe['ema_50'] < dataframe['ema_200'])
        conditions = [
            bull_aligned & (dataframe['atr_percentile'] < 0.7),
            bear_aligned & (dataframe['atr_percentile'] < 0.7),
            dataframe['atr_percentile'] >= 0.7,
        ]
        dataframe['regime'] = np.select(conditions, ['bull', 'bear', 'volatile'], default='range')
        slope = (dataframe['ema_50'] - dataframe['ema_50'].shift(10)) / dataframe['ema_50']
        _slope_up = slope > 0
        _adx_strong = dataframe['adx'] >= 22
        sub_conditions = [
            _slope_up & _adx_strong,
            _slope_up & ~_adx_strong,
            ~_slope_up & _adx_strong,
        ]
        dataframe['regime_sub'] = np.select(
            sub_conditions,
            ['accel_strong', 'accel_weak', 'cons_strong'],
            default='cons_weak',
        )
        dataframe['regime_full'] = np.where(
            dataframe['regime'] == 'volatile',
            'volatile',
            dataframe['regime'].astype(str) + '_' + dataframe['regime_sub'].astype(str)
        )
        dataframe['regime'] = dataframe['regime_full']
        dataframe['regime_confidence'] = 0.5
        return dataframe
'''
