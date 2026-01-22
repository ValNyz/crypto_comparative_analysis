# =============================================================================
# FILE: lib/regime/detector.py
# =============================================================================
"""
Multi-factor regime detection.

Components:
- Volatility Score: ATR percentile + BB width
- Trend Score: ADX + DI + EMA alignment
- Momentum Score: RSI position + MACD histogram
"""

import numpy as np
from pandas import DataFrame
from .classifier import classify_regime, get_regime_confidence


def detect_regime_v3(
    dataframe: DataFrame,
    lookback: int = 72,
    adx_threshold: int = 20,
    adx_strong: int = 25,
    atr_volatile: float = 0.75,
    atr_low: float = 0.25,
) -> DataFrame:
    """
    Détection de régime multi-facteur améliorée.

    Args:
        dataframe: DataFrame with OHLCV and indicators
        lookback: Lookback period for percentile calculations
        adx_threshold: ADX threshold for trending detection
        adx_strong: ADX threshold for strong trends
        atr_volatile: ATR percentile threshold for volatile regime
        atr_low: ATR percentile threshold for quiet regime

    Returns:
        DataFrame with regime columns added
    """
    df = dataframe.copy()

    # Calculate scores
    df = calculate_volatility_score(df, lookback)
    df = calculate_trend_score(df)
    df = calculate_momentum_score(df)

    # Classify regime
    df = classify_regime(df, adx_threshold, atr_volatile, atr_low)

    # Calculate confidence
    df = get_regime_confidence(df)

    return df


def calculate_volatility_score(df: DataFrame, lookback: int = 72) -> DataFrame:
    """
    Calculate volatility score from ATR and BB width percentiles.

    Args:
        df: DataFrame with 'atr', 'bb_upper', 'bb_lower', 'bb_middle'
        lookback: Lookback period for percentile calculation

    Returns:
        DataFrame with vol_score added
    """
    # ATR percentile
    df["atr_pct"] = (
        df["atr"]
        .rolling(lookback)
        .apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
        )
    )

    # BB width percentile
    bb_width = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_width_pct"] = bb_width.rolling(lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
    )

    # Combined volatility score
    df["vol_score"] = (df["atr_pct"] + df["bb_width_pct"]) / 2

    return df


def calculate_trend_score(df: DataFrame) -> DataFrame:
    """
    Calculate trend score from ADX and EMA alignment.

    Args:
        df: DataFrame with 'adx', 'ema_8', 'ema_21', 'ema_50'

    Returns:
        DataFrame with trend_strength added
    """
    # ADX strength normalized to 0-1
    adx_score = np.clip(df["adx"] / 40, 0, 1)

    # EMA alignment (8 > 21 > 50 = bullish, inverse = bearish)
    ema_bull = ((df["ema_8"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])).astype(
        float
    )

    ema_bear = ((df["ema_8"] < df["ema_21"]) & (df["ema_21"] < df["ema_50"])).astype(
        float
    )

    # Trend strength: -1 to 1 scaled by ADX
    df["trend_strength"] = adx_score * (ema_bull - ema_bear + 1) / 2

    return df


def calculate_momentum_score(df: DataFrame) -> DataFrame:
    """
    Calculate momentum score from RSI and MACD.

    Args:
        df: DataFrame with 'rsi_14', 'macd_hist'

    Returns:
        DataFrame with momentum_score added
    """
    # RSI position (above/below 50)
    rsi_bull = (df["rsi_14"] > 50).astype(float)

    # MACD histogram positive
    macd_bull = (df["macd_hist"] > 0).astype(float)

    # MACD histogram rising
    macd_rising = (df["macd_hist"] > df["macd_hist"].shift(1)).astype(float)

    # Combined score
    df["momentum_score"] = (rsi_bull + macd_bull + macd_rising) / 3

    return df
