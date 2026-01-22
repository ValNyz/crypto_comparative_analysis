# =============================================================================
# FILE: lib/regime/classifier.py
# =============================================================================
"""Regime classification logic."""

import numpy as np
from pandas import DataFrame


def classify_regime(
    df: DataFrame,
    adx_threshold: int = 20,
    vol_high: float = 0.6,
    vol_low: float = 0.3,
) -> DataFrame:
    """
    Classify market regime based on scores.

    Regimes:
    - bull: Strong uptrend (trending + DI+ > DI- + momentum > 0.6)
    - bear: Strong downtrend (trending + DI- > DI+ + momentum < 0.4)
    - volatile: High volatility without clear direction
    - quiet: Low volatility ranging market
    - range: Default (moderate conditions)

    Args:
        df: DataFrame with vol_score, adx, di_plus, di_minus, momentum_score
        adx_threshold: Threshold for trending detection
        vol_high: Volatility score threshold for volatile regime
        vol_low: Volatility score threshold for quiet regime

    Returns:
        DataFrame with 'regime' column added
    """
    # Conditions
    is_high_vol = df["vol_score"] > vol_high
    is_low_vol = df["vol_score"] < vol_low
    is_trending = df["adx"] > adx_threshold

    # Strong bull: trending + bullish direction + positive momentum
    is_strong_bull = (
        is_trending & (df["di_plus"] > df["di_minus"]) & (df["momentum_score"] > 0.6)
    )

    # Strong bear: trending + bearish direction + negative momentum
    is_strong_bear = (
        is_trending & (df["di_minus"] > df["di_plus"]) & (df["momentum_score"] < 0.4)
    )

    # Volatile: high volatility but not strong trend
    is_volatile = is_high_vol & ~is_strong_bull & ~is_strong_bear

    # Quiet: low volatility and not trending
    is_quiet_range = is_low_vol & ~is_trending

    # Classification (order matters - first match wins)
    conditions = [is_strong_bull, is_strong_bear, is_volatile, is_quiet_range]
    choices = ["bull", "bear", "volatile", "quiet"]
    df["regime"] = np.select(conditions, choices, default="range")

    return df


def get_regime_confidence(df: DataFrame) -> DataFrame:
    """
    Calculate confidence score for regime classification.

    Args:
        df: DataFrame with regime, adx, vol_score

    Returns:
        DataFrame with 'regime_confidence' column added
    """
    # Confidence based on how strongly conditions are met
    is_strong_bull = df["regime"] == "bull"
    is_strong_bear = df["regime"] == "bear"
    is_volatile = df["regime"] == "volatile"

    df["regime_confidence"] = np.where(
        is_strong_bull | is_strong_bear,
        df["adx"] / 40,  # Higher ADX = more confidence in trend
        np.where(
            is_volatile,
            df["vol_score"],  # Higher vol score = more confidence in volatile
            0.5,  # Default confidence for range/quiet
        ),
    )

    return df


def get_regime_stats(df: DataFrame) -> dict:
    """
    Calculate statistics about regime distribution in DataFrame.

    Args:
        df: DataFrame with 'regime' column

    Returns:
        Dict with counts and percentages per regime
    """
    if "regime" not in df.columns:
        return {}

    counts = df["regime"].value_counts()
    total = len(df)

    return {
        regime: {"count": count, "pct": count / total * 100}
        for regime, count in counts.items()
    }
