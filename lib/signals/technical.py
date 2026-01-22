# =============================================================================
# FILE: lib/signals/technical.py
# =============================================================================
"""Technical analysis signal configurations."""

from typing import List
from .base import SignalConfig


def get_technical_signals() -> List[SignalConfig]:
    """
    Generate standard technical analysis signals.

    Includes: RSI, Bollinger, EMA Cross, Stochastic, MACD, Reversal, Z-Score, Multi-factor
    """
    signals = []

    # === RSI Signals ===
    for threshold in [25, 30, 35]:
        signals.append(
            SignalConfig(
                name=f"rsi_{threshold}_long",
                signal_type="rsi",
                direction="long",
                params={"threshold": threshold},
            )
        )

    for threshold in [65, 70, 75]:
        signals.append(
            SignalConfig(
                name=f"rsi_{threshold}_short",
                signal_type="rsi",
                direction="short",
                params={"threshold": threshold},
            )
        )

    # === Bollinger Band Signals ===
    for threshold in [1.0, 1.5, 2.0]:
        signals.append(
            SignalConfig(
                name=f"bb_{threshold}_long",
                signal_type="bollinger",
                direction="long",
                params={"threshold": threshold},
            )
        )
        signals.append(
            SignalConfig(
                name=f"bb_{threshold}_short",
                signal_type="bollinger",
                direction="short",
                params={"threshold": threshold},
            )
        )

    # === EMA Cross Signals ===
    signals.append(
        SignalConfig(name="ema_8_21_long", signal_type="ema_cross", direction="long")
    )
    signals.append(
        SignalConfig(name="ema_8_21_short", signal_type="ema_cross", direction="short")
    )

    # === Stochastic Signals ===
    signals.append(
        SignalConfig(
            name="stoch_os_long",
            signal_type="stochastic",
            direction="long",
            params={"threshold": 20},
        )
    )
    signals.append(
        SignalConfig(
            name="stoch_ob_short",
            signal_type="stochastic",
            direction="short",
            params={"threshold": 80},
        )
    )

    # === MACD Cross Signals ===
    signals.append(
        SignalConfig(name="macd_cross_long", signal_type="macd", direction="long")
    )
    signals.append(
        SignalConfig(name="macd_cross_short", signal_type="macd", direction="short")
    )

    # === Reversal Signals ===
    for candles in [3, 4, 5]:
        signals.append(
            SignalConfig(
                name=f"reversal_{candles}_long",
                signal_type="reversal",
                direction="long",
                params={"candles": candles},
            )
        )
        signals.append(
            SignalConfig(
                name=f"reversal_{candles}_short",
                signal_type="reversal",
                direction="short",
                params={"candles": candles},
            )
        )

    # === Z-Score Price Signals ===
    for z_threshold in [2.0, 2.5]:
        signals.append(
            SignalConfig(
                name=f"zscore_{z_threshold}_long",
                signal_type="zscore",
                direction="long",
                params={"threshold": z_threshold},
            )
        )
        signals.append(
            SignalConfig(
                name=f"zscore_{z_threshold}_short",
                signal_type="zscore",
                direction="short",
                params={"threshold": z_threshold},
            )
        )

    # === Multi-factor Signals ===
    signals.append(
        SignalConfig(name="multi_os_long", signal_type="multi", direction="long")
    )
    signals.append(
        SignalConfig(name="multi_ob_short", signal_type="multi", direction="short")
    )

    return signals
