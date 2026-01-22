# =============================================================================
# FILE: lib/signals/advanced.py
# =============================================================================
"""Advanced technical analysis signal configurations.

Includes: Williams %R, CCI, ROC, Divergences, Squeeze, Keltner,
          Donchian, VWAP, Volume Spike, OI, Liquidation
"""

from typing import List
from .base import SignalConfig


def get_advanced_signals() -> List[SignalConfig]:
    """
    Generate advanced technical analysis signals.

    Based on classic trading literature:
    - Williams %R: "How I Made One Million Dollars Trading Commodities"
    - CCI: "Commodities Channel Index: Tools for Trading Cyclic Trends"
    - ROC: "Technical Analysis of Stock Trends" - Edwards & Magee
    - Divergences: "New Concepts in Technical Trading Systems" - Welles Wilder
    - Squeeze: "Mastering the Trade" - John Carter
    - Keltner: "How to Make Money in Commodities" - Chester Keltner
    - Donchian: "Way of the Turtle" - Curtis Faith
    """
    signals = []

    # === Williams %R (Larry Williams) ===
    signals.append(
        SignalConfig(
            name="willr_os_long",
            signal_type="williams_r",
            direction="long",
            params={"threshold": -80},
        )
    )
    signals.append(
        SignalConfig(
            name="willr_ob_short",
            signal_type="williams_r",
            direction="short",
            params={"threshold": -20},
        )
    )

    # === CCI - Commodity Channel Index (Donald Lambert) ===
    for threshold in [100, 150, 200]:
        signals.append(
            SignalConfig(
                name=f"cci_{threshold}_long",
                signal_type="cci",
                direction="long",
                params={"threshold": -threshold},
            )
        )
        signals.append(
            SignalConfig(
                name=f"cci_{threshold}_short",
                signal_type="cci",
                direction="short",
                params={"threshold": threshold},
            )
        )

    # === ROC - Rate of Change with confirmation ===
    signals.append(
        SignalConfig(
            name="roc_reversal_long",
            signal_type="roc",
            direction="long",
            params={"period": 12, "threshold": -5},
        )
    )
    signals.append(
        SignalConfig(
            name="roc_reversal_short",
            signal_type="roc",
            direction="short",
            params={"period": 12, "threshold": 5},
        )
    )

    # === RSI Divergence ===
    signals.append(
        SignalConfig(
            name="rsi_div_bull",
            signal_type="divergence",
            direction="long",
            params={"indicator": "rsi", "lookback": 14},
        )
    )
    signals.append(
        SignalConfig(
            name="rsi_div_bear",
            signal_type="divergence",
            direction="short",
            params={"indicator": "rsi", "lookback": 14},
        )
    )

    # === OBV Divergence (Joseph Granville) ===
    signals.append(
        SignalConfig(
            name="obv_div_bull",
            signal_type="divergence",
            direction="long",
            params={"indicator": "obv", "lookback": 20},
        )
    )
    signals.append(
        SignalConfig(
            name="obv_div_bear",
            signal_type="divergence",
            direction="short",
            params={"indicator": "obv", "lookback": 20},
        )
    )

    # === Squeeze Momentum (John Carter) ===
    signals.append(
        SignalConfig(name="squeeze_long", signal_type="squeeze", direction="long")
    )
    signals.append(
        SignalConfig(name="squeeze_short", signal_type="squeeze", direction="short")
    )

    # === Keltner Channel Breakout ===
    signals.append(
        SignalConfig(
            name="keltner_break_long",
            signal_type="keltner",
            direction="long",
            params={"mult": 2.0},
        )
    )
    signals.append(
        SignalConfig(
            name="keltner_break_short",
            signal_type="keltner",
            direction="short",
            params={"mult": 2.0},
        )
    )

    # === Donchian Channel (Turtle Trading) ===
    signals.append(
        SignalConfig(
            name="donchian_20_long",
            signal_type="donchian",
            direction="long",
            params={"period": 20},
        )
    )
    signals.append(
        SignalConfig(
            name="donchian_20_short",
            signal_type="donchian",
            direction="short",
            params={"period": 20},
        )
    )

    # === VWAP Reversion ===
    signals.append(
        SignalConfig(
            name="vwap_long", signal_type="vwap", direction="long", params={"dev": -2.0}
        )
    )
    signals.append(
        SignalConfig(
            name="vwap_short",
            signal_type="vwap",
            direction="short",
            params={"dev": 2.0},
        )
    )

    # === Volume Spike + Price action ===
    signals.append(
        SignalConfig(
            name="vol_spike_long",
            signal_type="volume_spike",
            direction="long",
            params={"vol_mult": 2.0},
        )
    )
    signals.append(
        SignalConfig(
            name="vol_spike_short",
            signal_type="volume_spike",
            direction="short",
            params={"vol_mult": 2.0},
        )
    )

    # === Open Interest Divergence ===
    signals.append(
        SignalConfig(name="oi_div_long", signal_type="oi_divergence", direction="long")
    )
    signals.append(
        SignalConfig(
            name="oi_div_short", signal_type="oi_divergence", direction="short"
        )
    )

    # === Liquidation Cascade Detection ===
    signals.append(
        SignalConfig(
            name="liq_cascade_long",
            signal_type="liquidation",
            direction="long",
            params={"threshold": 2.0},
        )
    )
    signals.append(
        SignalConfig(
            name="liq_cascade_short",
            signal_type="liquidation",
            direction="short",
            params={"threshold": 2.0},
        )
    )

    return signals
