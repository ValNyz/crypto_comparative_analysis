# =============================================================================
# FILE: lib/signals/combo.py
# =============================================================================
"""Combined/confluence signal configurations."""

from typing import List
from .base import SignalConfig


def get_combo_signals() -> List[SignalConfig]:
    """
    Generate combined/confluence signal configurations.

    These signals combine multiple indicators for higher conviction entries.
    """
    signals = []

    # === Confluence Mean Reversion Long ===
    # Long when 3+ indicators are oversold + green candle confirmation
    signals.append(
        SignalConfig(
            name="combo_mr_3of5_long",
            signal_type="combo",
            direction="long",
            params={
                "min_signals": 3,
                "signals": ["rsi_os", "bb_low", "stoch_os", "zscore_low", "willr_os"],
                "confirm": "is_green",
            },
            roi={"0": 0.02},
            stoploss=-0.03,
        )
    )

    # === Confluence Mean Reversion Short ===
    # Short when 3+ indicators are overbought + red candle confirmation
    signals.append(
        SignalConfig(
            name="combo_mr_3of5_short",
            signal_type="combo",
            direction="short",
            params={
                "min_signals": 3,
                "signals": ["rsi_ob", "bb_high", "stoch_ob", "zscore_high", "willr_ob"],
                "confirm": "is_red",
            },
            roi={"0": 0.02},
            stoploss=-0.03,
        )
    )

    # === Trend + Momentum Long ===
    # Long in confirmed uptrend with momentum
    signals.append(
        SignalConfig(
            name="combo_trend_mom_long",
            signal_type="combo",
            direction="long",
            params={
                "conditions": [
                    "ema_bull",
                    "adx_strong",
                    "macd_positive",
                    "volume_above_avg",
                ],
            },
            roi={"0": 0.03},
            stoploss=-0.04,
        )
    )

    # === Trend + Momentum Short ===
    # Short in confirmed downtrend with momentum
    signals.append(
        SignalConfig(
            name="combo_trend_mom_short",
            signal_type="combo",
            direction="short",
            params={
                "conditions": [
                    "ema_bear",
                    "adx_strong",
                    "macd_negative",
                    "volume_above_avg",
                ],
            },
            roi={"0": 0.03},
            stoploss=-0.04,
        )
    )

    # === Volume Spike + Oversold ===
    # Long on volume spike when at least 2 oversold indicators + green candle
    signals.append(
        SignalConfig(
            name="combo_vol_os_long",
            signal_type="combo",
            direction="long",
            params={
                "min_signals": 2,
                "signals": ["rsi_os", "bb_low", "stoch_os"],
                "extra_conditions": ["volume_spike", "is_green"],
            },
            roi={"0": 0.025},
            stoploss=-0.03,
        )
    )

    return signals


def get_combo_signal_conditions() -> dict:
    """
    Get the atomic signal conditions used in combo signals.

    These are the building blocks that can be combined.
    Returns the same dict as in config but typed for reference.
    """
    return {
        # Oversold conditions
        "rsi_os": "dataframe['rsi_14'] < 30",
        "stoch_os": "dataframe['stoch_k'] < 20",
        "bb_low": "dataframe['bb_pos'] < -1",
        "mfi_os": "dataframe['mfi'] < 20",
        "willr_os": "dataframe['willr'] < -80",
        "zscore_low": "dataframe['zscore'] < -2",
        # Overbought conditions
        "rsi_ob": "dataframe['rsi_14'] > 70",
        "stoch_ob": "dataframe['stoch_k'] > 80",
        "bb_high": "dataframe['bb_pos'] > 1",
        "mfi_ob": "dataframe['mfi'] > 80",
        "willr_ob": "dataframe['willr'] > -20",
        "zscore_high": "dataframe['zscore'] > 2",
        # Trend conditions
        "ema_bull": "dataframe['ema_8'] > dataframe['ema_21']",
        "ema_bear": "dataframe['ema_8'] < dataframe['ema_21']",
        "adx_strong": "dataframe['adx'] > 25",
        "macd_positive": "dataframe['macd_hist'] > 0",
        "macd_negative": "dataframe['macd_hist'] < 0",
        # Volume conditions
        "volume_above_avg": "dataframe['volume'] > dataframe['volume'].rolling(20).mean()",
        "volume_spike": "dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 2",
        # Price action
        "is_green": "dataframe['is_green']",
        "is_red": "dataframe['is_red']",
    }
