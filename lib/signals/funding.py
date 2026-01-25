# =============================================================================
# FILE: lib/signals/funding.py
# =============================================================================
"""Funding contrarian signal configurations."""

from typing import List, Optional
from .base import SignalConfig
from ..config.base import Config
from ..exits.registry import get_exit_names


def get_funding_signals(
    config: Config,
    include_exits: bool = True,
    exit_configs: Optional[List[str]] = None,
) -> List[SignalConfig]:
    """
    Generate funding contrarian signal configurations.

    Args:
        include_exits: Whether to include exit method variations
        exit_configs: List of exit config names to test (None = all)
        configs_dir: Directory for config files

    Returns:
        List of SignalConfig instances
    """
    signals = []

    # Get available exit configs
    if exit_configs is None and include_exits:
        exit_configs = get_exit_names(config)

    # Parameter combinations to test
    zscore_thresholds = [0.75, 1.0, 1.5]
    lookbacks = [72, 168]

    for zscore_t in zscore_thresholds:
        for lookback in lookbacks:
            # Base signal with RSI filter
            base_signal = SignalConfig(
                name=f"f_z{zscore_t}_lb{lookback}_base",
                signal_type="funding",
                direction="both",
                params={
                    "zscore": zscore_t,
                    "lookback": lookback,
                    "use_rsi": True,
                    "rsi_min": 35,
                    "rsi_max": 65,
                },
                roi={"0": 0.02},
                stoploss=-0.03,
            )
            signals.append(base_signal)

            # Generate exit variations
            if include_exits and exit_configs:
                for exit_name in exit_configs:
                    if exit_name != "none":
                        signals.append(
                            SignalConfig(
                                name=f"f_z{zscore_t}_lb{lookback}_x{exit_name[:8]}",
                                signal_type="funding",
                                direction="both",
                                params={
                                    "zscore": zscore_t,
                                    "lookback": lookback,
                                    "use_rsi": True,
                                    "rsi_min": 35,
                                    "rsi_max": 65,
                                },
                                roi={"0": 0.2},  # High ROI to let exit decide
                                stoploss=-0.03,
                                exit_config=exit_name,
                            )
                        )

    return signals


def get_funding_with_filters() -> List[SignalConfig]:
    """
    Generate funding signals with various filter combinations.

    Returns additional filter variations (commented out in original but preserved).
    """
    signals = []

    # These are preserved from the original but typically disabled
    # Uncomment to enable specific filter combinations

    # Example: Funding + ATR filter
    # for zscore_t in [1.0, 1.5, 2.0]:
    #     for lookback in [72, 168]:
    #         signals.append(SignalConfig(
    #             f"f_z{zscore_t}_lb{lookback}_atr",
    #             "funding", "both",
    #             {"zscore": zscore_t, "lookback": lookback,
    #              "use_atr": True, "atr_min": 0.2, "atr_max": 0.8},
    #             roi={"0": 0.02}, stoploss=-0.03,
    #         ))

    # Example: Funding + Volume filter
    # signals.append(SignalConfig(
    #     f"f_z{zscore_t}_lb{lookback}_vol",
    #     "funding", "both",
    #     {"zscore": zscore_t, "lookback": lookback,
    #      "use_volume": True, "volume_min": 0.5, "volume_max": 3.0},
    #     roi={"0": 0.02}, stoploss=-0.03,
    # ))

    # Example: Funding + Anti-trend filter
    # for adx_max in [35]:
    #     signals.append(SignalConfig(
    #         f"f_z{zscore_t}_lb{lookback}_antitrend{adx_max}",
    #         "funding", "both",
    #         {"zscore": zscore_t, "lookback": lookback,
    #          "use_antitrend": True, "adx_max": adx_max},
    #         roi={"0": 0.02}, stoploss=-0.03,
    #     ))

    # Example: Strict multi-filter
    # signals.append(SignalConfig(
    #     f"f_z{zscore_t}_lb{lookback}_strict",
    #     "funding", "both",
    #     {"zscore": zscore_t, "lookback": lookback,
    #      "use_rsi": True, "rsi_min": 40, "rsi_max": 60,
    #      "use_volume": True, "volume_min": 0.7,
    #      "use_atr": True, "atr_min": 0.25, "atr_max": 0.75,
    #      "use_antitrend": True, "adx_max": 30},
    #     roi={"0": 0.025}, stoploss=-0.025,
    # ))

    return signals
