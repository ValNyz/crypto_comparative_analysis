# =============================================================================
# FILE: lib/exits/registry.py
# =============================================================================
"""Exit configurations registry - loads from YAML."""

from pathlib import Path
from typing import Dict, Optional, List
from .base import ExitConfig
from ..config.loader import load_yaml
from ..config.base import Config


# Cache for loaded exits
_exits_cache: Optional[Dict[str, ExitConfig]] = None


def load_exits_from_yaml(
    filepath: str = "./configs/exits.yaml",
) -> Dict[str, ExitConfig]:
    """
    Load all exit configurations from YAML file.

    Args:
        filepath: Path to exits YAML file

    Returns:
        Dict mapping exit name to ExitConfig
    """
    global _exits_cache

    path = Path(filepath)
    if not path.exists():
        # Return defaults if file doesn't exist
        return _get_default_exits()

    data = load_yaml(filepath)
    exits = {}

    for name, config in data.get("exits", {}).items():
        config["name"] = name
        exits[name] = ExitConfig.from_dict(config)

    _exits_cache = exits
    return exits


def get_exit_config(name: str, config: Config) -> ExitConfig:
    """
    Get a specific exit configuration by name.

    Args:
        name: Exit config name
        configs_dir: Directory containing config files

    Returns:
        ExitConfig instance
    """
    global _exits_cache

    if _exits_cache is None:
        filepath = Path(config.exits)
        _exits_cache = load_exits_from_yaml(str(filepath))

    if name not in _exits_cache:
        # Check defaults
        defaults = _get_default_exits()
        if name in defaults:
            return defaults[name]
        # Return empty config
        return ExitConfig(name=name)

    return _exits_cache[name]


def get_all_exit_configs(config: Config) -> Dict[str, ExitConfig]:
    """Get all available exit configurations."""
    global _exits_cache

    if _exits_cache is None:
        filepath = Path(config.exits)
        if Path(filepath).exists():
            _exits_cache = load_exits_from_yaml(str(filepath))
        else:
            _exits_cache = _get_default_exits()

    return _exits_cache


def get_exit_names(config: Config) -> List[str]:
    """Get list of all available exit config names."""
    return list(get_all_exit_configs(config).keys())


def clear_cache():
    """Clear the exits cache (useful for testing)."""
    global _exits_cache
    _exits_cache = None


def _get_default_exits() -> Dict[str, ExitConfig]:
    """Get default exit configurations (fallback if no YAML)."""
    return {
        "none": ExitConfig(name="none"),
        # BB Reversion
        "bb_reversion": ExitConfig(
            name="bb_reversion", use_bb_exit=True, bb_exit_long=0.8, bb_exit_short=-0.8
        ),
        # RSI variants
        "rsi_70": ExitConfig(
            name="rsi_70", use_rsi_exit=True, rsi_exit_long=70, rsi_exit_short=30
        ),
        "rsi_75": ExitConfig(
            name="rsi_75", use_rsi_exit=True, rsi_exit_long=75, rsi_exit_short=25
        ),
        "rsi_80": ExitConfig(
            name="rsi_80", use_rsi_exit=True, rsi_exit_long=80, rsi_exit_short=20
        ),
        # Zscore variants
        "zs_0.5": ExitConfig(
            name="zs_0.5", use_zscore_exit=True, zscore_exit_threshold=0.5
        ),
        "zs_0.8": ExitConfig(
            name="zs_0.8", use_zscore_exit=True, zscore_exit_threshold=0.8
        ),
        "zs_1.0": ExitConfig(
            name="zs_1.0", use_zscore_exit=True, zscore_exit_threshold=1.0
        ),
        "zs_1.5": ExitConfig(
            name="zs_1.5", use_zscore_exit=True, zscore_exit_threshold=1.5
        ),
        # Combo variants
        "combo_60": ExitConfig(
            name="combo_60",
            use_rsi_zscore_combo=True,
            combo_rsi_long=60,
            combo_rsi_short=40,
        ),
        "combo_65": ExitConfig(
            name="combo_65",
            use_rsi_zscore_combo=True,
            combo_rsi_long=65,
            combo_rsi_short=35,
        ),
        "combo_70": ExitConfig(
            name="combo_70",
            use_rsi_zscore_combo=True,
            combo_rsi_long=70,
            combo_rsi_short=30,
        ),
        # Full combos
        "combo_rsi_zscore": ExitConfig(
            name="combo_rsi_zscore",
            use_rsi_exit=True,
            rsi_exit_long=75,
            rsi_exit_short=25,
            use_zscore_exit=True,
            zscore_exit_threshold=0.8,
            use_rsi_zscore_combo=True,
            combo_rsi_long=65,
            combo_rsi_short=35,
        ),
        "full_tp": ExitConfig(
            name="full_tp",
            use_rsi_exit=True,
            rsi_exit_long=75,
            rsi_exit_short=25,
            use_zscore_exit=True,
            zscore_exit_threshold=0.8,
            use_rsi_zscore_combo=True,
            use_bb_exit=True,
            bb_exit_long=1.5,
            bb_exit_short=-1.5,
        ),
        "full_conservative": ExitConfig(
            name="full_conservative",
            use_rsi_exit=True,
            rsi_exit_long=80,
            rsi_exit_short=20,
            use_zscore_exit=True,
            zscore_exit_threshold=1.0,
            use_rsi_zscore_combo=True,
            combo_rsi_long=70,
            combo_rsi_short=30,
        ),
        "full_aggressive": ExitConfig(
            name="full_aggressive",
            use_rsi_exit=True,
            rsi_exit_long=70,
            rsi_exit_short=30,
            use_zscore_exit=True,
            zscore_exit_threshold=0.5,
            use_rsi_zscore_combo=True,
            combo_rsi_long=60,
            combo_rsi_short=40,
        ),
        # Stochastic
        "stoch_exit": ExitConfig(
            name="stoch_exit",
            use_stoch_exit=True,
            stoch_exit_long=80,
            stoch_exit_short=20,
        ),
        # Funding velocity
        "fvel4_2": ExitConfig(
            name="fvel_4_2",
            use_funding_velocity=True,
            funding_vel_period=4,
            funding_vel_threshold=0.2,
        ),
        "fvel8_3": ExitConfig(
            name="fvel_8_3",
            use_funding_velocity=True,
            funding_vel_period=8,
            funding_vel_threshold=0.3,
        ),
        "fvel12_4": ExitConfig(
            name="fvel_12_4",
            use_funding_velocity=True,
            funding_vel_period=12,
            funding_vel_threshold=0.4,
        ),
        # Funding acceleration
        "faccel_4": ExitConfig(
            name="faccel_4", use_funding_accel=True, funding_accel_period=4
        ),
        "faccel_8": ExitConfig(
            name="faccel_8", use_funding_accel=True, funding_accel_period=8
        ),
        # Funding neutral
        "fneu_03": ExitConfig(
            name="fneu_03", use_funding_neutral=True, funding_neutral_zone=0.3
        ),
        "fneu_05": ExitConfig(
            name="fneu_05", use_funding_neutral=True, funding_neutral_zone=0.5
        ),
        "fneu_08": ExitConfig(
            name="fneu_08", use_funding_neutral=True, funding_neutral_zone=0.8
        ),
        # Volatility exits
        "vol_15": ExitConfig(
            name="vol_15", use_vol_regime_exit=True, vol_expansion_mult=1.5
        ),
        "vol_20": ExitConfig(
            name="vol_20", use_vol_regime_exit=True, vol_expansion_mult=2.0
        ),
        "vol_25": ExitConfig(
            name="vol_25", use_vol_regime_exit=True, vol_expansion_mult=2.5
        ),
        # Volume spike
        "volspi25": ExitConfig(
            name="volspi25", use_volume_spike_exit=True, volume_spike_mult=2.5
        ),
        "volspi30": ExitConfig(
            name="volspi30", use_volume_spike_exit=True, volume_spike_mult=3.0
        ),
        # Crypto trailing
        "ctrail15": ExitConfig(
            name="ctrail15", use_crypto_trail=True, crypto_trail_mult=1.5
        ),
        "ctrail20": ExitConfig(
            name="ctrail20", use_crypto_trail=True, crypto_trail_mult=2.0
        ),
        "ctrail25": ExitConfig(
            name="ctrail25", use_crypto_trail=True, crypto_trail_mult=2.5
        ),
        "ctrail30": ExitConfig(
            name="ctrail30", use_crypto_trail=True, crypto_trail_mult=3.0
        ),
        # RSI divergence
        "rsidiv10": ExitConfig(
            name="rsidiv10", use_rsi_divergence=True, divergence_lookback=10
        ),
        "rsidiv14": ExitConfig(
            name="rsidiv14", use_rsi_divergence=True, divergence_lookback=14
        ),
        # Crypto combos
        "fvel_neu": ExitConfig(
            name="fvel_neu",
            use_funding_velocity=True,
            funding_vel_period=8,
            funding_vel_threshold=0.3,
            use_funding_neutral=True,
            funding_neutral_zone=0.5,
        ),
        "faccelvol": ExitConfig(
            name="faccelvol",
            use_funding_accel=True,
            funding_accel_period=4,
            use_vol_regime_exit=True,
            vol_expansion_mult=2.0,
        ),
        "triplecr": ExitConfig(
            name="triplecr",
            use_funding_velocity=True,
            funding_vel_period=8,
            funding_vel_threshold=0.3,
            use_funding_neutral=True,
            funding_neutral_zone=0.5,
            use_crypto_trail=True,
            crypto_trail_mult=2.5,
        ),
        "cryptoco": ExitConfig(
            name="cryptoco",
            use_funding_velocity=True,
            funding_vel_period=12,
            funding_vel_threshold=0.4,
            use_funding_neutral=True,
            funding_neutral_zone=0.3,
            use_vol_regime_exit=True,
            vol_expansion_mult=2.5,
            use_crypto_trail=True,
            crypto_trail_mult=3.0,
        ),
        "cryptoag": ExitConfig(
            name="cryptoag",
            use_funding_velocity=True,
            funding_vel_period=4,
            funding_vel_threshold=0.2,
            use_funding_neutral=True,
            funding_neutral_zone=0.8,
            use_vol_regime_exit=True,
            vol_expansion_mult=1.5,
            use_crypto_trail=True,
            crypto_trail_mult=1.5,
        ),
    }
