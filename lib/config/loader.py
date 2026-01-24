# =============================================================================
# FILE: lib/config/loader.py
# =============================================================================
"""Configuration loader from YAML files."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from .base import Config


def load_yaml(filepath: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    filepath: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        filepath: Path to YAML config file. If None, uses defaults.
        overrides: Dict of values to override after loading.

    Returns:
        Config instance
    """
    config_dict = {}

    if filepath:
        config_dict = load_yaml(filepath)

    # Apply overrides
    if overrides:
        config_dict.update(overrides)

    # Filter to only valid Config fields
    valid_fields = {f.name for f in Config.__dataclass_fields__.values()}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}

    return Config(**filtered)


def load_regime_mappings(config: Config) -> Dict[str, List[str]]:
    """Load regime filter mappings from YAML."""
    filepath = Path(config.regime_mappings)
    if not filepath.exists():
        # Return defaults if file doesn't exist
        return _default_regime_mappings()

    data = load_yaml(filepath)
    return data.get("regime_filters", _default_regime_mappings())


def load_signal_conditions(config: Config) -> Dict[str, str]:
    """Load atomic signal conditions from YAML."""
    filepath = Path(config.signal_conditions)
    if not filepath.exists():
        return _default_signal_conditions()

    data = load_yaml(filepath)
    return data.get("conditions", _default_signal_conditions())


def _default_regime_mappings() -> Dict[str, List[str]]:
    """Default regime filter mappings."""
    return {
        "funding_*": ["bull", "bear", "range", "volatile"],
        "rsi_*_long": ["range", "volatile"],
        "rsi_*_short": ["range", "volatile"],
        "bb_*_long": ["range", "volatile"],
        "bb_*_short": ["range", "volatile"],
        "zscore_*_long": ["range", "volatile"],
        "zscore_*_short": ["bull", "range"],
        "stoch_os_long": ["range", "volatile"],
        "stoch_ob_short": ["range", "volatile"],
        "reversal_*_long": ["bear", "range"],
        "reversal_*_short": ["bull", "range"],
        "multi_os_long": ["range", "volatile"],
        "multi_ob_short": ["range", "volatile"],
        "ema_8_21_long": ["bull"],
        "ema_8_21_short": ["bear"],
        "macd_cross_long": ["bull", "range"],
        "macd_cross_short": ["bear", "range"],
    }


def _default_signal_conditions() -> Dict[str, str]:
    """Default atomic signal conditions for combos."""
    return {
        # Oversold / Survendu
        "rsi_os": "dataframe['rsi_14'] < 30",
        "stoch_os": "dataframe['stoch_k'] < 20",
        "bb_low": "dataframe['bb_pos'] < -1",
        "mfi_os": "dataframe['mfi'] < 20",
        "willr_os": "dataframe['willr'] < -80",
        "zscore_low": "dataframe['zscore'] < -2",
        # Overbought / Suracheté
        "rsi_ob": "dataframe['rsi_14'] > 70",
        "stoch_ob": "dataframe['stoch_k'] > 80",
        "bb_high": "dataframe['bb_pos'] > 1",
        "mfi_ob": "dataframe['mfi'] > 80",
        "willr_ob": "dataframe['willr'] > -20",
        "zscore_high": "dataframe['zscore'] > 2",
        # Trend
        "ema_bull": "dataframe['ema_8'] > dataframe['ema_21']",
        "ema_bear": "dataframe['ema_8'] < dataframe['ema_21']",
        "adx_strong": "dataframe['adx'] > 25",
        "macd_positive": "dataframe['macd_hist'] > 0",
        "macd_negative": "dataframe['macd_hist'] < 0",
        # Volume
        "volume_above_avg": "dataframe['volume'] > dataframe['volume'].rolling(20).mean()",
        "volume_spike": "dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 2",
        # Price action
        "is_green": "dataframe['is_green']",
        "is_red": "dataframe['is_red']",
    }
