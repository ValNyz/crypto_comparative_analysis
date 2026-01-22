# =============================================================================
# FILE: lib/signals/registry.py
# =============================================================================
"""Signal registry - combines all signal types and loads from YAML."""

from typing import List, Optional, Dict, Any
from pathlib import Path
from .base import SignalConfig
from .funding import get_funding_signals
from .technical import get_technical_signals
from .advanced import get_advanced_signals
from .combo import get_combo_signals
from ..config.loader import load_yaml
from ..exits.registry import get_exit_names


def get_signal_configs(
    signal_filter: Optional[str] = None,
    include_exits: bool = True,
    configs_dir: str = "./configs",
) -> List[SignalConfig]:
    """
    Get all signal configurations.

    Args:
        signal_filter: Filter by signal type ('funding', 'technical', 'advanced', 'combo')
        include_exits: Whether to include exit method variations for funding signals
        configs_dir: Directory containing config files

    Returns:
        List of SignalConfig instances
    """
    signals = []

    # Try to load from YAML first
    yaml_path = Path(configs_dir) / "signals.yaml"
    if yaml_path.exists():
        yaml_signals = load_signals_from_yaml(
            str(yaml_path), signal_filter, include_exits, configs_dir
        )
        if yaml_signals:
            return yaml_signals

    # Fall back to programmatic generation
    if signal_filter is None or signal_filter == "funding":
        signals.extend(
            get_funding_signals(include_exits=include_exits, configs_dir=configs_dir)
        )

    if signal_filter is None or signal_filter == "technical":
        signals.extend(get_technical_signals())

    if signal_filter is None or signal_filter == "advanced":
        signals.extend(get_advanced_signals())

    if signal_filter is None or signal_filter == "combo":
        signals.extend(get_combo_signals())

    return signals


def load_signals_from_yaml(
    filepath: str,
    signal_filter: Optional[str] = None,
    include_exits: bool = True,
    configs_dir: str = "./configs",
) -> List[SignalConfig]:
    """
    Load signal configurations from YAML file.

    Supports multiple structures:
    1. Simple: signals: [...]
    2. Grouped: funding_signals: {signals: [...]}
    3. With auto-generation: generate_with_exits: {...}

    Args:
        filepath: Path to signals YAML file
        signal_filter: Optional filter by signal_type
        include_exits: Whether to generate exit variations
        configs_dir: Directory for config files (to load exit names)

    Returns:
        List of SignalConfig instances
    """
    data = load_yaml(filepath)
    signals = []

    # === Structure 1: Simple flat list ===
    # signals:
    #   - name: "..."
    if "signals" in data and isinstance(data["signals"], list):
        signals.extend(_parse_signal_list(data["signals"], signal_filter))

    # === Structure 2: Grouped by type ===
    # funding_signals:
    #   signals: [...]
    # technical_signals:
    #   signals: [...]
    for key in [
        "funding_signals",
        "technical_signals",
        "advanced_signals",
        "combo_signals",
    ]:
        if key in data:
            group_data = data[key]

            # Parse explicit signals
            if "signals" in group_data:
                group_signals = _parse_signal_list(group_data["signals"], signal_filter)
                signals.extend(group_signals)

            # Handle auto-generation with exits
            if include_exits and "generate_with_exits" in group_data:
                gen_config = group_data["generate_with_exits"]
                if gen_config.get("enabled", False):
                    generated = _generate_signals_with_exits(
                        gen_config, signal_filter, configs_dir
                    )
                    signals.extend(generated)

    # === Structure 3: Top-level generate_with_exits ===
    if include_exits and "generate_with_exits" in data:
        gen_config = data["generate_with_exits"]
        if gen_config.get("enabled", False):
            generated = _generate_signals_with_exits(
                gen_config, signal_filter, configs_dir
            )
            signals.extend(generated)

    return signals


def _parse_signal_list(
    signal_list: List[Dict[str, Any]], signal_filter: Optional[str] = None
) -> List[SignalConfig]:
    """Parse a list of signal dicts into SignalConfig objects."""
    signals = []

    for sig_data in signal_list:
        # Filter by type if specified
        if signal_filter and sig_data.get("signal_type") != signal_filter:
            continue

        signals.append(SignalConfig.from_dict(sig_data))

    return signals


def _generate_signals_with_exits(
    gen_config: Dict[str, Any],
    signal_filter: Optional[str] = None,
    configs_dir: str = "./configs",
) -> List[SignalConfig]:
    """
    Generate signal variations with different exit methods.

    Config structure:
    generate_with_exits:
      enabled: true
      signal_type: funding  # optional, defaults to funding
      base_signals:
        - {zscore: 0.75, lookback: 72}
        - {zscore: 1.0, lookback: 168}
      common_params:
        use_rsi: true
        rsi_min: 35
        rsi_max: 65
      roi_with_exit: {"0": 0.2}
      stoploss: -0.03
      exit_methods: [...]  # optional, defaults to all
    """
    signals = []

    signal_type = gen_config.get("signal_type", "funding")

    # Filter check
    if signal_filter and signal_type != signal_filter:
        return signals

    base_signals = gen_config.get("base_signals", [])
    common_params = gen_config.get("common_params", {})
    roi_with_exit = gen_config.get("roi_with_exit", {"0": 0.2})
    stoploss = gen_config.get("stoploss", -0.03)

    # Get exit methods to use
    specified_exits = gen_config.get("exit_methods")
    if specified_exits:
        exit_methods = specified_exits
    else:
        # Load all available exits except 'none'
        exit_methods = [e for e in get_exit_names(configs_dir) if e != "none"]

    # Generate signals
    for base in base_signals:
        zscore = base.get("zscore", 1.5)
        lookback = base.get("lookback", 72)

        # Merge base params with common params
        params = {
            "zscore": zscore,
            "lookback": lookback,
            **common_params,
            **base,  # Allow base to override common
        }
        # Remove zscore/lookback from params if they were in base (they're already set)
        params.pop("zscore", None)
        params.pop("lookback", None)
        params["zscore"] = zscore
        params["lookback"] = lookback

        # Generate variant for each exit method
        for exit_name in exit_methods:
            # Truncate exit name for signal name (max 8 chars like original)
            exit_short = exit_name[:8]

            signal = SignalConfig(
                name=f"f_z{zscore}_lb{lookback}_x{exit_short}",
                signal_type=signal_type,
                direction="both",
                params=params.copy(),
                roi=roi_with_exit,
                stoploss=stoploss,
                exit_config=exit_name,
            )
            signals.append(signal)

    return signals


def get_funding_signals_only(
    include_exits: bool = True, configs_dir: str = "./configs"
) -> List[SignalConfig]:
    """Get only funding contrarian signals."""
    return get_signal_configs(
        signal_filter="funding", include_exits=include_exits, configs_dir=configs_dir
    )


def get_technical_signals_only() -> List[SignalConfig]:
    """Get only standard technical signals."""
    return get_technical_signals()


def get_advanced_signals_only() -> List[SignalConfig]:
    """Get only advanced technical signals."""
    return get_advanced_signals()


def get_combo_signals_only() -> List[SignalConfig]:
    """Get only combo/confluence signals."""
    return get_combo_signals()


def get_signal_by_name(
    name: str, configs_dir: str = "./configs"
) -> Optional[SignalConfig]:
    """Get a specific signal by name."""
    all_signals = get_signal_configs(configs_dir=configs_dir)
    for signal in all_signals:
        if signal.name == name:
            return signal
    return None


def get_signal_types() -> List[str]:
    """Get list of all available signal types."""
    return [
        "funding",
        "rsi",
        "bollinger",
        "ema_cross",
        "stochastic",
        "macd",
        "reversal",
        "zscore",
        "multi",
        "williams_r",
        "cci",
        "roc",
        "divergence",
        "squeeze",
        "keltner",
        "donchian",
        "vwap",
        "volume_spike",
        "oi_divergence",
        "liquidation",
        "combo",
    ]
