# =============================================================================
# FILE: lib/signals/registry.py
# =============================================================================
"""Signal registry - combines all signal types and loads from YAML."""

import itertools
from copy import deepcopy
from typing import List, Optional, Dict, Union, Any
from pathlib import Path
from .base import SignalConfig
from .funding import get_funding_signals
from .technical import get_technical_signals
from .advanced import get_advanced_signals
from .combo import get_combo_signals
from ..config.loader import load_yaml
from ..config.base import Config
from ..exits.registry import get_exit_names


def get_signal_configs(
    config: Config,
    signal_filter: Optional[str] = None,
    include_exits: bool = True,
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
    yaml_path = Path(config.signals)
    if yaml_path.exists():
        yaml_signals = load_signals_from_yaml(config, str(yaml_path), signal_filter)
        if yaml_signals:
            return yaml_signals

    # Fall back to programmatic generation
    if signal_filter is None or signal_filter == "funding":
        signals.extend(get_funding_signals(include_exits=include_exits, config=config))

    if signal_filter is None or signal_filter == "technical":
        signals.extend(get_technical_signals())

    if signal_filter is None or signal_filter == "advanced":
        signals.extend(get_advanced_signals())

    if signal_filter is None or signal_filter == "combo":
        signals.extend(get_combo_signals())

    return signals


def load_signals_from_yaml(
    config: Config,
    filepath: str,
    signal_filter: Optional[str] = None,
) -> List[SignalConfig]:
    """
    Load signal configurations from YAML file with advanced expansion.

    Supports:
    - Parameter lists for automatic expansion
    - Name placeholders like {zscore}, {lookback}
    - exit_config: "all" to generate all exit variations

    Args:
        filepath: Path to signals YAML file
        signal_filter: Optional filter by category
        configs_dir: Directory for config files (to load exits)

    Returns:
        List of SignalConfig instances
    """
    data = load_yaml(filepath)
    signals = []

    # Map of category names to signal_type
    category_map = {
        "funding_signals": "funding",
        "technical_signals": None,  # Keep original type
        "advanced_signals": None,
        "combo_signals": "combo",
    }

    # Get available exit configs for "all" expansion
    available_exits = get_exit_names(config)

    for category, signal_type_override in category_map.items():
        if category not in data:
            continue

        # Filter by category if requested
        if signal_filter:
            if signal_filter == "funding" and category != "funding_signals":
                continue
            elif signal_filter == "technical" and category != "technical_signals":
                continue
            elif signal_filter == "advanced" and category != "advanced_signals":
                continue
            elif signal_filter == "combo" and category != "combo_signals":
                continue

        for sig_template in data[category]:
            if not sig_template.get("enabled", True):
                continue

            expanded = expand_signal_template(
                sig_template,
                signal_type_override
                or sig_template.get("signal_type", _infer_signal_type(category)),
                available_exits,
            )
            signals.extend(expanded)

    return signals


def expand_signal_template(
    template: Dict[str, Any],
    signal_type: str,
    available_exits: List[str],
) -> List[SignalConfig]:
    """
    Expand a signal template into multiple SignalConfig instances.

    Handles:
    - Parameter lists (cartesian product expansion)
    - Name placeholders
    - exit_config: "all"

    Args:
        template: Signal template from YAML
        signal_type: Type of signal
        available_exits: List of available exit config names

    Returns:
        List of expanded SignalConfig instances
    """
    signals = []

    # Paramètres qui sont des listes de valeurs, pas des options à expandre
    PROTECTED_LIST_PARAMS = {"signals", "conditions", "extra_conditions"}

    # Extract template fields
    name_template = template.get("name", "unnamed")
    direction = template.get("direction", "both")
    params = template.get("params", {})
    roi = template.get("roi", {"0": 0.02})
    stoploss = template.get("stoploss", -0.03)
    timeframe_override = template.get("timeframe_override")
    allowed_regimes = template.get("allowed_regimes")
    exit_config = template.get("exit_config", "none")

    # Separate expandable params (lists) from fixed params
    expand_params = {}
    fixed_params = {}

    for key, value in params.items():
        if isinstance(value, list) and key not in PROTECTED_LIST_PARAMS:
            expand_params[key] = value
        else:
            fixed_params[key] = value

        # Handle ROI expansion
    roi_list = _ensure_list(roi)

    # Handle stoploss expansion
    stoploss_list = _ensure_list(stoploss)

    # Handle exit_config expansion
    exit_list = _expand_exit_configs(exit_config, available_exits)

    # Generate all combinations
    signals = []

    # Build list of (param_name, values) for expansion
    expansion_items = list(expand_params.items())

    # Add roi, stoploss, exit to expansion if they have multiple values
    if len(roi_list) > 1:
        expansion_items.append(("_roi", roi_list))
    if len(stoploss_list) > 1:
        expansion_items.append(("_stoploss", stoploss_list))
    if len(exit_list) > 1:
        expansion_items.append(("_exit", exit_list))

    if expansion_items:
        # Generate cartesian product of all expandable values
        keys = [item[0] for item in expansion_items]
        value_lists = [item[1] for item in expansion_items]

        for combination in itertools.product(*value_lists):
            # Build params for this combination
            combo_params = deepcopy(fixed_params)
            combo_dict = dict(zip(keys, combination))

            # Extract special keys
            roi = combo_dict.pop("_roi", roi_list[0])
            stoploss = combo_dict.pop("_stoploss", stoploss_list[0])
            exit_config = combo_dict.pop("_exit", exit_list[0])

            # Add remaining to params
            combo_params.update(combo_dict)

            # Build name from template
            name = _format_name(name_template, combo_params, exit_config)

            # Create signal config
            signal = SignalConfig(
                name=name,
                signal_type=signal_type,
                direction=direction,
                params=combo_params,
                roi=roi if isinstance(roi, dict) else {"0": roi},
                stoploss=stoploss,
                timeframe_override=timeframe_override,
                allowed_regimes=allowed_regimes,
                exit_config=exit_config,
            )
            signals.append(signal)
    else:
        # No expansion needed - single config
        name = _format_name(name_template, fixed_params, exit_list[0])

        signal = SignalConfig(
            name=name,
            signal_type=signal_type,
            direction=direction,
            params=fixed_params,
            roi=roi_list[0] if isinstance(roi_list[0], dict) else {"0": roi_list[0]},
            stoploss=stoploss_list[0],
            timeframe_override=timeframe_override,
            allowed_regimes=allowed_regimes,
            exit_config=exit_list[0],
        )
        signals.append(signal)

    return signals


def _ensure_list(value: Any) -> List:
    """Ensure value is a list."""
    if isinstance(value, list):
        return value
    return [value]


def _expand_exit_configs(
    exit_spec: Union[str, List[str]], available_exits: List[str]
) -> List[str]:
    """
    Expand exit configuration specification.

    Args:
        exit_spec: "all", single name, or list of names
        configs_dir: Directory for config files

    Returns:
        List of exit config names
    """
    if exit_spec == "all":
        # Get all available exit configs
        return [e for e in available_exits if e != "none"]
    elif isinstance(exit_spec, list):
        return [e for e in exit_spec if e in available_exits]
    else:
        return [exit_spec if exit_spec in available_exits else "none"]


def _format_name(template: str, params: Dict[str, Any], exit_config: str) -> str:
    """
    Format a name template with parameter values.

    Args:
        template: Name template with placeholders like {zscore}
        params: Parameter dict for substitution
        exit_config: Exit config name

    Returns:
        Formatted name string
    """
    # Build substitution dict
    subs = {}

    # Add params
    for key, value in params.items():
        # Format numeric values nicely
        if isinstance(value, float):
            if value == int(value):
                subs[key] = str(int(value))
            else:
                subs[key] = str(value).replace(".", "_")
        else:
            subs[key] = str(value)

    # Add exit
    subs["exit_config"] = exit_config[:8] if exit_config != "none" else ""

    # Format template
    try:
        name = template.format(**subs)
    except KeyError:
        # Missing placeholder - just use what we can
        for key, val in subs.items():
            template = template.replace(f"{{{key}}}", val)
        name = template

    # Clean up double underscores and trailing underscores
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")

    return name


def _infer_signal_type(category: str) -> str:
    """Infer signal type from category name."""
    if "funding" in category:
        return "funding"
    elif "combo" in category:
        return "combo"
    elif "advanced" in category:
        return "technical"  # Default for advanced
    return "technical"


def get_signal_by_name(
    name: str, configs_dir: str = "./configs", signals_file: str = "signals.yaml"
) -> Optional[SignalConfig]:
    """Get a specific signal by name."""
    all_signals = get_signal_configs(configs_dir=configs_dir, signals_file=signals_file)
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
