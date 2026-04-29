# =============================================================================
# FILE: lib/signals/registry.py
# =============================================================================
"""Signal registry - combines all signal types and loads from YAML."""

import fnmatch
import itertools
from copy import deepcopy
from typing import List, Optional, Dict, Union, Any
from pathlib import Path
from .base import SignalConfig
from ..config.loader import load_yaml
from ..config.base import Config
from ..exits.registry import get_exit_names


# Legacy filter aliases — kept for backward compat. Maps a short type-name
# (the old --filter values) to the canonical YAML category it used to match.
_LEGACY_FILTER_ALIASES = {
    "funding": "funding_signals",
    "technical": "technical_signals",
    "advanced": "advanced_signals",
    "combo": "combo_signals",
}


def _matches_category_filter(category: str, filter_spec: str) -> bool:
    """True if `category` matches `filter_spec`.

    filter_spec rules (in order):
    - if filter_spec is a legacy alias (funding/technical/advanced/combo) →
      strict match on the canonical *_signals category.
    - if filter_spec contains a glob char (* or ?) → fnmatch wildcard.
    - otherwise → strict equality.
    """
    if filter_spec in _LEGACY_FILTER_ALIASES:
        return category == _LEGACY_FILTER_ALIASES[filter_spec]
    if "*" in filter_spec or "?" in filter_spec:
        return fnmatch.fnmatch(category, filter_spec)
    return category == filter_spec


def _signal_type_for_category(category: str) -> Optional[str]:
    """Force a signal_type based on the YAML group prefix.

    Returns the forced type ('funding' / 'combo'), or None if the entry's own
    signal_type field should be used (technical / advanced / cross_coin / …).
    """
    if category.startswith("funding"):    # funding_signals, funding_baseline, funding_macro_*, funding_mode_*
        return "funding"
    if category.startswith("combo"):       # combo_signals, combo_baseline, combo_advanced, …
        return "combo"
    return None  # technical_signals, advanced_signals, cross_coin_*, advanced_*, … → use entry's signal_type


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
    yaml_path = Path(config.signals)
    if yaml_path.exists():
        yaml_signals = load_signals_from_yaml(config, str(yaml_path), signal_filter)
        if yaml_signals:
            return yaml_signals

    return []


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

    # Get available exit configs for "all" expansion
    available_exits = get_exit_names(config)

    # Iterate every top-level YAML key. Free-form group names are supported:
    # {funding_baseline, funding_macro_fng, cross_coin_triggers, …}. The
    # _signal_type_for_category prefix logic decides which signal_type each
    # group forces (funding/combo) or whether it lets the entry decide.
    for category, entries in data.items():
        if not isinstance(entries, list):
            continue  # skip non-list top-level keys (e.g. metadata blocks)

        # Filter by category if requested (strict equality, legacy alias, or fnmatch wildcard)
        if signal_filter and not _matches_category_filter(category, signal_filter):
            continue

        signal_type_override = _signal_type_for_category(category)

        for sig_template in entries:
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
    PROTECTED_LIST_PARAMS = {"signals", "conditions", "extra_conditions", "btc_regime_allowed"}

    # Extract template fields
    name_template = template.get("name", "unnamed")
    direction = template.get("direction", "both")
    params = template.get("params", {})
    roi = template.get("roi", {"0": 0.02})
    stoploss = template.get("stoploss", -0.03)
    timeframe_override = template.get("timeframe_override")
    allowed_regimes = template.get("allowed_regimes")
    exit_config = template.get("exit_config", "none")
    regime_classifier = template.get("regime_classifier", "v3")
    multi_lookback = template.get("multi_lookback")
    lookback_combine = template.get("lookback_combine", "all")

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

            # Auto-disambiguate: when SL / ROI / exit_config are in the sweep
            # but not referenced in the name template, the resulting strategy
            # class names collide and only the last variant survives in the
            # strategies dir. Append a compact suffix per swept dimension.
            name = _disambiguate_name(
                name, name_template,
                stoploss, len(stoploss_list),
                roi, len(roi_list),
                exit_config, len(exit_list),
            )

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
                regime_classifier=regime_classifier,
                multi_lookback=multi_lookback,
                lookback_combine=lookback_combine,
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
            regime_classifier=regime_classifier,
            multi_lookback=multi_lookback,
            lookback_combine=lookback_combine,
        )
        signals.append(signal)

    return signals


def _short_exit_name(exit_config: str) -> str:
    """Short, unique-friendly suffix for an exit_config name.

    Strips common prefixes (`trailing_roi_` → `tr_`, `zscore_roi_` → `zr_`,
    `atr_roi_` → `ar_`) so names stay readable without colliding (the previous
    `[:8]` slice mapped both `trailing_roi_fixed` and `trailing_roi_2_1` to
    `trailing`, breaking disambiguation).
    """
    if not exit_config or exit_config == "none":
        return "none"
    s = (exit_config
         .replace("trailing_roi_", "tr_")
         .replace("zscore_roi_", "zr_")
         .replace("atr_roi_", "ar_")
         .replace("regime_roi_", "rr_")
         .replace("partial_exit_", "pe_"))
    return s[:12]


def _disambiguate_name(
    name: str,
    name_template: str,
    stoploss: float,
    n_stoploss: int,
    roi: Any,
    n_roi: int,
    exit_config: str,
    n_exit: int,
) -> str:
    """Append disambiguation suffix when SL/ROI/exit sweeps aren't in the name template.

    Without this, a sweep like `stoploss: [-0.03, -0.05, -0.10]` collapses to a
    single class name (because the template doesn't include `{stoploss}`), so
    the generated .py files overwrite each other and only the last variant is
    backtested. We append `_sl{N}` / `_r{N}` / `_x{name}` only for swept dims
    that the user didn't already encode in the template.
    """
    suffixes: List[str] = []

    if n_stoploss > 1 and "{stoploss}" not in name_template:
        sl_pct = int(round(abs(float(stoploss)) * 100))
        suffixes.append(f"sl{sl_pct}")

    if n_roi > 1 and "{roi}" not in name_template:
        if isinstance(roi, dict) and roi:
            first_roi = next(iter(roi.values()))
            roi_pct = int(round(float(first_roi) * 100))
        else:
            roi_pct = int(round(float(roi) * 100))
        suffixes.append(f"r{roi_pct}")

    if n_exit > 1 and "{exit_config}" not in name_template:
        suffixes.append(f"x{_short_exit_name(exit_config)}")

    if suffixes:
        return f"{name}_{'_'.join(suffixes)}"
    return name


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


def get_signal_by_name(name: str, config: Config) -> Optional[SignalConfig]:
    """Get a specific signal by name."""
    all_signals = get_signal_configs(config)
    for signal in all_signals:
        if signal.name == name:
            return signal
    return None
