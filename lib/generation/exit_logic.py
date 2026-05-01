# =============================================================================
# FILE: lib/generation/exit_logic.py
# =============================================================================
"""Exit logic code generation."""

from ..exits.base import ExitConfig


def generate_exit_logic(exit_config: ExitConfig, signal_type: str) -> str:
    """
    Generate exit logic code for a strategy.

    Args:
        exit_config: ExitConfig instance
        signal_type: Type of signal (affects which exits are valid)

    Returns:
        Python code string for exit logic
    """
    indent = "        "

    lines = [
        f"{indent}dataframe.loc[:, 'exit_long'] = 0",
        f"{indent}dataframe.loc[:, 'exit_short'] = 0",
        f"{indent}dataframe.loc[:, 'exit_tag'] = ''",
    ]

    exit_long_conds = []
    exit_short_conds = []

    # RSI Exit
    if exit_config.use_rsi_exit:
        exit_long_conds.append(
            (f"dataframe['rsi_14'] >= {exit_config.rsi_exit_long}", "tp_rsi")
        )
        exit_short_conds.append(
            (f"dataframe['rsi_14'] <= {exit_config.rsi_exit_short}", "tp_rsi")
        )

    # Zscore Exit (funding only)
    if exit_config.use_zscore_exit and signal_type == "funding":
        exit_long_conds.append(
            (
                f"dataframe['funding_zscore'] >= {exit_config.zscore_exit_threshold}",
                "zscore_rev",
            )
        )
        exit_short_conds.append(
            (
                f"dataframe['funding_zscore'] <= -{exit_config.zscore_exit_threshold}",
                "zscore_rev",
            )
        )

    # BB Exit
    if exit_config.use_bb_exit:
        exit_long_conds.append(
            (f"dataframe['bb_pos'] >= {exit_config.bb_exit_long}", "bb_tp")
        )
        exit_short_conds.append(
            (f"dataframe['bb_pos'] <= {exit_config.bb_exit_short}", "bb_tp")
        )

    # RSI + Zscore Combo (funding only)
    if exit_config.use_rsi_zscore_combo and signal_type == "funding":
        exit_long_conds.append(
            (
                f"(dataframe['rsi_14'] >= {exit_config.combo_rsi_long}) & (dataframe['funding_zscore'] >= 0)",
                "combo_tp",
            )
        )
        exit_short_conds.append(
            (
                f"(dataframe['rsi_14'] <= {exit_config.combo_rsi_short}) & (dataframe['funding_zscore'] <= 0)",
                "combo_tp",
            )
        )

    # Stoch Exit
    if exit_config.use_stoch_exit:
        exit_long_conds.append(
            (f"dataframe['stoch_k'] >= {exit_config.stoch_exit_long}", "stoch_tp")
        )
        exit_short_conds.append(
            (f"dataframe['stoch_k'] <= {exit_config.stoch_exit_short}", "stoch_tp")
        )

    # Funding Velocity (funding only)
    if exit_config.use_funding_velocity and signal_type == "funding":
        lines.append(
            f"{indent}funding_vel = dataframe['funding_zscore'].diff({exit_config.funding_vel_period})"
        )
        exit_long_conds.append(
            (f"funding_vel > {exit_config.funding_vel_threshold}", "fvel")
        )
        exit_short_conds.append(
            (f"funding_vel < -{exit_config.funding_vel_threshold}", "fvel")
        )

    # Funding Acceleration (funding only)
    if exit_config.use_funding_accel and signal_type == "funding":
        lines.append(
            f"{indent}fvel = dataframe['funding_zscore'].diff({exit_config.funding_accel_period})"
        )
        lines.append(f"{indent}faccel = fvel.diff({exit_config.funding_accel_period})")
        exit_long_conds.append(("(fvel > 0) & (faccel > 0)", "faccel"))
        exit_short_conds.append(("(fvel < 0) & (faccel < 0)", "faccel"))

    # Funding Neutral Zone (funding only)
    if exit_config.use_funding_neutral and signal_type == "funding":
        lines.append(f"\n{indent}# Funding Neutral Zone Exit")
        lines.append(f"{indent}funding_abs = dataframe['funding_zscore'].abs()")
        exit_long_conds.append(
            (f"funding_abs < {exit_config.funding_neutral_zone}", "fneutral")
        )
        exit_short_conds.append(
            (f"funding_abs < {exit_config.funding_neutral_zone}", "fneutral")
        )

    # Vol Regime Exit
    if exit_config.use_vol_regime_exit:
        lines.append(f"\n{indent}# Volatility Regime Exit")
        lines.append(f"{indent}vol_sma = dataframe['atr'].rolling(72).mean()")
        lines.append(
            f"{indent}vol_expansion = dataframe['atr'] > vol_sma * {exit_config.vol_expansion_mult}"
        )
        exit_long_conds.append(("vol_expansion", "vol_exp"))
        exit_short_conds.append(("vol_expansion", "vol_exp"))

    # Volume Spike Exit
    if exit_config.use_volume_spike_exit:
        lines.append(f"\n{indent}# Volume Spike Exit")
        lines.append(f"{indent}vol_sma = dataframe['volume'].rolling(20).mean()")
        lines.append(
            f"{indent}vol_spike = dataframe['volume'] > vol_sma * {exit_config.volume_spike_mult}"
        )
        exit_long_conds.append(
            ("vol_spike & (dataframe['close'] < dataframe['open'])", "volspike")
        )
        exit_short_conds.append(
            ("vol_spike & (dataframe['close'] > dataframe['open'])", "volspike")
        )

    # Crypto Trailing Stop
    if exit_config.use_crypto_trail:
        lines.append(f"\n{indent}# Crypto Trailing Stop")
        lines.append(
            f"{indent}atr_trail = dataframe['atr'].rolling({exit_config.crypto_trail_period}).mean()"
        )
        lines.append(
            f"{indent}trail_long = dataframe['high'].rolling(24).max() - atr_trail * {exit_config.crypto_trail_mult}"
        )
        lines.append(
            f"{indent}trail_short = dataframe['low'].rolling(24).min() + atr_trail * {exit_config.crypto_trail_mult}"
        )
        exit_long_conds.append(("dataframe['close'] < trail_long", "ctrail"))
        exit_short_conds.append(("dataframe['close'] > trail_short", "ctrail"))

    # RSI Divergence Exit
    if exit_config.use_rsi_divergence:
        lb = exit_config.divergence_lookback
        lines.append(f"\n{indent}# RSI Divergence Exit")
        lines.append(
            f"{indent}price_hh = dataframe['close'] > dataframe['close'].rolling({lb}).max().shift(1)"
        )
        lines.append(
            f"{indent}rsi_lh = dataframe['rsi_14'] < dataframe['rsi_14'].rolling({lb}).max().shift(1)"
        )
        lines.append(
            f"{indent}bear_div = price_hh & rsi_lh & (dataframe['rsi_14'] > 60)"
        )
        lines.append(
            f"{indent}price_ll = dataframe['close'] < dataframe['close'].rolling({lb}).min().shift(1)"
        )
        lines.append(
            f"{indent}rsi_hl = dataframe['rsi_14'] > dataframe['rsi_14'].rolling({lb}).min().shift(1)"
        )
        lines.append(
            f"{indent}bull_div = price_ll & rsi_hl & (dataframe['rsi_14'] < 40)"
        )
        exit_long_conds.append(("bear_div", "rsidiv"))
        exit_short_conds.append(("bull_div", "rsidiv"))

    # Generate exit long conditions
    if exit_long_conds:
        lines.append(f"\n{indent}# Exit Long conditions")
        for cond, tag in exit_long_conds:
            lines.append(f"{indent}mask_el = {cond}")
            lines.append(f"{indent}dataframe.loc[mask_el, 'exit_long'] = 1")
            lines.append(f"{indent}dataframe.loc[mask_el, 'exit_tag'] = '{tag}'")

    # Generate exit short conditions
    if exit_short_conds:
        lines.append(f"\n{indent}# Exit Short conditions")
        for cond, tag in exit_short_conds:
            lines.append(f"{indent}mask_es = {cond}")
            lines.append(f"{indent}dataframe.loc[mask_es, 'exit_short'] = 1")
            lines.append(f"{indent}dataframe.loc[mask_es, 'exit_tag'] = '{tag}'")

    return "\n".join(lines)


def generate_custom_exit_method(exit_config: ExitConfig) -> str:
    """
    Emit a `custom_exit()` method body for regime_roi, atr_roi, or zscore_roi.
    Returns "" when none of those flags is set.

    `use_trailing_roi` is handled by native freqtrade `trailing_stop_*` class
    attributes (see _trailing_attrs in generator.py) — no custom_exit needed
    for trailing because the native mechanism evaluates intra-bar via HIGH/LOW
    without lookahead and exits at the stop level instead of OPEN.

    Priority if multiple flags set: regime > atr > zscore.
    YAML should set only one.
    """
    flags = (
        exit_config.use_regime_roi,
        exit_config.use_atr_roi,
        exit_config.use_zscore_roi,
    )
    if not any(flags):
        return ""

    ind = "    "
    lines = [
        f"{ind}def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):",
    ]

    if exit_config.use_regime_roi:
        items = ", ".join(
            f"{k!r}: {v!r}" for k, v in sorted(exit_config.regime_roi_map.items())
        )
        lines += [
            f"{ind}{ind}roi_map = {{{items}}}",
            f"{ind}{ind}fallback = {exit_config.regime_roi_fallback}",
            f"{ind}{ind}tag = trade.enter_tag or ''",
            # Token-scan for direction and regime — order-agnostic vs enter_tag format.
            f"{ind}{ind}_parts = tag.split('_')",
            f"{ind}{ind}direction = next((p for p in _parts if p in ('long', 'short')), None)",
            f"{ind}{ind}regime = next((p for p in _parts if p in ('bull', 'bear', 'range', 'volatile')), None)",
            f"{ind}{ind}if not direction or not regime:",
            f"{ind}{ind}{ind}return None",
            f"{ind}{ind}target = roi_map.get(f'{{direction}}_{{regime}}', fallback)",
            f"{ind}{ind}if current_profit >= target:",
            f"{ind}{ind}{ind}return f'roi_{{direction}}_{{regime}}_{{int(target*10000)}}bps'",
            f"{ind}{ind}return None",
        ]
    elif exit_config.use_atr_roi:
        k = exit_config.atr_roi_k
        floor = exit_config.atr_roi_floor
        cap = exit_config.atr_roi_cap
        lines += [
            f"{ind}{ind}k, floor, cap = {k}, {floor}, {cap}",
            f"{ind}{ind}tag = trade.enter_tag or ''",
            f"{ind}{ind}atr_pct = 0.01",
            f"{ind}{ind}for tok in tag.split('_'):",
            f"{ind}{ind}{ind}if tok.startswith('atr'):",
            f"{ind}{ind}{ind}{ind}try:",
            f"{ind}{ind}{ind}{ind}{ind}atr_pct = float(tok[3:])",
            f"{ind}{ind}{ind}{ind}except ValueError:",
            f"{ind}{ind}{ind}{ind}{ind}pass",
            f"{ind}{ind}{ind}{ind}break",
            f"{ind}{ind}target = max(floor, min(cap, k * atr_pct))",
            f"{ind}{ind}if current_profit >= target:",
            f"{ind}{ind}{ind}return f'roi_atr{{atr_pct:.4f}}_{{int(target*10000)}}bps'",
            f"{ind}{ind}return None",
        ]
    elif exit_config.use_zscore_roi:
        base = exit_config.zscore_roi_base
        slope = exit_config.zscore_roi_slope
        cap = exit_config.zscore_roi_cap
        lines += [
            f"{ind}{ind}base, slope, cap = {base}, {slope}, {cap}",
            f"{ind}{ind}tag = trade.enter_tag or ''",
            f"{ind}{ind}abs_z = 1.0",
            f"{ind}{ind}for tok in tag.split('_'):",
            f"{ind}{ind}{ind}if tok.startswith('z') and not tok.startswith('zs'):",
            f"{ind}{ind}{ind}{ind}try:",
            f"{ind}{ind}{ind}{ind}{ind}abs_z = float(tok[1:])",
            f"{ind}{ind}{ind}{ind}except ValueError:",
            f"{ind}{ind}{ind}{ind}{ind}pass",
            f"{ind}{ind}{ind}{ind}break",
            f"{ind}{ind}target = max(base, min(cap, base + slope * (abs_z - 1.0)))",
            f"{ind}{ind}if current_profit >= target:",
            f"{ind}{ind}{ind}return f'roi_zs{{abs_z:.1f}}_{{int(target*10000)}}bps'",
            f"{ind}{ind}return None",
        ]
    return "\n".join(lines)


def generate_partial_exit_method(exit_config: ExitConfig) -> str:
    """
    Emit an `adjust_trade_position()` method that exits a fraction of the
    position at a target profit, leaving the rest to run until signal/SL.
    Returns "" when use_partial_exit is False.

    Also emits `position_adjustment_enable = True` as a class attribute so
    freqtrade wires up the adjustment path.
    """
    if not exit_config.use_partial_exit:
        return ""

    ind = "    "
    trigger = exit_config.partial_trigger_pct
    frac = exit_config.partial_frac
    lines = [
        f"{ind}position_adjustment_enable = True",
        "",
        f"{ind}def adjust_trade_position(self, trade, current_time, current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, **kwargs):",
        f"{ind}{ind}# Exit {int(frac*100)}% at +{trigger*100:.1f}% profit (once).",
        f"{ind}{ind}if current_profit < {trigger}:",
        f"{ind}{ind}{ind}return None",
        f"{ind}{ind}if getattr(trade, '_partial_done', False):",
        f"{ind}{ind}{ind}return None",
        f"{ind}{ind}filled_stake = trade.stake_amount",
        f"{ind}{ind}exit_stake = -filled_stake * {frac}",
        f"{ind}{ind}try:",
        f"{ind}{ind}{ind}trade._partial_done = True  # type: ignore",
        f"{ind}{ind}except Exception:",
        f"{ind}{ind}{ind}pass",
        f"{ind}{ind}return exit_stake, f'partial_{{int({trigger}*10000)}}bps'",
    ]
    return "\n".join(lines)
