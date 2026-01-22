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
