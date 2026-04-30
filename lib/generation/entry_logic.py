# =============================================================================
# FILE: lib/generation/entry_logic.py
# =============================================================================
"""Entry logic code generation."""

from ..signals.base import SignalConfig
from ..config.loader import load_signal_conditions
from ..config import Config


def generate_entry_logic(
    signal: SignalConfig, enable_filter: bool, config: Config
) -> str:
    """
    Generate entry logic code for a signal.

    Args:
        signal: SignalConfig instance
        enable_filter: Whether regime filtering is enabled

    Returns:
        Python code string for entry logic
    """
    indent = "        "

    # Special handling for combo signals
    if signal.signal_type == "combo":
        return _generate_combo_logic(signal, indent, config)

    # Get condition for this signal type
    condition = _get_signal_condition(signal)
    entry_col = "enter_long" if signal.direction == "long" else "enter_short"

    # Tag layout: <name>_<direction>_<regime>. Direction must be CONTIGUOUS
    # with the regime suffix because the parser regex matches
    # `(long|short)_(bull|bear|range|volatile)` as adjacent tokens
    # (lib/backtest/parser.py:160).
    return f"""{indent}base_cond = {condition} & regime_ok
{indent}for reg in self.ALLOWED_REGIMES:
{indent}    mask = base_cond & (regime == reg)
{indent}    dataframe.loc[mask, '{entry_col}'] = 1
{indent}    dataframe.loc[mask, 'enter_tag'] = f'{signal.name}_{signal.direction}_{{reg}}'"""


def _get_signal_condition(signal: SignalConfig) -> str:
    """Get the condition expression for a signal type."""
    t = signal.params.get("threshold", 30)
    c = signal.params.get("candles", 3)
    dev = signal.params.get("dev", 2.0)
    vol_mult = signal.params.get("vol_mult", 2.0)

    adx_min = signal.params.get("adx_min", 25)
    adx_threshold = signal.params.get("adx_threshold", 25)
    di_strong = signal.params.get("di_strong_threshold", 40)
    di_weak = signal.params.get("di_weak_threshold", 15)
    use_adx_filter = signal.params.get("use_adx_filter", False)
    rsi_max = signal.params.get("rsi_max", 35)
    rsi_min = signal.params.get("rsi_min", 65)
    # P1+P2 new triggers params
    zscore_thr = signal.params.get("zscore", 2.0)
    vwap_zscore_thr = signal.params.get("vwap_zscore_threshold", 2.0)
    trend_thr = signal.params.get("trend_threshold", 1.0)
    trend_chg_thr = signal.params.get("trend_chg_threshold", 0.3)

    # Map of (signal_type, direction) -> condition
    logic_map = {
        ("rsi", "long"): f"(dataframe['rsi_14'] < {t}) & (dataframe['is_green'])",
        ("rsi", "short"): f"(dataframe['rsi_14'] > {t}) & (dataframe['is_red'])",
        (
            "bollinger",
            "long",
        ): f"(dataframe['bb_pos'] < -{t}) & (dataframe['is_green'])",
        ("bollinger", "short"): f"(dataframe['bb_pos'] > {t}) & (dataframe['is_red'])",
        (
            "ema_cross",
            "long",
        ): "(dataframe['ema_8'] > dataframe['ema_21']) & (dataframe['ema_8'].shift(1) <= dataframe['ema_21'].shift(1)) & (dataframe['adx'] > 20)",
        (
            "ema_cross",
            "short",
        ): "(dataframe['ema_8'] < dataframe['ema_21']) & (dataframe['ema_8'].shift(1) >= dataframe['ema_21'].shift(1)) & (dataframe['adx'] > 20)",
        (
            "stochastic",
            "long",
        ): f"(dataframe['stoch_k'] < {t}) & (dataframe['stoch_k'] > dataframe['stoch_d'])",
        (
            "stochastic",
            "short",
        ): f"(dataframe['stoch_k'] > {t}) & (dataframe['stoch_k'] < dataframe['stoch_d'])",
        (
            "macd",
            "long",
        ): "(dataframe['macd'] > dataframe['macd_signal']) & (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1))",
        (
            "macd",
            "short",
        ): "(dataframe['macd'] < dataframe['macd_signal']) & (dataframe['macd'].shift(1) >= dataframe['macd_signal'].shift(1))",
        (
            "reversal",
            "long",
        ): f"(dataframe['consec_red'].shift(1) >= {c}) & (dataframe['is_green']) & (dataframe['rsi_14'] < 45)",
        (
            "reversal",
            "short",
        ): f"(dataframe['consec_green'].shift(1) >= {c}) & (dataframe['is_red']) & (dataframe['rsi_14'] > 55)",
        ("zscore", "long"): f"(dataframe['zscore'] < -{t}) & (dataframe['is_green'])",
        ("zscore", "short"): f"(dataframe['zscore'] > {t}) & (dataframe['is_red'])",
        (
            "multi",
            "long",
        ): "((dataframe['rsi_14'] < 30).astype(int) + (dataframe['bb_pos'] < -1).astype(int) + (dataframe['stoch_k'] < 20).astype(int) + (dataframe['zscore'] < -2).astype(int) >= 3) & (dataframe['is_green'])",
        (
            "multi",
            "short",
        ): "((dataframe['rsi_14'] > 70).astype(int) + (dataframe['bb_pos'] > 1).astype(int) + (dataframe['stoch_k'] > 80).astype(int) + (dataframe['zscore'] > 2).astype(int) >= 3) & (dataframe['is_red'])",
        # Williams %R
        (
            "williams_r",
            "long",
        ): f"(dataframe['willr'] < {t}) & (dataframe['willr'].shift(1) < dataframe['willr'])",
        (
            "williams_r",
            "short",
        ): f"(dataframe['willr'] > {t}) & (dataframe['willr'].shift(1) > dataframe['willr'])",
        # CCI
        (
            "cci",
            "long",
        ): f"(dataframe['cci'] < {t}) & (dataframe['cci'] > dataframe['cci'].shift(1))",
        (
            "cci",
            "short",
        ): f"(dataframe['cci'] > {t}) & (dataframe['cci'] < dataframe['cci'].shift(1))",
        # Squeeze (BB inside Keltner then breakout)
        ("squeeze", "long"): """(
        (dataframe['bb_lower'] > dataframe['kc_lower']) &
        (dataframe['bb_lower'].shift(1) > dataframe['kc_lower'].shift(1)) &
        (dataframe['close'] > dataframe['bb_upper']) &
        (dataframe['macd_hist'] > 0)
    )""",
        ("squeeze", "short"): """(
        (dataframe['bb_upper'] < dataframe['kc_upper']) &
        (dataframe['bb_upper'].shift(1) < dataframe['kc_upper'].shift(1)) &
        (dataframe['close'] < dataframe['bb_lower']) &
        (dataframe['macd_hist'] < 0)
    )""",
        # Donchian Breakout
        (
            "donchian",
            "long",
        ): "(dataframe['close'] > dataframe['donchian_high'].shift(1)) & (dataframe['adx'] > 20)",
        (
            "donchian",
            "short",
        ): "(dataframe['close'] < dataframe['donchian_low'].shift(1)) & (dataframe['adx'] > 20)",
        # VWAP Reversion
        (
            "vwap",
            "long",
        ): f"(dataframe['close'] < dataframe['vwap'] - {abs(dev)} * dataframe['vwap_std']) & (dataframe['is_green'])",
        (
            "vwap",
            "short",
        ): f"(dataframe['close'] > dataframe['vwap'] + {abs(dev)} * dataframe['vwap_std']) & (dataframe['is_red'])",
        # Divergence
        ("divergence", "long"): """(
        (dataframe['close'] < dataframe['close'].shift(5)) &
        (dataframe['rsi_14'] > dataframe['rsi_14'].shift(5)) &
        (dataframe['rsi_14'] < 40)
    )""",
        ("divergence", "short"): """(
        (dataframe['close'] > dataframe['close'].shift(5)) &
        (dataframe['rsi_14'] < dataframe['rsi_14'].shift(5)) &
        (dataframe['rsi_14'] > 60)
    )""",
        # Volume Spike
        ("volume_spike", "long"): f"""(
        (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * {vol_mult}) &
        (dataframe['is_green']) &
        (dataframe['close'] > dataframe['open'] * 1.01)
    )""",
        ("volume_spike", "short"): f"""(
        (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * {vol_mult}) &
        (dataframe['is_red']) &
        (dataframe['close'] < dataframe['open'] * 0.99)
    )""",
        # Keltner
        (
            "keltner",
            "long",
        ): "(dataframe['close'] > dataframe['kc_upper']) & (dataframe['adx'] > 20)",
        (
            "keltner",
            "short",
        ): "(dataframe['close'] < dataframe['kc_lower']) & (dataframe['adx'] > 20)",
        # ROC
        (
            "roc",
            "long",
        ): f"(dataframe['roc'] > {t}) & (dataframe['roc'].shift(1) <= {t})",
        (
            "roc",
            "short",
        ): f"(dataframe['roc'] < {t}) & (dataframe['roc'].shift(1) >= {t})",
        # OI Divergence (placeholder)
        (
            "oi_divergence",
            "long",
        ): "(dataframe['volume'] > 0) & (dataframe['is_green'])",
        ("oi_divergence", "short"): "(dataframe['volume'] > 0) & (dataframe['is_red'])",
        # Liquidation (placeholder)
        (
            "liquidation",
            "long",
        ): "(dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 2) & (dataframe['is_green'])",
        (
            "liquidation",
            "short",
        ): "(dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 2) & (dataframe['is_red'])",
        # =====================================================================
        # DI CROSSOVER (simple, sans filtre ADX)
        # =====================================================================
        ("di_cross", "long"): f"""(
            (dataframe['di_plus'] > dataframe['di_minus']) &
            (dataframe['di_plus'].shift(1) <= dataframe['di_minus'].shift(1))
            {f"& (dataframe['adx'] > {adx_min})" if use_adx_filter else ""}
        )""",
        ("di_cross", "short"): f"""(
            (dataframe['di_minus'] > dataframe['di_plus']) &
            (dataframe['di_minus'].shift(1) <= dataframe['di_plus'].shift(1))
            {f"& (dataframe['adx'] > {adx_min})" if use_adx_filter else ""}
        )""",
        # =====================================================================
        # ADX BREAKOUT (tendance naissante)
        # =====================================================================
        ("adx_breakout", "long"): f"""(
            (dataframe['adx'] > {adx_threshold}) &
            (dataframe['adx'].shift(1) <= {adx_threshold}) &
            (dataframe['di_plus'] > dataframe['di_minus'])
        )""",
        ("adx_breakout", "short"): f"""(
            (dataframe['adx'] > {adx_threshold}) &
            (dataframe['adx'].shift(1) <= {adx_threshold}) &
            (dataframe['di_minus'] > dataframe['di_plus'])
        )""",
        # =====================================================================
        # DI EXTREME (mean reversion sur DI extrêmes)
        # =====================================================================
        ("di_extreme", "long"): f"""(
            (dataframe['di_minus'] > {di_strong}) &
            (dataframe['di_plus'] < {di_weak}) &
            (dataframe['is_green'])
        )""",
        ("di_extreme", "short"): f"""(
            (dataframe['di_plus'] > {di_strong}) &
            (dataframe['di_minus'] < {di_weak}) &
            (dataframe['is_red'])
        )""",
        # =====================================================================
        # DI + RSI COMBO
        # =====================================================================
        ("di_rsi", "long"): f"""(
            (dataframe['di_plus'] > dataframe['di_minus']) &
            (dataframe['adx'] > {adx_min}) &
            (dataframe['rsi_14'] < {rsi_max})
        )""",
        ("di_rsi", "short"): f"""(
            (dataframe['di_minus'] > dataframe['di_plus']) &
            (dataframe['adx'] > {adx_min}) &
            (dataframe['rsi_14'] > {rsi_min})
        )""",
        # =====================================================================
        # P2 / P1: NEW TRIGGERS
        # =====================================================================
        # Ichimoku Kumo cross (rising edge above kumo_top / below kumo_bottom)
        (
            "ichimoku_kumo",
            "long",
        ): "((dataframe['close'] > dataframe['kumo_top']) & (dataframe['close'].shift(1) <= dataframe['kumo_top'].shift(1)))",
        (
            "ichimoku_kumo",
            "short",
        ): "((dataframe['close'] < dataframe['kumo_bottom']) & (dataframe['close'].shift(1) >= dataframe['kumo_bottom'].shift(1)))",
        # Ratio coin/BTC extreme (mean-rev) — needs cross_coin_block
        (
            "ratio_btc_extreme",
            "long",
        ): f"(dataframe['ratio_btc_zscore'] < -{zscore_thr})",
        (
            "ratio_btc_extreme",
            "short",
        ): f"(dataframe['ratio_btc_zscore'] > {zscore_thr})",
        # Ratio coin/BTC breakout (momentum)
        (
            "ratio_btc_breakout",
            "long",
        ): "(dataframe['ratio_btc'] > dataframe['ratio_btc_high_break'])",
        (
            "ratio_btc_breakout",
            "short",
        ): "(dataframe['ratio_btc'] < dataframe['ratio_btc_low_break'])",
        # Ratio coin/ETH extreme + breakout
        (
            "ratio_eth_extreme",
            "long",
        ): f"(dataframe['ratio_eth_zscore'] < -{zscore_thr})",
        (
            "ratio_eth_extreme",
            "short",
        ): f"(dataframe['ratio_eth_zscore'] > {zscore_thr})",
        (
            "ratio_eth_breakout",
            "long",
        ): "(dataframe['ratio_eth'] > dataframe['ratio_eth_high_break'])",
        (
            "ratio_eth_breakout",
            "short",
        ): "(dataframe['ratio_eth'] < dataframe['ratio_eth_low_break'])",
        # Bull climax = bull-side capitulation (top) → only short
        (
            "bull_climax",
            "short",
        ): "((dataframe['volume_ratio'] > 2.5) & (dataframe['ret_1d'] > dataframe['ret_1d'].rolling(24).quantile(0.95)))",
        # Bear climax = bear-side capitulation (bottom) → only long
        (
            "bear_climax",
            "long",
        ): "((dataframe['volume_ratio'] > 2.5) & (dataframe['ret_1d'] < dataframe['ret_1d'].rolling(24).quantile(0.05)))",
        # VWAP intraday z-score — mean-reversion
        (
            "vwap_zscore",
            "long",
        ): f"(dataframe['vwap_zscore'] < -{vwap_zscore_thr})",
        (
            "vwap_zscore",
            "short",
        ): f"(dataframe['vwap_zscore'] > {vwap_zscore_thr})",
        # Trend weakening — strong trend losing momentum (rebound point)
        (
            "trend_weakening",
            "long",
        ): f"((dataframe['trend_strength_c'] < -{trend_thr}) & (dataframe['trend_strength_chg'] > {trend_chg_thr}))",
        (
            "trend_weakening",
            "short",
        ): f"((dataframe['trend_strength_c'] > {trend_thr}) & (dataframe['trend_strength_chg'] < -{trend_chg_thr}))",
    }

    return logic_map.get((signal.signal_type, signal.direction), "False")


def _generate_combo_logic(signal: SignalConfig, indent: str, config: Config) -> str:
    """Generate logic for combo/confluence signals."""
    entry_col = "enter_long" if signal.direction == "long" else "enter_short"

    # Load signal conditions
    signal_conditions = load_signal_conditions(config)

    conditions_parts = []

    # Mode 1: Scoring (min_signals from signals list)
    if signal.params.get("min_signals"):
        scoring_signals = signal.params.get("signals", [])
        score_parts = []
        for s in scoring_signals:
            if s in signal_conditions:
                score_parts.append(f"({signal_conditions[s]}).astype(int)")
        if score_parts:
            conditions_parts.append(
                f"(({' + '.join(score_parts)}) >= {signal.params['min_signals']})"
            )

    # Mode 2: All conditions required
    if signal.params.get("conditions"):
        for cond in signal.params["conditions"]:
            if cond in signal_conditions:
                conditions_parts.append(f"({signal_conditions[cond]})")

    # Extra conditions (mandatory)
    if signal.params.get("extra_conditions"):
        for cond in signal.params["extra_conditions"]:
            if cond in signal_conditions:
                conditions_parts.append(f"({signal_conditions[cond]})")

    # Confirmation candle
    if signal.params.get("confirm"):
        confirm = signal.params["confirm"]
        if confirm in signal_conditions:
            conditions_parts.append(f"({signal_conditions[confirm]})")

    # Always check volume
    conditions_parts.append("(dataframe['volume'] > 0)")

    full_condition = " & ".join(conditions_parts) if conditions_parts else "False"

    return f"""{indent}base_cond = {full_condition} & regime_ok
{indent}for reg in ['bull', 'bear', 'range', 'volatile']:
{indent}    mask = base_cond & (regime == reg)
{indent}    dataframe.loc[mask, '{entry_col}'] = 1
{indent}    dataframe.loc[mask, 'enter_tag'] = f'{signal.name}_{signal.direction}_{{reg}}'"""
