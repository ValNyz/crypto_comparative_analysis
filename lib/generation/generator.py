# =============================================================================
# FILE: lib/generation/generator.py
# =============================================================================
"""Strategy code generator - assembles templates with logic."""

import json
from pathlib import Path
from typing import Tuple

from ..signals.base import SignalConfig
from ..exits.base import ExitConfig
from ..exits.registry import get_exit_config
from ..config.base import Config
from ..utils.helpers import sanitize_class_name

from .templates.base import (
    INDICATORS_BLOCK,
    REGIME_DETECTION_BLOCK,
    REGIME_DETECTION_BLOCK_V4EMA,
    REGIME_DETECTION_BLOCK_V4EMA_SLOPE,
    REGIME_DETECTION_BLOCK_V4EMA_ADX,
    REGIME_DETECTION_BLOCK_V4EMA_ATR,
    REGIME_DETECTION_BLOCK_V4EMA_COMBO,
)
from .templates.standard import STRATEGY_TEMPLATE_STANDARD
from .templates.funding import STRATEGY_TEMPLATE_FUNDING
from .templates.external_block import EXTERNAL_LOADERS_BLOCK
from .templates.cross_coin_block import CROSS_COIN_LOADERS_BLOCK


def _needs_external_block(signal: SignalConfig) -> bool:
    """True if any external-data filter is enabled in signal.params."""
    p = signal.params
    return any(p.get(f) for f in [
        "use_fng", "use_vix", "use_dxy", "use_etf_flow", "use_funding_spread",
    ])


def _needs_cross_coin_block(signal: SignalConfig) -> bool:
    """True if signal needs BTC/ETH OHLCV (regime filter or ratio trigger)."""
    p = signal.params
    if p.get("use_btc_regime") or p.get("use_funding_spread"):
        return True
    return signal.signal_type in {
        "ratio_btc_extreme", "ratio_btc_breakout",
        "ratio_eth_extreme", "ratio_eth_breakout",
    }


def _external_data_dir(data_dir: str) -> str:
    """Auto-derive external data path from data_dir.

    Layout: data_dir = <root>/user_data/data/hyperliquid → external = <root>/user_data/data/external.
    """
    return str(Path(data_dir).parent / "external")


def _funding_extra_lookbacks_literal(primary_lookback: int, multi_lookback) -> str:
    """Render the EXTRA_LOOKBACKS class-attribute value as a Python literal.

    Extras exclude the primary lookback; the template always includes the primary
    via `funding_zscore`. Returns '[]' when no multi-lookback is configured.
    """
    if not multi_lookback:
        return "[]"
    primary = int(primary_lookback)
    extras = [int(x) for x in multi_lookback if int(x) != primary]
    return repr(extras)


def _funding_direction_loop(direction: str) -> str:
    """Direction-loop literal for the funding template.

    Uses base_long / base_short which are dispatched by FUNDING_MODE in
    populate_entry_trend (zscore | flip | cum_7d). The template's reduction
    block sets zscore_long == zscore_short when EXTRA_LOOKBACKS is empty,
    preserving single-lookback behavior in mode 'zscore'.
    """
    long_entry = '("long", base_long, "enter_long")'
    short_entry = '("short", base_short, "enter_short")'
    if direction == "long":
        entries = [long_entry]
    elif direction == "short":
        entries = [short_entry]
    else:
        entries = [long_entry, short_entry]
    return "[" + ", ".join(entries) + "]"
from .entry_logic import generate_entry_logic
from .exit_logic import (
    generate_exit_logic,
    generate_custom_exit_method,
    generate_partial_exit_method,
)


_REGIME_BLOCKS = {
    "v3": REGIME_DETECTION_BLOCK,
    "v4ema": REGIME_DETECTION_BLOCK_V4EMA,
    "v4ema_slope": REGIME_DETECTION_BLOCK_V4EMA_SLOPE,
    "v4ema_adx": REGIME_DETECTION_BLOCK_V4EMA_ADX,
    "v4ema_atr": REGIME_DETECTION_BLOCK_V4EMA_ATR,
    "v4ema_combo": REGIME_DETECTION_BLOCK_V4EMA_COMBO,
}


def _regime_block_for(classifier: str) -> str:
    """Return the regime-detection code block matching a classifier name.

    Unknown names fall back to 'v3' (ADX/ATR) so the generator stays
    strictly backward-compatible.
    """
    return _REGIME_BLOCKS.get(classifier, REGIME_DETECTION_BLOCK)


class StrategyGenerator:
    """Generates Python strategy files from SignalConfig."""

    def __init__(self, config: Config):
        """
        Initialize the generator.

        Args:
            config: Config instance with paths and parameters
        """
        self.config = config
        self.output_dir = Path(config.strategies_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, signal: SignalConfig, timeframe: str) -> Tuple[str, Path]:
        """
        Generate a strategy file from a SignalConfig.

        Args:
            signal: SignalConfig instance
            timeframe: Timeframe to use (can be overridden by signal)

        Returns:
            Tuple of (class_name, file_path)
        """
        # Determine actual timeframe
        actual_tf = signal.timeframe_override or timeframe

        # Generate class name
        class_name = f"S_{sanitize_class_name(signal.name)}_{actual_tf}"

        # Get exit config
        exit_cfg = get_exit_config(signal.exit_config, self.config)

        # Generate code based on signal type
        if signal.signal_type == "funding":
            code = self._generate_funding_strategy(
                signal, exit_cfg, actual_tf, class_name
            )
        else:
            code = self._generate_standard_strategy(
                signal, exit_cfg, actual_tf, class_name
            )

        # Write file
        filepath = self.output_dir / f"{class_name}.py"
        filepath.write_text(code)

        return class_name, filepath

    def _generate_funding_strategy(
        self,
        signal: SignalConfig,
        exit_cfg: ExitConfig,
        timeframe: str,
        class_name: str,
    ) -> str:
        """Generate a funding contrarian strategy."""

        # Generate exit logic
        exit_logic = generate_exit_logic(exit_cfg, "funding")

        return STRATEGY_TEMPLATE_FUNDING.format(
            name=signal.name,
            class_name=class_name,
            timeframe=timeframe,
            roi=json.dumps({str(k): v for k, v in signal.roi.items()}),
            stoploss=signal.stoploss,
            # Funding parameters
            zscore_threshold=signal.params.get("zscore", 1.5),
            funding_lookback=signal.params.get("lookback", 168),
            # RSI Filter
            use_rsi_filter=signal.params.get("use_rsi", False),
            rsi_min=signal.params.get("rsi_min", 35),
            rsi_max=signal.params.get("rsi_max", 65),
            # Volume Filter
            use_volume_filter=signal.params.get("use_volume", False),
            volume_min=signal.params.get("volume_min", 0.5),
            volume_max=signal.params.get("volume_max", 3.0),
            # ATR Filter
            use_atr_filter=signal.params.get("use_atr", False),
            atr_min=signal.params.get("atr_min", 0.2),
            atr_max=signal.params.get("atr_max", 0.8),
            # Bollinger Filter
            use_bb_filter=signal.params.get("use_bb", False),
            bb_long_max=signal.params.get("bb_long_max", 0.0),
            bb_short_min=signal.params.get("bb_short_min", 0.0),
            # Stochastic Filter
            use_stoch_filter=signal.params.get("use_stoch", False),
            stoch_long_max=signal.params.get("stoch_long_max", 50),
            stoch_short_min=signal.params.get("stoch_short_min", 50),
            use_stoch_cross_filter=signal.params.get("use_stoch_cross", False),
            # MACD Filter
            use_macd_filter=signal.params.get("use_macd", False),
            # Candle Confirmation
            use_candle_filter=signal.params.get("use_candle", False),
            # Engulfing Pattern
            use_engulfing_filter=signal.params.get("use_engulfing", False),
            # Anti-Trend Filter
            use_antitrend_filter=signal.params.get("use_antitrend", False),
            adx_max=signal.params.get("adx_max", 30),
            # Pro-Trend Confirmation (B3: MTF-confluence proxy via strong ADX + DI direction)
            use_adx_min_filter=signal.params.get("use_adx_min", False),
            adx_min_threshold=signal.params.get("adx_min", 25),
            # EMA Contra-Trend
            use_ema_contra_filter=signal.params.get("use_ema_contra", False),
            # Inter-coin divergence filter (E14)
            use_intercoin_filter=signal.params.get("use_intercoin", False),
            intercoin_ref=signal.params.get("intercoin_ref", "BTC"),
            intercoin_neutral_threshold=signal.params.get("intercoin_neutral_threshold", 0.5),
            # Funding velocity filter (B4)
            use_velocity_filter=signal.params.get("use_velocity", False),
            velocity_period=signal.params.get("velocity_period", 4),
            velocity_zscore_min=signal.params.get("velocity_zscore_min", 0.5),
            velocity_revert=signal.params.get("velocity_revert", True),
            # Regime-transition filter (C7)
            use_transition_filter=signal.params.get("use_transition", False),
            transition_prev_regime=signal.params.get("transition_prev_regime", ""),
            transition_window=signal.params.get("transition_window", 1),
            # Hour-of-day filter (C10)
            use_hour_filter=signal.params.get("use_hour_filter", False),
            hour_window_start=signal.params.get("hour_window_start", 0),
            hour_window_end=signal.params.get("hour_window_end", 23),
            # Regime parameters
            regime_lookback=self.config.regime_lookback,
            regime_adx_threshold=self.config.regime_adx_threshold,
            regime_adx_strong=self.config.regime_adx_strong,
            regime_atr_volatile=self.config.regime_atr_volatile,
            regime_atr_low=self.config.regime_atr_low,
            allowed_regimes=signal.allowed_regimes,
            enable_filter=self.config.enable_regime_filter,
            # Data directory
            data_dir=self.config.data_dir,
            # Regime detection block
            regime_detection_block=_regime_block_for(signal.regime_classifier),
            # Entry-direction loop (respects signal.direction)
            direction_loop=_funding_direction_loop(signal.direction),
            # Exit logic
            exit_logic=exit_logic,
            # Dynamic-ROI methods (empty strings when disabled → no code emitted)
            custom_exit_method=generate_custom_exit_method(exit_cfg),
            partial_exit_method=generate_partial_exit_method(exit_cfg),
            # Multi-lookback support (empty list → single-lookback behavior)
            extra_lookbacks_list=_funding_extra_lookbacks_literal(
                signal.params.get("lookback", 168),
                signal.multi_lookback,
            ),
            lookback_combine=signal.lookback_combine,
            # External & cross-coin block injection (conditional)
            external_loaders_block=(
                EXTERNAL_LOADERS_BLOCK.format(external_data_dir=_external_data_dir(self.config.data_dir))
                if _needs_external_block(signal) else ""
            ),
            cross_coin_block=(CROSS_COIN_LOADERS_BLOCK.format() if _needs_cross_coin_block(signal) else ""),
            # New macro filter flags
            use_fng_filter=signal.params.get("use_fng", False),
            fng_fear=signal.params.get("fng_fear", 25),
            fng_greed=signal.params.get("fng_greed", 75),
            fng_consec_days=signal.params.get("fng_consec", 3),
            use_vix_filter=signal.params.get("use_vix", False),
            vix_max_long=signal.params.get("vix_max_long", 25),
            vix_min_short=signal.params.get("vix_min_short", 25),
            use_dxy_filter=signal.params.get("use_dxy", False),
            dxy_slope_max_long=signal.params.get("dxy_slope_max_long", 0.01),
            dxy_slope_min_short=signal.params.get("dxy_slope_min_short", 0.01),
            use_etf_flow_filter=signal.params.get("use_etf_flow", False),
            etf_ref=signal.params.get("etf_ref", "btc"),
            etf_inflow_min_long=signal.params.get("etf_inflow_min_long", 200),
            etf_outflow_max_short=signal.params.get("etf_outflow_max_short", 200),
            use_funding_spread_filter=signal.params.get("use_funding_spread", False),
            spread_ref=signal.params.get("spread_ref", "BTC"),
            spread_z_threshold=signal.params.get("spread_z_threshold", 2.0),
            use_btc_regime_filter=signal.params.get("use_btc_regime", False),
            btc_regime_allowed=signal.params.get("btc_regime_allowed", ["bull", "range"]),
            # Volume-zscore + BBW-squeeze filters (non-directional)
            use_volume_zscore_filter=signal.params.get("use_volume_zscore", False),
            volume_zscore_min=signal.params.get("volume_zscore_min", 2.5),
            use_bbw_squeeze_filter=signal.params.get("use_bbw_squeeze", False),
            bbw_pct_max=signal.params.get("bbw_pct_max", 0.15),
            # Funding mode dispatch
            funding_mode=signal.params.get("funding_mode", "zscore"),
        )

    def _generate_standard_strategy(
        self,
        signal: SignalConfig,
        exit_cfg: ExitConfig,
        timeframe: str,
        class_name: str,
    ) -> str:
        """Generate a standard technical strategy."""

        # Generate entry and exit logic
        entry_logic = generate_entry_logic(
            signal, self.config.enable_regime_filter, self.config
        )
        exit_logic = generate_exit_logic(exit_cfg, signal.signal_type)

        # Determine if trailing stop should be used
        trailing_stop = "True" if signal.signal_type == "ema_cross" else "False"

        return STRATEGY_TEMPLATE_STANDARD.format(
            name=signal.name,
            class_name=class_name,
            timeframe=timeframe,
            roi=json.dumps({str(k): v for k, v in signal.roi.items()}),
            stoploss=signal.stoploss,
            trailing_stop=trailing_stop,
            # Regime parameters
            regime_lookback=self.config.regime_lookback,
            regime_adx_threshold=self.config.regime_adx_threshold,
            regime_adx_strong=self.config.regime_adx_strong,
            regime_atr_volatile=self.config.regime_atr_volatile,
            regime_atr_low=self.config.regime_atr_low,
            allowed_regimes=signal.allowed_regimes,
            enable_filter=self.config.enable_regime_filter,
            # ATR Filter (post-regime gate, default off)
            use_atr_filter=signal.params.get("use_atr", False),
            atr_min=signal.params.get("atr_min", 0.2),
            atr_max=signal.params.get("atr_max", 0.8),
            # Indicators block
            indicators_block=INDICATORS_BLOCK,
            # Regime detection block
            regime_detection_block=_regime_block_for(signal.regime_classifier),
            # Entry and exit logic
            entry_logic=entry_logic,
            exit_logic=exit_logic,
            # External & cross-coin block injection (conditional)
            external_loaders_block=(
                EXTERNAL_LOADERS_BLOCK.format(external_data_dir=_external_data_dir(self.config.data_dir))
                if _needs_external_block(signal) else ""
            ),
            cross_coin_block=(CROSS_COIN_LOADERS_BLOCK.format() if _needs_cross_coin_block(signal) else ""),
            data_dir=self.config.data_dir,
        )

    def generate_batch(self, signals: list, timeframe: str) -> list:
        """
        Generate multiple strategies.

        Args:
            signals: List of SignalConfig instances
            timeframe: Default timeframe

        Returns:
            List of (class_name, filepath) tuples
        """
        results = []
        for signal in signals:
            try:
                result = self.generate(signal, timeframe)
                results.append(result)
            except Exception as e:
                print(f"Error generating {signal.name}: {e}")
        return results

    def clean_generated(self):
        """Remove all generated strategy files."""
        for filepath in self.output_dir.glob("S_*.py"):
            filepath.unlink()
