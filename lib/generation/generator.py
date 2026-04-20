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
)
from .templates.standard import STRATEGY_TEMPLATE_STANDARD
from .templates.funding import STRATEGY_TEMPLATE_FUNDING


def _funding_direction_loop(direction: str) -> str:
    long_entry = '("long", zscore <= -threshold, "enter_long")'
    short_entry = '("short", zscore >= threshold, "enter_short")'
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
            # EMA Contra-Trend
            use_ema_contra_filter=signal.params.get("use_ema_contra", False),
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
            # Indicators block
            indicators_block=INDICATORS_BLOCK,
            # Regime detection block
            regime_detection_block=_regime_block_for(signal.regime_classifier),
            # Entry and exit logic
            entry_logic=entry_logic,
            exit_logic=exit_logic,
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
