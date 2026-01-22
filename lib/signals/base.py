# =============================================================================
# FILE: lib/signals/base.py
# =============================================================================
"""Signal configuration dataclass."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class SignalConfig:
    """Configuration for a trading signal."""

    name: str
    signal_type: str  # funding, rsi, bollinger, ema_cross, etc.
    direction: str  # long, short, both

    # Signal-specific parameters
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            # === Funding de base ===
            "zscore": 1.5,
            "lookback": 168,
            # === RSI Filter ===
            "use_rsi": False,
            "rsi_min": 35,
            "rsi_max": 65,
            # === Volume Filter ===
            "use_volume": False,
            "volume_min": 0.5,
            "volume_max": 3.0,
            # === ATR/Volatility Filter ===
            "use_atr": False,
            "atr_min": 0.2,
            "atr_max": 0.8,
            # === Bollinger Filter ===
            "use_bb": False,
            "bb_long_max": 0.0,
            "bb_short_min": 0.0,
            # === Stochastic Filter ===
            "use_stoch": False,
            "stoch_long_max": 50,
            "stoch_short_min": 50,
            "use_stoch_cross": False,
            # === MACD Filter ===
            "use_macd": False,
            # === Candle Confirmation ===
            "use_candle": False,
            # === Engulfing Pattern ===
            "use_engulfing": False,
            # === Anti-Trend Filter ===
            "use_antitrend": False,
            "adx_max": 30,
            # === EMA Contra-Trend ===
            "use_ema_contra": False,
        }
    )

    # ROI and stoploss
    roi: Dict[str, float] = field(default_factory=lambda: {"0": 0.02})
    stoploss: float = -0.03

    # Timeframe override (None = use default)
    timeframe_override: Optional[str] = None

    # Allowed regimes (None = auto-detect based on name)
    allowed_regimes: Optional[List[str]] = None

    # Exit configuration name
    exit_config: str = "none"

    def __post_init__(self):
        """Auto-detect allowed regimes if not specified."""
        if self.allowed_regimes is None:
            from ..utils.helpers import get_allowed_regimes

            self.allowed_regimes = get_allowed_regimes(self.name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalConfig":
        """Create SignalConfig from dictionary."""
        # Handle nested params
        params = data.get("params", {})

        # Merge with defaults
        default_params = cls.__dataclass_fields__["params"].default_factory()
        merged_params = {**default_params, **params}

        return cls(
            name=data["name"],
            signal_type=data["signal_type"],
            direction=data.get("direction", "both"),
            params=merged_params,
            roi=data.get("roi", {"0": 0.02}),
            stoploss=data.get("stoploss", -0.03),
            timeframe_override=data.get("timeframe_override"),
            allowed_regimes=data.get("allowed_regimes"),
            exit_config=data.get("exit_config", "none"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "params": self.params,
            "roi": self.roi,
            "stoploss": self.stoploss,
            "timeframe_override": self.timeframe_override,
            "allowed_regimes": self.allowed_regimes,
            "exit_config": self.exit_config,
        }

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with default fallback."""
        return self.params.get(key, default)

    def with_exit(self, exit_name: str) -> "SignalConfig":
        """Create a copy with different exit config."""
        return SignalConfig(
            name=f"{self.name}_x{exit_name[:8]}",
            signal_type=self.signal_type,
            direction=self.direction,
            params=self.params.copy(),
            roi={"0": 0.2},  # High ROI to let exits decide
            stoploss=self.stoploss,
            timeframe_override=self.timeframe_override,
            allowed_regimes=self.allowed_regimes,
            exit_config=exit_name,
        )
