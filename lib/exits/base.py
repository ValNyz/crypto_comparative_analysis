# =============================================================================
# FILE: lib/exits/base.py
# =============================================================================
"""Exit configuration dataclass."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ExitConfig:
    """Configuration complète des méthodes de sortie."""

    name: str

    # === Classic exits ===
    use_rsi_exit: bool = False
    rsi_exit_long: int = 75
    rsi_exit_short: int = 25

    use_zscore_exit: bool = False
    zscore_exit_threshold: float = 0.8

    use_bb_exit: bool = False
    bb_exit_long: float = 1.0
    bb_exit_short: float = -1.0

    use_rsi_zscore_combo: bool = False
    combo_rsi_long: int = 65
    combo_rsi_short: int = 35

    use_stoch_exit: bool = False
    stoch_exit_long: int = 80
    stoch_exit_short: int = 20

    # === Funding Rate Exits ===
    use_funding_velocity: bool = False
    funding_vel_period: int = 8
    funding_vel_threshold: float = 0.3

    use_funding_accel: bool = False
    funding_accel_period: int = 4

    use_funding_halflife: bool = False
    halflife_mult: float = 2.0

    use_funding_neutral: bool = False
    funding_neutral_zone: float = 0.5

    use_basis_exit: bool = False
    basis_threshold: float = 0.001

    # === Volatility Exits ===
    use_vol_regime_exit: bool = False
    vol_expansion_mult: float = 1.5

    use_volume_spike_exit: bool = False
    volume_spike_mult: float = 3.0

    # === Divergence Exit ===
    use_rsi_divergence: bool = False
    divergence_lookback: int = 14

    # === Multi-Timeframe ===
    use_mtf_exit: bool = False

    # === Trailing ===
    use_crypto_trail: bool = False
    crypto_trail_mult: float = 2.0
    crypto_trail_period: int = 14

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExitConfig":
        """Create ExitConfig from dictionary."""
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}

    def is_empty(self) -> bool:
        """Check if no exit method is enabled."""
        exit_flags = [
            self.use_rsi_exit,
            self.use_zscore_exit,
            self.use_bb_exit,
            self.use_rsi_zscore_combo,
            self.use_stoch_exit,
            self.use_funding_velocity,
            self.use_funding_accel,
            self.use_funding_neutral,
            self.use_vol_regime_exit,
            self.use_volume_spike_exit,
            self.use_rsi_divergence,
            self.use_crypto_trail,
        ]
        return not any(exit_flags)
