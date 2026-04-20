# =============================================================================
# FILE: lib/exits/base.py
# =============================================================================
"""Exit configuration dataclass."""

from dataclasses import dataclass, field
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

    # =========================================================================
    # DYNAMIC ROI (custom_exit-based, orthogonal to signal exits above)
    # =========================================================================
    # Each dynamic-ROI method replaces freqtrade's static minimal_roi with a
    # per-trade target computed at exit-check time. Set only one per ExitConfig.
    # Empty defaults preserve backwards compatibility.

    # Design 1: Regime-indexed ROI. Keys are "<direction>_<regime>"
    # e.g. "long_volatile". Uses fallback when key not found.
    use_regime_roi: bool = False
    regime_roi_map: Dict[str, float] = field(default_factory=dict)
    regime_roi_fallback: float = 0.02

    # Design 2: ATR-scaled ROI. target = clip(k * atr_pct_at_entry, floor, cap).
    # ATR% is carried in entry tag as suffix `_atr<pct:.4f>`.
    use_atr_roi: bool = False
    atr_roi_k: float = 1.5          # multiplier
    atr_roi_floor: float = 0.008    # min 0.8%
    atr_roi_cap: float = 0.04       # max 4%

    # Design 3: |Z|-scaled ROI. target = clip(base + slope*(|z|-1.0), base, cap).
    # |z| is carried in entry tag as suffix `_z<|z|:.2f>`.
    use_zscore_roi: bool = False
    zscore_roi_base: float = 0.01
    zscore_roi_slope: float = 0.005
    zscore_roi_cap: float = 0.03

    # Design 4: Trailing ROI. Arms once profit >= trail_activate_pct;
    # exits when profit drops below (peak - trail_distance_pct).
    # Uses freqtrade's trade.max_rate / trade.min_rate to reconstruct peak.
    use_trailing_roi: bool = False
    trail_activate_pct: float = 0.005
    trail_distance_pct: float = 0.005

    # Design 5: Partial exit. At trigger_pct, exit `partial_frac` of position.
    # Remaining position continues until stoploss/signal-exit. Requires
    # freqtrade `position_adjustment_enable: True` (emitted by generator when
    # this flag is on).
    use_partial_exit: bool = False
    partial_trigger_pct: float = 0.010   # trigger at +1%
    partial_frac: float = 0.5            # exit 50% of stake

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
