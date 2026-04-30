# =============================================================================
# FILE: lib/report/base.py
# =============================================================================
"""Main report generator class."""

import pandas as pd
from typing import Optional

from ..config.base import Config
from .sections.global_metrics import print_global_metrics
from .sections.regime import (
    print_regime_distribution,
    print_regime_performance,
    print_signal_regime_matrix,
)
from .sections.exit_analysis import print_exit_analysis
from .sections.rankings import print_top_by_sharpe
from .sections.recommendations import print_recommendations
from .sections.coin import (
    print_per_coin_summary,
    print_coin_comparison_matrix,
)
from .sections.winners import print_winners
from .sections.drill_down import print_drill_down
from .sections.blacklist import print_blacklist
from .formatters import print_header
from .utils import add_signal_root


class ReportGenerator:
    """Generates comprehensive backtest reports."""

    def __init__(self, df: pd.DataFrame, config: Optional[Config] = None):
        """
        Initialize the report generator.

        Args:
            df: DataFrame with backtest results
            config: Optional Config instance
        """
        self.df = df
        self.config = config or Config()

    def print_full_report(self, top_n: int = 50, show_regime: bool = False):
        """
        Print the complete report.

        Args:
            top_n: Number of top results to show in rankings
            show_regime: If True, include the three regime sections
                (distribution, performance, signal × regime matrix). Default
                False — those sections are verbose and rarely actionable
                day-to-day; opt in via flag when you need them.
        """
        if len(self.df) == 0:
            print("\n❌ Aucun résultat valide!")
            return

        self._filter_low_frequency()
        if len(self.df) == 0:
            print("\n❌ Tous les résultats ont été filtrés (< trades/mois)!")
            return

        # Add `signal_root` (signal name minus exit suffix) once, so all
        # sections can dedup display by (signal_root, pair, tf) without
        # re-computing it. Idempotent: no-op on later calls.
        self.df = add_signal_root(self.df)

        self._print_header()
        print_global_metrics(self.df)
        # Winners first: surfaces FDR-significant methods immediately so the
        # user doesn't have to triangulate across 6 sections to find them.
        # The Tier-1/Tier-2 + cross-coin + temporal sub-blocks subsume the
        # legacy `print_consistent_performers` (cross-coin) and 2 of the 3
        # legacy `print_temporal_analysis` sub-blocks (top consistency, top
        # robust). Both legacy calls removed below.
        print_winners(self.df, top_n=top_n)
        # Drill-down on top 10 deduped: per-month (with intra-month DD +
        # market change) and per-regime — answers "is it robust or a fluke?"
        print_drill_down(self.df, config=self.config, top_n=10)
        if show_regime:
            print_regime_distribution(self.df)
            print_regime_performance(self.df)
            print_signal_regime_matrix(self.df)
        print_exit_analysis(self.df)
        print_top_by_sharpe(self.df, top_n)
        if "pair" in self.df.columns and self.df["pair"].nunique() > 1:
            print_per_coin_summary(self.df, top_n=5)
            print_coin_comparison_matrix(self.df)
        print_recommendations(self.df, self.config)
        # Blacklist last: actionable cleanup the user pastes back into
        # the YAML grid before the next run. Empty when no signal is bad
        # enough to flag.
        print_blacklist(self.df)

        print(f"\n{'=' * 120}")

    def _filter_low_frequency(self):
        """Drop rows that fail the trade-volume gates before any section runs.

        Two cumulative gates:
          - `min_trades_total` (absolute): a strat with too few trades has
            no statistical mass — Sharpe explodes when std collapses to 0
            on 2-3 nearly-identical trades.
          - `min_trades_per_month` (rate): a strat that fires sporadically
            has no temporal stability even if total count is decent.
        Both must hold. Either gate at 0 disables it.
        """
        rate_threshold = float(getattr(self.config, "min_trades_per_month", 0.0) or 0.0)
        abs_threshold = int(getattr(self.config, "min_trades_total", 0) or 0)
        if rate_threshold <= 0 and abs_threshold <= 0:
            return
        if "trades" not in self.df.columns or "months_total" not in self.df.columns:
            return
        before = len(self.df)
        months = self.df["months_total"].fillna(0).astype(float)
        trades = self.df["trades"].fillna(0).astype(float)
        rate = trades / months.where(months > 0, other=1.0)
        keep = months > 0
        if abs_threshold > 0:
            keep &= trades >= abs_threshold
        if rate_threshold > 0:
            keep &= rate >= rate_threshold
        self.df = self.df[keep].copy()
        dropped = before - len(self.df)
        if dropped > 0:
            parts = []
            if abs_threshold > 0:
                parts.append(f"trades<{abs_threshold}")
            if rate_threshold > 0:
                parts.append(f"<{rate_threshold:g}/mois")
            print(
                f"\n  Filtre volume: {dropped}/{before} strats retirées "
                f"({' OU '.join(parts)})"
            )

    def _print_header(self):
        """Print report header."""
        filter_status = (
            "AVEC FILTRAGE CONDITIONNEL"
            if self.config.enable_regime_filter
            else "SANS FILTRAGE"
        )
        print_header(f"📊 RAPPORT V3 - {filter_status}")

    def get_summary(self) -> dict:
        """
        Get a summary dict of key metrics.

        Returns:
            Dict with summary statistics
        """
        if len(self.df) == 0:
            return {}

        profitable = self.df[self.df["profit_pct"] > 0]

        return {
            "total_results": len(self.df),
            "profitable": len(profitable),
            "profitable_pct": len(profitable) / len(self.df) * 100,
            "sharpe_positive": len(self.df[self.df["sharpe"] > 0]),
            "sharpe_above_1": len(self.df[self.df["sharpe"] > 1]),
            "sharpe_above_2": len(self.df[self.df["sharpe"] > 2]),
            "avg_sharpe": self.df["sharpe"].mean(),
            "avg_profit": self.df["profit_pct"].mean(),
            "best_signal": self.df.loc[self.df["sharpe"].idxmax(), "signal"]
            if len(self.df) > 0
            else None,
            "best_sharpe": self.df["sharpe"].max(),
        }

    def export_summary(self, filepath: str):
        """Export summary to CSV."""
        summary = self.get_summary()
        pd.DataFrame([summary]).to_csv(filepath, index=False)
