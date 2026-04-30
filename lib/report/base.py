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
from .sections.temporal import print_temporal_analysis
from .sections.rankings import print_top_by_sharpe
from .sections.recommendations import print_recommendations
from .sections.coin import (
    print_per_coin_summary,
    print_coin_comparison_matrix,
    print_consistent_performers,
)
from .sections.winners import print_winners
from .sections.drill_down import print_drill_down
from .formatters import print_header


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

    def print_full_report(self, top_n: int = 50):
        """
        Print the complete report.

        Args:
            top_n: Number of top results to show in rankings
        """
        if len(self.df) == 0:
            print("\n❌ Aucun résultat valide!")
            return

        self._print_header()
        print_global_metrics(self.df)
        # Winners first: surfaces FDR-significant methods immediately so the
        # user doesn't have to triangulate across 6 sections to find them.
        print_winners(self.df, top_n=top_n)
        # Drill-down on top 10 deduped: per-month (with intra-month DD +
        # market change) and per-regime — answers "is it robust or a fluke?"
        print_drill_down(self.df, config=self.config, top_n=10)
        print_regime_distribution(self.df)
        print_regime_performance(self.df)
        print_signal_regime_matrix(self.df)
        print_exit_analysis(self.df)
        print_temporal_analysis(self.df, self.config)
        print_top_by_sharpe(self.df, top_n)
        if "pair" in self.df.columns and self.df["pair"].nunique() > 1:
            print_per_coin_summary(self.df, top_n=5)
            print_coin_comparison_matrix(self.df)
            print_consistent_performers(self.df, min_coins=2)
        print_recommendations(self.df, self.config)

        print(f"\n{'=' * 120}")

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
