# =============================================================================
# FILE: lib/report/sections/recommendations.py
# =============================================================================
"""Recommendations section."""

import pandas as pd
from typing import Optional
from ...config.base import Config
from ..formatters import print_header, print_section
from ...utils.helpers import short_pair


def print_recommendations(df: pd.DataFrame, config: Optional[Config] = None):
    """Print recommendations based on analysis."""
    print_header("💡 RECOMMANDATIONS V3")
    print()

    # Best balanced score
    _print_balanced_recommendations(df)

    # Summary by signal type
    _print_signal_type_summary(df)

    # Filtering info
    _print_filtering_info(config)


def _print_balanced_recommendations(df: pd.DataFrame):
    """Print recommendations based on balanced score."""
    df_with_monthly = df[df["months_total"] > 2].copy()

    if len(df_with_monthly) > 0:
        # Calculate balanced score: Performance + Stability + Risk
        df_with_monthly["balanced_score"] = (
            df_with_monthly["sharpe"] * 0.4
            + df_with_monthly["consistency"] / 100 * 0.3
            + (1 - df_with_monthly["max_dd_pct"] / 20) * 0.3
        )

        best_balanced = df_with_monthly.nlargest(5, "balanced_score")
        print("  🎯 MEILLEUR ÉQUILIBRE (Performance + Stabilité + Risque):\n")

        for i, (_, r) in enumerate(best_balanced.iterrows(), 1):
            avg_m = r.get("avg_month", 0)
            print(
                f"     {i}. {r['signal']:<26} ({short_pair(r['pair'])}, {r['timeframe']})\n"
                f"        Sharpe={r['sharpe']:+.2f} │ Consistance={r['consistency']:.0f}% │ "
                f"DD={r['max_dd_pct']:.1f}% │ PnL={r['profit_pct']:+.1f}% │ Avg/mois={avg_m:+.1f} USDC\n"
            )


def _print_signal_type_summary(df: pd.DataFrame):
    """Print summary by signal type."""
    print_section("📈 RÉSUMÉ PAR TYPE DE SIGNAL")

    for sig_type in df["signal_type"].unique():
        type_df = df[df["signal_type"] == sig_type]
        if len(type_df) > 0:
            best = type_df.loc[type_df["sharpe"].idxmax()]
            avg_sharpe = type_df["sharpe"].mean()
            avg_cons = (
                type_df["consistency"].mean() if "consistency" in type_df.columns else 0
            )

            if avg_sharpe > 0.5:
                status = "✅"
            elif avg_sharpe > 0:
                status = "⚪"
            else:
                status = "❌"

            print(
                f"  {status} {sig_type:<12} │ {len(type_df):>3} tests │ "
                f"Sharpe={avg_sharpe:+.2f} Cons={avg_cons:.0f}% │ "
                f"Best: {best['signal'][:22]:<22} ({best['sharpe']:+.2f})"
            )


def _print_filtering_info(config: Optional[Config]):
    """Print information about regime filtering."""
    if config is None:
        return

    if config.enable_regime_filter:
        print("\n  ✅ Filtrage conditionnel ACTIF")
        print("  → Comparer avec --no-filter pour mesurer l'amélioration")
    else:
        print("\n  ℹ️  Filtrage conditionnel INACTIF")
        print("  → Utiliser --enable-filter pour activer le filtrage par régime")
