# =============================================================================
# FILE: lib/report/sections/temporal.py
# =============================================================================
"""Temporal stability analysis section."""

import pandas as pd
from typing import Optional
from ...config.base import Config
from ..formatters import print_header, print_section
from ...utils.helpers import short_pair


def print_temporal_analysis(df: pd.DataFrame, config: Optional[Config] = None):
    """Print temporal stability analysis."""
    min_months = config.min_months if config else 2

    df_monthly = df[df["months_total"] > min_months].copy()
    if len(df_monthly) == 0:
        print_header("📅 ANALYSE TEMPORELLE")
        print()
        print("  ⚠️  Pas assez de données mensuelles pour l'analyse de stabilité")
        return

    print_header("📅 ANALYSE DE STABILITÉ TEMPORELLE (MENSUELLE)")

    # Top by consistency
    _print_top_by_consistency(df_monthly)

    # Stability score
    _print_stability_score(df_monthly)

    # Robust signals
    _print_robust_signals(df_monthly)


def _print_top_by_consistency(df: pd.DataFrame):
    """Print top signals by consistency."""
    print_section("🎯 TOP 15 PAR CONSISTANCE (% de mois profitables)")

    top_consistent = df.nlargest(15, "consistency")

    print(
        f"{'#':<3} {'Signal':<26} {'Pair':<6} │ {'Sharpe':<7} {'PnL%':<7} │ "
        f"{'Mois+':<5} {'Tot':<4} {'Cons%':<6} │ {'Avg/M':<8} {'Std/M':<8} {'Best':<8} {'Worst':<8}"
    )
    print("─" * 105)

    for i, (_, r) in enumerate(top_consistent.iterrows(), 1):
        print(
            f"{i:<3} {r['signal']:<26} {short_pair(r['pair']):<6} │ "
            f"{r['sharpe']:<+7.2f} {r['profit_pct']:<+7.1f} │ "
            f"{r['months_profitable']:<5} {r['months_total']:<4} {r['consistency']:<6.1f} │ "
            f"{r['avg_month']:<+8.2f} {r['std_month']:<8.2f} "
            f"{r['best_month']:<+8.1f} {r['worst_month']:<+8.1f}"
        )


def _print_stability_score(df: pd.DataFrame):
    """Print stability score ranking."""
    # Calculate stability score
    df = df.copy()
    df["stability_score"] = df["sharpe"] / (df["std_month"].abs() + 1)

    print_section("📊 TOP 15 PAR STABILITÉ (Sharpe / Volatilité mensuelle)")

    top_stable = df.nlargest(15, "stability_score")

    print(
        f"{'#':<3} {'Signal':<26} {'Pair':<6} │ {'Sharpe':<7} {'Std/M':<8} "
        f"{'Score':<7} │ {'Cons%':<6} {'Avg/M':<8} {'MinPF':<6}"
    )
    print("─" * 100)

    for i, (_, r) in enumerate(top_stable.iterrows(), 1):
        print(
            f"{i:<3} {r['signal']:<26} {short_pair(r['pair']):<6} │ "
            f"{r['sharpe']:<+7.2f} {r['std_month']:<8.2f} {r['stability_score']:<7.2f} │ "
            f"{r['consistency']:<6.1f} {r['avg_month']:<+8.2f} {r.get('min_monthly_pf', 0):<6.2f}"
        )


def _print_robust_signals(df: pd.DataFrame):
    """Print robust signals meeting all criteria."""
    print_section("🛡️  SIGNAUX ROBUSTES (Consistance≥60%, DD≤10%, Sharpe>0.5)")

    robust = df[
        (df["consistency"] >= 60) & (df["max_dd_pct"] <= 10) & (df["sharpe"] > 0.5)
    ].nlargest(15, "sharpe")

    if len(robust) > 0:
        for _, r in robust.iterrows():
            avg_pf = r.get("avg_monthly_pf", 0)
            print(
                f"  ✅ {r['signal']:<26} ({short_pair(r['pair'])}, {r['timeframe']}) │ "
                f"Sharpe={r['sharpe']:+.2f} DD={r['max_dd_pct']:.1f}% "
                f"Cons={r['consistency']:.0f}% ({r['months_profitable']}/{r['months_total']} mois) "
                f"AvgPF={avg_pf:.2f}"
            )
    else:
        print("  ⚠️  Aucun signal ne remplit tous les critères de robustesse")
