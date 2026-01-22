# =============================================================================
# FILE: lib/report/sections/exit_analysis.py
# =============================================================================
"""Exit method analysis section."""

import pandas as pd
from ..formatters import print_header, print_section


def print_exit_analysis(df: pd.DataFrame):
    """Print comparative analysis of exit methods."""
    if "exit_config" not in df.columns:
        return

    print_header("🚪 ANALYSE COMPARATIVE DES MÉTHODES D'EXIT")
    print()

    # Group by exit method
    exit_stats = []
    for exit_method in df["exit_config"].unique():
        subset = df[df["exit_config"] == exit_method]
        if len(subset) > 0:
            exit_stats.append(
                {
                    "method": exit_method,
                    "count": len(subset),
                    "avg_sharpe": subset["sharpe"].mean(),
                    "avg_profit": subset["profit_pct"].mean(),
                    "avg_wr": subset["win_rate"].mean(),
                    "avg_dd": subset["max_dd_pct"].mean(),
                    "profitable": len(subset[subset["profit_pct"] > 0]),
                }
            )

    if not exit_stats:
        return

    exit_df = pd.DataFrame(exit_stats).sort_values("avg_sharpe", ascending=False)

    print(
        f"  {'Exit Method':<20} │ {'Tests':<6} │ {'Sharpe':<8} │ "
        f"{'Profit%':<9} │ {'WR%':<7} │ {'DD%':<6} │ {'Rentables':<10}"
    )
    print("  " + "─" * 90)

    for _, r in exit_df.iterrows():
        pct_profitable = r["profitable"] / r["count"] * 100
        print(
            f"  {r['method']:<20} │ {r['count']:<6} │ {r['avg_sharpe']:<+8.2f} │ "
            f"{r['avg_profit']:<+9.2f} │ {r['avg_wr']:<7.1f} │ {r['avg_dd']:<6.1f} │ "
            f"{r['profitable']}/{r['count']} ({pct_profitable:.0f}%)"
        )

    # Best exit by signal type
    print_section("Meilleure méthode d'exit par type de signal")

    for sig_type in df["signal_type"].unique():
        type_df = df[df["signal_type"] == sig_type]
        if len(type_df) > 0:
            best_exit = type_df.groupby("exit_config")["sharpe"].mean().idxmax()
            best_sharpe = type_df.groupby("exit_config")["sharpe"].mean().max()
            print(f"    {sig_type:<15} → {best_exit:<20} (Sharpe: {best_sharpe:+.2f})")
