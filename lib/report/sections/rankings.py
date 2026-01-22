# =============================================================================
# FILE: lib/report/sections/rankings.py
# =============================================================================
"""Rankings sections."""

import pandas as pd
from ..formatters import print_header


def print_top_by_sharpe(df: pd.DataFrame, top_n: int = 25):
    """Print top results by Sharpe ratio."""
    print_header(f"🏆 TOP {min(top_n, len(df))} GLOBAL PAR SHARPE RATIO")

    top = df.nlargest(top_n, "sharpe")

    print(
        f"\n{'#':<3} {'Signal':<26} {'Pair':<16} {'TF':<4} │ "
        f"{'Tr':<4} {'WR%':<6} {'PnL%':<7} {'Sharpe':<7} {'DD%':<5} │ "
        f"{'Cons%':<5} {'Best':<7} {'Worst':<7}"
    )
    print("─" * 115)

    for i, (_, r) in enumerate(top.iterrows(), 1):
        cons = r.get("consistency", 0)
        best_m = r.get("best_month", 0)
        worst_m = r.get("worst_month", 0)
        print(
            f"{i:<3} {r['signal']:<26} {r['pair']:<16} {r['timeframe']:<4} │ "
            f"{r['trades']:<4} {r['win_rate']:<6.1f} {r['profit_pct']:<+7.1f} "
            f"{r['sharpe']:<+7.2f} {r['max_dd_pct']:<5.1f} │ "
            f"{cons:<5.0f} {best_m:<+7.1f} {worst_m:<+7.1f}"
        )


def print_polyvalent_signals(df: pd.DataFrame):
    """Print signals performing well across multiple regimes."""
    # This functionality is now integrated into print_signal_regime_matrix
    # Kept for backwards compatibility if called directly
    pass
