# =============================================================================
# FILE: lib/report/sections/rankings.py
# =============================================================================
"""Rankings sections."""

import pandas as pd
from ..formatters import print_header
from ..utils import dedup_for_display
from ...utils.helpers import short_pair


def print_top_by_sharpe(df: pd.DataFrame, top_n: int = 25):
    """Print top results by Sharpe ratio (with p-value when available).

    Display dedup: collapse exit siblings via signal_root before nlargest,
    so we don't show 3 near-identical rows for the same (signal, pair, tf)
    differing only in exit. SL/ROI stay distinct (they're separate strats).
    """
    df_disp = dedup_for_display(df, sort_cols="sharpe")
    print_header(f"🏆 TOP {min(top_n, len(df_disp))} GLOBAL PAR SHARPE RATIO")

    top = df_disp.nlargest(top_n, "sharpe")
    has_p = "p_value" in df_disp.columns and df_disp["p_value"].notna().any()

    if has_p:
        print(
            f"\n{'#':<3} {'Signal':<26} {'Pair':<6} {'TF':<4} │ "
            f"{'Tr':<4} {'WR%':<6} {'PnL%':<7} {'Sharpe':<7} {'DD%':<5} │ "
            f"{'Cons%':<5} {'μ_m':<6} {'σ_m':<6} │ {'p':<6}"
        )
        print("─" * 115)
    else:
        print(
            f"\n{'#':<3} {'Signal':<26} {'Pair':<6} {'TF':<4} │ "
            f"{'Tr':<4} {'WR%':<6} {'PnL%':<7} {'Sharpe':<7} {'DD%':<5} │ "
            f"{'Cons%':<5} {'μ_m':<6} {'σ_m':<6}"
        )
        print("─" * 103)

    for i, (_, r) in enumerate(top.iterrows(), 1):
        cons = r.get("consistency", 0)
        mu_m = r.get("avg_month", 0) or 0
        sd_m = r.get("std_month", 0) or 0
        line = (
            f"{i:<3} {r['signal']:<26} {short_pair(r['pair']):<6} {r['timeframe']:<4} │ "
            f"{r['trades']:<4} {r['win_rate']:<6.1f} {r['profit_pct']:<+7.1f} "
            f"{r['sharpe']:<+7.2f} {r['max_dd_pct']:<5.1f} │ "
            f"{cons:<5.0f} {mu_m:<+6.1f} {sd_m:<6.1f}"
        )
        if has_p:
            pv = r.get("p_value")
            if pv is None or pd.isna(pv):
                line += " │  n/a "
            else:
                marker = "*" if pv < 0.05 else ("•" if pv < 0.10 else " ")
                line += f" │ {pv:.3f}{marker}"
        print(line)


def print_polyvalent_signals(df: pd.DataFrame):
    """Print signals performing well across multiple regimes."""
    # This functionality is now integrated into print_signal_regime_matrix
    # Kept for backwards compatibility if called directly
    pass
