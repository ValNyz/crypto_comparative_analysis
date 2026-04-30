# =============================================================================
# FILE: lib/report/sections/winners.py
# =============================================================================
"""Top winners section — surfaces statistically validated best methods.

Designed as the FIRST section the user sees: a focused, decisive ranking that
answers "which methods actually beat random?" without making the reader scan
6 other sections to triangulate. Sorted by FDR-adjusted p-value first, then
Sharpe, so the top entries are both significant and high-performing.
"""

import pandas as pd
from ..formatters import print_header
from ...utils.helpers import short_pair


def print_winners(df: pd.DataFrame, top_n: int = 20):
    """Print the top significance-ranked methods.

    Tier 1: signals with adj p-value < 0.05 (FDR-controlled). These survived
            multiple-testing correction → high confidence not random luck.
    Tier 2: signals with raw p < 0.10 but adj > 0.05. Suggestive but not
            FDR-significant; useful when the test is exploratory.
    Tier 3 (fallback): top-N by Sharpe when null-pool wasn't run.
    """
    if len(df) == 0:
        return

    has_pvals = "p_value_adj" in df.columns and df["p_value_adj"].notna().any()

    print_header("🏆 TOP MÉTHODES — Significativité statistique")

    if has_pvals:
        # Tier 1: FDR-significant
        tier1 = df[(df["p_value_adj"].notna()) & (df["p_value_adj"] < 0.05)]
        tier1 = tier1.sort_values(["p_value_adj", "sharpe"], ascending=[True, False])

        # Tier 2: raw-significant but not FDR-significant
        tier2 = df[
            (df["p_value"].notna())
            & (df["p_value"] < 0.10)
            & ((df["p_value_adj"].isna()) | (df["p_value_adj"] >= 0.05))
        ]
        tier2 = tier2.sort_values(["p_value", "sharpe"], ascending=[True, False])

        if len(tier1) > 0:
            print(
                f"\n  ✅ TIER 1 — FDR-significatifs (p_adj < 0.05) : "
                f"{len(tier1)} signaux"
            )
            _print_block(tier1.head(top_n))
        else:
            print(
                "\n  ⚠️  Aucun signal n'est significatif après correction FDR (p_adj < 0.05)."
            )
            print(
                "     Cela peut signifier (a) signaux marginaux face au null random,"
            )
            print(
                "     (b) trop peu de trades par signal, ou (c) bruit pur.\n"
            )

        if len(tier2) > 0:
            print(
                f"\n  🔶 TIER 2 — Suggestifs raw p<0.10, non FDR-significatifs : "
                f"{min(len(tier2), top_n // 2)} affichés"
            )
            _print_block(tier2.head(top_n // 2))
    else:
        # Fallback: pure Sharpe ranking when null-pool absent
        print(
            "\n  ⚠️  Pas de p-values disponibles (null pool désactivé ou pools manquants)."
        )
        print(f"     Ranking par Sharpe pur :\n")
        top = df.nlargest(top_n, "sharpe")
        _print_block(top, show_pvals=False)


def _print_block(rows: pd.DataFrame, show_pvals: bool = True):
    """Print a block of rows with the standard winner format."""
    if len(rows) == 0:
        return

    if show_pvals:
        header = (
            f"\n  {'#':<3} {'Signal':<30} {'Pair':<6} {'TF':<4} {'Exit':<14} │ "
            f"{'Tr':<4} {'PnL%':<7} {'Sharpe':<7} {'DD%':<5} {'Cons%':<5} │ "
            f"{'p':<6} {'p_adj':<6}"
        )
        sep_w = 130
    else:
        header = (
            f"\n  {'#':<3} {'Signal':<30} {'Pair':<6} {'TF':<4} {'Exit':<14} │ "
            f"{'Tr':<4} {'PnL%':<7} {'Sharpe':<7} {'DD%':<5} {'Cons%':<5}"
        )
        sep_w = 110

    print(header)
    print("  " + "─" * sep_w)

    for i, (_, r) in enumerate(rows.iterrows(), 1):
        exit_cfg = r.get("exit_config", "none") or "none"
        if pd.isna(exit_cfg):
            exit_cfg = "none"
        cons = r.get("consistency", 0) or 0
        line = (
            f"  {i:<3} {r['signal']:<30} {short_pair(r['pair']):<6} "
            f"{r['timeframe']:<4} {str(exit_cfg)[:14]:<14} │ "
            f"{r['trades']:<4d} {r['profit_pct']:<+7.1f} "
            f"{r['sharpe']:<+7.2f} {r['max_dd_pct']:<5.1f} {cons:<5.0f}"
        )
        if show_pvals:
            pv = r.get("p_value")
            pv_adj = r.get("p_value_adj")
            pv_s = f"{pv:.3f}" if pv is not None and not pd.isna(pv) else " n/a "
            adj_s = (
                f"{pv_adj:.3f}"
                if pv_adj is not None and not pd.isna(pv_adj)
                else " n/a "
            )
            marker = ""
            if pv is not None and not pd.isna(pv):
                marker = "*" if pv < 0.05 else ("•" if pv < 0.10 else " ")
            line += f" │ {pv_s}{marker} {adj_s}"
        print(line)
