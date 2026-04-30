# =============================================================================
# FILE: lib/report/sections/winners.py
# =============================================================================
"""Top winners section — surfaces statistically validated best methods.

Designed as the FIRST section the user sees: a focused, decisive ranking that
answers "which methods actually beat random?" without making the reader scan
6 other sections to triangulate.

Three views, each answering a different question about robustness:
  • Tier 1 / Tier 2  — statistical significance vs random baseline
  • Cross-coin       — does the same signal_name win on multiple pairs?
  • Temporal         — does it stay profitable across months?
"""

import pandas as pd
from ..formatters import print_header
from ...utils.helpers import short_pair


# Trades floor for ranking display: Sharpe is unreliable below this and the
# bootstrap p-value loses power. 20 matches NULL_POOL_MIN_TRADES.
RANK_MIN_TRADES = 20


def print_winners(df: pd.DataFrame, top_n: int = 50, min_trades: int = RANK_MIN_TRADES):
    """Print the top significance-ranked methods.

    Tier 1: signals with adj p-value < 0.05 (FDR-controlled). UNCAPPED — at
            scale (6k+ candidates), Tier 1 is rare by construction so dumping
            all of them is the right call.
    Tier 2: raw p < 0.10 but adj > 0.05. Capped at top_n.
    Tier 3 (fallback): top-N by Sharpe when null-pool wasn't run.

    Below this, two robustness sub-sections add cross-coin and temporal views.
    """
    if len(df) == 0:
        return

    has_pvals = "p_value_adj" in df.columns and df["p_value_adj"].notna().any()
    # Apply min_trades floor — we don't want to celebrate signals on 5 trades
    df_f = df[df["trades"].fillna(0) >= min_trades] if "trades" in df.columns else df

    print_header(
        f"🏆 TOP MÉTHODES — Significativité statistique (N≥{min_trades} trades)"
    )

    if has_pvals:
        tier1 = df_f[(df_f["p_value_adj"].notna()) & (df_f["p_value_adj"] < 0.05)]
        tier1 = tier1.sort_values(["p_value_adj", "sharpe"], ascending=[True, False])

        tier2 = df_f[
            (df_f["p_value"].notna())
            & (df_f["p_value"] < 0.10)
            & ((df_f["p_value_adj"].isna()) | (df_f["p_value_adj"] >= 0.05))
        ]
        tier2 = tier2.sort_values(["p_value", "sharpe"], ascending=[True, False])

        if len(tier1) > 0:
            print(
                f"\n  ✅ TIER 1 — FDR-significatifs (p_adj < 0.05) : "
                f"{len(tier1)} signaux (tous affichés)"
            )
            _print_block(tier1)
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
                f"{min(len(tier2), top_n)} affichés"
            )
            _print_block(tier2.head(top_n))

        # Robustness sub-sections: only meaningful when p-values exist
        _print_cross_coin_robustness(df_f, top_n=top_n)
        _print_temporal_robustness(df_f, top_n=top_n)
    else:
        # Fallback: pure Sharpe ranking when null-pool absent
        print(
            "\n  ⚠️  Pas de p-values disponibles (null pool désactivé ou pools manquants)."
        )
        print(f"     Ranking par Sharpe pur (N≥{min_trades}) :\n")
        top = df_f.nlargest(top_n, "sharpe")
        _print_block(top, show_pvals=False)


def _print_cross_coin_robustness(df: pd.DataFrame, top_n: int = 50):
    """Signals that win on ≥2 distinct pairs.

    Groups by signal name (without pair). A name appearing on multiple pairs
    with consistent sub-significance (raw p < 0.10 on each) is much more
    likely a real edge than a single-coin coincidence.
    """
    if "signal" not in df.columns or "pair" not in df.columns:
        return
    if "p_value" not in df.columns or not df["p_value"].notna().any():
        return

    sig_df = df[df["p_value"].notna() & (df["p_value"] < 0.10)]
    if len(sig_df) == 0:
        return

    # Group by signal name → set of distinct pairs where it's significant
    grouped = (
        sig_df.groupby("signal")
        .agg(
            n_pairs=("pair", "nunique"),
            mean_sharpe=("sharpe", "mean"),
            mean_pnl=("profit_pct", "mean"),
            mean_pval=("p_value", "mean"),
            best_padj=("p_value_adj", "min"),
            n_total=("pair", "count"),
        )
        .reset_index()
    )
    multi = grouped[grouped["n_pairs"] >= 2].sort_values(
        ["n_pairs", "mean_sharpe"], ascending=[False, False]
    )
    if len(multi) == 0:
        return

    print(
        f"\n  🌍 ROBUSTESSE CROSS-COIN — signaux significatifs sur ≥2 pairs : "
        f"{len(multi)} candidats"
    )
    print(
        f"\n  {'#':<3} {'Signal':<35} {'#Pairs':<7} {'Tests':<6} │ "
        f"{'Avg Sharpe':<11} {'Avg PnL%':<10} {'Avg p':<7} {'Best p_adj':<10}"
    )
    print("  " + "─" * 95)
    for i, (_, r) in enumerate(multi.head(top_n).iterrows(), 1):
        print(
            f"  {i:<3} {r['signal']:<35} {int(r['n_pairs']):<7} "
            f"{int(r['n_total']):<6} │ "
            f"{r['mean_sharpe']:<+11.2f} {r['mean_pnl']:<+10.1f} "
            f"{r['mean_pval']:<7.3f} {r['best_padj']:<10.3f}"
        )


def _print_temporal_robustness(df: pd.DataFrame, top_n: int = 50):
    """Signals consistent across months.

    Combines: raw p<0.10 + consistency ≥ 60% (≥6 of 10 months profitable).
    Sorted by `consistency × Sharpe` so a 90% consistency, 0.5 Sharpe beats
    a 60% consistency, 0.7 Sharpe.
    """
    if "consistency" not in df.columns or "p_value" not in df.columns:
        return
    df_t = df[
        (df["p_value"].notna())
        & (df["p_value"] < 0.10)
        & (df["consistency"] >= 60)
    ].copy()
    if len(df_t) == 0:
        return
    df_t["robust_score"] = df_t["consistency"].astype(float) * df_t["sharpe"]
    df_t = df_t.sort_values("robust_score", ascending=False)

    print(
        f"\n  📅 ROBUSTESSE TEMPORELLE — p<0.10 ET ≥60% mois profitables : "
        f"{len(df_t)} candidats"
    )
    print(
        f"\n  {'#':<3} {'Signal':<30} {'Pair':<6} {'TF':<4} │ "
        f"{'Sharpe':<7} {'Cons%':<6} {'Mois+':<6} {'PnL%':<7} {'p':<6} {'Score':<7}"
    )
    print("  " + "─" * 100)
    for i, (_, r) in enumerate(df_t.head(top_n).iterrows(), 1):
        cons = r.get("consistency", 0) or 0
        m_prof = r.get("months_profitable", 0) or 0
        m_tot = r.get("months_total", 0) or 0
        mois = f"{int(m_prof)}/{int(m_tot)}"
        print(
            f"  {i:<3} {r['signal']:<30} {short_pair(r['pair']):<6} "
            f"{r['timeframe']:<4} │ "
            f"{r['sharpe']:<+7.2f} {cons:<6.0f} {mois:<6} "
            f"{r['profit_pct']:<+7.1f} {r['p_value']:<6.3f} "
            f"{r['robust_score']:<7.1f}"
        )


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
