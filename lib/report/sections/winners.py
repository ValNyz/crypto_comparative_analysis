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
from ..utils import strip_exit_suffix, dedup_for_display


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
    if "signal_root" not in df_f.columns:
        df_f = df_f.copy()
        df_f["signal_root"] = df_f.apply(
            lambda r: strip_exit_suffix(str(r["signal"]), r.get("exit_config")),
            axis=1,
        )

    print_header(
        f"🏆 TOP MÉTHODES — Significativité statistique (N≥{min_trades} trades)"
    )

    if has_pvals:
        # Total variants per (signal_root, pair) — denominator of n_var ratio.
        # Collapses both TF AND exit siblings: the displayed row keeps the
        # best (TF, exit) by sort order ; n_var = X/Y where Y is total variants
        # tested for this (signal_root, pair) combo (across all TFs and exits).
        total_var = (
            df_f.groupby(["signal_root", "pair"]).size().rename("n_total")
            if {"signal_root", "pair"}.issubset(df_f.columns)
            else None
        )

        tier1 = df_f[(df_f["p_value_adj"].notna()) & (df_f["p_value_adj"] < 0.05)]
        # Tiebreak by Calmar (risk-adjusted return) instead of Sharpe — under
        # fixed stake, Sharpe is diluted by idle capital but Calmar stays valid.
        tier1 = tier1.sort_values(["p_value_adj", "calmar"], ascending=[True, False])
        tier1 = _dedup_with_count(tier1, total_var)

        tier2 = df_f[
            (df_f["p_value"].notna())
            & (df_f["p_value"] < 0.10)
            & ((df_f["p_value_adj"].isna()) | (df_f["p_value_adj"] >= 0.05))
        ]
        tier2 = tier2.sort_values(["p_value", "calmar"], ascending=[True, False])
        tier2 = _dedup_with_count(tier2, total_var)

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
        # Same dedup as Tier 1/2: collapse TF + exit siblings.
        df_pool = dedup_for_display(df_f, sort_cols="sharpe")
        top = df_pool.nlargest(top_n, "sharpe")
        _print_block(top, show_pvals=False)


def _dedup_with_count(
    sub: pd.DataFrame, total_var
) -> pd.DataFrame:
    """Keep best (already pre-sorted) row per (signal_root, pair) and tag n_var.

    Collapses BOTH TF and exit siblings — input is pre-sorted by criterion
    (e.g., p_adj asc), so the kept row carries the best (TF, exit) for that
    (signal_root, pair). `n_var` = "X/Y" where X = FDR-sig variants of the
    combo, Y = total variants tested (across all TFs and exits). 9/9 means
    every TF×exit variant of this combo passed FDR → very robust ; 2/9 means
    fragile, dependent on a specific (TF, exit) pair.
    """
    if len(sub) == 0 or not {"signal_root", "pair"}.issubset(sub.columns):
        sub = sub.copy()
        sub["n_var"] = ""
        return sub
    n_sig = sub.groupby(["signal_root", "pair"]).size().rename("_n_sig")
    deduped = sub.drop_duplicates(
        subset=["signal_root", "pair"], keep="first"
    ).copy()
    deduped = deduped.merge(n_sig, on=["signal_root", "pair"], how="left")
    if total_var is not None:
        deduped = deduped.merge(
            total_var, on=["signal_root", "pair"], how="left"
        )
        deduped["n_var"] = deduped.apply(
            lambda r: f"{int(r['_n_sig'])}/{int(r['n_total'])}", axis=1
        )
        deduped = deduped.drop(columns=["_n_sig", "n_total"])
    else:
        deduped["n_var"] = deduped["_n_sig"].astype(int).astype(str)
        deduped = deduped.drop(columns=["_n_sig"])
    return deduped


def _print_cross_coin_robustness(df: pd.DataFrame, top_n: int = 50):
    """Strategies that win on >=2 distinct pairs.

    A "strategy" here = `signal_root` (signal name minus exit suffix). Exit
    siblings of the same core signal are collapsed since their results are
    often identical (trail never fires) and showing them as separate rows
    is just noise. The displayed exit / SL come from the best (lowest p_adj)
    representative of the group. Aggregates: mean Calmar / mean PnL across
    the surviving (pair, tf) rows ; best (lowest) p_adj.
    """
    if "signal_root" not in df.columns or "pair" not in df.columns:
        return
    if "p_value" not in df.columns or not df["p_value"].notna().any():
        return

    sig_df = df[df["p_value"].notna() & (df["p_value"] < 0.10)].copy()
    if len(sig_df) == 0:
        return

    grouped = (
        sig_df.groupby(["signal_root"], dropna=False)
        .agg(
            n_pairs=("pair", "nunique"),
            mean_calmar=("calmar", "mean"),
            mean_pnl=("profit_pct", "mean"),
            best_padj=("p_value_adj", "min"),
            n_total=("pair", "count"),
        )
        .reset_index()
    )
    multi = grouped[grouped["n_pairs"] >= 2].sort_values(
        ["n_pairs", "mean_calmar"], ascending=[False, False]
    )
    if len(multi) == 0:
        return

    # Per (signal_root, pair) best TF — lowest p_value_adj wins.
    best_tf = (
        sig_df.sort_values("p_value_adj", ascending=True)
        .drop_duplicates(subset=["signal_root", "pair"], keep="first")
    )
    # Per signal_root: pick the best (lowest p_adj) row for the displayed
    # exit/SL representative of the group.
    best_repr = (
        sig_df.sort_values("p_value_adj", ascending=True)
        .drop_duplicates(subset=["signal_root"], keep="first")
        .set_index("signal_root")
    )

    def _tf_breakdown(signal_root: str) -> str:
        """E.g. 'BTC(4h), ENA(1h), SOL(30m)' for a strat winning on 3 coins."""
        rows = best_tf[best_tf["signal_root"] == signal_root].sort_values(
            "p_value_adj", ascending=True
        )
        return ", ".join(
            f"{short_pair(str(r['pair']))}({r['timeframe']})"
            for _, r in rows.iterrows()
        )

    print(
        f"\n  🌍 ROBUSTESSE CROSS-COIN — signal_root sur >=2 pairs : "
        f"{len(multi)} candidats. Calmar/PnL = MOYENNE inter-paires, "
        f"best_p_adj = paire la plus sig."
    )
    print(
        f"\n  {'#':<3} {'Signal_root':<28} {'Best Exit':<14} {'SL':<5} {'#':<3} │ "
        f"{'Calmar':<7} {'PnL%':<8} {'Best p_adj':<11} │ Best TF par coin"
    )
    print("  " + "─" * 130)
    for i, (_, r) in enumerate(multi.head(top_n).iterrows(), 1):
        sr = r["signal_root"]
        breakdown = _tf_breakdown(sr)
        repr_row = best_repr.loc[sr] if sr in best_repr.index else None
        if repr_row is not None:
            sl_val = repr_row.get("stoploss")
            sl_str = (
                f"{abs(int(round(float(sl_val) * 100)))}%"
                if pd.notna(sl_val) else "  -  "
            )
            exit_str = str(repr_row.get("exit_config", "none"))[:14]
        else:
            sl_str, exit_str = "  -  ", "none"
        print(
            f"  {i:<3} {str(sr)[:28]:<28} {exit_str:<14} "
            f"{sl_str:<5} {int(r['n_pairs']):<3} │ "
            f"{r['mean_calmar']:<+7.2f} {r['mean_pnl']:<+8.1f} "
            f"{r['best_padj']:<11.4f} │ {breakdown}"
        )


def _print_temporal_robustness(df: pd.DataFrame, top_n: int = 50):
    """Signals consistent across months.

    Combines: raw p<0.10 + consistency ≥ 60% (≥6 of 10 months profitable).
    Sorted by `consistency × Sharpe` so a 90% consistency, 0.5 Sharpe beats
    a 60% consistency, 0.7 Sharpe. Display dedup: same exit-suffix collapse
    as Tier 1 — keep best (robust_score) per (signal_root, pair, tf).
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
    # Score = consistency × Calmar (risk-adjusted return matters more than
    # raw Sharpe under fixed stake; multiplied by mois-profitables ratio).
    df_t["robust_score"] = df_t["consistency"].astype(float) * df_t["calmar"]
    df_t = df_t.sort_values("robust_score", ascending=False)
    # Dedup display: collapse TF + exit siblings, keep best robust_score
    if "signal_root" in df_t.columns:
        df_t = df_t.drop_duplicates(
            subset=["signal_root", "pair"], keep="first"
        )

    print(
        f"\n  📅 ROBUSTESSE TEMPORELLE — p<0.10 ET ≥60% mois profitables : "
        f"{len(df_t)} candidats"
    )
    print(
        f"\n  {'#':<3} {'Signal':<30} {'Pair':<6} {'TF':<4} │ "
        f"{'Calmar':<7} {'Cons%':<6} {'Mois+':<6} {'PnL%':<7} "
        f"{'μ_m':<6} {'σ_m':<6} "
        f"{'p':<6} {'p_adj':<6} {'q':<6} {'Score':<7}"
    )
    print("  " + "─" * 131)
    for i, (_, r) in enumerate(df_t.head(top_n).iterrows(), 1):
        cons = r.get("consistency", 0) or 0
        m_prof = r.get("months_profitable", 0) or 0
        m_tot = r.get("months_total", 0) or 0
        mois = f"{int(m_prof)}/{int(m_tot)}"
        pv_adj = r.get("p_value_adj")
        adj_s = (
            f"{pv_adj:.3f}"
            if pv_adj is not None and not pd.isna(pv_adj)
            else " n/a "
        )
        qv = r.get("q_value")
        q_s = f"{qv:.3f}" if qv is not None and not pd.isna(qv) else " n/a "
        mu_m = r.get("avg_month", 0) or 0
        sd_m = r.get("std_month", 0) or 0
        cal = r.get("calmar", 0) or 0
        print(
            f"  {i:<3} {r['signal']:<30} {short_pair(r['pair']):<6} "
            f"{r['timeframe']:<4} │ "
            f"{cal:<+7.2f} {cons:<6.0f} {mois:<6} "
            f"{r['profit_pct']:<+7.1f} {mu_m:<+6.1f} {sd_m:<6.1f} "
            f"{r['p_value']:<6.3f} {adj_s:<6} {q_s:<6} "
            f"{r['robust_score']:<7.1f}"
        )


def _print_block(rows: pd.DataFrame, show_pvals: bool = True):
    """Print a block of rows with the standard winner format."""
    if len(rows) == 0:
        return

    if show_pvals:
        header = (
            f"\n  {'#':<3} {'Signal':<30} {'Pair':<6} {'TF':<4} {'Exit':<14} │ "
            f"{'Tr':<4} {'PnL%':<7} {'Calmar':<7} {'PF':<6} {'DD%':<5} {'Cons%':<5} "
            f"{'μ_m':<6} {'σ_m':<6} │ "
            f"{'p':<6} {'p_adj':<6} {'q':<6} {'n_var':<6}"
        )
        sep_w = 165
    else:
        header = (
            f"\n  {'#':<3} {'Signal':<30} {'Pair':<6} {'TF':<4} {'Exit':<14} │ "
            f"{'Tr':<4} {'PnL%':<7} {'Calmar':<7} {'PF':<6} {'DD%':<5} {'Cons%':<5} "
            f"{'μ_m':<6} {'σ_m':<6}"
        )
        sep_w = 131

    print(header)
    print("  " + "─" * sep_w)

    for i, (_, r) in enumerate(rows.iterrows(), 1):
        exit_cfg = r.get("exit_config", "none") or "none"
        if pd.isna(exit_cfg):
            exit_cfg = "none"
        cons = r.get("consistency", 0) or 0
        mu_m = r.get("avg_month", 0) or 0
        sd_m = r.get("std_month", 0) or 0
        cal = r.get("calmar", 0) or 0
        pf = r.get("profit_factor", 0) or 0
        pf_s = f"{pf:<6.2f}" if pf != float("inf") else "  inf"
        line = (
            f"  {i:<3} {r['signal']:<30} {short_pair(r['pair']):<6} "
            f"{r['timeframe']:<4} {str(exit_cfg)[:14]:<14} │ "
            f"{r['trades']:<4d} {r['profit_pct']:<+7.1f} "
            f"{cal:<+7.2f} {pf_s} {r['max_dd_pct']:<5.1f} {cons:<5.0f} "
            f"{mu_m:<+6.1f} {sd_m:<6.1f}"
        )
        if show_pvals:
            pv = r.get("p_value")
            pv_adj = r.get("p_value_adj")
            qv = r.get("q_value")
            pv_s = f"{pv:.3f}" if pv is not None and not pd.isna(pv) else " n/a "
            adj_s = (
                f"{pv_adj:.3f}"
                if pv_adj is not None and not pd.isna(pv_adj)
                else " n/a "
            )
            q_s = (
                f"{qv:.3f}" if qv is not None and not pd.isna(qv) else " n/a "
            )
            marker = ""
            if pv is not None and not pd.isna(pv):
                marker = "*" if pv < 0.05 else ("•" if pv < 0.10 else " ")
            n_var_s = str(r.get("n_var", "") or "")
            line += f" │ {pv_s}{marker} {adj_s} {q_s} {n_var_s:<6}"
        print(line)
