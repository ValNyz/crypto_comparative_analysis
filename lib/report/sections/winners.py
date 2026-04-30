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
        # Total variants per (signal, pair, tf) — denominator of n_var ratio.
        # Computed on the FULL trade-floor-filtered df, before significance gates.
        total_var = (
            df_f.groupby(["signal", "pair", "timeframe"]).size().rename("n_total")
            if {"signal", "pair", "timeframe"}.issubset(df_f.columns)
            else None
        )

        tier1 = df_f[(df_f["p_value_adj"].notna()) & (df_f["p_value_adj"] < 0.05)]
        tier1 = tier1.sort_values(["p_value_adj", "sharpe"], ascending=[True, False])
        tier1 = _dedup_with_count(tier1, total_var)

        tier2 = df_f[
            (df_f["p_value"].notna())
            & (df_f["p_value"] < 0.10)
            & ((df_f["p_value_adj"].isna()) | (df_f["p_value_adj"] >= 0.05))
        ]
        tier2 = tier2.sort_values(["p_value", "sharpe"], ascending=[True, False])
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
        top = df_f.nlargest(top_n, "sharpe")
        _print_block(top, show_pvals=False)


def _dedup_with_count(
    sub: pd.DataFrame, total_var: pd.Series | None
) -> pd.DataFrame:
    """Keep best (already pre-sorted) row per (signal, pair, tf) and tag n_var.

    `n_var` = "X/Y" where X = FDR-sig variants in the group (or rows in `sub`),
    Y = total variants tested for that combo (from `total_var`). Surfaces
    parameter-robustness: 9/9 means every exit/SL variant of this combo passed
    FDR → very robust ; 2/9 means it's an edge case dependent on specific
    exit/SL params.
    """
    if len(sub) == 0 or not {"signal", "pair", "timeframe"}.issubset(sub.columns):
        sub = sub.copy()
        sub["n_var"] = ""
        return sub
    n_sig = sub.groupby(["signal", "pair", "timeframe"]).size().rename("_n_sig")
    deduped = sub.drop_duplicates(
        subset=["signal", "pair", "timeframe"], keep="first"
    ).copy()
    deduped = deduped.merge(n_sig, on=["signal", "pair", "timeframe"], how="left")
    if total_var is not None:
        deduped = deduped.merge(
            total_var, on=["signal", "pair", "timeframe"], how="left"
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

    A "strategy" = (signal, exit_config, stoploss). Two rows with the same
    signal name but different exit or SL are different strategies. For each
    surviving strategy, we show the BEST TF per coin so the user sees at a
    glance "this strat wins on BTC(4h) but on ENA(30m)" without grepping.
    """
    if "signal" not in df.columns or "pair" not in df.columns:
        return
    if "p_value" not in df.columns or not df["p_value"].notna().any():
        return

    sig_df = df[df["p_value"].notna() & (df["p_value"] < 0.10)].copy()
    if len(sig_df) == 0:
        return

    # Strategy identity: (signal, exit_config, stoploss). Stoploss may be
    # absent in older result schemas; default to NaN-safe placeholder.
    strat_keys = ["signal", "exit_config"]
    if "stoploss" in sig_df.columns:
        strat_keys.append("stoploss")

    grouped = (
        sig_df.groupby(strat_keys, dropna=False)
        .agg(
            n_pairs=("pair", "nunique"),
            mean_sharpe=("sharpe", "mean"),
            mean_pnl=("profit_pct", "mean"),
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

    # Per-strategy-pair best TF: lowest p_value_adj within each
    # (signal, exit_config, stoploss, pair) group → keep its TF.
    best_tf = (
        sig_df.sort_values("p_value_adj", ascending=True)
        .drop_duplicates(subset=strat_keys + ["pair"], keep="first")
    )

    def _tf_breakdown(strat_row: pd.Series) -> str:
        """E.g. 'BTC(4h), ENA(1h), SOL(30m)' for a strat winning on 3 coins."""
        mask = (best_tf["signal"] == strat_row["signal"]) & (
            best_tf["exit_config"] == strat_row["exit_config"]
        )
        if "stoploss" in strat_keys:
            mask &= best_tf["stoploss"] == strat_row["stoploss"]
        rows = best_tf[mask].sort_values("p_value_adj", ascending=True)
        return ", ".join(
            f"{short_pair(str(r['pair']))}({r['timeframe']})"
            for _, r in rows.iterrows()
        )

    print(
        f"\n  🌍 ROBUSTESSE CROSS-COIN — strats sur >=2 pairs (signal+exit+SL) : "
        f"{len(multi)} candidats"
    )
    print(
        f"\n  {'#':<3} {'Signal':<28} {'Exit':<14} {'SL':<5} {'#':<3} │ "
        f"{'Sharpe':<7} {'PnL%':<8} {'Best p_adj':<11} │ Best TF par coin"
    )
    print("  " + "─" * 130)
    for i, (_, r) in enumerate(multi.head(top_n).iterrows(), 1):
        breakdown = _tf_breakdown(r)
        sl_str = (
            f"{abs(int(round(float(r['stoploss']) * 100)))}%"
            if "stoploss" in strat_keys and pd.notna(r.get("stoploss"))
            else "  -  "
        )
        exit_str = str(r["exit_config"])[:14]
        print(
            f"  {i:<3} {str(r['signal'])[:28]:<28} {exit_str:<14} "
            f"{sl_str:<5} {int(r['n_pairs']):<3} │ "
            f"{r['mean_sharpe']:<+7.2f} {r['mean_pnl']:<+8.1f} "
            f"{r['best_padj']:<11.4f} │ {breakdown}"
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
        f"{'Sharpe':<7} {'Cons%':<6} {'Mois+':<6} {'PnL%':<7} "
        f"{'p':<6} {'p_adj':<6} {'q':<6} {'Score':<7}"
    )
    print("  " + "─" * 117)
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
        print(
            f"  {i:<3} {r['signal']:<30} {short_pair(r['pair']):<6} "
            f"{r['timeframe']:<4} │ "
            f"{r['sharpe']:<+7.2f} {cons:<6.0f} {mois:<6} "
            f"{r['profit_pct']:<+7.1f} {r['p_value']:<6.3f} {adj_s:<6} {q_s:<6} "
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
            f"{'p':<6} {'p_adj':<6} {'q':<6} {'n_var':<6}"
        )
        sep_w = 144
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
