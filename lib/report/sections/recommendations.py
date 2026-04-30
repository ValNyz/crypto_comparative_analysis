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
    """Print recommendations: FDR-significant first, then balanced score.

    Two-stage filter: (1) keep only FDR-significant rows when available
    (p_value_adj < 0.10) — anything else is suspect at scale; (2) within
    the survivors, rank by Sharpe + consistency + risk balance.
    Also prints a hard reminder: bootstrap p-value protects against luck
    on the dataset, NOT against regime shift / overfit. OOS rolling
    backtest stays mandatory before going live.
    """
    df_with_monthly = df[df["months_total"] > 2].copy()
    if len(df_with_monthly) == 0:
        return

    # Stage 1: FDR filter (loose 0.10 — drill-down + winners use 0.05)
    has_padj = (
        "p_value_adj" in df_with_monthly.columns
        and df_with_monthly["p_value_adj"].notna().any()
    )
    if has_padj:
        df_with_monthly = df_with_monthly[
            df_with_monthly["p_value_adj"].notna()
            & (df_with_monthly["p_value_adj"] < 0.10)
        ]
        if len(df_with_monthly) == 0:
            print(
                "  ⚠️  Aucune strat ne passe le filtre FDR p_adj<0.10. "
                "Pas de recommandations sûres.\n"
                "     → Élargis la grille de signaux ou diminue les seuils, "
                "puis re-run."
            )
            return

    # Stage 2: balanced score within FDR-significant subset
    df_with_monthly["balanced_score"] = (
        df_with_monthly["sharpe"] * 0.4
        + df_with_monthly["consistency"] / 100 * 0.3
        + (1 - df_with_monthly["max_dd_pct"] / 20) * 0.3
    )
    best_balanced = df_with_monthly.nlargest(5, "balanced_score")

    if has_padj:
        print(
            "  🎯 MEILLEUR ÉQUILIBRE — FDR-validés (p_adj<0.10), "
            "Performance + Stabilité + Risque :\n"
        )
    else:
        print(
            "  🎯 MEILLEUR ÉQUILIBRE (Performance + Stabilité + Risque) — "
            "⚠️  pas de p_value disponible, ranking pur Sharpe :\n"
        )

    for i, (_, r) in enumerate(best_balanced.iterrows(), 1):
        avg_m = r.get("avg_month", 0)
        pv_adj = r.get("p_value_adj")
        pv_str = (
            f"p_adj={pv_adj:.3f}"
            if pv_adj is not None and not pd.isna(pv_adj)
            else "p_adj=n/a"
        )
        print(
            f"     {i}. {r['signal']:<26} ({short_pair(r['pair'])}, {r['timeframe']})\n"
            f"        Sharpe={r['sharpe']:+.2f} │ Cons={r['consistency']:.0f}% │ "
            f"DD={r['max_dd_pct']:.1f}% │ PnL={r['profit_pct']:+.1f}% │ "
            f"Avg/mois={avg_m:+.1f} USDC │ {pv_str}\n"
        )

    # Hard reminder: bootstrap doesn't protect against regime shift.
    if has_padj:
        print(
            "  ⚠️  Le filtre FDR garantit que ces strats ont battu un baseline "
            "random sur ce dataset, mais NE valide PAS la robustesse hors-sample.\n"
            "     → Avant live : valider via `--rolling 3 --step 1` "
            "(walk-forward OOS) sur les top candidats. Si elles survivent, "
            "alors edge probable.\n"
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
