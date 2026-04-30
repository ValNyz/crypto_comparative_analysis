# =============================================================================
# FILE: lib/report/sections/blacklist.py
# =============================================================================
"""Signals to drop from the next grid run.

Aggregates per signal across all contexts (pair × TF × exit × SL) and
flags those that:
  - Have a mean Sharpe deeply negative across contexts
  - Lose money in >half their contexts
  - Have no FDR-significant row (so we're not killing a signal that wins
    in one specific niche)
  - Were tested on >=3 contexts (single bad run isn't enough to blacklist)

Output format: commented YAML-style lines the user can paste into the
signals: section to comment them out.
"""

import pandas as pd
from ..formatters import print_header


# Tuned aggressively: only "obviously dead" signals make this list.
# Borderline ones (avg_sharpe ~ -0.1, mixed profitability) stay in for
# the user to inspect. The goal is "remove immediately, no thinking".
_MIN_CONTEXTS = 3
_MAX_AVG_SHARPE = -0.30
_MAX_PROFITABLE_PCT = 0.30
_MIN_BEST_PVAL = 0.20  # best p-value across contexts must be >= this


def print_blacklist(df: pd.DataFrame) -> None:
    if "signal" not in df.columns or len(df) == 0:
        return

    has_p = "p_value" in df.columns and df["p_value"].notna().any()
    has_padj = "p_value_adj" in df.columns and df["p_value_adj"].notna().any()

    # Per-signal aggregates
    agg = df.groupby("signal").agg(
        contexts=("signal", "count"),
        avg_sharpe=("sharpe", "mean"),
        profitable=("profit_pct", lambda s: (s > 0).sum()),
        best_p=("p_value", "min") if has_p else ("signal", "count"),
        best_padj=("p_value_adj", "min") if has_padj else ("signal", "count"),
    ).reset_index()
    agg["profitable_pct"] = agg["profitable"] / agg["contexts"]

    # Filter — must satisfy ALL gates to be flagged
    mask = (
        (agg["contexts"] >= _MIN_CONTEXTS)
        & (agg["avg_sharpe"] < _MAX_AVG_SHARPE)
        & (agg["profitable_pct"] < _MAX_PROFITABLE_PCT)
    )
    if has_p:
        # Exclude signals with at least one suggestive row
        mask &= agg["best_p"] >= _MIN_BEST_PVAL
    if has_padj:
        # Exclude any signal with FDR-significant context
        mask &= (agg["best_padj"].isna()) | (agg["best_padj"] >= 0.10)

    bad = agg[mask].sort_values("avg_sharpe", ascending=True)
    if len(bad) == 0:
        return

    print_header("🗑️  SIGNAUX À RETIRER DE LA GRILLE")
    print(
        "  Critères : Sharpe moyen < -0.30, profitable <30% des contextes, "
        f"≥{_MIN_CONTEXTS} tests, "
        "aucun contexte FDR-significatif."
    )
    print("  Copie-colle ces lignes dans signals: pour les retirer (préfixées #).\n")

    for _, r in bad.iterrows():
        bp = r.get("best_p")
        bp_s = (
            f"best_p={bp:.2f}"
            if bp is not None and not pd.isna(bp)
            else "best_p=n/a"
        )
        print(
            f"    # - {r['signal']:<28}"
            f"  SH_avg={r['avg_sharpe']:+.2f}  "
            f"prof={int(r['profitable'])}/{int(r['contexts'])} "
            f"({r['profitable_pct'] * 100:.0f}%)  {bp_s}"
        )
    print()
