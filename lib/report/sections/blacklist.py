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
# Borderline ones (avg_calmar ~ 0, mixed profitability) stay in for the
# user to inspect. The goal is "remove immediately, no thinking".
_MIN_CONTEXTS = 3
_MAX_AVG_CALMAR = -0.50  # Calmar < -0.5 = losing decisively vs DD
_MAX_PROFITABLE_PCT = 0.30
_MIN_BEST_PVAL = 0.20  # best p-value across contexts must be >= this


def print_blacklist(df: pd.DataFrame) -> None:
    if "signal" not in df.columns or len(df) == 0:
        return

    has_p = "p_value" in df.columns and df["p_value"].notna().any()
    has_padj = "p_value_adj" in df.columns and df["p_value_adj"].notna().any()

    # Aggregate at signal_root level: removing a "signal" with exit suffix from
    # the YAML doesn't make sense — the YAML defines the core signal and the
    # `exit_config: [list]` separately. Listing signal_root tells the user
    # "comment this whole entry out" rather than "edit one exit out of the list".
    group_key = "signal_root" if "signal_root" in df.columns else "signal"
    agg = df.groupby(group_key).agg(
        contexts=(group_key, "count"),
        avg_calmar=("calmar", "mean"),
        avg_sharpe=("sharpe", "mean"),
        profitable=("profit_pct", lambda s: (s > 0).sum()),
        best_p=("p_value", "min") if has_p else (group_key, "count"),
        best_padj=("p_value_adj", "min") if has_padj else (group_key, "count"),
    ).reset_index().rename(columns={group_key: "signal"})
    agg["profitable_pct"] = agg["profitable"] / agg["contexts"]

    # Filter — must satisfy ALL gates to be flagged
    mask = (
        (agg["contexts"] >= _MIN_CONTEXTS)
        & (agg["avg_calmar"] < _MAX_AVG_CALMAR)
        & (agg["profitable_pct"] < _MAX_PROFITABLE_PCT)
    )
    if has_p:
        # Exclude signals with at least one suggestive row
        mask &= agg["best_p"] >= _MIN_BEST_PVAL
    if has_padj:
        # Exclude any signal with FDR-significant context
        mask &= (agg["best_padj"].isna()) | (agg["best_padj"] >= 0.10)

    bad = agg[mask].sort_values("avg_calmar", ascending=True)
    if len(bad) == 0:
        return

    print_header("🗑️  SIGNAUX À RETIRER DE LA GRILLE")
    print(
        f"  Critères : Calmar moyen < {_MAX_AVG_CALMAR:+.2f}, profitable <{int(_MAX_PROFITABLE_PCT * 100)}% des contextes, "
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
            f"  CAL_avg={r['avg_calmar']:+.2f}  "
            f"prof={int(r['profitable'])}/{int(r['contexts'])} "
            f"({r['profitable_pct'] * 100:.0f}%)  {bp_s}"
        )
    print()
