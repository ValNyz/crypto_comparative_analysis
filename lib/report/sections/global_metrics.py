# =============================================================================
# FILE: lib/report/sections/global_metrics.py
# =============================================================================
"""Global metrics section."""

import pandas as pd


def print_global_metrics(df: pd.DataFrame):
    """Print global metrics summary."""
    profitable = df[df["profit_pct"] > 0]
    avg_cons = df["consistency"].mean() if "consistency" in df.columns else 0

    # Significance counts (filled when null pool ran)
    has_p = "p_value" in df.columns and df["p_value"].notna().any()
    has_padj = "p_value_adj" in df.columns and df["p_value_adj"].notna().any()
    has_q = "q_value" in df.columns and df["q_value"].notna().any()
    n_sig_raw = (
        int(((df["p_value"].notna()) & (df["p_value"] < 0.05)).sum()) if has_p else 0
    )
    n_sig_adj = (
        int(((df["p_value_adj"].notna()) & (df["p_value_adj"] < 0.05)).sum())
        if has_padj
        else 0
    )
    n_sig_q = (
        int(((df["q_value"].notna()) & (df["q_value"] < 0.05)).sum())
        if has_q
        else 0
    )

    sig_line = ""
    if has_p:
        sig_line = (
            f"│  Significatifs p<0.05:  {n_sig_raw:<6} brut "
            f"/ {n_sig_adj:<3} BH-FDR / {n_sig_q:<3} Storey q          │\n"
        )

    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│  MÉTRIQUES GLOBALES                                              │
├──────────────────────────────────────────────────────────────────┤
│  Résultats valides:     {len(df):<6}                                   │
│  Profitables:           {len(profitable):<6} ({len(profitable) / len(df) * 100:3.1f}%)                          │
│  Sharpe > 0:            {len(df[df["sharpe"] > 0]):<6}                                   │
│  Sharpe > 1:            {len(df[df["sharpe"] > 1]):<6}                                   │
│  Sharpe > 2:            {len(df[df["sharpe"] > 2]):<6}                                   │
│  Sharpe moyen:          {df["sharpe"].mean():<+6.2f}                                   │
│  Profit moyen:          {df["profit_pct"].mean():<+6.1f}%                                  │
│  Consistance moyenne:   {avg_cons:<6.1f}% (mois profitables)               │
{sig_line}└──────────────────────────────────────────────────────────────────┘""")
