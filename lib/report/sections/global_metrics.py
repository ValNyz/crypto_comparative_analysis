# =============================================================================
# FILE: lib/report/sections/global_metrics.py
# =============================================================================
"""Global metrics section."""

import pandas as pd


def print_global_metrics(df: pd.DataFrame):
    """Print global metrics summary."""
    profitable = df[df["profit_pct"] > 0]
    avg_cons = df["consistency"].mean() if "consistency" in df.columns else 0

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
└──────────────────────────────────────────────────────────────────┘""")
