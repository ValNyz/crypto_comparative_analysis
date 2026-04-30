# =============================================================================
# FILE: lib/report/sections/per_coin.py
# =============================================================================
"""Per-coin analysis and comparison sections."""

import pandas as pd
from typing import Dict, Any
from ..formatters import print_header
from ..utils import dedup_for_display
from ...utils.helpers import short_pair


def print_per_coin_summary(df: pd.DataFrame, top_n: int = 5):
    """
    Print a summary of best strategies per coin.

    Args:
        df: DataFrame with backtest results
        top_n: Number of top strategies to show per coin
    """
    if len(df) == 0 or "pair" not in df.columns:
        return

    pairs = df["pair"].unique()
    if len(pairs) <= 1:
        return  # Pas besoin de comparaison si une seule paire

    print_header(f"🪙 MEILLEURES STRATÉGIES PAR COIN ({len(pairs)} paires)")

    for pair in sorted(pairs):
        pair_df = df[df["pair"] == pair]
        if len(pair_df) == 0:
            continue

        # Stats globales pour cette paire
        profitable = pair_df[pair_df["profit_pct"] > 0]
        avg_sharpe = pair_df["sharpe"].mean()

        # Coin name (extract from pair like "BTC/USDC:USDC" -> "BTC")
        coin = short_pair(pair)

        print(f"\n{'─' * 110}")
        print(f"  📈 {coin}")
        print(
            f"     Stratégies: {len(pair_df)} │ "
            f"Profitables: {len(profitable)} ({len(profitable) / len(pair_df) * 100:.1f}%) │ "
            f"Sharpe moyen: {avg_sharpe:+.2f}"
        )
        print(f"{'─' * 110}")

        # Top N par Calmar pour cette paire — dedup TF + exit siblings.
        # pair est déjà fixé par la boucle, donc dedup par signal_root seul.
        pair_df_disp = dedup_for_display(
            pair_df, sort_cols="calmar", keys=("signal_root",)
        )
        top = pair_df_disp.nlargest(min(top_n, len(pair_df_disp)), "calmar")

        print(
            f"\n  {'#':<3} {'Signal':<30} {'TF':<4} │ "
            f"{'Tr':<4} {'WR%':<6} {'PnL%':<8} {'Calmar':<7} {'PF':<6} "
            f"{'μ_m':<6} {'σ_m':<6} │ {'Exit':<12}"
        )
        print("  " + "─" * 116)

        for i, (_, r) in enumerate(top.iterrows(), 1):
            exit_cfg = r.get("exit_config", "none")
            if pd.isna(exit_cfg):
                exit_cfg = "none"
            mu_m = r.get("avg_month", 0) or 0
            sd_m = r.get("std_month", 0) or 0
            cal = r.get("calmar", 0) or 0
            pf = r.get("profit_factor", 0) or 0
            pf_s = f"{pf:<6.2f}" if pf != float("inf") else "  inf"
            print(
                f"  {i:<3} {r['signal']:<30} {r['timeframe']:<4} │ "
                f"{r['trades']:<4} {r['win_rate']:<6.1f} {r['profit_pct']:<+8.1f} "
                f"{cal:<+7.2f} {pf_s} {mu_m:<+6.1f} {sd_m:<6.1f} │ {exit_cfg:<12}"
            )


def print_coin_comparison_matrix(df: pd.DataFrame):
    """
    Print a comparison matrix showing which signals work best across coins.

    Args:
        df: DataFrame with backtest results
    """
    if len(df) == 0 or "pair" not in df.columns:
        return

    pairs = df["pair"].unique()
    if len(pairs) <= 1:
        return

    print_header("📊 MATRICE DE COMPARAISON INTER-COINS")

    # Trouver les signaux communs
    signal_types = df["signal_type"].unique() if "signal_type" in df.columns else []

    # Créer une matrice signal_type x pair avec le meilleur Sharpe
    print("\n  Meilleur Sharpe par type de signal et par paire:\n")

    # Header
    coin_names = [p.split("/")[0][:6] for p in sorted(pairs)]
    header = (
        f"  {'Signal Type':<18} │ "
        + " │ ".join(f"{c:^8}" for c in coin_names)
        + " │ Écart"
    )
    print(header)
    print("  " + "─" * len(header))

    for sig_type in sorted(signal_types):
        row_values = []
        for pair in sorted(pairs):
            subset = df[(df["signal_type"] == sig_type) & (df["pair"] == pair)]
            if len(subset) > 0:
                best_sharpe = subset["sharpe"].max()
                row_values.append(best_sharpe)
            else:
                row_values.append(None)

        # Calculer l'écart (max - min) pour voir la variation inter-coins
        valid_values = [v for v in row_values if v is not None]
        if len(valid_values) >= 2:
            spread = max(valid_values) - min(valid_values)
        else:
            spread = 0

        # Formater la ligne
        formatted_values = []
        for v in row_values:
            if v is None:
                formatted_values.append(f"{'—':^8}")
            else:
                formatted_values.append(f"{v:+8.2f}")

        print(
            f"  {sig_type:<18} │ " + " │ ".join(formatted_values) + f" │ {spread:5.2f}"
        )


def print_consistent_performers(df: pd.DataFrame, min_coins: int = 2):
    """
    Print signals that perform consistently well across multiple coins.

    Args:
        df: DataFrame with backtest results
        min_coins: Minimum number of coins where signal must be profitable
    """
    if len(df) == 0 or "pair" not in df.columns:
        return

    pairs = df["pair"].unique()
    if len(pairs) < min_coins:
        return

    print_header(f"🏆 SIGNAUX PERFORMANTS SUR {min_coins}+ COINS")

    # Grouper par signal (sans la paire)
    # On veut trouver les signaux qui ont Sharpe > 0.5 sur plusieurs coins
    signal_performance = {}

    for signal in df["signal"].unique():
        signal_df = df[df["signal"] == signal]

        # Compter les coins où le signal a un bon Sharpe
        good_coins = []
        for pair in pairs:
            pair_signal = signal_df[signal_df["pair"] == pair]
            if len(pair_signal) > 0:
                best = pair_signal.loc[pair_signal["sharpe"].idxmax()]
                if best["sharpe"] > 0.5:
                    good_coins.append(
                        {
                            "pair": pair,
                            "sharpe": best["sharpe"],
                            "profit": best["profit_pct"],
                            "trades": best["trades"],
                        }
                    )

        if len(good_coins) >= min_coins:
            signal_performance[signal] = {
                "coins": good_coins,
                "count": len(good_coins),
                "avg_sharpe": sum(c["sharpe"] for c in good_coins) / len(good_coins),
                "signal_type": signal_df["signal_type"].iloc[0]
                if "signal_type" in signal_df.columns
                else "unknown",
            }

    if not signal_performance:
        print(f"\n  Aucun signal n'a un Sharpe > 0.5 sur {min_coins}+ coins.")
        return

    # Trier par nombre de coins puis par sharpe moyen
    sorted_signals = sorted(
        signal_performance.items(),
        key=lambda x: (x[1]["count"], x[1]["avg_sharpe"]),
        reverse=True,
    )

    print(
        f"\n  {'Signal':<35} │ {'Type':<12} │ {'Coins':<6} │ {'Avg Sharpe':<10} │ Détails"
    )
    print("  " + "─" * 100)

    for signal, perf in sorted_signals[:15]:  # Top 15
        coins_detail = ", ".join(
            f"{short_pair(c['pair'])}({c['sharpe']:+.1f})"
            for c in sorted(perf["coins"], key=lambda x: x["sharpe"], reverse=True)
        )
        print(
            f"  {signal:<35} │ {perf['signal_type']:<12} │ {perf['count']:^6} │ "
            f"{perf['avg_sharpe']:+10.2f} │ {coins_detail[:40]}"
        )


def get_per_coin_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Get statistics per coin for programmatic use.

    Args:
        df: DataFrame with backtest results

    Returns:
        Dict mapping pair to statistics dict
    """
    stats = {}

    if len(df) == 0 or "pair" not in df.columns:
        return stats

    for pair in df["pair"].unique():
        pair_df = df[df["pair"] == pair]
        profitable = pair_df[pair_df["profit_pct"] > 0]

        # Meilleure stratégie
        best_idx = pair_df["sharpe"].idxmax()
        best = pair_df.loc[best_idx]

        stats[pair] = {
            "total_strategies": len(pair_df),
            "profitable_count": len(profitable),
            "profitable_pct": len(profitable) / len(pair_df) * 100
            if len(pair_df) > 0
            else 0,
            "avg_sharpe": pair_df["sharpe"].mean(),
            "max_sharpe": pair_df["sharpe"].max(),
            "min_sharpe": pair_df["sharpe"].min(),
            "avg_profit": pair_df["profit_pct"].mean(),
            "best_signal": best["signal"],
            "best_signal_sharpe": best["sharpe"],
            "best_signal_profit": best["profit_pct"],
            "signal_types": pair_df["signal_type"].unique().tolist()
            if "signal_type" in pair_df.columns
            else [],
        }

    return stats
