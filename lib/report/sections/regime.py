# =============================================================================
# FILE: lib/report/sections/regime.py
# =============================================================================
"""Regime analysis sections."""

import pandas as pd
import numpy as np
from typing import Dict, List
from ..formatters import print_header, print_section
from ...utils.helpers import short_pair


def print_regime_distribution(df: pd.DataFrame):
    """Print regime distribution statistics."""
    regime_totals = {"bull": 0, "bear": 0, "range": 0, "volatile": 0}
    regime_dir_totals = {
        "bull": {"long": 0, "short": 0},
        "bear": {"long": 0, "short": 0},
        "range": {"long": 0, "short": 0},
        "volatile": {"long": 0, "short": 0},
    }

    for _, row in df.iterrows():
        if row.get("regime_stats"):
            for regime, directions in row["regime_stats"].items():
                if not directions:
                    continue
                for direction, stats in directions.items():
                    if stats and stats.get("trades", 0) > 0:
                        regime_totals[regime] += stats["trades"]
                        regime_dir_totals[regime][direction] += stats["trades"]

    total_trades = sum(regime_totals.values())
    if total_trades == 0:
        return

    print_header("📊 DISTRIBUTION DES TRADES PAR RÉGIME")
    print()

    print(
        f"  {'Régime':<12} │ {'Total':<8} │ {'%':<7} │ {'Long':<8} │ {'Short':<8} │ {'Barre':<30}"
    )
    print("  " + "─" * 90)

    for regime in ["bull", "bear", "range", "volatile"]:
        count = regime_totals[regime]
        pct = count / total_trades * 100
        long_count = regime_dir_totals[regime]["long"]
        short_count = regime_dir_totals[regime]["short"]
        bar = "█" * int(pct / 2.5)

        print(
            f"  {regime:<10} │ {count:<8} │ {pct:<6.1f}% │ "
            f"{long_count:<6} │ {short_count:<6} │ {bar}"
        )

    print(f"\n  Total: {total_trades} trades")


def print_regime_performance(df: pd.DataFrame):
    """Print performance by regime."""
    regime_data = _collect_regime_data(df)

    print_header("🎯 PERFORMANCE PAR RÉGIME DE MARCHÉ")
    print()

    print(
        f"  {'Régime':<12} │ {'Dir':<6} │ {'Signaux':<8} │ {'Trades':<8} │ "
        f"{'WR Moy%':<8} │ {'PnL Moy%':<10} │ {'Best Signal':<25}"
    )
    print("  " + "─" * 105)

    for regime in ["bull", "bear", "range", "volatile"]:
        for direction in ["long", "short"]:
            data = regime_data[regime][direction]
            if data:
                total_trades = sum(d["trades"] for d in data)
                avg_wr = np.mean([d["win_rate"] for d in data])
                avg_pnl = np.mean([d["profit_pct"] for d in data])
                best = max(data, key=lambda x: x["profit_pct"])

                dir_symbol = "L" if direction == "long" else "S"
                print(
                    f"  {regime:<10} │ {dir_symbol:<6} │ {len(data):<8} │ "
                    f"{total_trades:<8} │ {avg_wr:<8.1f} │ {avg_pnl:<+10.2f} │ "
                    f"{best['signal'][:25]:<25}"
                )

    # Print top by regime
    _print_top_by_regime(regime_data)


def _collect_regime_data(df: pd.DataFrame) -> Dict:
    """Collect regime data from DataFrame."""
    regime_data = {
        "bull": {"long": [], "short": []},
        "bear": {"long": [], "short": []},
        "range": {"long": [], "short": []},
        "volatile": {"long": [], "short": []},
    }

    for _, row in df.iterrows():
        if row.get("regime_stats"):
            for regime, directions in row["regime_stats"].items():
                if not directions:
                    continue
                for direction, stats in directions.items():
                    if stats and stats.get("trades", 0) > 0:
                        regime_data[regime][direction].append(
                            {
                                "signal": row["signal"],
                                "pair": row["pair"],
                                "timeframe": row["timeframe"],
                                **stats,
                            }
                        )

    return regime_data


def _print_top_by_regime(regime_data: Dict):
    """Print TOP 10 by regime and direction."""
    print_header("🏆 TOP 10 SIGNAUX PAR RÉGIME ET DIRECTION")

    for regime in ["bull", "bear", "range", "volatile"]:
        for direction in ["long", "short"]:
            data = regime_data[regime][direction]
            if not data:
                continue

            print(f"\n  {regime.upper()} {direction.upper()} - Top 10 par profit:\n")

            sorted_data = sorted(data, key=lambda x: x["profit_pct"], reverse=True)[:10]

            print(
                f"    {'#':<3} {'Signal':<28} {'Pair':<6} │ {'Tr':<4} {'WR%':<6} {'PnL%':<8}"
            )
            print("    " + "─" * 65)

            for i, d in enumerate(sorted_data, 1):
                print(
                    f"    {i:<3} {d['signal']:<28} {short_pair(d['pair']):<6} │ "
                    f"{d['trades']:<4} {d['win_rate']:<6.1f} {d['profit_pct']:<+8.2f}"
                )


def print_signal_regime_matrix(df: pd.DataFrame):
    """Print signal × regime matrix with long/short distinction."""
    print_header("📊 MATRICE SIGNAL × RÉGIME (Top 20 signaux)")
    print()

    matrix_summary = _build_matrix_summary(df)

    # Compact view
    print(
        f"  {'Signal':<28} │ {'Sharpe':<7} │ {'BULL (L/S)':<16} │ "
        f"{'BEAR (L/S)':<16} │ {'RANGE (L/S)':<16} │ {'VOLAT (L/S)':<16}"
    )
    print("  " + "─" * 115)

    for row in matrix_summary[:20]:
        parts = [f"  {row['signal']:<28} │ {row['global_sharpe']:<+7.2f} │"]

        for regime in ["bull", "bear", "range", "volatile"]:
            long_pnl = row.get(f"{regime}_long_pnl")
            long_n = row.get(f"{regime}_long_n", 0)
            short_pnl = row.get(f"{regime}_short_pnl")
            short_n = row.get(f"{regime}_short_n", 0)

            long_str = (
                f"{long_pnl:+.1f}" if long_pnl is not None and long_n > 0 else "---"
            )
            short_str = (
                f"{short_pnl:+.1f}" if short_pnl is not None and short_n > 0 else "---"
            )
            total_n = long_n + short_n

            if total_n > 0:
                cell = f" {long_str}/{short_str}({total_n:>2}) │"
            else:
                cell = f" {'---':^14} │"

            parts.append(cell)

        print("".join(parts))

    # Detailed matrices
    _print_detailed_matrices(matrix_summary)


def _build_matrix_summary(df: pd.DataFrame) -> List[Dict]:
    """Build matrix summary from DataFrame."""
    signal_regime_matrix = {}

    for _, row in df.iterrows():
        sig = row["signal"]
        if sig not in signal_regime_matrix:
            signal_regime_matrix[sig] = {
                "bull": {"long": [], "short": []},
                "bear": {"long": [], "short": []},
                "range": {"long": [], "short": []},
                "volatile": {"long": [], "short": []},
                "global_sharpe": row["sharpe"],
            }

        if row.get("regime_stats"):
            for regime, directions in row["regime_stats"].items():
                if not directions:
                    continue
                for direction, stats in directions.items():
                    if stats and stats.get("trades", 0) > 0:
                        signal_regime_matrix[sig][regime][direction].append(
                            {
                                "profit_pct": stats["profit_pct"],
                                "trades": stats["trades"],
                            }
                        )

    # Build summary
    matrix_summary = []
    for sig, data in signal_regime_matrix.items():
        row_data = {"signal": sig, "global_sharpe": data["global_sharpe"]}

        for regime in ["bull", "bear", "range", "volatile"]:
            for direction in ["long", "short"]:
                dir_data = data[regime][direction]
                if dir_data:
                    row_data[f"{regime}_{direction}_pnl"] = np.mean(
                        [d["profit_pct"] for d in dir_data]
                    )
                    row_data[f"{regime}_{direction}_n"] = sum(
                        d["trades"] for d in dir_data
                    )
                else:
                    row_data[f"{regime}_{direction}_pnl"] = None
                    row_data[f"{regime}_{direction}_n"] = 0

        matrix_summary.append(row_data)

    matrix_summary.sort(key=lambda x: x["global_sharpe"], reverse=True)
    return matrix_summary


def _print_detailed_matrices(matrix_summary: List[Dict]):
    """Print detailed LONG and SHORT matrices."""
    print_section("MATRICE DÉTAILLÉE - LONG")

    print(
        f"  {'Signal':<28} │ {'Sharpe':<7} │ {'BULL':<12} │ "
        f"{'BEAR':<12} │ {'RANGE':<12} │ {'VOLATILE':<12}"
    )
    print("  " + "─" * 95)

    for row in matrix_summary[:15]:
        parts = [f"  {row['signal']:<28} │ {row['global_sharpe']:<+7.2f} │"]

        for regime in ["bull", "bear", "range", "volatile"]:
            pnl = row.get(f"{regime}_long_pnl")
            n = row.get(f"{regime}_long_n", 0)

            if pnl is not None and n > 0:
                parts.append(f" {pnl:+6.2f}({n:>2}) │")
            else:
                parts.append(f" {'---':^10} │")

        print("".join(parts))

    print_section("MATRICE DÉTAILLÉE - SHORT")

    print(
        f"  {'Signal':<28} │ {'Sharpe':<7} │ {'BULL':<12} │ "
        f"{'BEAR':<12} │ {'RANGE':<12} │ {'VOLATILE':<12}"
    )
    print("  " + "─" * 95)

    for row in matrix_summary[:15]:
        parts = [f"  {row['signal']:<28} │ {row['global_sharpe']:<+7.2f} │"]

        for regime in ["bull", "bear", "range", "volatile"]:
            pnl = row.get(f"{regime}_short_pnl")
            n = row.get(f"{regime}_short_n", 0)

            if pnl is not None and n > 0:
                parts.append(f" {pnl:+6.2f}({n:>2}) │")
            else:
                parts.append(f" {'---':^10} │")

        print("".join(parts))


