# =============================================================================
# FILE: lib/backtest/parser.py
# =============================================================================
"""Parse freqtrade backtest output."""

import re
from typing import Dict
import numpy as np


def parse_freqtrade_output(output: str) -> Dict:
    """
    Parse freqtrade backtest output and extract metrics.

    Args:
        output: Raw stdout from freqtrade backtesting

    Returns:
        Dict with extracted metrics
    """
    result = {
        "trades": 0,
        "win_rate": 0.0,
        "profit_pct": 0.0,
        "avg_profit": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_dd_pct": 0.0,
        "dd_duration_days": 0.0,
        "market_change_pct": 0.0,
        "profit_factor": 0.0,
        "wins": 0,
        "losses": 0,
        "expectancy": 0.0,
        "sqn": 0.0,
        "monthly_profits": [],
        "monthly_trades": [],
        "monthly_pf": [],
        "monthly_wr": [],
        "months_profitable": 0,
        "months_total": 0,
        "best_month": 0.0,
        "worst_month": 0.0,
        "avg_month": 0.0,
        "std_month": 0.0,
        "consistency": 0.0,
        "avg_monthly_pf": 0.0,
        "avg_monthly_wr": 0.0,
        "min_monthly_pf": 0.0,
        "regime_stats": {},
    }

    # Parse TOTAL line
    result = _parse_total_line(output, result)

    # Parse metrics
    result = _parse_metrics(output, result)

    # Parse drawdown
    result = _parse_drawdown(output, result)

    # Parse monthly breakdown
    result = _parse_monthly(output, result)

    # Parse regime tags
    result = _parse_regime_tags(output, result)

    return result


def _parse_total_line(output: str, result: Dict) -> Dict:
    """Parse the TOTAL line from backtest output."""
    pattern = re.compile(
        r"[│┃|]\s*(TOTAL|S_\w+)\s*[│┃|]\s*(\d+)\s*[│┃|]\s*([-\d.]+)\s*[│┃|]\s*([-\d.]+)\s*[│┃|]\s*([-\d.]+)\s*[│┃|]\s*([^│┃|]+)[│┃|]\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)"
    )

    for match in pattern.finditer(output):
        _, trades, avg_pft, _, tot_pft_pct, _, wins, _, losses, win_rate = (
            match.groups()
        )
        result.update(
            {
                "trades": int(trades),
                "avg_profit": float(avg_pft),
                "profit_pct": float(tot_pft_pct),
                "wins": int(wins),
                "losses": int(losses),
                "win_rate": float(win_rate),
            }
        )
        break  # Only need first match

    return result


def _parse_metrics(output: str, result: Dict) -> Dict:
    """Parse performance metrics."""
    metrics = [
        ("sharpe", r"Sharpe\s*[│┃|]\s*([-\d.]+)"),
        ("sortino", r"Sortino\s*[│┃|]\s*([-\d.]+)"),
        ("calmar", r"Calmar\s*[│┃|]\s*([-\d.]+)"),
        ("sqn", r"SQN\s*[│┃|]\s*([-\d.]+)"),
        ("profit_factor", r"Profit factor\s*[│┃|]\s*([\d.]+)"),
        ("expectancy", r"Expectancy[^│┃|]*[│┃|]\s*([-\d.]+)"),
    ]

    for key, pattern in metrics:
        match = re.search(pattern, output)
        if match:
            try:
                result[key] = float(match.group(1))
            except ValueError:
                pass

    return result


def _parse_drawdown(output: str, result: Dict) -> Dict:
    """Parse maximum drawdown + duration."""
    match = re.search(
        r"(?:Absolute drawdown|Max % of account)[^│┃|]*[│┃|][^│┃|]*([\d.]+)%", output
    )
    if match:
        try:
            result["max_dd_pct"] = float(match.group(1))
        except ValueError:
            pass

    # Drawdown duration — freqtrade reports as "12 days 16:00:00" or "0 days …"
    dur_match = re.search(
        r"Drawdown\s+duration[^│┃|]*[│┃|][^│┃|]*?(\d+)\s+day", output
    )
    if dur_match:
        try:
            result["dd_duration_days"] = float(dur_match.group(1))
        except ValueError:
            pass

    # Market change over backtest period (BTC/pair price change)
    mkt_match = re.search(
        r"Market\s+change[^│┃|]*[│┃|][^│┃|]*?([-\d.]+)%", output
    )
    if mkt_match:
        try:
            result["market_change_pct"] = float(mkt_match.group(1))
        except ValueError:
            pass

    return result


def _parse_monthly(output: str, result: Dict) -> Dict:
    """Parse monthly breakdown."""
    pattern = re.compile(
        r"[│┃]\s*(\d{2}/\d{2}/\d{4})\s*[│┃]\s*(\d+)\s*[│┃]\s*([-\d.]+)\s*[│┃]\s*([\d.]+)\s*[│┃]\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*[│┃]"
    )

    monthly_profits = []
    monthly_trades = []
    monthly_pf = []
    monthly_wr = []

    for match in pattern.finditer(output):
        _, trades, profit, pf, wins, draws, losses, wr = match.groups()
        try:
            monthly_profits.append(float(profit))
            monthly_trades.append(int(trades))
            monthly_pf.append(float(pf))
            monthly_wr.append(float(wr))
        except ValueError:
            pass

    if monthly_profits:
        result.update(
            {
                "monthly_profits": monthly_profits,
                "monthly_trades": monthly_trades,
                "monthly_pf": monthly_pf,
                "monthly_wr": monthly_wr,
                "months_total": len(monthly_profits),
                "months_profitable": sum(1 for p in monthly_profits if p > 0),
                "best_month": max(monthly_profits),
                "worst_month": min(monthly_profits),
                "avg_month": np.mean(monthly_profits),
                "std_month": np.std(monthly_profits) if len(monthly_profits) > 1 else 0,
                "consistency": sum(1 for p in monthly_profits if p > 0)
                / len(monthly_profits)
                * 100,
                "avg_monthly_pf": np.mean(monthly_pf),
                "avg_monthly_wr": np.mean(monthly_wr),
                "min_monthly_pf": min(monthly_pf),
            }
        )

    return result


def _parse_regime_tags(output: str, result: Dict) -> Dict:
    """Parse entry tags with regime information."""
    pattern = re.compile(
        r"[│┃]\s*([\w.]+(?:long|short)_(?:bull|bear|range|volatile))\s*[│┃]\s*(\d+)\s*[│┃]\s*([-\d.]+)\s*[│┃]\s*([-\d.]+)\s*[│┃]\s*([-\d.]+)\s*[│┃]\s*[^│┃|]+[│┃]\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)"
    )

    regime_stats = {
        "bull": {"long": [], "short": []},
        "bear": {"long": [], "short": []},
        "range": {"long": [], "short": []},
        "volatile": {"long": [], "short": []},
    }

    for match in pattern.finditer(output):
        tag, trades, avg_pft, _, tot_pft_pct, wins, _, losses, win_rate = match.groups()

        # Determine direction
        if "_long_" in tag:
            direction = "long"
        elif "_short_" in tag:
            direction = "short"
        else:
            continue

        # Determine regime
        for regime in regime_stats.keys():
            if tag.endswith(f"_{regime}"):
                regime_stats[regime][direction].append(
                    {
                        "trades": int(trades),
                        "profit_pct": float(tot_pft_pct),
                        "wins": int(wins),
                        "losses": int(losses),
                        "win_rate": float(win_rate),
                    }
                )
                break

    # Aggregate stats
    for regime, directions in regime_stats.items():
        result["regime_stats"][regime] = {}

        for direction in ["long", "short"]:
            stats_list = directions[direction]
            if stats_list:
                total_trades = sum(s["trades"] for s in stats_list)
                total_wins = sum(s["wins"] for s in stats_list)
                total_losses = sum(s["losses"] for s in stats_list)
                weighted_profit = sum(s["profit_pct"] * s["trades"] for s in stats_list)

                result["regime_stats"][regime][direction] = {
                    "trades": total_trades,
                    "profit_pct": weighted_profit / total_trades
                    if total_trades > 0
                    else 0,
                    "win_rate": total_wins / (total_wins + total_losses) * 100
                    if (total_wins + total_losses) > 0
                    else 0,
                    "wins": total_wins,
                    "losses": total_losses,
                }

    return result
