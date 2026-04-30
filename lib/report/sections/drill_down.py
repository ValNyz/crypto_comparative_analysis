# =============================================================================
# FILE: lib/report/sections/drill_down.py
# =============================================================================
"""Drill-down view for the top FDR-significant strategies.

For each of the top N FDR-significant signals (deduped to one row per
(signal, pair) — keeping the best timeframe), prints a focused block:

  - Per-month : PnL, trades, profit_factor, win-rate
  - Per-regime: trades, PnL%, WR (from regime_stats already in df)

The user gets, in one section, the answer to "is this signal robust over
months and regimes, or is it a fluke from one good period?". Replaces
the need to grep across temporal/regime sections per signal.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

from ..formatters import print_header
from ..utils.monthly_stats import (
    compute_monthly_breakdown,
    compute_quarterly_breakdown,
    compute_monthly_market_change,
    build_export_index,
    extract_trades_from_zip_safe,
)
from ...utils.helpers import short_pair, sanitize_class_name


def print_drill_down(
    df: pd.DataFrame, config=None, top_n: int = 10
) -> None:
    """Print drill-down details for top N FDR-significant strategies.

    Dedup rule: when the same `signal` runs on the same `pair` at multiple
    TFs and several pass FDR, keep only the best TF (lowest p_value_adj).
    Rationale: avoids 5x repetition of the same strategy across timeframes
    when all behave similarly. Cross-coin diversity (same signal on BTC
    AND ETH) IS preserved.

    `config` (Config) provides freqtrade_path / data_dir / timerange used
    to find each strategy's export zip and feather. When config is None,
    falls back to df-derived monthly stats (no DD column, no MKT column).
    """
    if "p_value_adj" not in df.columns or not df["p_value_adj"].notna().any():
        return

    sig_df = df[df["p_value_adj"].notna() & (df["p_value_adj"] < 0.05)].copy()
    if len(sig_df) == 0:
        return

    # Dedupe (signal, pair) → keep row with lowest p_value_adj
    sig_df = sig_df.sort_values("p_value_adj", ascending=True)
    sig_df = sig_df.drop_duplicates(subset=["signal", "pair"], keep="first")
    sig_df = sig_df.head(top_n)
    if len(sig_df) == 0:
        return

    print_header(
        f"🔬 DRILL-DOWN — Top {len(sig_df)} signaux validés (FDR p_adj<0.05, "
        f"déduppés par (signal, pair) au meilleur TF)"
    )

    # Build the export index ONCE — many drill-down rows share the same
    # backtest_results dir, so we don't want to glob *.meta.json N times.
    export_index = None
    data_dir = None
    timerange = None
    if config is not None:
        export_dir = Path(config.freqtrade_path) / "user_data/backtest_results"
        timerange = config.timerange
        data_dir = config.data_dir
        export_index = build_export_index(export_dir, timerange)

    for i, (_, r) in enumerate(sig_df.iterrows(), 1):
        _print_one_drill(i, r, export_index, data_dir, timerange)


def _print_one_drill(
    idx: int,
    r: pd.Series,
    export_index: Optional[Dict],
    data_dir: Optional[str],
    timerange: Optional[str],
) -> None:
    """Print the detail block for a single strategy row."""
    pair = short_pair(str(r["pair"]))
    sig_name = str(r["signal"])
    tf = str(r["timeframe"])
    exit_cfg = r.get("exit_config", "none") or "none"
    if pd.isna(exit_cfg):
        exit_cfg = "none"

    pv = r.get("p_value")
    pv_adj = r.get("p_value_adj")
    pv_s = f"{pv:.4f}" if pv is not None and not pd.isna(pv) else "n/a"
    pv_adj_s = (
        f"{pv_adj:.4f}"
        if pv_adj is not None and not pd.isna(pv_adj)
        else "n/a"
    )

    # Header line
    print(
        f"\n  [{idx:>2}] {sig_name}  ({pair}, {tf}, {exit_cfg})"
        f"   p={pv_s}  p_adj={pv_adj_s}*"
    )
    print(
        f"       Tr={int(r.get('trades', 0)):d}  "
        f"PnL={r.get('profit_pct', 0):+.1f}%  "
        f"Sharpe={r.get('sharpe', 0):+.2f}  "
        f"DD={r.get('max_dd_pct', 0):.1f}% ({int(r.get('dd_duration_days', 0) or 0)}d)  "
        f"WR={r.get('win_rate', 0):.1f}%  "
        f"MKT={r.get('market_change_pct', 0):+.1f}%"
    )

    # Extract per-trade detail once (used by both monthly and quarterly).
    trades_df = _try_extract_drill_trades(r, export_index, timerange)
    if trades_df is not None:
        monthly_rows = compute_monthly_breakdown(trades_df)
        if monthly_rows and data_dir:
            mkt = compute_monthly_market_change(
                str(r["pair"]), str(r["timeframe"]), data_dir
            )
            for row in monthly_rows:
                row["market_pct"] = mkt.get(row["month"], None)
        if monthly_rows:
            _print_enriched_monthly_table(monthly_rows)
            # Quarterly Sharpe: regime-stability indicator at horizon where
            # trade-level Sharpe stops being noise (n>=20-30 trades).
            quarterly_rows = compute_quarterly_breakdown(trades_df)
            _print_quarterly_table(quarterly_rows)
        else:
            _print_monthly_table_from_df(r)
            _print_quarterly_fallback_from_df(r)
    else:
        _print_monthly_table_from_df(r)
        _print_quarterly_fallback_from_df(r)

    # Per-regime sub-table from regime_stats dict
    _print_regime_table(r)


def _try_extract_drill_trades(
    r: pd.Series,
    export_index: Optional[Dict],
    timerange: Optional[str],
):
    """Locate this strategy's export zip and extract its trades DataFrame.

    Returns None when the zip can't be found (e.g., backtest_results/ was
    cleared) — caller falls back to df-aggregated arrays.
    """
    if export_index is None or timerange is None:
        return None
    sig_name = str(r["signal"])
    tf = str(r["timeframe"])
    pair = str(r["pair"])
    class_name = f"S_{sanitize_class_name(sig_name)}_{tf}"
    zip_path = export_index.get((class_name, tf, timerange, pair))
    if zip_path is None:
        return None
    trades = extract_trades_from_zip_safe(zip_path, class_name)
    if trades is None or len(trades) == 0:
        return None
    return trades


def _print_enriched_monthly_table(rows: List[Dict]) -> None:
    print(
        f"\n       {'Mois':<8} {'PnL%':<8} {'MKT%':<8} {'Tr':<5} "
        f"{'WR%':<6} {'PF':<6} {'DD%':<7}"
    )
    print("       " + "─" * 55)
    for row in rows:
        mkt = row.get("market_pct")
        mkt_s = f"{mkt:+.1f}" if mkt is not None else "  -  "
        pf = row["profit_factor"]
        pf_s = f"{pf:.2f}" if pf != float("inf") else "  inf"
        print(
            f"       {row['month']:<8} {row['profit_pct']:<+8.1f} "
            f"{mkt_s:<8} {row['trades']:<5d} {row['win_rate']:<6.1f} "
            f"{pf_s:<6} {row['max_dd_pct']:<+7.1f}"
        )


def _print_quarterly_table(rows: List[Dict]) -> None:
    """Sharpe-per-quarter view. Skipped when only 1 quarter — no stability info.

    Sharpe column is trade-level (mean/std of profit_ratio), not annualized.
    Comparable across quarters at face value: a stable strat shows ~constant
    Sharpe; a regime-shift victim shows e.g. +1.5 in 2024Q4 then -0.4 in
    2025Q2.
    """
    if not rows or len(rows) < 2:
        return
    print(
        f"\n       {'Trim':<8} {'PnL%':<8} {'Tr':<5} {'SH':<7} {'DD%':<7}"
    )
    print("       " + "─" * 38)
    for row in rows:
        sh = row["sharpe"]
        sh_s = f"{sh:+.2f}" if not pd.isna(sh) else "  n/a "
        print(
            f"       {row['quarter']:<8} {row['profit_pct']:<+8.1f} "
            f"{row['trades']:<5d} {sh_s:<7} {row['max_dd_pct']:<+7.1f}"
        )


def _print_monthly_table_from_df(r: pd.Series) -> None:
    """Fallback: use df arrays (no DD, no MKT) when zip unavailable."""
    profits = r.get("monthly_profits") or []
    trades = r.get("monthly_trades") or []
    pfs = r.get("monthly_pf") or []
    wrs = r.get("monthly_wr") or []
    if not profits:
        return
    print(
        f"\n       {'Mois':<6} {'PnL':<8} {'Tr':<5} {'PF':<6} {'WR%':<6}"
    )
    print("       " + "─" * 35)
    for j in range(len(profits)):
        p = profits[j] if j < len(profits) else 0
        tr = trades[j] if j < len(trades) else 0
        pf = pfs[j] if j < len(pfs) else 0
        wr = wrs[j] if j < len(wrs) else 0
        print(
            f"       M{j + 1:<5} {p:<+8.2f} {int(tr):<5d} "
            f"{pf:<6.2f} {wr:<6.1f}"
        )


def _print_quarterly_fallback_from_df(r: pd.Series) -> None:
    """Aggregate monthly_profits / monthly_trades into 3-month buckets.

    Used when no zip is available (no per-trade returns) — Sharpe can't be
    computed honestly from 3 monthly aggregates so it's omitted. The PnL +
    trade count per quarter still surfaces regime shifts (e.g., +30 USDC/Q
    in 2024 then -10 in 2025 = clear drift).
    """
    profits = r.get("monthly_profits") or []
    trades = r.get("monthly_trades") or []
    if not profits or len(profits) < 3:
        return
    print(f"\n       {'Trim':<6} {'PnL':<8} {'Tr':<5} {'Mois+/3':<8}")
    print("       " + "─" * 30)
    for q_start in range(0, len(profits), 3):
        q_idx = q_start // 3 + 1
        q_profits = profits[q_start:q_start + 3]
        q_trades = trades[q_start:q_start + 3] if q_start < len(trades) else []
        pnl = sum(q_profits)
        tr = sum(int(t) for t in q_trades)
        n_pos = sum(1 for p in q_profits if p > 0)
        print(
            f"       Q{q_idx:<5} {pnl:<+8.2f} {tr:<5d} {n_pos}/{len(q_profits):<6}"
        )


def _print_regime_table(r: pd.Series) -> None:
    rs = r.get("regime_stats") or {}
    if not rs:
        return
    rows = []
    for regime in ("bull", "bear", "range", "volatile"):
        rd = rs.get(regime, {}) or {}
        for direction in ("long", "short"):
            d = rd.get(direction, {}) or {}
            tr = int(d.get("trades", 0) or 0)
            if tr == 0:
                continue
            rows.append(
                {
                    "regime": regime,
                    "direction": direction,
                    "trades": tr,
                    "profit_pct": d.get("profit_pct", 0),
                    "wr": d.get("win_rate", 0),
                }
            )
    if not rows:
        return
    print(
        f"\n       {'Régime':<8} {'Dir':<6} {'Tr':<5} "
        f"{'PnL%':<8} {'WR%':<6}"
    )
    print("       " + "─" * 38)
    for row in sorted(rows, key=lambda x: -x["trades"]):
        print(
            f"       {row['regime']:<8} {row['direction']:<6} "
            f"{row['trades']:<5d} {row['profit_pct']:<+8.2f} "
            f"{row['wr']:<6.1f}"
        )
