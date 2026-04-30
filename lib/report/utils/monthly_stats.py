# =============================================================================
# FILE: lib/report/utils/monthly_stats.py
# =============================================================================
"""Monthly per-trade aggregates for the drill-down section.

`compute_monthly_breakdown` derives per-month profit, trade count, win-rate,
profit-factor, and intra-month max drawdown by replaying the trade list
(profit_ratio + close_date). Drawdown is computed on the cumulative
equity curve within each month — answers "what was the worst peak-to-trough
during this month?" not "monthly PnL is negative" (different things).

`compute_monthly_market_change` reads the pair's feather and returns
per-month price change %, so the drill-down can show MKT alongside the
strategy's PnL — distinguishes "strategy beats market" from "strategy
follows the tide".
"""

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_monthly_breakdown(trades_df: pd.DataFrame) -> List[Dict]:
    """Per-month aggregates from the trade list.

    Returns rows ordered chronologically with keys:
      month (YYYY-MM), profit_pct, trades, win_rate, profit_factor, max_dd_pct
    """
    if trades_df is None or len(trades_df) == 0:
        return []
    if "close_date" not in trades_df.columns or "profit_ratio" not in trades_df.columns:
        return []

    df = trades_df.copy()
    df["close_date"] = pd.to_datetime(df["close_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["close_date"]).sort_values("close_date")
    if len(df) == 0:
        return []
    df["month"] = df["close_date"].dt.strftime("%Y-%m")

    rows: List[Dict] = []
    for month, grp in df.groupby("month", sort=True):
        ratios = grp["profit_ratio"].astype(float).values
        wins = int((ratios > 0).sum())
        n = len(ratios)
        gp = float(ratios[ratios > 0].sum() * 100) if (ratios > 0).any() else 0.0
        gl = float(abs(ratios[ratios < 0].sum()) * 100) if (ratios < 0).any() else 0.0
        pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)

        # Intra-month equity curve and max drawdown
        cum = np.cumprod(1.0 + ratios)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd_pct = float(dd.min() * 100) if len(dd) else 0.0

        rows.append(
            {
                "month": str(month),
                "profit_pct": float(ratios.sum() * 100),
                "trades": n,
                "win_rate": (wins / n * 100) if n else 0.0,
                "profit_factor": pf,
                "max_dd_pct": max_dd_pct,
            }
        )
    return rows


def compute_monthly_market_change(
    pair: str, timeframe: str, data_dir: str
) -> Dict[str, float]:
    """Per-month price change percentage for `pair` from the feather.

    Returns {YYYY-MM: pct_change}. Empty dict on any I/O / format issue.
    """
    safe_pair = pair.replace("/", "_").replace(":", "_")
    candidate_paths = [
        Path(data_dir) / "futures" / f"{safe_pair}-{timeframe}-futures.feather",
        Path(data_dir) / "futures" / f"{safe_pair}-{timeframe}.feather",
        Path(data_dir) / f"{safe_pair}-{timeframe}-futures.feather",
        Path(data_dir) / f"{safe_pair}-{timeframe}.feather",
    ]
    fp = next((p for p in candidate_paths if p.exists()), None)
    if fp is None:
        return {}
    try:
        df = pd.read_feather(fp)
    except Exception:
        return {}
    if "date" not in df.columns or "close" not in df.columns:
        return {}
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["month"] = df["date"].dt.strftime("%Y-%m")
    out: Dict[str, float] = {}
    for month, grp in df.groupby("month", sort=True):
        first_close = float(grp["close"].iloc[0])
        last_close = float(grp["close"].iloc[-1])
        if first_close > 0:
            out[str(month)] = (last_close / first_close - 1) * 100
    return out


def find_export_zip_for(
    class_name: str,
    timeframe: str,
    timerange: str,
    export_dir: Path,
    cached_index: Optional[Dict] = None,
) -> Optional[Path]:
    """Return the freqtrade export zip containing `class_name` (or None).

    Builds an index of (class_name, tf, timerange) -> zip_path by scanning
    *.meta.json files in `export_dir`. Pass a `cached_index` dict to amortize
    the scan across many lookups in the same report run.
    """
    if cached_index is None:
        cached_index = build_export_index(export_dir, timerange)
    return cached_index.get((class_name, timeframe, timerange))


def build_export_index(export_dir: Path, timerange: str) -> Dict:
    """Scan *.meta.json -> {(class_name, tf, timerange): zip_path}.

    Same logic as runner._build_export_index but without runner state.
    """
    index: Dict = {}
    if not export_dir.exists():
        return index
    for meta_path in export_dir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        zip_path = Path(str(meta_path).replace(".meta.json", ".zip"))
        if not zip_path.exists():
            continue
        for class_name, info in meta.items():
            tf = info.get("timeframe", "") or ""
            start_ts = info.get("backtest_start_ts")
            end_ts = info.get("backtest_end_ts")
            if not (start_ts and end_ts):
                continue
            if start_ts > 10**11:
                start_ts = start_ts / 1000
                end_ts = end_ts / 1000
            from datetime import datetime, timezone
            start_str = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y%m%d")
            end_str = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y%m%d")
            cached_range = f"{start_str}-{end_str}"
            if cached_range and cached_range != timerange:
                continue
            index[(class_name, tf, timerange)] = zip_path
    return index


def extract_trades_from_zip_safe(
    zip_path: Path, class_name: str
) -> Optional[pd.DataFrame]:
    """Same as null_pool.extract_trades_from_zip but silent on failure.

    Used by the report — log spam during summary printing is undesirable;
    a missing zip just means the drill-down skips that signal.
    """
    try:
        with zipfile.ZipFile(zip_path) as z:
            inner = next(
                (n for n in z.namelist() if n.endswith(".json") and "_config" not in n),
                None,
            )
            if not inner:
                return None
            data = json.loads(z.read(inner))
        strat = (data.get("strategy") or {}).get(class_name, {})
        trades = strat.get("trades") or []
        if not trades:
            return None
        df = pd.DataFrame(trades)
        if "profit_ratio" not in df.columns:
            return None
        return df
    except Exception:
        return None
