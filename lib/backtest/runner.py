# =============================================================================
# FILE: lib/backtest/runner.py
# =============================================================================
"""Backtest runner with parallel execution."""

import json
import subprocess
import time
import zipfile

import re
import random

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from ..signals.base import SignalConfig
from ..generation.generator import StrategyGenerator
from ..utils.logging import print_lock
from .parser import parse_freqtrade_output

# Patterns d'erreurs de rate limiting à détecter
RATE_LIMIT_PATTERNS = [
    r"rate.?limit",
    r"too.?many.?requests",
    r"throttl",
    r"quota.?exceeded",
    r"request.?limit",
    r"api.?limit",
    r"temporarily.?unavailable",
    r"try.?again.?later",
    r"overloaded",
]

# Compilé pour performance
RATE_LIMIT_REGEX = re.compile("|".join(RATE_LIMIT_PATTERNS), re.IGNORECASE)


def is_rate_limit_error(stdout: str, stderr: str) -> bool:
    """Détecte si l'erreur est due à un rate limiting."""
    combined = f"{stdout}\n{stderr}"
    return bool(RATE_LIMIT_REGEX.search(combined))


class BacktestRunner:
    """Runs backtests in parallel."""

    def __init__(
        self,
        config,  # Config
        debug: bool = False,
        max_retries: int = 5,
        base_delay: float = 15.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize the runner.

        Args:
            config: Config instance
            debug: Enable debug output
            max_retries: Nombre max de tentatives en cas de rate limit
            base_delay: Délai de base en secondes avant retry
            max_delay: Délai maximum en secondes
        """
        self.config = config
        self.debug = debug
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.generator = StrategyGenerator(config)
        self.completed = 0
        self.total = 0
        self.retries_total = 0  # Compteur de retries
        self.abbrev = {"bull": "Bu", "bear": "Be", "range": "Ra", "volatile": "Vo"}

        self.export_dir = Path(config.freqtrade_path) / "user_data/backtest_results"
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Cache index: built once in run_all (only when --use-cache is on) to
        # avoid paying for the meta.json scan when not needed. Keyed by
        # (class_name, timeframe, timerange) → zip path. Cf.
        # _build_export_index docstring for invalidation rules.
        self.use_cache: bool = bool(getattr(config, "use_cache", False))
        self._export_index: Optional[Dict[Tuple[str, str, str], Path]] = None
        self.cache_hits: int = 0

    def _get_export_name(self, signal: SignalConfig, pair: str, timeframe: str) -> str:
        """Génère un nom de fichier unique pour l'export."""
        safe_pair = pair.replace("/", "_").replace(":", "_")
        return f"{signal.name}_{safe_pair}_{timeframe}"

    def _build_export_index(self) -> Dict[Tuple[str, str, str], Path]:
        """Scan user_data/backtest_results/*.meta.json → {(class_name, timeframe, timerange): zip_path}.

        We key on `(class_name, timeframe, timerange)` so a cached entry is only
        reused when the strategy class AND the backtest window match. Stake /
        wallet / max_open_trades changes don't invalidate (less critical for
        relative ranking) — bump cache via `--force-fresh` if needed.
        """
        index: Dict[Tuple[str, str, str], Path] = {}
        timerange = self.config.timerange
        for meta_path in self.export_dir.glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            zip_path = meta_path.with_suffix("").with_suffix(".zip")
            # Replace .meta.zip → .zip if the .with_suffix chain produced wrong path
            if not zip_path.exists():
                zip_path = Path(str(meta_path).replace(".meta.json", ".zip"))
            if not zip_path.exists():
                continue
            for class_name, info in meta.items():
                tf = info.get("timeframe", "") or ""
                # Reconstitute timerange from start/end ts (in ms or s) → "YYYYMMDD-YYYYMMDD"
                start_ts = info.get("backtest_start_ts")
                end_ts = info.get("backtest_end_ts")
                if start_ts and end_ts:
                    # Some exports have ms, some seconds. Normalize.
                    if start_ts > 10**11:  # ms
                        start_ts = start_ts / 1000
                        end_ts = end_ts / 1000
                    from datetime import datetime, timezone
                    start_str = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y%m%d")
                    end_str = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y%m%d")
                    cached_range = f"{start_str}-{end_str}"
                else:
                    cached_range = ""
                # Only index if the timerange matches the current run's
                if cached_range and cached_range != timerange:
                    continue
                index[(class_name, tf, timerange)] = zip_path
        return index

    def _load_cached_result(self, class_name: str, zip_path: Path) -> Optional[Dict]:
        """Parse the inner backtest JSON from `zip_path` and return a result
        dict matching `parse_freqtrade_output`'s schema.

        Mirrors stdout-parser fields: trades, win_rate, profit_pct, sharpe,
        sortino, calmar, max_dd_pct, dd_duration_days, profit_factor, wins,
        losses, expectancy, sqn, monthly_*, regime_stats.
        Returns None on any parse error so the caller falls back to running
        freqtrade.
        """
        try:
            with zipfile.ZipFile(zip_path) as z:
                inner_name = next(
                    (n for n in z.namelist()
                     if n.endswith(".json") and "_config" not in n),
                    None,
                )
                if not inner_name:
                    return None
                data = json.loads(z.read(inner_name))
            strat = data.get("strategy", {}).get(class_name)
            if not strat:
                return None
        except Exception:
            return None

        result: Dict = {
            "trades": int(strat.get("total_trades", 0) or 0),
            "win_rate": float(strat.get("winrate", 0) or 0) * 100.0,
            "profit_pct": float(strat.get("profit_total", 0) or 0) * 100.0,
            "avg_profit": float(strat.get("profit_mean", 0) or 0) * 100.0,
            "sharpe": float(strat.get("sharpe", 0) or 0),
            "sortino": float(strat.get("sortino", 0) or 0),
            "calmar": float(strat.get("calmar", 0) or 0),
            "max_dd_pct": float(strat.get("max_drawdown_account", 0) or 0) * 100.0,
            "dd_duration_days": float(strat.get("drawdown_duration_s", 0) or 0) / 86400.0,
            "market_change_pct": float(strat.get("market_change", 0) or 0) * 100.0,
            "profit_factor": float(strat.get("profit_factor", 0) or 0),
            "wins": int(strat.get("wins", 0) or 0),
            "losses": int(strat.get("losses", 0) or 0),
            "expectancy": float(strat.get("expectancy", 0) or 0),
            "sqn": float(strat.get("sqn", 0) or 0),
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

        # Monthly breakdown from periodic_breakdown.month
        months = ((strat.get("periodic_breakdown") or {}).get("month") or [])
        for m in months:
            try:
                result["monthly_profits"].append(float(m.get("profit_abs", 0) or 0))
                result["monthly_trades"].append(int(m.get("trade_count", 0) or 0))
                # profit_factor / win_rate may be missing on monthly; fall back to 0
                result["monthly_pf"].append(float(m.get("profit_factor", 0) or 0))
                wins_m = int(m.get("wins", 0) or 0)
                tot_m = int(m.get("trade_count", 0) or 0)
                result["monthly_wr"].append((wins_m / tot_m * 100) if tot_m else 0.0)
            except (ValueError, TypeError):
                pass
        if result["monthly_profits"]:
            mp = result["monthly_profits"]
            import numpy as _np
            result["months_total"] = len(mp)
            result["months_profitable"] = sum(1 for p in mp if p > 0)
            result["best_month"] = max(mp)
            result["worst_month"] = min(mp)
            result["avg_month"] = float(_np.mean(mp))
            result["std_month"] = float(_np.std(mp)) if len(mp) > 1 else 0.0
            result["consistency"] = result["months_profitable"] / len(mp) * 100.0
            if result["monthly_pf"]:
                result["avg_monthly_pf"] = float(_np.mean(result["monthly_pf"]))
                result["min_monthly_pf"] = float(min(result["monthly_pf"]))
            if result["monthly_wr"]:
                result["avg_monthly_wr"] = float(_np.mean(result["monthly_wr"]))

        # Regime stats from mix_tag_stats — entries are {key: 'tag', trades, profit_total, ...}
        regime_buckets: Dict[str, Dict[str, list]] = {
            r: {"long": [], "short": []} for r in ("bull", "bear", "range", "volatile")
        }
        for entry in (strat.get("mix_tag_stats") or []):
            # `key` is either a string OR a list [enter_tag, exit_reason] in
            # newer freqtrade exports. Normalize to the enter_tag string.
            key_field = entry.get("key")
            if isinstance(key_field, list) and key_field:
                tag = str(key_field[0])
            else:
                tag = str(key_field or "")
            if "_long_" in tag:
                direction = "long"
            elif "_short_" in tag:
                direction = "short"
            else:
                continue
            for regime in regime_buckets:
                if tag.endswith(f"_{regime}"):
                    trades_e = int(entry.get("trades", 0) or 0)
                    wins_e = int(entry.get("wins", 0) or 0)
                    losses_e = int(entry.get("losses", 0) or 0)
                    profit_pct_e = float(entry.get("profit_total_pct", 0) or 0)
                    regime_buckets[regime][direction].append({
                        "trades": trades_e,
                        "profit_pct": profit_pct_e,
                        "wins": wins_e,
                        "losses": losses_e,
                        "win_rate": (wins_e / trades_e * 100) if trades_e else 0.0,
                    })
                    break
        for regime, dirs in regime_buckets.items():
            result["regime_stats"][regime] = {}
            for direction in ("long", "short"):
                lst = dirs[direction]
                if not lst:
                    continue
                tot_tr = sum(s["trades"] for s in lst)
                tot_w = sum(s["wins"] for s in lst)
                tot_l = sum(s["losses"] for s in lst)
                avg_pft = sum(s["profit_pct"] * s["trades"] for s in lst) / tot_tr if tot_tr else 0.0
                result["regime_stats"][regime][direction] = {
                    "trades": tot_tr,
                    "profit_pct": avg_pft,
                    "wins": tot_w,
                    "losses": tot_l,
                    "win_rate": (tot_w / tot_tr * 100) if tot_tr else 0.0,
                }

        return result

    def run_single(
        self,
        signal,
        pair: str,
        timeframe: str,  # SignalConfig
    ) -> Optional[Dict]:
        """
        Run a single backtest with retry on rate limit.

        Args:
            signal: SignalConfig to test
            pair: Trading pair
            timeframe: Timeframe

        Returns:
            Dict with results or None if failed
        """
        # Generate strategy
        class_name, strategy_path = self.generator.generate(signal, timeframe)
        actual_tf = signal.timeframe_override or timeframe

        # Cache hit check — skip freqtrade entirely if a previous export
        # for the same (class_name, tf, timerange) is on disk.
        # The index is built once in run_all() before the worker pool starts
        # to avoid race conditions / duplicate logs.
        if self.use_cache and self._export_index is not None:
            cache_key = (class_name, actual_tf, self.config.timerange)
            zip_path = self._export_index.get(cache_key)
            if zip_path is not None:
                cached = self._load_cached_result(class_name, zip_path)
                if cached is not None and cached.get("trades", 0) > self.config.min_trades:
                    cached.update({
                        "signal": signal.name,
                        "signal_type": signal.signal_type,
                        "direction": signal.direction,
                        "pair": pair,
                        "timeframe": actual_tf,
                        "allowed_regimes": signal.allowed_regimes,
                        "exit_config": signal.exit_config,
                        "_cached": True,
                    })
                    self.completed += 1
                    self.cache_hits += 1
                    self._print_result(cached)
                    return cached

        export_name = self._get_export_name(signal, pair, actual_tf)

        # Build command
        cmd = [
            "freqtrade",
            "backtesting",
            "--config",
            self.config.config_file,
            "--strategy",
            class_name,
            "--strategy-path",
            str(self.generator.output_dir),
            "--timeframe",
            actual_tf,
            "--timerange",
            self.config.timerange,
            "--pairs",
            pair,
        ]
        if self.config.timeframe_detail:
            cmd += ["--timeframe-detail", self.config.timeframe_detail]
        cmd += [
            "--datadir",
            self.config.data_dir,
            "--dry-run-wallet",
            str(self.config.dry_run_wallet),
            "--stake-amount",
            str(self.config.stake_amount),
            "--export",
            "trades",
            "--export-filename",
            export_name,
            "--max-open-trades",
            str(self.config.max_open_trades),
            "--breakdown",
            "month",
        ]

        # Retry loop
        for attempt in range(self.max_retries + 1):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=self.config.freqtrade_path,
                )

                # Only treat as rate-limit if the process actually failed.
                # CCXT has its own internal rate-limit retry — if freqtrade
                # exited 0, the backtest succeeded even if "RateLimit" appears
                # in the stderr traceback of an internally-recovered API call.
                if result.returncode != 0 and is_rate_limit_error(
                    result.stdout, result.stderr
                ):
                    if attempt < self.max_retries:
                        delay = self._calculate_delay(attempt)
                        self._debug_output(signal.name, result)
                        self._log_rate_limit(signal.name, attempt, delay)
                        time.sleep(delay)
                        self.retries_total += 1
                        continue
                    else:
                        self._log_rate_limit_failed(signal.name)
                        self.completed += 1
                        return None

                # Success or other error
                self.completed += 1

                if self.debug:
                    self._debug_output(signal.name, result)

                if result.returncode != 0:
                    return None

                parsed = parse_freqtrade_output(result.stdout)

                if parsed and parsed.get("trades", 0) > self.config.min_trades:
                    parsed.update(
                        {
                            "signal": signal.name,
                            "signal_type": signal.signal_type,
                            "direction": signal.direction,
                            "pair": pair,
                            "timeframe": actual_tf,
                            "allowed_regimes": signal.allowed_regimes,
                            "exit_config": signal.exit_config,
                        }
                    )
                    self._print_result(parsed)
                    return parsed

                return None

            except subprocess.TimeoutExpired:
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self._log_timeout(signal.name, attempt, delay)
                    time.sleep(delay)
                    self.retries_total += 1
                    continue
                else:
                    self.completed += 1
                    return None

            except Exception as e:
                self.completed += 1
                if self.debug:
                    with print_lock:
                        print(f"  💥 {signal.name}: {e}")
                return None

        self.completed += 1
        return None

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calcule le délai avec backoff exponentiel + jitter.

        Args:
            attempt: Numéro de la tentative (0-indexed)

        Returns:
            Délai en secondes
        """
        # Backoff exponentiel: base * 2^attempt
        delay = self.base_delay * (2**attempt)

        # Ajouter du jitter (±25%) pour éviter la synchronisation
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        delay += jitter

        # Plafonner au max
        return min(delay, self.max_delay)

    def _log_rate_limit(self, signal_name: str, attempt: int, delay: float):
        """Log un rate limit avec retry."""
        with print_lock:
            print(
                f"  ⏳ {signal_name}: Rate limit détecté, "
                f"tentative {attempt + 1}/{self.max_retries}, "
                f"attente {delay:.0f}s..."
            )

    def _log_rate_limit_failed(self, signal_name: str):
        """Log un échec après tous les retries."""
        with print_lock:
            print(
                f"  ❌ {signal_name}: Rate limit persistant après "
                f"{self.max_retries} tentatives"
            )

    def _log_timeout(self, signal_name: str, attempt: int, delay: float):
        """Log un timeout avec retry."""
        with print_lock:
            print(
                f"  ⏳ {signal_name}: Timeout, "
                f"tentative {attempt + 1}/{self.max_retries}, "
                f"attente {delay:.0f}s..."
            )

    def run_all(
        self, signals: List[SignalConfig], pairs: List[str], timeframes: List[str]
    ) -> pd.DataFrame:
        """
        Run all backtests in parallel.

        Args:
            signals: List of SignalConfig to test
            pairs: List of trading pairs
            timeframes: List of timeframes

        Returns:
            DataFrame with all results
        """
        # Build task list
        tasks = []
        for sig in signals:
            for pair in pairs:
                for tf in (
                    [sig.timeframe_override] if sig.timeframe_override else timeframes
                ):
                    tasks.append((sig, pair, tf))

        # Remove duplicates
        seen = set()
        unique_tasks = []
        for t in tasks:
            key = (t[0].name, t[1], t[2])
            if key not in seen:
                seen.add(key)
                unique_tasks.append(t)

        self.total = len(unique_tasks)
        self.completed = 0

        # Build cache index once (before worker pool) so workers see a pre-populated
        # _export_index without racing each other. Logged with the count so the
        # user can see immediately how many strats will be served from cache.
        if self.use_cache and self._export_index is None:
            self._export_index = self._build_export_index()

        # Print header
        filter_status = (
            "🔒 FILTRAGE ACTIF"
            if self.config.enable_regime_filter
            else "🔓 SANS FILTRE"
        )
        with print_lock:
            print(f"\n{'=' * 100}")
            print(
                f"🚀 V3 - {self.total} backtests ({self.config.max_workers} workers) - {filter_status}"
            )
            if self.use_cache and self._export_index is not None:
                print(
                    f"⚡ Export cache : {len(self._export_index)} (class, tf) entrées indexées "
                    f"pour timerange={self.config.timerange}"
                )
            print(f"{'=' * 100}\n")

        # Run in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.run_single, *task): task for task in unique_tasks
            }

            # Stagger start to avoid overwhelming the system
            for i in range(min(self.config.max_workers - 1, len(unique_tasks))):
                time.sleep(1.5)

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        if self.retries_total > 0:
            with print_lock:
                print(
                    f"\n  ℹ️  Total retries (rate limit/timeout): {self.retries_total}"
                )
        if self.cache_hits > 0:
            with print_lock:
                print(
                    f"  ⚡ Cache hits (skipped freqtrade): {self.cache_hits} / {self.total}"
                )

        return pd.DataFrame(results)

    def _print_result(self, r: Dict):
        """Print a single result line: trades L/S, WR L/S, PnL, DD + duration, Sharpe."""
        with print_lock:
            # Aggregate L/S trade counts + wins across regimes
            long_n = long_w = short_n = short_w = 0
            for reg in ["bull", "bear", "range", "volatile"]:
                rd = (r.get("regime_stats") or {}).get(reg, {})
                long_d = rd.get("long", {}) or {}
                short_d = rd.get("short", {}) or {}
                long_n += int(long_d.get("trades", 0) or 0)
                short_n += int(short_d.get("trades", 0) or 0)
                long_w += int(long_d.get("wins", 0) or 0)
                short_w += int(short_d.get("wins", 0) or 0)
            long_wr = (long_w / long_n * 100) if long_n > 0 else 0.0
            short_wr = (short_w / short_n * 100) if short_n > 0 else 0.0

            ls_part = f"({long_n:>3d}L/{short_n:>3d}S)"
            wr_part = f"WR={r['win_rate']:5.1f}%(L:{long_wr:>3.0f}%/S:{short_wr:>3.0f}%)"
            dd_dur = r.get("dd_duration_days", 0) or 0
            dd_part = f"DD={r['max_dd_pct']:5.1f}%({dd_dur:>3.0f}d)"
            mkt = r.get("market_change_pct", 0) or 0
            mkt_part = f"MKT={mkt:+6.1f}%"

            # ⚡ prefix on cache hits (no freqtrade subprocess), ▶ on fresh runs
            tag = "⚡" if r.get("_cached") else "▶"

            print(
                f"  {tag} [{self.completed:3d}/{self.total}] "
                f"{r['signal']:<35} {r['pair']:<18} {r['timeframe']:<4} │ "
                f"Tr={r['trades']:>4d} {ls_part} "
                f"{wr_part} "
                f"PnL={r['profit_pct']:+6.1f}% "
                f"{mkt_part} "
                f"{dd_part} "
                f"Sharpe={r['sharpe']:+5.2f}"
            )

    def _debug_output(self, signal_name: str, result):
        """Print debug information."""
        with print_lock:
            print(f"\nDEBUG: {signal_name} | RC={result.returncode}")
            if "TOTAL" in result.stdout:
                for line in result.stdout.split("\n"):
                    if "TOTAL" in line:
                        print(f"  TOTAL LINE: {line}")
                        break
            else:
                print("  ⚠ Pas de TOTAL dans stdout")
                lines = result.stdout.strip().split("\n")
                for l in lines:
                    print(f"    {l}")

            parsed = parse_freqtrade_output(result.stdout)
            print(
                f"  PARSED: trades={parsed.get('trades')}, profit={parsed.get('profit_pct')}"
            )
