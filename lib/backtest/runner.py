# =============================================================================
# FILE: lib/backtest/runner.py
# =============================================================================
"""Backtest runner with parallel execution."""

import json
import shutil
import subprocess
import tempfile
import time
import threading
import zipfile

import re
import random

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

from ..signals.base import SignalConfig
from ..generation.generator import StrategyGenerator
from ..utils.logging import print_lock
from ..utils.colors import color_calmar, color_dd, color_pvalue, color_pnl_composite
from ..utils.helpers import short_pair, sanitize_class_name
from ..null_pool import (
    compute_cache_key,
    load_pool,
    save_pool,
    extract_trades_from_zip,
    pvalue_vs_null,
    pvalue_vs_null_mixed,
    bh_adjusted_pvalues,
    storey_q_values,
    estimate_pi0,
)
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
        # (class_name, timeframe, timerange, pair) → zip path. The pair is
        # essential — without it, per-pair zips for the same (class_name, tf)
        # collide on the index and the cache returns one pair's results
        # mislabeled as another's (silent corruption).
        self.use_cache: bool = bool(getattr(config, "use_cache", False))
        self._export_index: Optional[Dict[Tuple[str, str, str, str], Path]] = None
        self.cache_hits: int = 0

        # Pool cache for live p-value computation during Phase 2 (so p-values
        # can show on the per-line print, not only in the Phase 3 summary).
        # Keyed by cache_key str → DataFrame or None (negative cache).
        self._live_pool_cache: Dict[str, Optional[pd.DataFrame]] = {}
        self._live_pool_lock = threading.Lock()

        # tqdm bar for Phase 1 (set in _build_null_pools_phase). Workers route
        # diagnostic logs through `_phase1_log()` so prints don't drown the bar.
        self._phase1_pbar = None

        # Counter lock — guards self.completed against the race where two
        # workers read the same value, both add 1, both write the same result
        # (lost increment → duplicate IDs in display). Cost: ~1-2 µs per
        # acquire, negligible vs subprocess time.
        self._counter_lock = threading.Lock()

        # Memoize expected timerange per (pair, tf). Computing it requires
        # opening the pair's feather to read its date min/max, ~50ms each ;
        # we do it once per (pair, tf) and serve from cache thereafter.
        self._expected_tr_cache: Dict[Tuple[str, str], str] = {}

    @staticmethod
    def _parse_timerange(tr: str) -> Tuple[Optional[str], Optional[str]]:
        """Split 'YYYYMMDD-YYYYMMDD' (with optionally-empty bounds) into a
        (start, end) pair. Empty bound → None (open-ended on that side).
        """
        if "-" not in (tr or ""):
            return None, None
        parts = (tr or "").split("-", 1)
        return (parts[0].strip() or None, parts[1].strip() or None)

    def _expected_timerange(self, pair: str, tf: str) -> str:
        """Predict the YYYYMMDD-YYYYMMDD freqtrade will stamp for this (pair, tf).

        Freqtrade clips `--timerange` to actual data bounds in the feather:
            effective_start = max(config_start, data_start)
            effective_end   = min(config_end,   data_end)
        We compute the same thing here so cache keys match what freqtrade
        wrote in past meta.json files. Without this, the cached_range varies
        per (pair, tf) (data ends at different days) and strict equality
        against `config.timerange` always fails.

        Falls back to `config.timerange` when the feather can't be located —
        better to miss cache than to pretend we know.
        """
        ck = (pair, tf)
        cached = self._expected_tr_cache.get(ck)
        if cached is not None:
            return cached

        from datetime import datetime, timezone
        safe_pair = pair.replace("/", "_").replace(":", "_")
        candidates = [
            Path(self.config.data_dir) / "futures" / f"{safe_pair}-{tf}-futures.feather",
            Path(self.config.data_dir) / "futures" / f"{safe_pair}-{tf}.feather",
            Path(self.config.data_dir) / f"{safe_pair}-{tf}-futures.feather",
            Path(self.config.data_dir) / f"{safe_pair}-{tf}.feather",
        ]
        feather = next((p for p in candidates if p.exists()), None)
        result = self.config.timerange
        if feather is not None:
            try:
                df = pd.read_feather(feather, columns=["date"])
                df["date"] = pd.to_datetime(df["date"], utc=True)
                data_start = df["date"].min()
                data_end = df["date"].max()
                cfg_start, cfg_end = self._parse_timerange(self.config.timerange)
                if cfg_start:
                    cfg_start_dt = datetime.strptime(cfg_start, "%Y%m%d").replace(tzinfo=timezone.utc)
                    eff_start = max(cfg_start_dt, data_start)
                else:
                    eff_start = data_start
                if cfg_end:
                    cfg_end_dt = datetime.strptime(cfg_end, "%Y%m%d").replace(tzinfo=timezone.utc)
                    eff_end = min(cfg_end_dt, data_end)
                else:
                    eff_end = data_end
                result = f"{eff_start.strftime('%Y%m%d')}-{eff_end.strftime('%Y%m%d')}"
            except Exception:
                pass
        self._expected_tr_cache[ck] = result
        return result

    def _bump_completed(self) -> int:
        """Atomically increment self.completed and return the new value.

        Worker threads must capture the returned id and pass it to
        `_print_result(line_id=...)` to display their OWN id, not whatever
        value `self.completed` happens to hold at print time.
        """
        with self._counter_lock:
            self.completed += 1
            return self.completed

    def _get_export_name(self, signal: SignalConfig, pair: str, timeframe: str) -> str:
        """Génère un nom de fichier unique pour l'export."""
        safe_pair = pair.replace("/", "_").replace(":", "_")
        return f"{signal.name}_{safe_pair}_{timeframe}"

    def _make_isolated_user_data(self, prefix: str) -> Path:
        """Create an isolated temporary user_data_dir for a single freqtrade run.

        Each parallel worker gets its own dir → no collision on the
        backtest-result-{TS}.zip filename freqtrade writes (it ignores
        --export-filename). Created under the project's user_data/ so
        rename() across filesystems is never needed when consolidating.
        Caller is responsible for shutil.rmtree.
        """
        parent = self.export_dir.parent
        tmp = Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent)))
        (tmp / "backtest_results").mkdir(parents=True, exist_ok=True)
        return tmp

    def _consolidate_export(self, isolated: Path, export_name: str) -> None:
        """Move freqtrade's backtest-result-{TS}.zip + .meta.json from the
        isolated worker dir to the central export_dir under a deterministic
        name (`{export_name}.zip` / `.meta.json`).

        This is what freqtrade WOULD do if it honored --export-filename. By
        consolidating ourselves we keep the cache index (`_build_export_index`)
        functional across runs while avoiding the timestamp-collision bug
        when multiple workers write concurrently.
        """
        src_dir = isolated / "backtest_results"
        if not src_dir.exists():
            return
        zips = list(src_dir.glob("*.zip"))
        if not zips:
            return
        src_zip = max(zips, key=lambda p: p.stat().st_mtime)
        # freqtrade names meta as `{stem}.meta.json` (see get_backtest_metadata_filename)
        src_meta = src_zip.parent / f"{src_zip.stem}.meta.json"
        dst_zip = self.export_dir / f"{export_name}.zip"
        dst_meta = self.export_dir / f"{export_name}.meta.json"
        # Replace any prior result for the same (signal, pair, tf) atomically.
        for dst in (dst_zip, dst_meta):
            if dst.exists():
                dst.unlink()
        src_zip.rename(dst_zip)
        if src_meta.exists():
            src_meta.rename(dst_meta)

    def _build_export_index(self) -> Dict[Tuple[str, str, str, str], Path]:
        """Scan user_data/backtest_results/*.meta.json → {(class_name, tf, timerange, pair): zip_path}.

        We key on `(class_name, timeframe, timerange, pair)` so a cached entry
        is only reused when ALL of strategy class, backtest window, AND pair
        match. The pair is read from inside each zip (meta.json doesn't carry
        it). Without the pair in the key, per-pair zips collide and one pair's
        result silently masquerades as another's. Stake / wallet / MOT changes
        don't invalidate — bump cache via `--refresh` if needed.
        """
        index: Dict[Tuple[str, str, str, str], Path] = {}
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
                    if start_ts > 10**11:  # ms
                        start_ts = start_ts / 1000
                        end_ts = end_ts / 1000
                    from datetime import datetime, timezone
                    start_str = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y%m%d")
                    end_str = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y%m%d")
                    cached_range = f"{start_str}-{end_str}"
                else:
                    cached_range = ""
                pair = self._extract_pair_from_zip(zip_path, class_name)
                if pair is None:
                    continue
                # Compare cached_range to the EXPECTED range freqtrade would
                # use for this (pair, tf) under the current config — that's
                # config.timerange clipped to feather data bounds. Without
                # this, cached zips with end shifted by data clipping never
                # match config.timerange (closed) and the cache always misses.
                expected = self._expected_timerange(pair, tf)
                if cached_range and cached_range != expected:
                    continue
                index[(class_name, tf, expected, pair)] = zip_path
        return index

    def _extract_pair_from_zip(self, zip_path: Path, class_name: str) -> Optional[str]:
        """Read the first trade's pair from the freqtrade backtest zip.

        Used to disambiguate per-pair zips that share a class_name. Falls back
        to top-level `pairs` list when trades is empty. Returns None on any
        I/O / parse error so the caller skips that zip.
        """
        try:
            with zipfile.ZipFile(zip_path) as z:
                inner = next(
                    (n for n in z.namelist()
                     if n.endswith(".json") and "_config" not in n),
                    None,
                )
                if not inner:
                    return None
                data = json.loads(z.read(inner))
            strat = (data.get("strategy") or {}).get(class_name) or {}
            trades = strat.get("trades") or []
            if trades:
                p = trades[0].get("pair")
                if p:
                    return str(p)
            pairs = data.get("pairs") or []
            if pairs and isinstance(pairs, list):
                return str(pairs[0])
        except Exception:
            return None
        return None

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
            "profit_pct_long": float(strat.get("profit_total_long", 0) or 0) * 100.0,
            "profit_pct_short": float(strat.get("profit_total_short", 0) or 0) * 100.0,
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

        # Cache hit check — skip freqtrade entirely if a previous export for
        # the same (class_name, tf, expected_range, pair) is on disk. The
        # expected_range is what freqtrade WILL stamp for this (pair, tf)
        # given config.timerange clipped to feather data bounds — same value
        # used by `_build_export_index`, so strict equality matches.
        if self.use_cache and self._export_index is not None:
            expected_tr = self._expected_timerange(pair, actual_tf)
            cache_key = (class_name, actual_tf, expected_tr, pair)
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
                        "stoploss": signal.stoploss,
                        "roi": json.dumps(signal.roi, sort_keys=True),
                        "_cached": True,
                    })
                    pv = self._live_pvalue_for_result(cached)
                    if pv is not None:
                        cached["p_value"] = pv
                    my_id = self._bump_completed()
                    self.cache_hits += 1
                    self._print_result(cached, line_id=my_id)
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
            # Pool runs override wallet to 10000 so 20 parallel trades at
            # stake=100 each can fit (20 × 100 = 2000 < 10000 with margin).
            "--dry-run-wallet",
            str(10000.0 if signal.signal_type == "random_baseline" else self.config.dry_run_wallet),
            # Fixed stake (not "unlimited") so every trade exposes the same
            # capital — profit_ratio is directly comparable across the pool.
            # 100 USDC matches the user's normal config.json stake.
            "--stake-amount",
            str(100.0 if signal.signal_type == "random_baseline" else self.config.stake_amount),
            "--export",
            "trades",
            "--export-filename",
            export_name,
            # MOT=50 is the sweet spot: enough parallelism for ~all 2000
            # random entries to find a slot over the timerange (50 parallel
            # x ~6500 bars / ~30 bar trade duration ≈ 10k capacity), while
            # staying fast (per-bar cost scales with concurrent trades).
            # MOT=999 made freqtrade churn 100 simultaneous positions and
            # was visibly slow.
            "--max-open-trades",
            str(50 if signal.signal_type == "random_baseline" else self.config.max_open_trades),
            "--breakdown",
            "month",
        ]

        # Isolate this freqtrade run in its own user_data_dir so concurrent
        # workers don't collide on backtest_results/backtest-result-{TS}.zip
        # (freqtrade ignores --export-filename → all workers write to the
        # default path with second-resolution timestamps → corrupt zips
        # AND corrupt .meta.json that later poison every backtest's
        # load_prior_backtest scan). On success we consolidate the result
        # to self.export_dir under a deterministic name so the cache index
        # still works across runs.
        #
        # Pool builds set _user_data_dir_override themselves because they
        # need to read the zip back from the isolated dir before cleanup.
        pool_uds = signal.params.get("_user_data_dir_override")
        isolated_dir: Optional[Path] = None
        if pool_uds:
            cmd += ["--user-data-dir", str(pool_uds)]
        else:
            isolated_dir = self._make_isolated_user_data("ftrun_")
            cmd += ["--user-data-dir", str(isolated_dir)]

        try:
            return self._run_freqtrade_loop(
                cmd, signal, pair, actual_tf, isolated_dir, export_name,
            )
        finally:
            if isolated_dir is not None:
                shutil.rmtree(isolated_dir, ignore_errors=True)

    def _run_freqtrade_loop(
        self,
        cmd: list,
        signal,
        pair: str,
        actual_tf: str,
        isolated_dir: Optional[Path],
        export_name: str,
    ) -> Optional[Dict]:
        """Subprocess retry loop extracted so the parent run_single can
        wrap it in try/finally for tempdir cleanup without restructuring
        every early-return path.
        """
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

                # Bail on signal-kill (rc<0 means killed by signal: SIGINT=-2,
                # SIGTERM=-15, SIGKILL=-9). Retrying is pointless and the
                # rate-limit regex can false-match CCXT init logs → infinite
                # retry loop on user Ctrl-C otherwise.
                if result.returncode < 0:
                    my_id = self._bump_completed()
                    sig_label = {-2: "SIGINT", -9: "SIGKILL", -15: "SIGTERM"}.get(
                        result.returncode, f"sig{-result.returncode}"
                    )
                    self._print_error(my_id, signal.name, pair, actual_tf, f"killed by {sig_label}")
                    if signal.signal_type == "random_baseline" or self.debug:
                        sig_name = {-2: "SIGINT", -9: "SIGKILL", -15: "SIGTERM"}.get(
                            result.returncode, f"sig{-result.returncode}"
                        )
                        with print_lock:
                            print(
                                f"  🛑 freqtrade killed by {sig_name} "
                                f"({signal.name} {pair} {actual_tf}) — bailing without retry"
                            )
                    return None

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
                        my_id = self._bump_completed()
                        self._print_error(my_id, signal.name, pair, actual_tf, "rate-limit retries exhausted")
                        return None

                # Success or other error — capture our own id for the print path
                my_id = self._bump_completed()

                if self.debug:
                    self._debug_output(signal.name, result)

                if result.returncode != 0:
                    self._print_error(my_id, signal.name, pair, actual_tf, f"freqtrade rc={result.returncode}")
                    if signal.signal_type == "random_baseline" or self.debug:
                        with print_lock:
                            err = (result.stderr or "")[-500:]
                            out_tail = (result.stdout or "")[-300:]
                            print(
                                f"    stderr: {err!r}\n"
                                f"    stdout(tail): {out_tail!r}"
                            )
                    return None

                # Move freqtrade's output from the isolated worker dir to
                # the central export_dir under a deterministic name. Pool
                # runs (isolated_dir is None) manage their own dir and read
                # the zip themselves before cleanup.
                if isolated_dir is not None:
                    self._consolidate_export(isolated_dir, export_name)

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
                            "stoploss": signal.stoploss,
                            "roi": json.dumps(signal.roi, sort_keys=True),
                        }
                    )
                    pv = self._live_pvalue_for_result(parsed)
                    if pv is not None:
                        parsed["p_value"] = pv
                    self._print_result(parsed, line_id=my_id)
                    return parsed

                # rc=0 but no parseable trades — common for filter-strict
                # signals on quiet pairs. Show as one-liner so IDs don't jump
                # silently; debug adds the stdout tail for diagnosis.
                n_tr = (parsed or {}).get("trades", 0)
                parsed_ok = parsed is not None
                self._print_error(
                    my_id, signal.name, pair, actual_tf,
                    f"rc=0 trades={n_tr} (parsed={parsed_ok})",
                )
                if signal.signal_type == "random_baseline" or self.debug:
                    out_tail = (result.stdout or "")[-300:]
                    with print_lock:
                        print(f"    stdout(tail): {out_tail!r}")
                return None

            except subprocess.TimeoutExpired:
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self._log_timeout(signal.name, attempt, delay)
                    time.sleep(delay)
                    self.retries_total += 1
                    continue
                else:
                    my_id = self._bump_completed()
                    self._print_error(my_id, signal.name, pair, actual_tf, "timeout (retries exhausted)")
                    return None

            except Exception as e:
                my_id = self._bump_completed()
                self._print_error(
                    my_id, signal.name, pair, actual_tf,
                    f"{type(e).__name__}: {e!s:.80s}",
                )
                return None

        my_id = self._bump_completed()
        self._print_error(my_id, signal.name, pair, actual_tf, "retry loop exhausted (no result)")
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

    # ============================================================
    # === Null-pool phases ===
    # ============================================================

    # Pool dims: (pair, tf, exit_config, direction). Exit logic dramatically
    # changes per-trade return distribution (trailing vs fixed vs none) so it
    # MUST be a pool dim. SL/ROI are held neutral within each cell — minor
    # variations don't shift the distribution enough to justify exploding
    # the pool count. Net pool count for a typical grid:
    # 6 pairs × 3 TFs × ~3 exits × 2 directions = ~108 pools.
    NULL_POOL_NEUTRAL_SL: float = -0.05
    NULL_POOL_NEUTRAL_ROI: Dict[str, float] = {"0": 0.02}

    def _null_pool_cells(
        self,
        signals: List[SignalConfig],
        pairs: List[str],
        timeframes: List[str],
    ) -> List[Tuple[str, str, str, str]]:
        """Distinct (pair, tf, exit_config, direction) cells."""
        base: set = set()
        for sig in signals:
            tfs = (
                [sig.timeframe_override] if sig.timeframe_override else timeframes
            )
            for pair in pairs:
                for tf in tfs:
                    base.add((pair, tf, sig.exit_config))
        cells = []
        for pair, tf, exit_cfg in sorted(base):
            for direction in ("long", "short"):
                cells.append((pair, tf, exit_cfg, direction))
        return cells

    def _synthesize_pool_signal(
        self, exit_cfg: str, direction: str
    ) -> SignalConfig:
        """Build the SignalConfig for a null-pool run on a given exit config."""
        seed = int(self.config.null_pool_seed)
        dir_short = {"long": "L", "short": "S", "both": "B"}.get(direction, "B")
        # Class-name uniqueness: encode exit + direction so each cell maps
        # to a distinct freqtrade strategy file / cached export.
        name = f"random_baseline_seed{seed}_{exit_cfg}_dir{dir_short}"
        return SignalConfig(
            name=name,
            signal_type="random_baseline",
            direction=direction,
            params={
                "seed": seed,
                "target_trades": int(self.config.null_pool_target_trades),
            },
            roi=dict(self.NULL_POOL_NEUTRAL_ROI),
            stoploss=float(self.NULL_POOL_NEUTRAL_SL),
            exit_config=exit_cfg,
            allowed_regimes=["bull", "bear", "range", "volatile"],
        )

    def _find_pool_export_zip(
        self, class_name: str, since: float
    ) -> Optional[Path]:
        """Locate the freshly-written export zip containing `class_name`.

        Scans only zips with mtime >= since to avoid expensive full scans.
        Returns the first match or None.
        """
        candidates = sorted(
            (p for p in self.export_dir.glob("*.zip") if p.stat().st_mtime >= since),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for zp in candidates:
            try:
                with zipfile.ZipFile(zp) as z:
                    inner = next(
                        (
                            n
                            for n in z.namelist()
                            if n.endswith(".json") and "_config" not in n
                        ),
                        None,
                    )
                    if not inner:
                        continue
                    data = json.loads(z.read(inner))
                if class_name in (data.get("strategy") or {}):
                    return zp
            except Exception:
                continue
        return None

    # Minimum trades a pool must have to be usable. Lowered from 50 → 20
    # because higher TFs (4h+) over short timeranges naturally produce
    # fewer trades. Below 20 the bootstrap is too noisy.
    NULL_POOL_MIN_TRADES: int = 20

    def _phase1_log(self, msg: str) -> None:
        """Print without breaking the tqdm bar.

        tqdm provides .write() which coordinates with the rendered bar (clears
        it, prints the message, redraws). When the bar isn't active (e.g.
        running outside Phase 1) this falls back to a plain locked print.
        """
        if self._phase1_pbar is not None:
            self._phase1_pbar.write(msg)
        else:
            with print_lock:
                print(msg, flush=True)

    def _build_null_pool_for_cell(
        self,
        cell: Tuple[str, str, str, str],
    ) -> bool:
        """Build (or reuse) the null pool for a single cell.

        Each cell runs freqtrade in an isolated temp `user_data_dir` so the
        export goes to a unique `{tmpdir}/backtest_results/` — avoids the
        race condition where parallel workers write `backtest-result-{TS}.zip`
        with identical second-resolution timestamps and corrupt each other's
        zips (BadZipFile / CRC errors).

        Returns True on cache hit / successful build, False on failure.
        """
        pair, tf, exit_cfg, direction = cell
        cache_key = compute_cache_key(
            pair, tf, exit_cfg,
            self.NULL_POOL_NEUTRAL_SL,
            self.NULL_POOL_NEUTRAL_ROI,
            self.config.timerange, int(self.config.null_pool_seed),
            direction=direction,
        )
        cache_dir = self.config.null_pool_cache_dir

        if not self.config.refresh_null_pool:
            existing = load_pool(cache_key, cache_dir)
            if existing is not None and len(existing) >= self.NULL_POOL_MIN_TRADES:
                return True

        pool_sig = self._synthesize_pool_signal(exit_cfg, direction)
        class_name = f"S_{sanitize_class_name(pool_sig.name)}_{tf}"

        # Isolate this freqtrade run: its export goes to a private temp dir.
        pool_user_data = self._make_isolated_user_data("ftpool_")
        # Smuggled through signal.params so run_single can append --user-data-dir
        pool_sig.params["_user_data_dir_override"] = str(pool_user_data)

        try:
            result = self.run_single(pool_sig, pair, tf)
            if result is None:
                self._phase1_log(
                    f"  ! Pool build failed (run_single None): "
                    f"{pair} {tf} {exit_cfg} dir={direction}"
                )
                return False

            zips = list((pool_user_data / "backtest_results").glob("*.zip"))
            if not zips:
                self._phase1_log(
                    f"  ! Pool build failed (no zip in {pool_user_data.name}): "
                    f"{class_name}"
                )
                return False
            # Only one zip should exist in the isolated dir; pick the most
            # recent if somehow multiple were written.
            zp = max(zips, key=lambda p: p.stat().st_mtime)

            df = extract_trades_from_zip(zp, class_name, log=self._phase1_log)
            if df is None:
                self._phase1_log(
                    f"  ! Pool build failed (extract None): {class_name} from {zp.name}"
                )
                return False
            if len(df) < self.NULL_POOL_MIN_TRADES:
                self._phase1_log(
                    f"  ! Pool too small ({len(df)} < {self.NULL_POOL_MIN_TRADES}): "
                    f"{class_name}"
                )
                return False
            save_pool(df, cache_key, cache_dir)
            return True
        finally:
            shutil.rmtree(pool_user_data, ignore_errors=True)

    def _build_null_pools_phase(
        self,
        signals: List[SignalConfig],
        pairs: List[str],
        timeframes: List[str],
    ) -> None:
        """Phase 1: ensure every (pair, tf, exit, sl, roi) cell has a null pool."""
        cells = self._null_pool_cells(signals, pairs, timeframes)
        if not cells:
            return

        cache_dir = self.config.null_pool_cache_dir
        seed = int(self.config.null_pool_seed)
        n_cached = 0
        if not self.config.refresh_null_pool:
            for c in cells:
                pair, tf, exit_cfg, direction = c
                key = compute_cache_key(
                    pair, tf, exit_cfg,
                    self.NULL_POOL_NEUTRAL_SL,
                    self.NULL_POOL_NEUTRAL_ROI,
                    self.config.timerange, seed, direction=direction,
                )
                if (cache_dir / f"{key}.parquet").exists():
                    n_cached += 1

        with print_lock:
            print(f"\n{'=' * 100}")
            print(
                f"🎲 PHASE 1 — Null pool builder: {len(cells)} cells "
                f"({n_cached} cached, {len(cells) - n_cached} to build, seed={seed}, "
                f"target={self.config.null_pool_target_trades} trades)"
            )
            print(f"{'=' * 100}\n")

        # Build missing pools (parallel, like run_all). Each cell synthesizes
        # a distinct random_baseline strategy so threads don't collide.
        # Pre-flight: pre-build the export index if not already done so the
        # cache lookup inside run_single sees the fresh state.
        if self.use_cache and self._export_index is None:
            self._export_index = self._build_export_index()

        cells_to_run = cells if self.config.refresh_null_pool else [
            c for c in cells
            if not (cache_dir / (
                compute_cache_key(
                    c[0], c[1], c[2],
                    self.NULL_POOL_NEUTRAL_SL,
                    self.NULL_POOL_NEUTRAL_ROI,
                    self.config.timerange, seed, direction=c[3],
                ) + ".parquet"
            )).exists()
        ]
        # Save & restore total/completed so phase 1 progress doesn't leak into phase 2
        saved_total, saved_completed = self.total, self.completed
        self.total = len(cells_to_run)
        self.completed = 0

        n_built = 0
        if cells_to_run:
            import sys
            # Defensive: try .auto first, then plain tqdm, then None.
            # If neither imports we fall back to a one-shot startup print so
            # the user sees activity (very rare: tqdm is in requirements).
            tqdm_cls = None
            try:
                from tqdm.auto import tqdm as tqdm_cls
            except Exception:
                try:
                    from tqdm import tqdm as tqdm_cls
                except Exception as e:
                    with print_lock:
                        print(
                            f"  Note: tqdm unavailable ({type(e).__name__}: {e}); "
                            f"Phase 1 will run without progress bar.",
                            flush=True,
                        )

            pbar = None
            if tqdm_cls is not None:
                # ncols=80 fixed (some SSH sessions report 0 cols and tqdm
                # silently disables itself). mininterval=0 + miniters=1 so
                # every update redraws. ascii=True for non-Unicode terminals.
                # refresh() + flush so the initial "0/N" bar shows before
                # the first completion lands.
                try:
                    pbar = tqdm_cls(
                        total=len(cells_to_run),
                        desc="Pool builds",
                        unit="cell",
                        file=sys.stdout,
                        ncols=80,
                        mininterval=0,
                        miniters=1,
                        ascii=True,
                        leave=True,
                        position=0,
                    )
                    pbar.refresh()
                    sys.stdout.flush()
                except Exception as e:
                    with print_lock:
                        print(
                            f"  Note: tqdm init failed ({type(e).__name__}: {e}); "
                            f"continuing without bar.",
                            flush=True,
                        )
                    pbar = None

            self._phase1_pbar = pbar

            try:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
                    futures = {
                        ex.submit(self._build_null_pool_for_cell, c): c
                        for c in cells_to_run
                    }
                    for f in as_completed(futures):
                        if f.result():
                            n_built += 1
                        if pbar is not None:
                            pbar.update(1)
            finally:
                if pbar is not None:
                    pbar.close()
                self._phase1_pbar = None

        self.total, self.completed = saved_total, saved_completed
        with print_lock:
            print(
                f"\n🎲 Pool cache: {n_cached} hits + {n_built} built / {len(cells)} cells\n"
            )

    def _load_pool_cached(
        self, cache_key: str
    ) -> Optional[pd.DataFrame]:
        """Thread-safe pool load with negative cache."""
        with self._live_pool_lock:
            if cache_key in self._live_pool_cache:
                return self._live_pool_cache[cache_key]
            df = load_pool(cache_key, self.config.null_pool_cache_dir)
            self._live_pool_cache[cache_key] = df
            return df

    def _observed_ls_split(self, r: Dict) -> Tuple[int, int]:
        """Recover the long/short trade counts of an observed result.

        Tries regime_stats first (the parser's canonical source); falls back
        to attributing wholesale via signal.direction when buckets are empty
        (random_baseline tag missing, or older cached results).
        """
        n_long = n_short = 0
        for reg in ("bull", "bear", "range", "volatile"):
            rd = (r.get("regime_stats") or {}).get(reg, {})
            n_long += int((rd.get("long") or {}).get("trades", 0) or 0)
            n_short += int((rd.get("short") or {}).get("trades", 0) or 0)
        if n_long + n_short == 0:
            total = int(r.get("trades", 0) or 0)
            sig_dir = (r.get("direction") or "both").lower()
            if sig_dir == "long":
                n_long = total
            elif sig_dir == "short":
                n_short = total
            # direction="both" with empty regime_stats → 50/50 fallback
            elif total > 0:
                n_long = total // 2
                n_short = total - n_long
        return n_long, n_short

    def _live_pvalue_for_result(self, r: Dict) -> Optional[float]:
        """Compute the empirical p-value for a single result on the fly.

        Dispatches on signal.direction to the appropriate pool(s):
          - "long"  → bootstrap against long-only pool
          - "short" → bootstrap against short-only pool
          - "both"  → mixed bootstrap (sample n_long from L pool + n_short
                      from S pool, matching the observed signal's L/S split)
        """
        if not getattr(self.config, "enable_null_pool", False):
            return None
        if r.get("signal_type") == "random_baseline":
            return None  # don't bootstrap a pool against itself

        pair = str(r["pair"])
        tf = str(r["timeframe"])
        exit_cfg = str(r.get("exit_config") or "none")
        seed = int(self.config.null_pool_seed)
        timerange = self.config.timerange
        n_boot = int(self.config.null_pool_n_bootstrap)
        cap_pct = float(self.config.null_pool_capital_pct)
        block_len = float(self.config.null_pool_block_len)
        observed_pct = float(r.get("profit_pct", 0) or 0)
        sig_dir = (r.get("direction") or "both").lower()
        # Pool key matches the signal's exit_config (drives trade dist) but
        # uses NEUTRAL SL/ROI (minor variations don't shift dist enough to
        # justify the cost of a separate pool per SL/ROI combo).
        ns, nr = self.NULL_POOL_NEUTRAL_SL, self.NULL_POOL_NEUTRAL_ROI

        try:
            if sig_dir == "long":
                key = compute_cache_key(
                    pair, tf, exit_cfg, ns, nr, timerange, seed,
                    direction="long",
                )
                pool = self._load_pool_cached(key)
                if pool is None or "profit_ratio" not in pool.columns:
                    return None
                return pvalue_vs_null(
                    pool["profit_ratio"].values,
                    observed_return_pct=observed_pct,
                    n_trades=int(r.get("trades", 0) or 0),
                    capital_pct_per_trade=cap_pct,
                    n_bootstrap=n_boot,
                    seed=seed,
                    mean_block_len=block_len,
                )
            elif sig_dir == "short":
                key = compute_cache_key(
                    pair, tf, exit_cfg, ns, nr, timerange, seed,
                    direction="short",
                )
                pool = self._load_pool_cached(key)
                if pool is None or "profit_ratio" not in pool.columns:
                    return None
                return pvalue_vs_null(
                    pool["profit_ratio"].values,
                    observed_return_pct=observed_pct,
                    n_trades=int(r.get("trades", 0) or 0),
                    capital_pct_per_trade=cap_pct,
                    n_bootstrap=n_boot,
                    seed=seed,
                    mean_block_len=block_len,
                )
            else:  # "both" — mixed bootstrap
                n_long, n_short = self._observed_ls_split(r)
                if n_long + n_short == 0:
                    return None
                key_l = compute_cache_key(
                    pair, tf, exit_cfg, ns, nr, timerange, seed, direction="long",
                )
                key_s = compute_cache_key(
                    pair, tf, exit_cfg, ns, nr, timerange, seed, direction="short",
                )
                pool_l = self._load_pool_cached(key_l)
                pool_s = self._load_pool_cached(key_s)
                long_arr = (
                    pool_l["profit_ratio"].values
                    if pool_l is not None and "profit_ratio" in pool_l.columns
                    else np.array([])
                )
                short_arr = (
                    pool_s["profit_ratio"].values
                    if pool_s is not None and "profit_ratio" in pool_s.columns
                    else np.array([])
                )
                if (n_long > 0 and len(long_arr) == 0) or (n_short > 0 and len(short_arr) == 0):
                    return None
                return pvalue_vs_null_mixed(
                    long_arr, short_arr,
                    observed_return_pct=observed_pct,
                    n_long=n_long, n_short=n_short,
                    capital_pct_per_trade=cap_pct,
                    n_bootstrap=n_boot,
                    seed=seed,
                    mean_block_len=block_len,
                )
        except Exception:
            return None

    def _compute_pvalues_phase(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 3: ensure every row has p_value (filled live during Phase 2),
        then add p_value_adj (BH-FDR over the whole batch).

        Live computation in run_single populates p_value as each backtest
        finishes. This phase backfills any rows the live path skipped (e.g.,
        rows missing stoploss/roi metadata or with parse errors), then adds
        FDR — which is by definition a global step.
        """
        if df is None or len(df) == 0:
            return df

        # Pre-fill p_value column from any live-computed values
        if "p_value" not in df.columns:
            df = df.copy()
            df["p_value"] = float("nan")

        cache_dir = self.config.null_pool_cache_dir
        seed = int(self.config.null_pool_seed)
        n_boot = int(self.config.null_pool_n_bootstrap)
        cap_pct = float(self.config.null_pool_capital_pct)
        block_len = float(self.config.null_pool_block_len)
        timerange = self.config.timerange

        pvals = []
        for _, r in df.iterrows():
            # Honor live-computed p-value if present (avoid double bootstrap)
            existing = r.get("p_value")
            if existing is not None and not (
                isinstance(existing, float) and pd.isna(existing)
            ):
                pvals.append(float(existing))
                continue
            # Backfill via the same dispatcher used live (consistent direction
            # handling, mixed bootstrap for "both", thread-safe pool cache).
            p = self._live_pvalue_for_result(dict(r))
            pvals.append(float(p) if p is not None else float("nan"))

        # Silence unused-vars warning when pool_cache path is removed
        _ = (cache_dir, seed, n_boot, cap_pct, block_len, timerange)

        df = df.copy()
        df["p_value"] = pvals
        # BH-FDR + Storey q-values over non-NaN p-values only.
        # BH stays the conservative reference; q-value (BH × π₀) recovers
        # power when many tests have a real edge. Always q ≤ p_adj_BH.
        valid = df["p_value"].notna()
        if valid.sum() > 0:
            p_valid = df.loc[valid, "p_value"].values
            adj = bh_adjusted_pvalues(p_valid)
            q = storey_q_values(p_valid)
            pi0 = estimate_pi0(p_valid)
            df["p_value_adj"] = float("nan")
            df["q_value"] = float("nan")
            df.loc[valid, "p_value_adj"] = adj
            df.loc[valid, "q_value"] = q
            print(
                f"  Storey: pi0={pi0:.3f} estime, q_value = BH x {pi0:.3f} "
                f"(plus permissif que BH si pi0 < 1)"
            )
        else:
            df["p_value_adj"] = float("nan")
            df["q_value"] = float("nan")
        return df

    # ============================================================
    # === run_all (orchestrator) ===
    # ============================================================

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
        # === Phase 1: build null pools (only when --null-pool enabled) ===
        if getattr(self.config, "enable_null_pool", False):
            self._build_null_pools_phase(signals, pairs, timeframes)

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

        df = pd.DataFrame(results)

        # === Phase 3: empirical p-values via bootstrap on the null pools ===
        if getattr(self.config, "enable_null_pool", False) and len(df) > 0:
            df = self._compute_pvalues_phase(df)
            n_sig = int(((df["p_value"].notna()) & (df["p_value"] < 0.05)).sum())
            n_sig_adj = int(((df["p_value_adj"].notna()) & (df["p_value_adj"] < 0.05)).sum())
            with print_lock:
                print(
                    f"  🎲 Bootstrap p-values: {n_sig}/{len(df)} signals at p<0.05 raw, "
                    f"{n_sig_adj} after BH-FDR"
                )
                self._print_phase3_summary(df)

        return df

    def _print_phase3_summary(self, df: pd.DataFrame) -> None:
        """Re-print result lines with p-values attached, sorted by raw p-value.

        Phase 2 prints lines as backtests complete — at that point the bootstrap
        hasn't run yet, so p_value isn't known. This summary fills the gap by
        re-emitting each line with the same format plus `p=X.XXX*` suffix
        (* = p<0.05, • = p<0.10, blank otherwise).
        """
        if len(df) == 0:
            return
        # Sort: significant first (smallest p), then by sharpe descending
        df_sorted = df.copy()
        df_sorted["_p_sort"] = df_sorted["p_value"].fillna(1.0)
        df_sorted = df_sorted.sort_values(
            ["_p_sort", "sharpe"], ascending=[True, False]
        )

        print(f"\n{'=' * 100}")
        print("🎲 PHASE 3 — RÉSULTATS AVEC P-VALUES (triés p-value, puis Sharpe)")
        print(f"{'=' * 100}")

        for i, (_, r) in enumerate(df_sorted.iterrows(), 1):
            # Build same display blocks as _print_result for consistency
            long_n = long_w = short_n = short_w = 0
            for reg in ["bull", "bear", "range", "volatile"]:
                rd = (r.get("regime_stats") or {}).get(reg, {})
                long_n += int((rd.get("long") or {}).get("trades", 0) or 0)
                short_n += int((rd.get("short") or {}).get("trades", 0) or 0)
                long_w += int((rd.get("long") or {}).get("wins", 0) or 0)
                short_w += int((rd.get("short") or {}).get("wins", 0) or 0)
            total_tr = int(r.get("trades", 0) or 0)
            if (long_n + short_n) == 0 and total_tr > 0:
                sig_dir = (r.get("direction") or "both").lower()
                wins_total = int(r.get("wins", 0) or 0)
                if sig_dir == "long":
                    long_n, long_w = total_tr, wins_total
                elif sig_dir == "short":
                    short_n, short_w = total_tr, wins_total

            pv = r.get("p_value")
            pv_adj = r.get("p_value_adj")
            if pv is None or (isinstance(pv, float) and pd.isna(pv)):
                pv_str = "  n/a "
                marker = " "
            else:
                if pv < 0.05:
                    marker = "*"
                elif pv < 0.10:
                    marker = "•"
                else:
                    marker = " "
                pv_str = f"{pv:.3f}"
            adj_str = (
                f"{pv_adj:.3f}"
                if pv_adj is not None and not (isinstance(pv_adj, float) and pd.isna(pv_adj))
                else "  n/a"
            )

            print(
                f"  [{i:3d}/{len(df_sorted)}] "
                f"{r['signal']:<35} {short_pair(r['pair']):<6} {r['timeframe']:<4} │ "
                f"Tr={total_tr:>4d} ({long_n:>3d}L/{short_n:>3d}S) "
                f"PnL={r['profit_pct']:+6.1f}% "
                f"CAL={r.get('calmar', 0) or 0:+6.2f} │ "
                f"p={pv_str}{marker} (adj={adj_str})"
            )

    def _print_error(self, line_id: int, signal_name: str, pair: str, tf: str, reason: str):
        """One-liner for failed backtests, same column structure as success.

        Used for hard failures (rc!=0, signal kill, timeout, exception). NOT
        used for `rc=0 with 0 trades` — that's expected for filter-strict
        signals on quiet pairs and would just clutter the output. The
        verbose stderr/stdout dump remains gated behind --debug.
        """
        with print_lock:
            print(
                f"  ❌ [{line_id:3d}/{self.total}] "
                f"{signal_name:<35} {short_pair(pair):<6} {tf:<4} │ {reason}"
            )

    def _print_result(self, r: Dict, line_id: Optional[int] = None):
        """Print a single result line: trades L/S, WR L/S, PnL, DD + duration, Calmar.

        Suppresses output for random_baseline pool builds — Phase 1 has its own
        tqdm bar; per-line 🎲 prints would drown out the bar and clutter the
        diagnostic. Failures still log via the dedicated paths.

        `line_id` (when provided by the caller via `_bump_completed()`) lets
        each worker display ITS OWN counter value instead of `self.completed`
        which moves under concurrent workers (duplicate IDs in display).
        """
        if r.get("signal_type") == "random_baseline":
            return
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

            # Fallback for cached results predating the tag-format fix or for
            # tags the regex parser couldn't bucket: attribute via
            # signal.direction wholesale (works for direction="long" or "short";
            # direction="both" rows stay 0/0 since we'd need per-trade detail).
            total_tr = int(r.get("trades", 0) or 0)
            if (long_n + short_n) == 0 and total_tr > 0:
                sig_dir = (r.get("direction") or "both").lower()
                wins_total = int(r.get("wins", 0) or 0)
                if sig_dir == "long":
                    long_n, long_w = total_tr, wins_total
                elif sig_dir == "short":
                    short_n, short_w = total_tr, wins_total
            long_wr = (long_w / long_n * 100) if long_n > 0 else 0.0
            short_wr = (short_w / short_n * 100) if short_n > 0 else 0.0

            ls_part = f"({long_n:>3d}L/{short_n:>3d}S)"
            wr_part = f"WR={r['win_rate']:5.1f}%(L:{long_wr:>3.0f}%/S:{short_wr:>3.0f}%)"
            dd_dur = r.get("dd_duration_days", 0) or 0
            # Color DD if low (good): <5% bold green, <10% green
            dd_val = r["max_dd_pct"]
            dd_colored = color_dd(dd_val, f"{dd_val:5.1f}%")
            dd_part = f"DD={dd_colored}({dd_dur:>3.0f}d)"
            mkt = r.get("market_change_pct", 0) or 0
            mkt_part = f"MKT={mkt:+6.1f}%"
            pnl_l = r.get("profit_pct_long", 0) or 0
            pnl_s = r.get("profit_pct_short", 0) or 0
            # Same fallback as L/S counts: when parser couldn't split, attribute
            # global PnL to signal.direction wholesale.
            if pnl_l == 0 and pnl_s == 0 and r.get("profit_pct", 0):
                sig_dir = (r.get("direction") or "both").lower()
                if sig_dir == "long":
                    pnl_l = r["profit_pct"]
                elif sig_dir == "short":
                    pnl_s = r["profit_pct"]
            # Composite color on global PnL: green only when both DD AND p
            # reach their threshold (bold when both at strict).
            pnl_global = color_pnl_composite(
                f"{r['profit_pct']:+6.1f}%",
                dd=r["max_dd_pct"],
                p=r.get("p_value"),
            )
            pnl_part = f"PnL={pnl_global}(L:{pnl_l:+5.1f}%/S:{pnl_s:+5.1f}%)"

            # Tag prefix:
            #   🎲 = null-pool random_baseline build
            #   ⚡ = freqtrade cache hit (no subprocess)
            #   ▶  = fresh freqtrade run
            if r.get("signal_type") == "random_baseline":
                tag = "🎲"
            elif r.get("_cached"):
                tag = "⚡"
            else:
                tag = "▶"

            # Optional p-value suffix (only present in Phase 3 results).
            # Color: <0.01 bold green, <0.05 green.
            p_part = ""
            pv = r.get("p_value")
            if pv is not None and not (isinstance(pv, float) and pd.isna(pv)):
                if pv < 0.05:
                    sig_marker = "*"
                elif pv < 0.10:
                    sig_marker = "•"
                else:
                    sig_marker = " "
                pv_colored = color_pvalue(pv, f"{pv:.3f}")
                p_part = f" p={pv_colored}{sig_marker}"

            # Color Calmar: >=3 bold green, >=1 green. Replaces Sharpe in
            # the live line — under fixed stake / idle capital, Sharpe is
            # diluted but Calmar (annualized return / max DD) stays meaningful.
            calmar_val = r.get("calmar", 0) or 0
            calmar_colored = color_calmar(calmar_val, f"{calmar_val:+6.2f}")

            line_no = line_id if line_id is not None else self.completed
            print(
                f"  {tag} [{line_no:3d}/{self.total}] "
                f"{r['signal']:<35} {short_pair(r['pair']):<6} {r['timeframe']:<4} │ "
                f"Tr={r['trades']:>4d} {ls_part} "
                f"{wr_part} "
                f"{pnl_part} "
                f"{mkt_part} "
                f"{dd_part} "
                f"CAL={calmar_colored}"
                f"{p_part}"
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
