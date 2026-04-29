# =============================================================================
# FILE: lib/backtest/runner.py
# =============================================================================
"""Backtest runner with parallel execution."""

import subprocess
import time

import re
import random

from typing import List, Dict, Optional
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

    def _get_export_name(self, signal: SignalConfig, pair: str, timeframe: str) -> str:
        """Génère un nom de fichier unique pour l'export."""
        safe_pair = pair.replace("/", "_").replace(":", "_")
        return f"{signal.name}_{safe_pair}_{timeframe}"

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

        return pd.DataFrame(results)

    def _print_result(self, r: Dict):
        """Print a single result with regime info."""
        with print_lock:
            regime_info = ""
            if r.get("regime_stats"):
                parts = []
                for reg in ["bull", "bear", "range", "volatile"]:
                    reg_data = r["regime_stats"].get(reg, {})
                    if reg_data:
                        long_tr = reg_data.get("long", {}).get("trades", 0)
                        short_tr = reg_data.get("short", {}).get("trades", 0)
                        total_tr = long_tr + short_tr
                        if total_tr > 0:
                            parts.append(f"{self.abbrev[reg]}: {long_tr}L/{short_tr}S")
                regime_info = f" │ {chr(9).join(parts)}" if parts else ""

            print(
                f"  [{self.completed:3d}/{self.total}] "
                f"{r['signal']:<25} {r['pair']:<18} {r['timeframe']:<4} │ "
                f"Tr={r['trades']:<3} WR={r['win_rate']:5.1f}% "
                f"PnL={r['profit_pct']:+6.1f}% "
                f"Sharpe={r['sharpe']:+5.2f}"
                f"{regime_info}"
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
