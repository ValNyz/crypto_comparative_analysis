# =============================================================================
# FILE: lib/backtest/runner.py
# =============================================================================
"""Backtest runner with parallel execution."""

import subprocess
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from ..config.base import Config
from ..signals.base import SignalConfig
from ..generation.generator import StrategyGenerator
from ..utils.logging import print_lock
from .parser import parse_freqtrade_output


class BacktestRunner:
    """Runs backtests in parallel."""

    def __init__(self, config: Config, debug: bool = False):
        """
        Initialize the runner.

        Args:
            config: Config instance
            debug: Enable debug output
        """
        self.config = config
        self.debug = debug
        self.generator = StrategyGenerator(config)
        self.completed = 0
        self.total = 0
        self.abbrev = {"bull": "Bu", "bear": "Be", "range": "Ra", "volatile": "Vo"}

    def run_single(
        self, signal: SignalConfig, pair: str, timeframe: str
    ) -> Optional[Dict]:
        """
        Run a single backtest.

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
            "--datadir",
            self.config.data_dir,
            "--dry-run-wallet",
            str(self.config.dry_run_wallet),
            "--stake-amount",
            str(self.config.stake_amount),
            "--export",
            "none",
            "--max-open-trades",
            str(self.config.max_open_trades),
            "--breakdown",
            "month",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                cwd=self.config.freqtrade_path,
            )
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
            self.completed += 1
            return None
        except Exception as e:
            self.completed += 1
            if self.debug:
                with print_lock:
                    print(f"  💥 {signal.name}: {e}")
            return None

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
                lines = result.stdout.strip().split("\n")[-20:]
                for l in lines:
                    print(f"    {l[:100]}")

            parsed = parse_freqtrade_output(result.stdout)
            print(
                f"  PARSED: trades={parsed.get('trades')}, profit={parsed.get('profit_pct')}"
            )
