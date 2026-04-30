#!/usr/bin/env python3
"""
Comparative Backtest V3 - Modular Architecture
===============================================

Usage:
    python scripts/backtest_v3.py --pairs BTC/USDC:USDC --timeframes 1h
    python scripts/backtest_v3.py -p "*/USDC:*" -t 1h 30m --filter funding
    python scripts/backtest_v3.py --list  # List available pairs
    python scripts/backtest_v3.py --config configs/custom.yaml

V3 Changelog:
- Modular architecture with separate packages
- YAML-based configuration
- Preserved all V2.3 functionality
- Improved code organization and maintainability
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import Config, load_config
from lib.signals import get_signal_configs
from lib.backtest import BacktestRunner, RollingConfig, run_rolling_backtest
from lib.data import discover_pairs, expand_pair_patterns
from lib.report import ReportGenerator, RollingReportGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="V3 - Analyse comparative avec architecture modulaire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/comparative_analysis_v3.py -p BTC/USDC:USDC -t 1h

  # Multiple pairs with pattern
  python scripts/comparative_analysis_v3.py -p "*/USDC:*" -t 1h 30m

  # Filter by signal type
  python scripts/comparative_analysis_v3.py -p BTC/USDC:USDC --filter funding

  # With regime filtering enabled
  python scripts/comparative_analysis_v3.py -p BTC/USDC:USDC --enable-filter

  # Custom config file
  python scripts/comparative_analysis_v3.py --config configs/custom.yaml

  # List available pairs
  python scripts/comparative_analysis_v3.py --list
        """,
    )

    # Pair/timeframe selection
    parser.add_argument(
        "--pairs",
        "-p",
        nargs="+",
        default=None,
        help="Trading pairs (supports wildcards like */USDC:*)",
    )
    parser.add_argument(
        "--timeframes",
        "-t",
        nargs="+",
        default=["1h"],
        help="Timeframes to test (default: 1h)",
    )

    # Time range
    parser.add_argument(
        "--timerange",
        "-r",
        type=str,
        default=None,
        help="Timerange for backtest (default: from config)",
    )
    parser.add_argument(
        "--timeframe-detail",
        type=str,
        default=None,
        help="Fine-grained timeframe for exit fills, e.g. '1m'. Enables freqtrade's high-fidelity exit detection.",
    )

    # Configuration
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="./configs",
        help="Directory containing config files",
    )
    parser.add_argument(
        "--signals",
        type=str,
        default="./configs/signals.yaml",
        help="YAML file specifying the signal to use.",
    )

    # Execution
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    # Signal filtering
    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        default=None,
        help=(
            "YAML group name to test. Exact match (e.g. 'funding_baseline') or "
            "wildcard (e.g. 'funding_macro_*'). Legacy values 'funding', "
            "'technical', 'advanced', 'combo' map to the canonical *_signals groups."
        ),
    )
    parser.add_argument(
        "--no-exits",
        action="store_true",
        help="Don't generate exit method variations for funding signals",
    )

    # Regime filtering
    parser.add_argument(
        "--enable-filter",
        action="store_true",
        help="Enable conditional regime filtering",
    )
    parser.add_argument(
        "--regime-lookback",
        type=int,
        default=72,
        help="Lookback period for regime detection (default: 72)",
    )
    parser.add_argument(
        "--adx-threshold",
        type=int,
        default=20,
        help="ADX threshold for trending detection (default: 20)",
    )

    # Rolling backtest
    parser.add_argument(
        "--rolling", action="store_true", help="Mode rolling/walk-forward"
    )
    parser.add_argument(
        "--window", type=int, default=3, help="Taille fenêtre en mois (défaut: 3)"
    )
    parser.add_argument(
        "--step", type=int, default=3, help="Décalage en mois (défaut: 1)"
    )
    parser.add_argument(
        "--min-windows", type=int, default=2, help="Min fenêtres pour stats (défaut: 3)"
    )

    # Display options
    parser.add_argument(
        "--show-regime",
        action="store_true",
        help=(
            "Include the three regime sections (distribution, performance, "
            "signal x regime matrix) in the report. Off by default — verbose "
            "and rarely actionable day-to-day."
        ),
    )

    # Debug/utility
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug output"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Force a full re-run via freqtrade, bypassing cached backtest "
            "results. By default, the runner reuses any matching export in "
            "user_data/backtest_results/ (class_name + timeframe + timerange "
            "match). Pass --refresh when you've changed strategy code or "
            "data and need fresh results."
        ),
    )
    # Null-pool comparison (always-on; tunable via these knobs)
    parser.add_argument(
        "--null-pool-seed",
        type=int,
        default=42,
        help="RNG seed for random-baseline entries (default: 42, also used in bootstrap)",
    )
    parser.add_argument(
        "--null-pool-target-trades",
        type=int,
        default=1000,
        help="Target entries per pool run (default: 1000; adaptive per TF length)",
    )
    parser.add_argument(
        "--refresh-null-pool",
        action="store_true",
        help="Force rebuild of parquet pool cache (keeps freqtrade signal cache)",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        dest="list_pairs",
        help="List available pairs and exit",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output CSV path"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()

    # Apply CLI overrides
    config.max_workers = args.workers
    config.debug = args.debug
    config.use_cache = not args.refresh
    config.enable_regime_filter = args.enable_filter
    config.null_pool_seed = args.null_pool_seed
    config.null_pool_target_trades = args.null_pool_target_trades
    config.refresh_null_pool = args.refresh_null_pool
    config.regime_lookback = args.regime_lookback
    config.regime_adx_threshold = args.adx_threshold
    config.configs_dir = args.configs_dir
    config.signals = args.signals

    if args.timerange:
        config.timerange = args.timerange
    if args.timeframe_detail:
        config.timeframe_detail = args.timeframe_detail

    # List pairs mode
    if args.list_pairs:
        print("\n📊 Paires disponibles:\n")
        for pair in discover_pairs(config.data_dir, args.timeframes[0]):
            print(f"  {pair}")
        return None

    # Résoudre paires
    pairs = args.pairs or ["BTC/USDC:USDC"]
    if any("*" in p for p in pairs):
        print("\n🔍 Expansion des patterns...")
        pairs = expand_pair_patterns(pairs, config.data_dir, args.timeframes[0])
        if not pairs:
            print("❌ Aucune paire trouvée")
            return

    # Charger signaux
    signals = get_signal_configs(
        config=config,
        signal_filter=args.filter,
        include_exits=not args.no_exits,
    )

    # Header commun
    _print_header(args, config, pairs, signals)

    # Exécution
    if args.rolling:
        consistency_df, raw_df = _run_rolling(config, signals, pairs, args)
        _save_rolling_results(consistency_df, raw_df, config, args)
        return consistency_df, raw_df
    else:
        df = _run_standard(config, signals, pairs, args)
        _save_standard_results(df, config, args)
        return df


def _print_header(args, config, pairs, signals):
    """Affiche le header commun."""
    mode = "🔄 ROLLING" if args.rolling else "📊 STANDARD"
    filter_status = "🔒 ACTIF" if config.enable_regime_filter else "🔓 INACTIF"

    print("\n" + "=" * 120)
    print(f"🔬 ANALYSE COMPARATIVE V3 - {mode}")
    print("=" * 120)
    print(f"""
   Paires:          {pairs if len(pairs) <= 5 else f"{len(pairs)} paires"}
   Timeframes:      {args.timeframes}
   Période:         {config.timerange}
   Signaux:         {len(signals)}
   Filtrage régime: {filter_status}""")

    if args.rolling:
        print(f"   Fenêtre:         {args.window} mois")
        print(f"   Décalage:        {args.step} mois")


def _run_standard(config, signals, pairs, args):
    """Exécute le mode standard."""
    total = len(pairs) * len(signals) * len(args.timeframes)
    print(f"   Total backtests: {total}\n")

    # Exécution
    runner = BacktestRunner(config, debug=args.debug)
    df = runner.run_all(signals, pairs, args.timeframes)

    # Rapport
    report = ReportGenerator(df, config)
    report.print_full_report(show_regime=args.show_regime)

    return df


def _run_rolling(config, signals, pairs, args):
    """Exécute le mode rolling."""
    rolling_config = RollingConfig(
        window_months=args.window,
        step_months=args.step,
        min_windows=args.min_windows,
    )

    # Exécution
    consistency_df, raw_df = run_rolling_backtest(
        config,
        signals,
        pairs,
        args.timeframes,
        rolling_config,
        debug=args.debug,
    )

    # Rapport via RollingReportGenerator
    if len(consistency_df) > 0:
        report = RollingReportGenerator(
            consistency_df,
            raw_df,
            config,
            window_months=args.window,
            step_months=args.step,
        )
        report.print_full_report()

    return consistency_df, raw_df


def _save_standard_results(df, config, args):
    """Sauvegarde les résultats standard.

    Writes three CSVs side-by-side:
      • v3_standard_TS.csv  — full results (every column, including
                              p_value/p_value_adj when null pool ran)
      • v3_winners_TS.csv   — only FDR-significant rows (p_adj < 0.05),
                              sorted by p_adj asc + Sharpe desc, key cols only
      • v3_monthly_TS.csv   — long-form monthly stats: one row per
                              (signal, pair, tf, month_idx) with profit,
                              trades, profit_factor, win_rate
    """
    import pandas as pd

    if df is None or len(df) == 0:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output or config.output_dir / f"v3_standard_{ts}.csv"
    base = str(csv_path).replace(".csv", "")

    df.to_csv(csv_path, index=False)
    print(f"\n📁 Résultats:")
    print(f"   • {csv_path}  ({len(df)} rows × {len(df.columns)} cols)")

    # Winners CSV: FDR-significant only, key columns
    if "p_value_adj" in df.columns:
        winners = df[df["p_value_adj"].notna() & (df["p_value_adj"] < 0.05)].copy()
        if len(winners) > 0:
            winners = winners.sort_values(
                ["p_value_adj", "sharpe"], ascending=[True, False]
            )
            keep_cols = [
                c for c in (
                    "signal", "signal_type", "direction", "pair", "timeframe",
                    "exit_config", "stoploss", "roi", "trades", "win_rate",
                    "profit_pct", "profit_pct_long", "profit_pct_short",
                    "sharpe", "max_dd_pct", "dd_duration_days",
                    "market_change_pct", "consistency", "p_value", "p_value_adj",
                )
                if c in winners.columns
            ]
            wpath = f"{base}_winners.csv"
            winners[keep_cols].to_csv(wpath, index=False)
            print(f"   • {wpath}  ({len(winners)} FDR-significant rows)")

    # Monthly long-form CSV: explode monthly_* lists into one row per month.
    if "monthly_profits" in df.columns:
        rows = []
        for _, r in df.iterrows():
            mp = r.get("monthly_profits") or []
            mt = r.get("monthly_trades") or []
            mpf = r.get("monthly_pf") or []
            mwr = r.get("monthly_wr") or []
            for i, profit in enumerate(mp):
                rows.append({
                    "signal": r.get("signal"),
                    "pair": r.get("pair"),
                    "timeframe": r.get("timeframe"),
                    "exit_config": r.get("exit_config"),
                    "month_idx": i + 1,
                    "profit_abs": profit,
                    "trades": mt[i] if i < len(mt) else None,
                    "profit_factor": mpf[i] if i < len(mpf) else None,
                    "win_rate": mwr[i] if i < len(mwr) else None,
                })
        if rows:
            mpath = f"{base}_monthly.csv"
            pd.DataFrame(rows).to_csv(mpath, index=False)
            print(f"   • {mpath}  ({len(rows)} monthly rows)")


def _save_rolling_results(consistency_df, raw_df, config, args):
    """Sauvegarde les résultats rolling."""
    if consistency_df is None or len(consistency_df) == 0:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (
        args.output.replace(".csv", "")
        if args.output
        else str(config.output_dir / f"rolling_{args.window}m_{args.step}s_{ts}")
    )

    consistency_df.to_csv(f"{base}_consistency.csv", index=False)
    raw_df.to_csv(f"{base}_raw.csv", index=False)
    print("\n📁 Résultats:")
    print(f"   • {base}_consistency.csv")
    print(f"   • {base}_raw.csv")


if __name__ == "__main__":
    main()
