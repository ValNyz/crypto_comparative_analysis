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
        choices=["funding", "technical", "advanced", "combo"],
        default=None,
        help="Filter by signal type",
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

    # Debug/utility
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug output"
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
    config.enable_regime_filter = args.enable_filter
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
    report.print_full_report()

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
    """Sauvegarde les résultats standard."""
    if df is None or len(df) == 0:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output or config.output_dir / f"v3_standard_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📁 Résultats: {csv_path}")


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
