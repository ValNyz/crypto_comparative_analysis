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
from lib.backtest import BacktestRunner
from lib.data import discover_pairs, expand_pair_patterns
from lib.report import ReportGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="V3 - Analyse comparative avec architecture modulaire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/backtest_v3.py -p BTC/USDC:USDC -t 1h

  # Multiple pairs with pattern
  python scripts/backtest_v3.py -p "*/USDC:*" -t 1h 30m

  # Filter by signal type
  python scripts/backtest_v3.py -p BTC/USDC:USDC --filter funding

  # With regime filtering enabled
  python scripts/backtest_v3.py -p BTC/USDC:USDC --enable-filter

  # Custom config file
  python scripts/backtest_v3.py --config configs/custom.yaml

  # List available pairs
  python scripts/backtest_v3.py --list
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

    # Execution
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=6,
        help="Number of parallel workers (default: 6)",
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

    if args.timerange:
        config.timerange = args.timerange

    # List pairs mode
    if args.list_pairs:
        print("\n📊 Paires disponibles:\n")
        for pair in discover_pairs(config.data_dir, args.timeframes[0]):
            print(f"  {pair}")
        return None

    # Resolve pairs
    if args.pairs is None:
        pairs = ["BTC/USDC:USDC"]
    else:
        if any("*" in p or "?" in p for p in args.pairs):
            print("\n🔍 Expansion des patterns...")
            pairs = expand_pair_patterns(
                args.pairs, config.data_dir, args.timeframes[0]
            )
            if not pairs:
                print("❌ Aucune paire trouvée pour les patterns spécifiés")
                return None
        else:
            pairs = args.pairs

    # Print header
    filter_status = "🔒 ACTIF" if config.enable_regime_filter else "🔓 INACTIF"
    print("=" * 120)
    print(f"🔬 ANALYSE COMPARATIVE V3 - FILTRAGE {filter_status}")
    print("=" * 120)
    print(f"""
   Pairs:           {pairs if len(pairs) <= 5 else f"{len(pairs)} paires"}
   Timeframes:      {args.timeframes}
   Période:         {config.timerange}
   Filtrage régime: {filter_status}
    """)

    # Load signals
    signals = get_signal_configs(
        signal_filter=args.filter,
        include_exits=not args.no_exits,
        configs_dir=config.configs_dir,
    )

    print(f"   Total pairs:      {len(pairs)}")
    print(f"   Total signaux:    {len(signals)}")
    print(f"   Total timeframes: {len(args.timeframes)}")
    print(f"   Total backtests:  {len(pairs) * len(signals) * len(args.timeframes)}")

    # Run backtests
    runner = BacktestRunner(config, debug=args.debug)
    df = runner.run_all(signals, pairs, args.timeframes)

    # Generate report
    report = ReportGenerator(df, config)
    report.print_full_report()

    # Save results
    if len(df) > 0:
        output_dir = config.output_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_filtered" if config.enable_regime_filter else "_unfiltered"

        if args.output:
            csv_path = Path(args.output)
        else:
            csv_path = output_dir / f"v3_{len(pairs)}pairs{suffix}_{ts}.csv"

        df.to_csv(csv_path, index=False)
        print(f"\n📁 Résultats: {csv_path}")

    return df


if __name__ == "__main__":
    main()
