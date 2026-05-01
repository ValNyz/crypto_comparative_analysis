# =============================================================================
# FILE: lib/backtest/rolling.py
# =============================================================================
"""Rolling backtest - génération de fenêtres et calcul de métriques.

Ce module ne fait AUCUN affichage. Il fournit uniquement:
- Dataclasses de configuration
- Génération des fenêtres temporelles
- Agrégation des résultats
- Calcul des métriques de consistance
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, cast
import pandas as pd
import numpy as np


@dataclass
class RollingConfig:
    """Configuration pour le rolling backtest."""

    window_months: int = 3
    step_months: int = 1
    min_windows: int = 3

    def __post_init__(self):
        if self.window_months < 1:
            raise ValueError("window_months doit être >= 1")
        if self.step_months < 1:
            raise ValueError("step_months doit être >= 1")


@dataclass
class RollingWindow:
    """Représente une fenêtre de backtest."""

    index: int
    start_date: datetime
    end_date: datetime

    @property
    def timerange(self) -> str:
        """Format freqtrade: YYYYMMDD-YYYYMMDD"""
        return (
            f"{self.start_date.strftime('%Y%m%d')}-{self.end_date.strftime('%Y%m%d')}"
        )

    @property
    def label(self) -> str:
        """Label lisible."""
        return f"W{self.index}: {self.start_date.strftime('%Y-%m-%d')} → {self.end_date.strftime('%Y-%m-%d')}"

    @property
    def duration_days(self) -> int:
        """Durée en jours."""
        return (self.end_date - self.start_date).days


def generate_windows(
    timerange: str,
    window_months: int,
    step_months: int,
) -> List[RollingWindow]:
    """
    Génère les fenêtres glissantes pour un rolling backtest.

    Args:
        timerange: Période totale au format "YYYYMMDD-YYYYMMDD"
        window_months: Taille de chaque fenêtre en mois
        step_months: Décalage entre fenêtres en mois

    Returns:
        Liste de RollingWindow

    Example:
        >>> generate_windows("20250101-20250630", 3, 1)
        [W0: Jan-Mar, W1: Feb-Apr, W2: Mar-May, W3: Apr-Jun]
    """
    if "-" not in timerange:
        raise ValueError(
            f"Format timerange invalide: '{timerange}'. Attendu: YYYYMMDD-YYYYMMDD"
        )

    start_str, end_str = timerange.split("-", 1)

    if not start_str:
        raise ValueError("Date de début requise (format: YYYYMMDD-YYYYMMDD)")

    if not end_str:
        raise ValueError(
            "Date de fin requise pour le rolling backtest (format: YYYYMMDD-YYYYMMDD)"
        )

    try:
        start_date = datetime.strptime(start_str, "%Y%m%d")
    except ValueError:
        raise ValueError(f"Date de début invalide: '{start_str}'. Attendu: YYYYMMDD")

    try:
        end_date = datetime.strptime(end_str, "%Y%m%d")
    except ValueError:
        raise ValueError(f"Date de fin invalide: '{end_str}'. Attendu: YYYYMMDD")

    windows = []
    idx = 0
    current_start = start_date

    while True:
        window_end = (
            current_start + relativedelta(months=window_months) - timedelta(days=1)
        )

        # Stop si la fenêtre dépasse la fin
        if window_end > end_date:
            # Fenêtre partielle si suffisamment grande (>50%)
            if current_start < end_date:
                windows.append(RollingWindow(idx, current_start, end_date))
            break

        windows.append(RollingWindow(idx, current_start, window_end))
        idx += 1
        current_start += relativedelta(months=step_months)

        # Sécurité
        if idx > 100:
            break

    return windows


def aggregate_window_results(
    window_results: List[Tuple[RollingWindow, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Agrège les résultats de toutes les fenêtres en un seul DataFrame.

    Args:
        window_results: Liste de tuples (window, dataframe)

    Returns:
        DataFrame avec colonnes window_idx et window_label ajoutées
    """
    all_dfs = []

    for window, df in window_results:
        if df is None or len(df) == 0:
            continue

        df_copy = df.copy()
        df_copy["window_idx"] = window.index
        df_copy["window_label"] = window.label
        all_dfs.append(df_copy)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def calculate_consistency(
    raw_df: pd.DataFrame,
    min_windows: int = 3,
) -> pd.DataFrame:
    """
    Calcule les métriques de consistance par signal.

    Args:
        raw_df: DataFrame agrégé avec tous les résultats
        min_windows: Nombre minimum de fenêtres pour calculer la consistance

    Returns:
        DataFrame avec métriques de consistance par signal
    """
    if len(raw_df) == 0:
        return pd.DataFrame()

    results = []

    # Grouper par signal/pair/timeframe
    grouped = raw_df.groupby(["signal", "pair", "timeframe"])

    for key, group in grouped:
        signal, pair, tf = cast(Tuple[str, str, str], key)
        n_windows = len(group)

        if n_windows < min_windows:
            continue

        def _col(name, default=0.0):
            if name in group.columns:
                return np.asarray(group[name].to_numpy(), dtype=float)
            return np.full(n_windows, default, dtype=float)

        sharpes = _col("sharpe")
        profits = _col("profit_pct")
        win_rates = _col("win_rate")
        trades = _col("trades")
        calmars = _col("calmar")
        max_dds = _col("max_dd_pct")
        pfs = _col("profit_factor")
        profit_longs = _col("profit_pct_long")
        profit_shorts = _col("profit_pct_short")

        n_profitable = np.sum(profits > 0)
        n_sharpe_pos = np.sum(sharpes > 0)

        # Calcul des scores
        pct_profitable = n_profitable / n_windows * 100
        pct_sharpe_pos = n_sharpe_pos / n_windows * 100

        sharpe_mean = np.mean(sharpes)
        sharpe_std = np.std(sharpes)
        profit_std = np.std(profits)

        # Score de stabilité: profitable et régulier
        stability = (n_profitable / n_windows) * (1 / (1 + profit_std))

        # Score de robustesse: combinaison performance/stabilité
        robustness = (
            (n_sharpe_pos / n_windows) * 0.4
            + (1 - min(sharpe_std, 2) / 2) * 0.3
            + (max(min(sharpe_mean, 2), -1) + 1) / 3 * 0.3
        )

        results.append(
            {
                "signal": signal,
                "pair": pair,
                "timeframe": tf,
                "signal_type": group["signal_type"].iloc[0]
                if "signal_type" in group.columns
                else "",
                "exit_config": group["exit_config"].iloc[0]
                if "exit_config" in group.columns
                else "",
                "n_windows": n_windows,
                # Sharpe
                "sharpe_mean": sharpe_mean,
                "sharpe_std": sharpe_std,
                "sharpe_min": np.min(sharpes),
                "sharpe_max": np.max(sharpes),
                # Profit
                "profit_mean": np.mean(profits),
                "profit_std": profit_std,
                "profit_min": np.min(profits),
                "profit_max": np.max(profits),
                # Calmar
                "calmar_mean": np.mean(calmars),
                "calmar_std": np.std(calmars),
                "calmar_min": np.min(calmars),
                "calmar_max": np.max(calmars),
                # Drawdown (max_dd_pct stocké négatif → min() = pire)
                "dd_mean": np.mean(max_dds),
                "dd_worst": np.min(max_dds),
                # Profit factor
                "pf_mean": np.mean(pfs),
                # L/S split (somme PnL long / short à travers les fenêtres)
                "profit_long_sum": np.sum(profit_longs),
                "profit_short_sum": np.sum(profit_shorts),
                # Win rate
                "wr_mean": np.mean(win_rates),
                "wr_std": np.std(win_rates),
                # Trades
                "trades_mean": np.mean(trades),
                "trades_total": np.sum(trades),
                # Consistance
                "pct_profitable": pct_profitable,
                "pct_sharpe_pos": pct_sharpe_pos,
                # Scores
                "stability": stability,
                "robustness": robustness,
            }
        )

    df = pd.DataFrame(results)

    if len(df) > 0:
        df = df.sort_values("robustness", ascending=False).reset_index(drop=True)

    return df


def get_window_details(
    raw_df: pd.DataFrame,
    signal: str,
    pair: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Récupère les détails par fenêtre pour un signal spécifique.

    Args:
        raw_df: DataFrame agrégé
        signal: Nom du signal
        pair: Paire de trading
        timeframe: Timeframe

    Returns:
        DataFrame avec une ligne par fenêtre
    """
    mask = (
        (raw_df["signal"] == signal)
        & (raw_df["pair"] == pair)
        & (raw_df["timeframe"] == timeframe)
    )
    return raw_df[mask].sort_values("window_idx")


# =============================================================================
# EXÉCUTION DU ROLLING BACKTEST
# =============================================================================


def run_rolling_backtest(
    config,  # Config
    signals: List,  # List[SignalConfig]
    pairs: List[str],
    timeframes: List[str],
    rolling_config: RollingConfig,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exécute un rolling backtest complet.

    Gère la boucle sur les fenêtres et l'affichage de progression,
    comme BacktestRunner.run_all() le fait pour le mode standard.

    Args:
        config: Configuration de base
        signals: Liste des signaux à tester
        pairs: Liste des paires
        timeframes: Liste des timeframes
        rolling_config: Configuration rolling
        debug: Mode debug

    Returns:
        (consistency_df, raw_df) - Métriques de consistance et résultats bruts
    """
    from copy import copy
    from .runner import BacktestRunner

    # Générer les fenêtres
    windows = generate_windows(
        config.timerange,
        rolling_config.window_months,
        rolling_config.step_months,
    )

    # Vérification
    if len(windows) < rolling_config.min_windows:
        print(
            f"\n⚠️  Seulement {len(windows)} fenêtres générées "
            f"(minimum requis: {rolling_config.min_windows})"
        )
        print("   → Augmentez la période ou réduisez window/step")
        return pd.DataFrame(), pd.DataFrame()

    # Header
    _print_rolling_header(windows, rolling_config)

    # Exécuter chaque fenêtre
    window_results = []

    for window in windows:
        _print_window_start(window, len(windows))

        # Config pour cette fenêtre
        window_config = copy(config)
        window_config.timerange = window.timerange

        # Réutiliser BacktestRunner
        runner = BacktestRunner(window_config, debug=debug)
        df = runner.run_all(signals, pairs, timeframes)

        window_results.append((window, df))

        # Résumé
        _print_window_end(df)

    # Agréger
    raw_df = aggregate_window_results(window_results)
    consistency_df = calculate_consistency(raw_df, rolling_config.min_windows)

    return consistency_df, raw_df


def _print_rolling_header(windows: List[RollingWindow], config: RollingConfig):
    """Affiche le header du rolling backtest."""
    print("\n" + "=" * 120)
    print("🔄 ROLLING BACKTEST / WALK-FORWARD ANALYSIS")
    print("=" * 120)
    overlap_pct = max(0.0, (1 - config.step_months / config.window_months) * 100)
    overlap_note = ""
    if overlap_pct > 0:
        overlap_note = (
            f"   ⚠️  Chevauchement {overlap_pct:.0f}% — fenêtres non-iid, "
            f"std de Sharpe/Calmar sous-estimée"
        )
    print(f"""
   Configuration:
   ├─ Fenêtre:    {config.window_months} mois
   ├─ Décalage:   {config.step_months} mois
   └─ Fenêtres:   {len(windows)}
{overlap_note}

   Périodes:""")
    for w in windows:
        print(f"   • {w.label} ({w.duration_days} jours)")
    print()


def _print_window_start(window: RollingWindow, total: int):
    """Affiche le début d'une fenêtre."""
    print(f"\n{'=' * 100}")
    print(f"📅 FENÊTRE {window.index + 1}/{total}: {window.label}")
    print(f"{'=' * 100}")


def _print_window_end(df: pd.DataFrame):
    """Affiche le résumé d'une fenêtre."""
    if df is not None and len(df) > 0:
        n_profitable = len(df[df["profit_pct"] > 0])
        pct = n_profitable / len(df) * 100
        print(f"\n   ✅ {len(df)} résultats, {n_profitable} profitables ({pct:.1f}%)")
    else:
        print("\n   ❌ Aucun résultat")
