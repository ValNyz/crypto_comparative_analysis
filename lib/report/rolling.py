# =============================================================================
# FILE: lib/report/rolling.py
# =============================================================================
"""Rolling report generator - hérite de ReportGenerator."""

import pandas as pd
import numpy as np
from typing import Optional

from .base import ReportGenerator
from .formatters import print_header, print_section
from ..config.base import Config


class RollingReportGenerator(ReportGenerator):
    """
    Générateur de rapport pour rolling backtest.

    Hérite de ReportGenerator pour réutiliser les sections communes
    et ajoute les sections spécifiques à l'analyse de consistance.
    """

    def __init__(
        self,
        consistency_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        config: Optional[Config] = None,
        window_months: int = 3,
        step_months: int = 1,
    ):
        """
        Args:
            consistency_df: Métriques de consistance agrégées
            raw_df: Résultats bruts de toutes les fenêtres
            config: Configuration
            window_months: Taille des fenêtres
            step_months: Décalage entre fenêtres
        """
        # Initialiser le parent avec raw_df pour les méthodes héritées
        super().__init__(raw_df, config)

        self.consistency_df = consistency_df
        self.raw_df = raw_df
        self.window_months = window_months
        self.step_months = step_months
        self.n_windows = (
            int(consistency_df["n_windows"].max()) if len(consistency_df) > 0 else 0
        )

    def print_full_report(self, top_n: int = 25):
        """Affiche le rapport rolling complet."""
        if len(self.consistency_df) == 0:
            print("\n❌ Aucun résultat de rolling backtest!")
            return

        self._print_header()
        self._print_rolling_global_metrics()
        self._print_consistency_distribution()
        self._print_top_consistent(top_n)
        self._print_window_heatmap(top_n=15)
        self._print_stability_analysis()
        self._print_consistency_by_type()

        # Analyses de régime adaptées au rolling
        self._print_regime_consistency()
        self._print_regime_signal_consistency_matrix()

        if "pair" in self.df.columns and self.df["pair"].nunique() > 1:
            self._print_per_coin_consistency(top_n)
            self._print_coin_consistency_matrix()
            self._print_cross_coin_performers(min_coins=2)

        # Analyse des exits (réutilisée telle quelle, pertinente sur raw_df agrégé)
        from .sections.exit_analysis import print_exit_analysis

        print_exit_analysis(self.raw_df)

        self._print_rolling_recommendations()

        print(f"\n{'=' * 120}")

    def _print_header(self):
        """Header spécifique rolling."""
        print_header(
            f"🔄 RAPPORT ROLLING BACKTEST ({self.window_months}m/{self.step_months}m)"
        )

    def _print_rolling_global_metrics(self):
        """Métriques globales adaptées au rolling."""
        df = self.consistency_df
        raw = self.raw_df

        n_signals = len(df)
        n_total_tests = len(raw)

        always_profitable = len(df[df["pct_profitable"] == 100])
        mostly_profitable = len(df[df["pct_profitable"] >= 75])
        half_profitable = len(df[df["pct_profitable"] >= 50])

        avg_robustness = df["robustness"].mean()
        avg_stability = df["stability"].mean()
        avg_sharpe = df["sharpe_mean"].mean()

        # Stats sur raw_df
        raw_profitable = len(raw[raw["profit_pct"] > 0])
        raw_pct = raw_profitable / len(raw) * 100 if len(raw) > 0 else 0

        print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│  MÉTRIQUES ROLLING GLOBALES                                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│  Configuration:     {self.window_months} mois fenêtre / {self.step_months} mois décalage                          │
│  Fenêtres:          {self.n_windows:<6}                                                       │
├────────────────────────────────────────────────────────────────────────────────┤
│  Signaux analysés:           {n_signals:<6}                                              │
│  Total backtests:            {n_total_tests:<6} ({raw_pct:.1f}% profitables)                      │
├────────────────────────────────────────────────────────────────────────────────┤
│  100% fenêtres profitables:  {always_profitable:<6} signaux                                    │
│  ≥75% fenêtres profitables:  {mostly_profitable:<6} signaux                                    │
│  ≥50% fenêtres profitables:  {half_profitable:<6} signaux                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│  Score robustesse moyen:     {avg_robustness:<6.3f}                                            │
│  Score stabilité moyen:      {avg_stability:<6.3f}                                            │
│  Sharpe moyen (des moyennes):{avg_sharpe:<+7.2f}                                            │
└────────────────────────────────────────────────────────────────────────────────┘""")

    def _print_consistency_distribution(self):
        """Distribution des niveaux de consistance."""
        print_section("📊 DISTRIBUTION DE LA CONSISTANCE")

        df = self.consistency_df

        # Buckets de consistance
        buckets = [
            ("100%", df["pct_profitable"] == 100),
            ("75-99%", (df["pct_profitable"] >= 75) & (df["pct_profitable"] < 100)),
            ("50-74%", (df["pct_profitable"] >= 50) & (df["pct_profitable"] < 75)),
            ("25-49%", (df["pct_profitable"] >= 25) & (df["pct_profitable"] < 50)),
            ("<25%", df["pct_profitable"] < 25),
        ]

        print(
            f"\n  {'Consistance':<12} │ {'Signaux':<8} │ {'%':<7} │ {'Sharpe moy':<11} │ Visualisation"
        )
        print("  " + "─" * 75)

        total = len(df)
        for label, mask in buckets:
            count = mask.sum()
            pct = count / total * 100 if total > 0 else 0
            avg_sharpe = df.loc[mask, "sharpe_mean"].mean() if count > 0 else 0
            bar = "█" * int(pct / 3)

            print(
                f"  {label:<12} │ {count:<8} │ {pct:<6.1f}% │ {avg_sharpe:<+11.2f} │ {bar}"
            )

    def _print_top_consistent(self, top_n: int = 25):
        """Top signaux par consistance."""
        print_header(
            f"🏆 TOP {min(top_n, len(self.consistency_df))} SIGNAUX LES PLUS CONSISTANTS"
        )

        df = self.consistency_df.head(top_n)

        print(
            f"\n  {'#':<3} {'Signal':<30} {'Pair':<14} {'TF':<4} │ "
            f"{'Win':<4} {'%Prof':<6} {'%Sh>0':<6} │ "
            f"{'Sharpe μ±σ':<14} │ {'Rob.':<5} {'Stab.':<5}"
        )
        print("  " + "─" * 115)

        for i, (_, r) in enumerate(df.iterrows(), 1):
            sharpe_str = f"{r['sharpe_mean']:+.2f}±{r['sharpe_std']:.2f}"

            # Icône selon consistance
            if r["pct_profitable"] >= 80 and r["robustness"] >= 0.6:
                icon = "🟢"
            elif r["pct_profitable"] >= 60:
                icon = "🟡"
            else:
                icon = "🔴"

            print(
                f"  {i:<3} {r['signal']:<30} {r['pair']:<14} {r['timeframe']:<4} │ "
                f"{int(r['n_windows']):<4} {r['pct_profitable']:<5.0f}% {r['pct_sharpe_pos']:<5.0f}% │ "
                f"{sharpe_str:<14} │ {r['robustness']:<5.2f} {r['stability']:<5.2f} {icon}"
            )

    def _print_window_heatmap(self, top_n: int = 15):
        """Heatmap des performances par fenêtre."""
        if "window_idx" not in self.raw_df.columns:
            return

        print_section("🗓️ HEATMAP PERFORMANCE PAR FENÊTRE")

        # Top signaux
        top_signals = self.consistency_df.head(top_n)[
            ["signal", "pair", "timeframe"]
        ].values.tolist()
        windows = sorted(self.raw_df["window_idx"].unique())

        # Header
        w_headers = " │ ".join(f"W{w}" for w in windows)
        print(f"\n  {'Signal':<38} │ {w_headers} │ {'Moy':>5}")
        print("  " + "─" * (45 + len(windows) * 5))

        for signal, pair, tf in top_signals:
            mask = (
                (self.raw_df["signal"] == signal)
                & (self.raw_df["pair"] == pair)
                & (self.raw_df["timeframe"] == tf)
            )
            signal_df = self.raw_df[mask]

            cells = []
            sharpes = []

            for w in windows:
                w_df = signal_df[signal_df["window_idx"] == w]
                if len(w_df) > 0:
                    s = w_df["sharpe"].iloc[0]
                    sharpes.append(s)
                    # Encoding visuel
                    if s >= 1.5:
                        cells.append("██")
                    elif s >= 1.0:
                        cells.append("▓▓")
                    elif s >= 0.5:
                        cells.append("▒▒")
                    elif s >= 0:
                        cells.append("░░")
                    else:
                        cells.append("××")
                else:
                    cells.append("--")

            avg = np.mean(sharpes) if sharpes else 0
            cells_str = " │ ".join(cells)

            # Tronquer le nom du signal si trop long
            display_name = signal[:38] if len(signal) <= 38 else signal[:35] + "..."
            print(f"  {display_name:<38} │ {cells_str} │ {avg:+5.2f}")

        print("\n  Légende: ██ ≥1.5 │ ▓▓ ≥1.0 │ ▒▒ ≥0.5 │ ░░ ≥0 │ ×× <0")

    def _print_stability_analysis(self):
        """Analyse des signaux les plus stables."""
        print_section("📈 SIGNAUX LES PLUS STABLES (faible variance)")

        # Filtrer signaux profitables avec assez de données
        df = self.consistency_df[
            (self.consistency_df["sharpe_mean"] > 0)
            & (self.consistency_df["n_windows"] >= 3)
        ].copy()

        if len(df) == 0:
            print("  Aucun signal stable et profitable trouvé.")
            return

        # Trier par variance Sharpe (plus faible = plus stable)
        df = df.nsmallest(15, "sharpe_std")

        print(
            f"\n  {'#':<3} {'Signal':<35} │ {'Sharpe μ':<9} │ {'Sharpe σ':<9} │ {'Min/Max':<15} │ {'Note'}"
        )
        print("  " + "─" * 95)

        for i, (_, r) in enumerate(df.iterrows(), 1):
            range_str = f"{r['sharpe_min']:+.2f}/{r['sharpe_max']:+.2f}"

            # Note de stabilité
            if r["sharpe_std"] < 0.25:
                note = "⭐⭐⭐ Excellent"
            elif r["sharpe_std"] < 0.5:
                note = "⭐⭐ Très bon"
            elif r["sharpe_std"] < 0.75:
                note = "⭐ Correct"
            else:
                note = "Variable"

            print(
                f"  {i:<3} {r['signal']:<35} │ {r['sharpe_mean']:<+9.2f} │ "
                f"{r['sharpe_std']:<9.3f} │ {range_str:<15} │ {note}"
            )

    def _print_consistency_by_type(self):
        """Consistance par type de signal."""
        if "signal_type" not in self.consistency_df.columns:
            return

        print_section("📊 CONSISTANCE PAR TYPE DE SIGNAL")

        type_stats = self.consistency_df.groupby("signal_type").agg(
            {
                "robustness": ["mean", "std"],
                "stability": "mean",
                "pct_profitable": "mean",
                "sharpe_mean": "mean",
                "signal": "count",
            }
        )
        type_stats.columns = [
            "rob_mean",
            "rob_std",
            "stab",
            "pct_prof",
            "sharpe",
            "count",
        ]
        type_stats = type_stats.sort_values("rob_mean", ascending=False)

        print(
            f"\n  {'Type':<15} │ {'N':<5} │ {'Robustesse':<12} │ {'Stabilité':<9} │ "
            f"{'%Prof':<6} │ {'Sharpe':<8} │ {'Verdict'}"
        )
        print("  " + "─" * 95)

        for sig_type, r in type_stats.iterrows():
            rob_str = f"{r['rob_mean']:.2f}±{r['rob_std']:.2f}"

            # Verdict
            if r["rob_mean"] >= 0.6 and r["pct_prof"] >= 70:
                verdict = "✅ Recommandé"
            elif r["rob_mean"] >= 0.4:
                verdict = "⚪ Acceptable"
            else:
                verdict = "❌ À éviter"

            print(
                f"  {sig_type:<15} │ {int(r['count']):<5} │ {rob_str:<12} │ "
                f"{r['stab']:<9.3f} │ {r['pct_prof']:<5.1f}% │ {r['sharpe']:<+8.2f} │ {verdict}"
            )

    def _print_regime_consistency(self):
        """Analyse de la consistance des performances par régime à travers les fenêtres."""
        if "regime_stats" not in self.raw_df.columns:
            return

        print_section("🎯 CONSISTANCE PAR RÉGIME DE MARCHÉ")

        regimes = ["bull", "bear", "range", "volatile"]
        windows = sorted(self.raw_df["window_idx"].unique())

        # Collecter les stats par régime et par fenêtre
        regime_window_stats = {
            r: {w: {"profits": [], "trades": 0} for w in windows} for r in regimes
        }

        for _, row in self.raw_df.iterrows():
            w_idx = row["window_idx"]
            regime_stats = row.get("regime_stats", {})

            if not regime_stats:
                continue

            for regime in regimes:
                if regime not in regime_stats:
                    continue

                regime_data = regime_stats[regime]
                for direction in ["long", "short"]:
                    if direction in regime_data and regime_data[direction]:
                        stats = regime_data[direction]
                        if stats.get("trades", 0) > 0:
                            regime_window_stats[regime][w_idx]["profits"].append(
                                stats.get("profit_pct", 0)
                            )
                            regime_window_stats[regime][w_idx]["trades"] += stats.get(
                                "trades", 0
                            )

        # Afficher la consistance par régime
        print("\n  Performance moyenne par régime et par fenêtre:\n")

        w_headers = " │ ".join(f"  W{w}  " for w in windows)
        print(f"  {'Régime':<10} │ {w_headers} │ {'Consist.':<10}")
        print("  " + "─" * (20 + len(windows) * 9))

        for regime in regimes:
            cells = []
            profitable_windows = 0

            for w in windows:
                profits = regime_window_stats[regime][w]["profits"]
                if profits:
                    avg = np.mean(profits)
                    if avg > 0:
                        profitable_windows += 1
                    cells.append(f"{avg:+6.1f}%")
                else:
                    cells.append("  ---  ")

            consistency = profitable_windows / len(windows) * 100 if windows else 0
            cells_str = " │ ".join(cells)

            # Icône
            if consistency >= 75:
                icon = "🟢"
            elif consistency >= 50:
                icon = "🟡"
            else:
                icon = "🔴"

            print(f"  {regime:<10} │ {cells_str} │ {consistency:>5.0f}% {icon}")

        # Résumé
        print(
            "\n  💡 Un régime est 'consistant' s'il est profitable sur la majorité des fenêtres."
        )

    def _print_regime_signal_consistency_matrix(self, top_n: int = 10):
        """Matrice: pour chaque régime, quels signaux sont consistants ?"""
        if "regime_stats" not in self.raw_df.columns:
            return

        print_section("📊 SIGNAUX CONSISTANTS PAR RÉGIME")

        regimes = ["bull", "bear", "range", "volatile"]
        windows = sorted(self.raw_df["window_idx"].unique())
        n_windows = len(windows)

        # Pour chaque signal, calculer la consistance par régime
        signal_regime_consistency = {}

        for (signal, pair, tf), group in self.raw_df.groupby(
            ["signal", "pair", "timeframe"]
        ):
            key = f"{signal}"
            if key not in signal_regime_consistency:
                signal_regime_consistency[key] = {
                    "signal": signal,
                    "pair": pair,
                    "tf": tf,
                }
                for regime in regimes:
                    signal_regime_consistency[key][f"{regime}_profitable_windows"] = 0
                    signal_regime_consistency[key][f"{regime}_total_windows"] = 0
                    signal_regime_consistency[key][f"{regime}_avg_profit"] = []

            for _, row in group.iterrows():
                regime_stats = row.get("regime_stats", {})
                if not regime_stats:
                    continue

                for regime in regimes:
                    if regime not in regime_stats:
                        continue

                    total_profit = 0
                    total_trades = 0

                    for direction in ["long", "short"]:
                        if (
                            direction in regime_stats[regime]
                            and regime_stats[regime][direction]
                        ):
                            stats = regime_stats[regime][direction]
                            trades = stats.get("trades", 0)
                            if trades > 0:
                                total_profit += stats.get("profit_pct", 0) * trades
                                total_trades += trades

                    if total_trades > 0:
                        avg_profit = total_profit / total_trades
                        signal_regime_consistency[key][f"{regime}_total_windows"] += 1
                        signal_regime_consistency[key][f"{regime}_avg_profit"].append(
                            avg_profit
                        )
                        if avg_profit > 0:
                            signal_regime_consistency[key][
                                f"{regime}_profitable_windows"
                            ] += 1

        # Afficher les meilleurs signaux par régime
        for regime in regimes:
            print(f"\n  {'─' * 100}")
            print(f"  🎯 TOP SIGNAUX CONSISTANTS EN RÉGIME {regime.upper()}")
            print(f"  {'─' * 100}\n")

            # Calculer le score de consistance pour ce régime
            regime_scores = []
            for key, data in signal_regime_consistency.items():
                total = data[f"{regime}_total_windows"]
                profitable = data[f"{regime}_profitable_windows"]
                profits = data[f"{regime}_avg_profit"]

                if total >= max(2, n_windows // 2):  # Au moins la moitié des fenêtres
                    consistency = profitable / total * 100
                    avg_profit = np.mean(profits) if profits else 0
                    profit_std = np.std(profits) if len(profits) > 1 else 0

                    regime_scores.append(
                        {
                            "signal": data["signal"],
                            "pair": data["pair"],
                            "windows": total,
                            "profitable": profitable,
                            "consistency": consistency,
                            "avg_profit": avg_profit,
                            "profit_std": profit_std,
                        }
                    )

            # Trier par consistance puis par profit moyen
            regime_scores.sort(
                key=lambda x: (x["consistency"], x["avg_profit"]), reverse=True
            )

            if not regime_scores:
                print("    Pas assez de données pour ce régime.")
                continue

            print(
                f"    {'#':<3} {'Signal':<35} │ {'Win':<5} │ {'%Cons':<7} │ {'Profit μ±σ':<15} │ Note"
            )
            print("    " + "─" * 85)

            for i, r in enumerate(regime_scores[:top_n], 1):
                profit_str = f"{r['avg_profit']:+.1f}%±{r['profit_std']:.1f}"

                if r["consistency"] >= 80 and r["avg_profit"] > 0:
                    note = "⭐ Excellent"
                elif r["consistency"] >= 60 and r["avg_profit"] > 0:
                    note = "✓ Bon"
                elif r["consistency"] >= 50:
                    note = "~ Moyen"
                else:
                    note = "✗ Faible"

                print(
                    f"    {i:<3} {r['signal']:<35} │ {r['windows']:<5} │ "
                    f"{r['consistency']:<6.0f}% │ {profit_str:<15} │ {note}"
                )

    def _print_rolling_recommendations(self):
        """Recommandations basées sur l'analyse rolling."""
        print_header("💡 RECOMMANDATIONS")

        df = self.consistency_df

        # Meilleurs signaux robustes ET stables
        best = df[
            (df["robustness"] >= 0.5)
            & (df["pct_profitable"] >= 60)
            & (df["sharpe_std"] < 1.0)
        ].head(5)

        print("\n  🎯 SIGNAUX RECOMMANDÉS POUR LE LIVE:\n")

        if len(best) > 0:
            for i, (_, r) in enumerate(best.iterrows(), 1):
                print(f"     {i}. {r['signal']}")
                print(f"        Pair: {r['pair']} │ TF: {r['timeframe']}")
                print(
                    f"        Robustesse: {r['robustness']:.2f} │ Stabilité: {r['stability']:.2f}"
                )
                print(f"        Profitable: {r['pct_profitable']:.0f}% des fenêtres")
                print(
                    f"        Sharpe: {r['sharpe_mean']:+.2f} (σ={r['sharpe_std']:.2f}, "
                    f"range: {r['sharpe_min']:+.2f} → {r['sharpe_max']:+.2f})"
                )
                print()
        else:
            print("     Aucun signal ne répond aux critères stricts.")
            print("     Critères: robustesse≥0.5, profitable≥60%, σ<1.0\n")

        # Signaux à éviter
        print("  ⚠️  SIGNAUX À ÉVITER:\n")

        worst = df[
            (df["pct_profitable"] < 40)
            | (df["sharpe_std"] > 1.5)
            | (df["robustness"] < 0.2)
        ].nsmallest(5, "robustness")

        if len(worst) > 0:
            for _, r in worst.iterrows():
                reasons = []
                if r["pct_profitable"] < 40:
                    reasons.append(f"profitable que {r['pct_profitable']:.0f}%")
                if r["sharpe_std"] > 1.5:
                    reasons.append(f"très variable (σ={r['sharpe_std']:.2f})")
                if r["robustness"] < 0.2:
                    reasons.append(f"robustesse faible ({r['robustness']:.2f})")

                print(f"     • {r['signal']}: {', '.join(reasons)}")
        else:
            print("     Aucun signal particulièrement problématique.")

        # Conseils
        print("""
  ────────────────────────────────────────────────────────────────────────────────
  📝 GUIDE D'INTERPRÉTATION:
  
  • Robustesse > 0.6      → Signal fiable pour le live trading
  • Sharpe σ < 0.5        → Performance stable dans le temps
  • % Profitable ≥ 75%    → Bonne consistance temporelle
  • Sharpe min > 0        → Jamais perdant sur une fenêtre
  
  💡 CONSEIL: Diversifiez avec 3-5 signaux robustes sur des paires différentes.
  ────────────────────────────────────────────────────────────────────────────────""")

    # =========================================================================
    # ANALYSE PAR COIN (adapté au rolling)
    # =========================================================================

    def _print_per_coin_consistency(self, top_n: int = 5):
        """Meilleures stratégies consistantes par coin."""
        df = self.consistency_df

        if len(df) == 0 or "pair" not in df.columns:
            return

        pairs = df["pair"].unique()
        if len(pairs) <= 1:
            return

        print_header(f"🪙 CONSISTANCE PAR COIN ({len(pairs)} paires)")

        for pair in sorted(pairs):
            pair_df = df[df["pair"] == pair]
            if len(pair_df) == 0:
                continue

            # Stats pour cette paire
            high_robust = len(pair_df[pair_df["robustness"] >= 0.5])
            avg_robust = pair_df["robustness"].mean()
            avg_pct_prof = pair_df["pct_profitable"].mean()

            print(f"\n{'─' * 110}")
            print(f"  📈 {pair}")
            print(
                f"     Signaux: {len(pair_df)} │ "
                f"Robustes (≥0.5): {high_robust} ({high_robust / len(pair_df) * 100:.1f}%) │ "
                f"Robustesse moy: {avg_robust:.2f} │ "
                f"% Prof moy: {avg_pct_prof:.1f}%"
            )
            print(f"{'─' * 110}")

            # Top N par robustesse
            top = pair_df.nlargest(min(top_n, len(pair_df)), "robustness")

            print(
                f"\n  {'#':<3} {'Signal':<32} │ "
                f"{'Win':<4} {'%Prof':<6} {'Rob.':<6} │ "
                f"{'Sharpe μ±σ':<14} │ {'Exit':<12}"
            )
            print("  " + "─" * 95)

            for i, (_, r) in enumerate(top.iterrows(), 1):
                exit_cfg = r.get("exit_config", "none") or "none"
                sharpe_str = f"{r['sharpe_mean']:+.2f}±{r['sharpe_std']:.2f}"

                print(
                    f"  {i:<3} {r['signal']:<32} │ "
                    f"{int(r['n_windows']):<4} {r['pct_profitable']:<5.0f}% {r['robustness']:<6.2f} │ "
                    f"{sharpe_str:<14} │ {exit_cfg:<12}"
                )

    def _print_coin_consistency_matrix(self):
        """Matrice comparant la consistance par type de signal et par coin."""
        df = self.consistency_df

        if len(df) == 0 or "pair" not in df.columns or "signal_type" not in df.columns:
            return

        pairs = df["pair"].unique()
        if len(pairs) <= 1:
            return

        print_section("📊 MATRICE CONSISTANCE PAR COIN")
        print("\n  Robustesse moyenne par type de signal et par paire:\n")

        signal_types = sorted(df["signal_type"].unique())
        coin_names = [p.split("/")[0][:6] for p in sorted(pairs)]

        # Header
        header = (
            f"  {'Type':<15} │ "
            + " │ ".join(f"{c:^8}" for c in coin_names)
            + " │ {'Moy':^6} │ Écart"
        )
        print(header)
        print("  " + "─" * len(header))

        for sig_type in signal_types:
            row_values = []
            for pair in sorted(pairs):
                subset = df[(df["signal_type"] == sig_type) & (df["pair"] == pair)]
                if len(subset) > 0:
                    avg_rob = subset["robustness"].mean()
                    row_values.append(avg_rob)
                else:
                    row_values.append(None)

            valid = [v for v in row_values if v is not None]
            avg_all = sum(valid) / len(valid) if valid else 0
            spread = max(valid) - min(valid) if len(valid) >= 2 else 0

            cells = []
            for v in row_values:
                if v is None:
                    cells.append(f"{'—':^8}")
                elif v >= 0.6:
                    cells.append(f"{v:^8.2f}")  # Bon
                elif v >= 0.4:
                    cells.append(f"{v:^8.2f}")
                else:
                    cells.append(f"{v:^8.2f}")  # Faible

            print(
                f"  {sig_type:<15} │ "
                + " │ ".join(cells)
                + f" │ {avg_all:^6.2f} │ {spread:5.2f}"
            )

    def _print_cross_coin_performers(self, min_coins: int = 2):
        """Signaux consistants sur plusieurs coins."""
        df = self.consistency_df

        if len(df) == 0 or "pair" not in df.columns:
            return

        pairs = df["pair"].unique()
        if len(pairs) < min_coins:
            return

        print_section(f"🏆 SIGNAUX ROBUSTES SUR {min_coins}+ COINS")

        # Grouper par signal (extraire le nom de base sans la paire)
        # Un signal peut avoir des résultats différents par paire
        signal_cross = {}

        for signal in df["signal"].unique():
            signal_df = df[df["signal"] == signal]

            # Coins où le signal est robuste
            robust_coins = []
            for pair in pairs:
                pair_signal = signal_df[signal_df["pair"] == pair]
                if len(pair_signal) > 0:
                    r = pair_signal.iloc[0]
                    if r["robustness"] >= 0.4 and r["pct_profitable"] >= 50:
                        robust_coins.append(
                            {
                                "pair": pair,
                                "robustness": r["robustness"],
                                "pct_profitable": r["pct_profitable"],
                                "sharpe_mean": r["sharpe_mean"],
                            }
                        )

            if len(robust_coins) >= min_coins:
                signal_cross[signal] = {
                    "coins": robust_coins,
                    "count": len(robust_coins),
                    "avg_robustness": sum(c["robustness"] for c in robust_coins)
                    / len(robust_coins),
                    "signal_type": signal_df["signal_type"].iloc[0]
                    if "signal_type" in signal_df.columns
                    else "unknown",
                }

        if not signal_cross:
            print(
                f"\n  Aucun signal robuste sur {min_coins}+ coins (rob≥0.4, %prof≥50%)."
            )
            return

        # Trier
        sorted_signals = sorted(
            signal_cross.items(),
            key=lambda x: (x[1]["count"], x[1]["avg_robustness"]),
            reverse=True,
        )

        print(
            f"\n  {'Signal':<35} │ {'Type':<12} │ {'Coins':<6} │ {'Rob.moy':<8} │ Détails"
        )
        print("  " + "─" * 100)

        for signal, perf in sorted_signals[:15]:
            details = ", ".join(
                f"{c['pair'].split('/')[0]}({c['robustness']:.2f})"
                for c in sorted(
                    perf["coins"], key=lambda x: x["robustness"], reverse=True
                )
            )
            print(
                f"  {signal:<35} │ {perf['signal_type']:<12} │ {perf['count']:^6} │ "
                f"{perf['avg_robustness']:^8.2f} │ {details[:35]}"
            )

    def get_summary(self) -> dict:
        """Résumé des métriques rolling."""
        df = self.consistency_df

        if len(df) == 0:
            return {}

        return {
            "mode": "rolling",
            "window_months": self.window_months,
            "step_months": self.step_months,
            "n_windows": self.n_windows,
            "n_signals": len(df),
            "n_total_tests": len(self.raw_df),
            "always_profitable": len(df[df["pct_profitable"] == 100]),
            "mostly_profitable": len(df[df["pct_profitable"] >= 75]),
            "avg_robustness": df["robustness"].mean(),
            "avg_stability": df["stability"].mean(),
            "best_signal": df.iloc[0]["signal"] if len(df) > 0 else None,
            "best_robustness": df["robustness"].max(),
        }
