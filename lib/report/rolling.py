# =============================================================================
# FILE: lib/report/rolling.py
# =============================================================================
"""Rolling report generator - hérite de ReportGenerator."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, cast

from .base import ReportGenerator
from .formatters import print_header, print_section
from ..config.base import Config
from ..utils.helpers import short_pair


class RollingReportGenerator(ReportGenerator):
    """
    Générateur de rapport pour rolling backtest.

    Hérite de ReportGenerator pour réutiliser les sections communes
    et ajoute les sections spécifiques à l'analyse de consistance.
    """

    # Annotations pour le mode portfolio (build dans _build_portfolio_frames)
    _strats: List[Tuple[str, str, str]]
    _wide: pd.DataFrame
    _port: pd.Series

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

    def print_full_report(self, top_n: int = 25, show_regime: bool = False):
        """Affiche le rapport rolling complet (show_regime ignoré ici).

        Auto-routage: ≤ 5 signaux → mode portefeuille (dense, focalisé go/no-go
        dry-run). Sinon → rapport grille classique.
        """
        _ = show_regime  # signature compat with parent
        if len(self.consistency_df) == 0:
            print("\n❌ Aucun résultat de rolling backtest!")
            return

        if len(self.consistency_df) <= 5:
            self._print_portfolio_report()
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

    # =========================================================================
    # PORTFOLIO MODE — rapport dense pour décider d'un dry-run sur N strats (N≤5)
    # =========================================================================

    def _print_portfolio_report(self):
        """Rapport focalisé pour valider un petit portefeuille avant dry-run."""
        self._build_portfolio_frames()
        self._print_portfolio_setup()
        self._print_portfolio_per_strat_summary()
        self._print_portfolio_per_window_table()
        self._print_portfolio_correlation()
        self._print_portfolio_worst_windows()
        self._print_portfolio_stability_over_time()
        self._print_portfolio_verdict()
        print(f"\n{'=' * 120}")

    def _build_portfolio_frames(self):
        """Construit deux DataFrames intermédiaires partagés par les sections.

        - self._strats : ordre stable des strats (= ordre du consistency_df).
        - self._wide   : pivot window_idx × strat avec colonnes profit/calmar/dd
                         + mkt_change_pct par fenêtre (commun aux strats).
        - self._port   : Series equal-weight portfolio PnL par window_idx.
        """
        cdf = self.consistency_df.reset_index(drop=True)
        rdf = self.raw_df

        # Identité ordonnée des strats
        self._strats = list(
            zip(cdf["signal"].tolist(), cdf["pair"].tolist(), cdf["timeframe"].tolist())
        )

        if "window_idx" not in rdf.columns:
            self._wide = pd.DataFrame()
            self._port = pd.Series(dtype=float)
            return

        windows = sorted(rdf["window_idx"].unique())
        rows = []
        for w in windows:
            row = {"window_idx": w}
            wdf = rdf[rdf["window_idx"] == w]
            label = wdf["window_label"].iloc[0] if "window_label" in wdf.columns and len(wdf) > 0 else f"W{w}"
            row["window_label"] = label
            row["mkt_change_pct"] = (
                wdf["market_change_pct"].mean() if "market_change_pct" in wdf.columns else np.nan
            )
            for i, (sig, pair, tf) in enumerate(self._strats, 1):
                m = (rdf["signal"] == sig) & (rdf["pair"] == pair) & (rdf["timeframe"] == tf) & (rdf["window_idx"] == w)
                sub = rdf[m]
                if len(sub) == 0:
                    row[f"s{i}_profit"] = np.nan
                    row[f"s{i}_calmar"] = np.nan
                    row[f"s{i}_dd"] = np.nan
                    row[f"s{i}_trades"] = 0
                else:
                    r = sub.iloc[0]
                    row[f"s{i}_profit"] = float(r.get("profit_pct", 0) or 0)
                    row[f"s{i}_calmar"] = float(r.get("calmar", 0) or 0)
                    row[f"s{i}_dd"] = float(r.get("max_dd_pct", 0) or 0)
                    row[f"s{i}_trades"] = int(r.get("trades", 0) or 0)
            rows.append(row)

        self._wide = pd.DataFrame(rows)

        # Equal-weight: moyenne arithmétique des PnL strats par fenêtre
        n = len(self._strats)
        if n > 0 and len(self._wide) > 0:
            cols = [f"s{i}_profit" for i in range(1, n + 1)]
            mean_result = self._wide[cols].mean(axis=1)
            self._port = pd.Series(np.asarray(mean_result, dtype=float))
        else:
            self._port = pd.Series(dtype=float)

    def _print_portfolio_setup(self):
        """Bloc d'entête : strats listées, période, n_windows, overlap."""
        cdf = self.consistency_df
        n = len(cdf)
        overlap = max(0.0, (1 - self.step_months / self.window_months) * 100) if self.window_months else 0
        print_header(f"🎯 RAPPORT PORTEFEUILLE ({n} strats × {self.n_windows} fenêtres)")
        print(f"\n   Fenêtre {self.window_months}m / décalage {self.step_months}m / chevauchement {overlap:.0f}%")
        if overlap >= 50:
            print(f"   ⚠️  Overlap élevé : la dispersion entre fenêtres est sous-estimée.")
        print()
        for i, (sig, pair, tf) in enumerate(self._strats, 1):
            row = cdf[(cdf["signal"] == sig) & (cdf["pair"] == pair) & (cdf["timeframe"] == tf)].iloc[0]
            exit_cfg = row.get("exit_config", "none") or "none"
            stype = row.get("signal_type", "") or ""
            print(f"   S{i}  {sig:<48} {short_pair(pair)} {tf:<4} ({stype}, exit={exit_cfg})")

    def _print_portfolio_per_strat_summary(self):
        """Tableau compact: une colonne par strat, métriques en lignes."""
        print_section("📊 SYNTHÈSE PAR STRATÉGIE")

        cdf = self.consistency_df
        rdf = self.raw_df

        # Pré-calculs par strat
        cols = []
        for i, (sig, pair, tf) in enumerate(self._strats, 1):
            cr = cdf[(cdf["signal"] == sig) & (cdf["pair"] == pair) & (cdf["timeframe"] == tf)].iloc[0]
            sub = rdf[(rdf["signal"] == sig) & (rdf["pair"] == pair) & (rdf["timeframe"] == tf)]
            n_w = int(cr.get("n_windows", 0))
            n_prof = int((sub["profit_pct"] > 0).sum()) if len(sub) > 0 else 0
            pl = float(cr.get("profit_long_sum", 0) or 0)
            ps = float(cr.get("profit_short_sum", 0) or 0)
            tot_ls = abs(pl) + abs(ps)
            ls_split = (
                f"{pl / (pl + ps) * 100:.0f}/{ps / (pl + ps) * 100:.0f}"
                if (pl + ps) > 0 else (f"{pl / tot_ls * 100:+.0f}/{ps / tot_ls * 100:+.0f}" if tot_ls > 0 else "n/a")
            )
            cols.append({
                "label": f"S{i}",
                "trades": int(cr.get("trades_total", 0) or 0),
                "wins": f"{n_prof}/{n_w}",
                "pct_prof": float(cr.get("pct_profitable", 0) or 0),
                "p_mean": float(cr.get("profit_mean", 0) or 0),
                "p_std": float(cr.get("profit_std", 0) or 0),
                "p_min": float(cr.get("profit_min", 0) or 0),
                "p_max": float(cr.get("profit_max", 0) or 0),
                "cal_mean": float(cr.get("calmar_mean", 0) or 0),
                "cal_min": float(cr.get("calmar_min", 0) or 0),
                "dd_worst": float(cr.get("dd_worst", 0) or 0),
                "pf": float(cr.get("pf_mean", 0) or 0),
                "ls": ls_split,
                "sharpe": float(cr.get("sharpe_mean", 0) or 0),
                "sharpe_std": float(cr.get("sharpe_std", 0) or 0),
            })

        col_w = 14
        header = f"   {'Métrique':<22}" + "".join(f"{c['label']:>{col_w}}" for c in cols)
        print(f"\n{header}")
        print("   " + "─" * (22 + col_w * len(cols)))

        def line(label, fmt):
            cells = "".join(f"{fmt(c):>{col_w}}" for c in cols)
            print(f"   {label:<22}{cells}")

        line("Trades total",       lambda c: f"{c['trades']}")
        line("Fenêtres profitab.", lambda c: f"{c['wins']} ({c['pct_prof']:.0f}%)")
        line("PnL μ ± σ",          lambda c: f"{c['p_mean']:+.2f}±{c['p_std']:.2f}")
        line("PnL min / max",      lambda c: f"{c['p_min']:+.1f}/{c['p_max']:+.1f}")
        line("Calmar μ (min)",     lambda c: f"{c['cal_mean']:+.0f} ({c['cal_min']:+.0f})")
        line("DD pire fenêtre",    lambda c: f"{c['dd_worst']:.1f}%")
        line("Profit factor μ",    lambda c: f"{c['pf']:.2f}")
        line("Sharpe μ ± σ",       lambda c: f"{c['sharpe']:+.2f}±{c['sharpe_std']:.2f}")
        line("L/S split (PnL)",    lambda c: c['ls'])

    def _print_portfolio_per_window_table(self):
        """Tableau dense: une ligne par fenêtre, PnL par strat + portfolio eq-weight."""
        if len(self._wide) == 0:
            return

        print_section("📅 DÉTAIL PAR FENÊTRE (PnL %, equal-weight)")

        n = len(self._strats)
        strat_w = 9
        header = f"\n   {'W#':<3} {'Période':<22} {'Mkt%':>7}  │ "
        header += " ".join(f"{f'S{i}':>{strat_w-1}}" for i in range(1, n + 1))
        header += f" │ {'Port':>6}  {'Best/Worst'}"
        print(header)
        print("   " + "─" * (40 + n * strat_w + 24))

        for raw_idx, row in self._wide.iterrows():
            idx = cast(int, raw_idx)
            w = int(row["window_idx"])
            label = str(row["window_label"])
            period = self._compact_label(label)
            raw_mkt = row["mkt_change_pct"]
            mkt = float(raw_mkt) if bool(pd.notna(raw_mkt)) else float("nan")
            mkt_str = f"{mkt:+.1f}%" if not np.isnan(mkt) else "  n/a "

            cells = []
            profits = []
            for i in range(1, n + 1):
                raw_p = row[f"s{i}_profit"]
                if bool(pd.isna(raw_p)):
                    cells.append(f"{'--':>{strat_w-1}}")
                else:
                    p = float(raw_p)
                    profits.append((i, p))
                    cells.append(f"{p:>+{strat_w-1}.2f}")

            port_val = self._port.iloc[idx] if idx < len(self._port) else float("nan")
            port = float(port_val) if bool(pd.notna(port_val)) else float("nan")
            port_str = f"{port:+6.2f}" if not np.isnan(port) else "  n/a "

            if profits:
                best = max(profits, key=lambda x: x[1])
                worst = min(profits, key=lambda x: x[1])
                bw = f"S{best[0]}/S{worst[0]}"
            else:
                bw = "—"

            mark = ""
            if not np.isnan(port):
                mark = "✓" if port > 0 else "✗"

            print(f"   W{w:<2} {period:<22} {mkt_str:>7}  │ " + " ".join(cells) + f" │ {port_str}  {bw:<8} {mark}")

    @staticmethod
    def _compact_label(label: str) -> str:
        """'W0: 2024-01-01 → 2024-03-31' → '2024-01 → 2024-03'."""
        if "→" not in label:
            return label
        try:
            _, dates = label.split(":", 1)
            a, b = dates.split("→")
            a = a.strip()[:7]
            b = b.strip()[:7]
            return f"{a} → {b}"
        except Exception:
            return label

    def _print_portfolio_correlation(self):
        """Matrice de corrélation NxN des PnL par fenêtre entre strats."""
        if len(self._wide) == 0 or len(self._strats) < 2:
            return

        print_section("🔗 CORRÉLATION INTER-STRATS (PnL par fenêtre)")

        n = len(self._strats)
        cols = [f"s{i}_profit" for i in range(1, n + 1)]
        sub = cast(pd.DataFrame, self._wide[cols].dropna())
        if len(sub) < 3:
            print("   Pas assez de fenêtres communes pour une corrélation fiable.")
            return

        corr = np.asarray(sub.corr().values, dtype=float)

        labels = [f"S{i}" for i in range(1, n + 1)]
        print("\n        " + "  ".join(f"{lab:>6}" for lab in labels))
        for i, lab in enumerate(labels):
            cells = []
            for j in range(n):
                if i == j:
                    cells.append(f"{'1.00':>6}")
                else:
                    v = corr[i, j]
                    cells.append(f"{v:>+6.2f}")
            print(f"   {lab:<4}  " + "  ".join(cells))

        # Pires paires
        worst = []
        for i in range(n):
            for j in range(i + 1, n):
                worst.append((i, j, corr[i, j]))
        worst.sort(key=lambda t: -abs(t[2]))
        if worst:
            i, j, v = worst[0]
            verdict = (
                "✓ décorrélation OK" if abs(v) < 0.3
                else "~ corrélation modérée" if abs(v) < 0.6
                else "✗ corrélation forte — diversification faible"
            )
            print(f"\n   Paire la plus corrélée : S{i+1}-S{j+1} = {v:+.2f}  ({verdict})")

    def _print_portfolio_worst_windows(self, top: int = 5):
        """Top fenêtres avec le pire PnL portefeuille — breakdown par strat."""
        if len(self._wide) == 0 or len(self._port) == 0:
            return

        print_section(f"⚠️  PIRES FENÊTRES PORTEFEUILLE (top {top})")

        df = self._wide.copy()
        df["port"] = self._port.values
        df = df.sort_values("port").head(top)

        n = len(self._strats)
        print()
        for _, row in df.iterrows():
            w = int(row["window_idx"])
            period = self._compact_label(str(row["window_label"]))
            mkt_raw = row["mkt_change_pct"]
            mkt_str = f"{float(mkt_raw):+.1f}%" if bool(pd.notna(mkt_raw)) else "n/a"
            port = float(row["port"])
            parts = []
            losses: list[tuple[int, float]] = []
            for i in range(1, n + 1):
                v = row[f"s{i}_profit"]
                if bool(pd.notna(v)):
                    fv = float(v)
                    parts.append(f"S{i}={fv:+.2f}")
                    losses.append((i, fv))
                else:
                    parts.append(f"S{i}=--")
            breakdown = "  ".join(parts)
            blame = min(losses, key=lambda x: x[1]) if losses else None
            blame_str = f"  → S{blame[0]} drag" if blame and blame[1] < 0 else ""
            print(f"   W{w:<2} {period:<22} mkt={mkt_str:>7}  port={port:+5.2f}%  │ {breakdown}{blame_str}")

    def _print_portfolio_stability_over_time(self):
        """Première moitié vs deuxième moitié — détecte la dégradation d'edge."""
        if len(self._port) < 6:
            return

        print_section("⏱  STABILITÉ TEMPORELLE (1ère vs 2nde moitié)")

        n_w = len(self._port)
        mid = n_w // 2
        first = self._port.iloc[:mid]
        second = self._port.iloc[mid:]

        def stats(s):
            n_prof = int((s > 0).sum())
            return {
                "n": len(s),
                "prof": n_prof,
                "pct": n_prof / len(s) * 100 if len(s) else 0,
                "mean": s.mean(),
                "std": s.std(),
            }

        a, b = stats(first), stats(second)
        delta_pct = b["pct"] - a["pct"]
        delta_mean = b["mean"] - a["mean"]

        print(f"\n   {'':16}{'Fenêtres':>11}{'%Prof':>9}{'PnL μ':>10}{'PnL σ':>10}")
        print(f"   {'1ère moitié':<16}{a['prof']}/{a['n']:<8}{a['pct']:>7.0f}% {a['mean']:>+9.2f}{a['std']:>+10.2f}")
        print(f"   {'2nde moitié':<16}{b['prof']}/{b['n']:<8}{b['pct']:>7.0f}% {b['mean']:>+9.2f}{b['std']:>+10.2f}")
        print(f"   {'Δ':<16}{'':<11}{delta_pct:>+7.0f}pp{delta_mean:>+9.2f}")

        if delta_pct < -20 or delta_mean < -1.0:
            print(f"\n   ✗ Dégradation marquée — l'edge faiblit en 2nde moitié.")
        elif delta_pct < -10:
            print(f"\n   ⚠️  Dégradation légère — surveiller en dry-run.")
        else:
            print(f"\n   ✓ Edge stable dans le temps.")

    def _print_portfolio_verdict(self):
        """Checklist explicite go/no-go dry-run avec critères chiffrés."""
        print_section("✅ CHECKLIST DRY-RUN")

        cdf = self.consistency_df
        n = len(cdf)
        n_w = self.n_windows

        checks = []

        # 1. Toutes les strats profitables sur ≥ 70% des fenêtres
        all_prof = bool((cdf["pct_profitable"] >= 70).all())
        worst_prof = float(cdf["pct_profitable"].min())
        checks.append((all_prof, f"Toutes strats ≥70% fenêtres profitables (min observé : {worst_prof:.0f}%)"))

        # 2. Pas de DD individuel pire que -10% sur une fenêtre
        if "dd_worst" in cdf.columns:
            worst_dd = float(cdf["dd_worst"].min())  # le plus négatif
            ok_dd = worst_dd > -10
            checks.append((ok_dd, f"DD pire fenêtre > -10% (observé : {worst_dd:.1f}%)"))

        # 3. Pire fenêtre portefeuille > -3%
        if len(self._port) > 0:
            worst_port = float(self._port.min())
            ok_port = worst_port > -3
            checks.append((ok_port, f"Pire fenêtre portfolio > -3% (observé : {worst_port:+.2f}%)"))

        # 4. Décorrélation : aucune paire > 0.6
        if len(self._wide) > 0 and n >= 2:
            cols = [f"s{i}_profit" for i in range(1, n + 1)]
            sub = cast(pd.DataFrame, self._wide[cols].dropna())
            if len(sub) >= 3:
                corr = np.asarray(sub.corr().values, dtype=float)
                upper = [corr[i, j] for i in range(n) for j in range(i + 1, n)]
                max_corr = max(abs(v) for v in upper) if upper else 0
                ok_corr = max_corr < 0.6
                checks.append((ok_corr, f"Aucune corrélation inter-strats > 0.6 (max observé : {max_corr:+.2f})"))

        # 5. PnL portefeuille positif sur la majorité des fenêtres
        if len(self._port) > 0:
            n_prof_port = int((self._port > 0).sum())
            pct_prof_port = n_prof_port / len(self._port) * 100
            ok_port_cons = pct_prof_port >= 70
            checks.append((ok_port_cons, f"Portfolio profitable ≥70% des fenêtres (observé : {pct_prof_port:.0f}%, {n_prof_port}/{len(self._port)})"))

        # 6. Profit factor moyen > 1.2 sur chaque strat
        if "pf_mean" in cdf.columns:
            min_pf = float(cdf["pf_mean"].min())
            ok_pf = min_pf > 1.2
            checks.append((ok_pf, f"Profit factor μ > 1.2 sur chaque strat (min observé : {min_pf:.2f})"))

        # Affichage
        n_pass = sum(1 for ok, _ in checks if ok)
        print()
        for ok, msg in checks:
            mark = "✓" if ok else "✗"
            print(f"   {mark}  {msg}")

        # Verdict global
        ratio = n_pass / len(checks) if checks else 0
        print()
        if ratio == 1.0:
            print(f"   🟢 GO dry-run — tous les critères ({n_pass}/{len(checks)}) sont remplis.")
        elif ratio >= 0.8:
            print(f"   🟡 GO conditionnel — {n_pass}/{len(checks)} critères. Surveiller le(s) point(s) ✗ en dry-run.")
        elif ratio >= 0.5:
            print(f"   🟠 GO partiel — {n_pass}/{len(checks)} critères. Considérer retirer/remplacer la strat la plus faible avant dry-run.")
        else:
            print(f"   🔴 NO-GO — {n_pass}/{len(checks)} critères seulement. Retravailler la sélection.")
        print(f"      Window={self.window_months}m  Step={self.step_months}m  N={n_w} fenêtres  ({n} strats)")

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
                f"  {i:<3} {r['signal']:<30} {short_pair(r['pair']):<6} {r['timeframe']:<4} │ "
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
        df = df.nsmallest(15, "sharpe_std")  # pyright: ignore[reportArgumentType]

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
        type_stats = type_stats.sort_values("rob_mean", ascending=False)  # pyright: ignore[reportCallIssue]

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
                print(f"        Pair: {short_pair(r['pair'])} │ TF: {r['timeframe']}")
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
        ].nsmallest(5, "robustness")  # pyright: ignore[reportArgumentType]

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
            print(f"  📈 {short_pair(pair)}")
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
            + f" │ {'Moy':^6} │ Écart"
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
                f"{short_pair(c['pair'])}({c['robustness']:.2f})"
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
