# =============================================================================
# FILE: lib/generation/templates/cross_coin_block.py
# =============================================================================
"""Cross-coin OHLCV loader + BTC regime + ratios coin/BTC, coin/ETH.

Injected into generated strategies when needs_cross_coin_block(signal) is True.
Reads feathers from DATA_DIR/futures/ (same path used by the funding loader).
"""

CROSS_COIN_LOADERS_BLOCK = '''
    def load_coin_ohlcv(self, coin: str, interval: str, reference_df: pd.DataFrame) -> pd.DataFrame:
        """Load OHLCV feather for a coin at given interval. Returns empty df if absent.

        Reuses DATA_DIR (already declared by funding template). Tries USDC and USDT patterns.
        """
        for pattern in [
            f"{{coin}}_USDC_USDC-{{interval}}-futures.feather",
            f"{{coin}}_USDC-{{interval}}-futures.feather",
            f"{{coin}}_USDC_USDC-{{interval}}.feather",
            f"{{coin}}_USDT_USDT-{{interval}}-futures.feather",
            f"{{coin}}_USDT-{{interval}}-futures.feather",
            f"{{coin}}_USDT_USDT-{{interval}}.feather",
        ]:
            path = self.DATA_DIR / pattern
            if path.exists():
                try:
                    df = pd.read_feather(path)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    ref_tz = reference_df["date"].dt.tz
                    if ref_tz is not None:
                        if df["date"].dt.tz is None:
                            df["date"] = df["date"].dt.tz_localize("UTC").dt.tz_convert(ref_tz)
                        else:
                            df["date"] = df["date"].dt.tz_convert(ref_tz)
                    else:
                        if df["date"].dt.tz is not None:
                            df["date"] = df["date"].dt.tz_localize(None)
                    return df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
                except Exception:
                    pass
        return pd.DataFrame()

    def load_cross_coin_funding(self, ref_coin: str, reference_df: pd.DataFrame) -> pd.DataFrame:
        """Load HL funding for a different coin than the trading pair. Reused by funding-spread filter."""
        for pattern in [
            f"{{ref_coin}}_USDC_USDC-1h-funding_rate.feather",
            f"{{ref_coin}}_USDC_USDC-8h-funding_rate.feather",
            f"{{ref_coin}}_USDC-1h-funding_rate.feather",
            f"{{ref_coin}}_USDT_USDT-1h-funding_rate.feather",
            f"{{ref_coin}}_USDT_USDT-8h-funding_rate.feather",
            f"{{ref_coin}}_USDT-1h-funding_rate.feather",
        ]:
            path = self.DATA_DIR / pattern
            if path.exists():
                try:
                    df = pd.read_feather(path)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    ref_tz = reference_df["date"].dt.tz
                    if ref_tz is not None:
                        if df["date"].dt.tz is None:
                            df["date"] = df["date"].dt.tz_localize("UTC").dt.tz_convert(ref_tz)
                        else:
                            df["date"] = df["date"].dt.tz_convert(ref_tz)
                    else:
                        if df["date"].dt.tz is not None:
                            df["date"] = df["date"].dt.tz_localize(None)
                    out = pd.DataFrame()
                    out["date"] = df["date"]
                    out["funding_rate"] = df["open"].astype(float)
                    return out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
                except Exception:
                    pass
        return pd.DataFrame()

    def merge_btc_regime(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute BTC daily regime, merge into dataframe. Adds 'btc_regime' (categorical).

        Falls back to resampling BTC 1h to 1d when no 1d feather exists in DATA_DIR
        (common when running on a Binance-style data dir that ships only 1h/30m/etc.).
        Logs a warning if neither 1d nor 1h is available — in that case the filter is
        effectively disabled (all bars tagged 'neutral').
        """
        btc_d = self.load_coin_ohlcv("BTC", "1d", dataframe)
        if btc_d.empty or "close" not in btc_d.columns:
            btc_h = self.load_coin_ohlcv("BTC", "1h", dataframe)
            if not btc_h.empty and "close" in btc_h.columns:
                idx = btc_h.set_index("date")
                btc_d = idx.resample("1D").agg({{
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                }}).dropna(subset=["close"]).reset_index()
            else:
                if not getattr(self, "_btc_regime_warned", False):
                    import logging
                    logging.getLogger(__name__).warning(
                        "BTC 1d/1h data not found in DATA_DIR=%s; btc_regime filter is "
                        "disabled (all bars set to 'neutral').", str(self.DATA_DIR),
                    )
                    self._btc_regime_warned = True
                dataframe["btc_regime"] = "neutral"
                return dataframe
        btc_d = btc_d.sort_values("date").reset_index(drop=True)
        ret_7d = btc_d["close"].pct_change(7)
        vol_1d = btc_d["close"].pct_change().rolling(20).std()
        btc_trend = ret_7d / (vol_1d * np.sqrt(7) + 1e-10)
        btc_d["btc_regime"] = np.select(
            [btc_trend > 1, btc_trend < -1, btc_trend.abs() < 0.5],
            ["bull", "bear", "range"],
            default="neutral",
        )
        slim = btc_d[["date", "btc_regime"]].copy()
        # Aligne la precision datetime (ms live OHLCV vs ns feather) sinon MergeError.
        if slim["date"].dtype != dataframe["date"].dtype:
            slim["date"] = slim["date"].astype(dataframe["date"].dtype)
        merged = pd.merge_asof(
            dataframe.sort_values("date").reset_index(drop=True),
            slim.sort_values("date").reset_index(drop=True),
            on="date", direction="backward",
        )
        merged["btc_regime"] = merged["btc_regime"].fillna("neutral").astype(str)
        return merged

    def merge_ratio_coin(self, dataframe: pd.DataFrame, ref_coin: str, own_coin: str = "", lookback_bars: int = 200) -> pd.DataFrame:
        """Merge ratio close/ref_close z-score and breakout binaries into dataframe.

        Adds: ratio_<ref>_zscore, ratio_<ref>_high_break, ratio_<ref>_low_break, ratio_<ref> (NaN->neutral).
        ref_coin in {{'BTC','ETH'}}. own_coin: skip self-ratio (caller passes the trading pair's coin).
        """
        ref_up = (ref_coin or "BTC").upper()
        col_z = f"ratio_{{ref_up.lower()}}_zscore"
        col_hi = f"ratio_{{ref_up.lower()}}_high_break"
        col_lo = f"ratio_{{ref_up.lower()}}_low_break"
        col_ratio = f"ratio_{{ref_up.lower()}}"
        if own_coin and own_coin.upper() == ref_up:
            dataframe[col_z] = 0.0
            dataframe[col_hi] = float("inf")
            dataframe[col_lo] = float("-inf")
            dataframe[col_ratio] = 1.0
            return dataframe
        ref_df = self.load_coin_ohlcv(ref_up, self.timeframe, dataframe)
        if ref_df.empty or "close" not in ref_df.columns:
            dataframe[col_z] = 0.0
            dataframe[col_hi] = float("inf")
            dataframe[col_lo] = float("-inf")
            dataframe[col_ratio] = 1.0
            return dataframe
        ref_slim = ref_df[["date", "close"]].rename(columns={{"close": f"_{{ref_up.lower()}}_close"}}).copy()
        # Aligne la precision datetime (ms live OHLCV vs ns feather) sinon MergeError.
        if ref_slim["date"].dtype != dataframe["date"].dtype:
            ref_slim["date"] = ref_slim["date"].astype(dataframe["date"].dtype)
        merged = pd.merge_asof(
            dataframe.sort_values("date").reset_index(drop=True),
            ref_slim.sort_values("date").reset_index(drop=True),
            on="date", direction="backward",
        )
        ref_close_col = f"_{{ref_up.lower()}}_close"
        ratio = merged["close"] / merged[ref_close_col].replace(0, np.nan)
        w = max(lookback_bars, 20)
        mu = ratio.rolling(w).mean()
        sd = ratio.rolling(w).std()
        merged[col_z] = ((ratio - mu) / (sd + 1e-10)).fillna(0.0).astype(float)
        w_break = max(lookback_bars // 14 * 5, 10)
        merged[col_hi] = ratio.rolling(w_break).max().shift(1).fillna(float("inf")).astype(float)
        merged[col_lo] = ratio.rolling(w_break).min().shift(1).fillna(float("-inf")).astype(float)
        merged[col_ratio] = ratio.fillna(1.0).astype(float)
        merged = merged.drop(columns=[ref_close_col], errors="ignore")
        return merged
'''
