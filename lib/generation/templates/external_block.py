# =============================================================================
# FILE: lib/generation/templates/external_block.py
# =============================================================================
"""External-data loader methods injected into generated strategies.

All loaders are methods of the strategy class. They read parquets from
EXTERNAL_DATA_DIR (auto-derived as Path(data_dir).parent / "external").
All merge_asof are direction="backward" (no lookahead).
All loaders are permissive: missing/empty parquet → no-op (filter passes).
"""

EXTERNAL_LOADERS_BLOCK = '''
    EXTERNAL_DATA_DIR = Path("{external_data_dir}")

    def _load_external_parquet(self, fname: str) -> pd.DataFrame:
        """Read parquet from EXTERNAL_DATA_DIR. Returns empty df if absent/unreadable."""
        path = self.EXTERNAL_DATA_DIR / fname
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        return df.sort_values("timestamp").reset_index(drop=True)

    def _align_to_dataframe_tz(self, df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
        """Convert df['timestamp'] tz to match reference_df['date'] tz, return rename'd."""
        if df.empty:
            return df
        ref_tz = reference_df["date"].dt.tz
        out = df.copy()
        if ref_tz is not None:
            if out["timestamp"].dt.tz is None:
                out["timestamp"] = out["timestamp"].dt.tz_localize("UTC").dt.tz_convert(ref_tz)
            else:
                out["timestamp"] = out["timestamp"].dt.tz_convert(ref_tz)
        else:
            if out["timestamp"].dt.tz is not None:
                out["timestamp"] = out["timestamp"].dt.tz_localize(None)
        return out.rename(columns={{"timestamp": "date"}})

    def merge_external_fng(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Merge Fear & Greed daily into dataframe. Adds column 'fng_value' (int 0-100, NaN→50 = neutral)."""
        fng = self._load_external_parquet("fng_daily.parquet")
        if fng.empty or "fng_value" not in fng.columns:
            dataframe["fng_value"] = 50
            return dataframe
        fng = self._align_to_dataframe_tz(fng[["timestamp", "fng_value"]], dataframe)
        merged = pd.merge_asof(
            dataframe.sort_values("date").reset_index(drop=True),
            fng.sort_values("date").reset_index(drop=True),
            on="date", direction="backward",
        )
        merged["fng_value"] = merged["fng_value"].fillna(50).astype(float)
        return merged

    def merge_external_vix(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Merge VIX daily into dataframe. Adds column 'vix_close' (NaN→15 = neutral)."""
        vix = self._load_external_parquet("yfinance_VIX.parquet")
        if vix.empty or "close" not in vix.columns:
            dataframe["vix_close"] = 15.0
            return dataframe
        v = vix[["timestamp", "close"]].rename(columns={{"close": "vix_close"}})
        v = self._align_to_dataframe_tz(v, dataframe)
        merged = pd.merge_asof(
            dataframe.sort_values("date").reset_index(drop=True),
            v.sort_values("date").reset_index(drop=True),
            on="date", direction="backward",
        )
        merged["vix_close"] = merged["vix_close"].fillna(15.0).astype(float)
        return merged

    def merge_external_dxy(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Merge DXY daily into dataframe. Adds 'dxy_close' (NaN→100 = neutral) and 'dxy_slope10' (NaN→0)."""
        dxy = self._load_external_parquet("yfinance_DXY.parquet")
        if dxy.empty or "close" not in dxy.columns:
            dataframe["dxy_close"] = 100.0
            dataframe["dxy_slope10"] = 0.0
            return dataframe
        d = dxy[["timestamp", "close"]].rename(columns={{"close": "dxy_close"}}).sort_values("timestamp").reset_index(drop=True)
        d["dxy_slope10"] = d["dxy_close"].pct_change(10)
        d = self._align_to_dataframe_tz(d, dataframe)
        merged = pd.merge_asof(
            dataframe.sort_values("date").reset_index(drop=True),
            d.sort_values("date").reset_index(drop=True),
            on="date", direction="backward",
        )
        merged["dxy_close"] = merged["dxy_close"].fillna(100.0).astype(float)
        merged["dxy_slope10"] = merged["dxy_slope10"].fillna(0.0).astype(float)
        return merged

    def merge_external_etf_flow(self, dataframe: pd.DataFrame, ref: str) -> pd.DataFrame:
        """Merge ETF flow daily into dataframe. ref ∈ {{'btc','eth'}}. Adds 'etf_flow_usd_m' (NaN→0)."""
        ref_low = (ref or "btc").lower()
        if ref_low not in ("btc", "eth"):
            ref_low = "btc"
        etf = self._load_external_parquet(f"etf_flows_{{ref_low}}.parquet")
        if etf.empty or "flow_usd_m" not in etf.columns:
            dataframe["etf_flow_usd_m"] = 0.0
            return dataframe
        e = etf[["timestamp", "flow_usd_m"]].rename(columns={{"flow_usd_m": "etf_flow_usd_m"}})
        e = self._align_to_dataframe_tz(e, dataframe)
        merged = pd.merge_asof(
            dataframe.sort_values("date").reset_index(drop=True),
            e.sort_values("date").reset_index(drop=True),
            on="date", direction="backward",
        )
        merged["etf_flow_usd_m"] = merged["etf_flow_usd_m"].fillna(0.0).astype(float)
        return merged

    def _load_binance_funding(self, symbol: str) -> pd.DataFrame:
        """Load Binance funding for symbol (e.g. 'BTCUSDT'). Returns df with timestamp + funding_rate, floored to hour."""
        df = self._load_external_parquet(f"binance_funding_{{symbol}}.parquet")
        if df.empty or "funding_rate" not in df.columns:
            return df
        out = df[["timestamp", "funding_rate"]].copy()
        out["timestamp"] = out["timestamp"].dt.floor("h")
        return out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    def merge_external_funding_spread(self, dataframe: pd.DataFrame, ref: str, w_zscore: int = 90) -> pd.DataFrame:
        """Merge HL-Binance funding spread z-score into dataframe.

        ref in {{'BTC','ETH'}}. Computed on Binance 8h grid, w_zscore in 8h periods (90 = 30 days).
        Adds 'funding_spread_zscore' (NaN->0). HL funding side comes from load_cross_coin_funding
        (provided by cross_coin_block); when that block is absent the filter no-ops.
        """
        ref_up = (ref or "BTC").upper()
        if ref_up not in ("BTC", "ETH"):
            ref_up = "BTC"
        bn = self._load_binance_funding(f"{{ref_up}}USDT")
        if bn.empty:
            dataframe["funding_spread_zscore"] = 0.0
            return dataframe
        hl_df = self.load_cross_coin_funding(ref_up, dataframe) if hasattr(self, "load_cross_coin_funding") else pd.DataFrame()
        if hl_df.empty or "funding_rate" not in hl_df.columns:
            dataframe["funding_spread_zscore"] = 0.0
            return dataframe
        bn = bn.rename(columns={{"funding_rate": "fr_binance"}}).sort_values("timestamp")
        hl = hl_df[["date", "funding_rate"]].rename(columns={{"date": "timestamp", "funding_rate": "fr_hl"}}).sort_values("timestamp")
        merged = pd.merge_asof(bn, hl, on="timestamp", direction="backward")
        merged["spread"] = merged["fr_hl"] - merged["fr_binance"]
        mu = merged["spread"].rolling(w_zscore).mean()
        sd = merged["spread"].rolling(w_zscore).std()
        merged["funding_spread_zscore"] = (merged["spread"] - mu) / (sd + 1e-10)
        sp = merged[["timestamp", "funding_spread_zscore"]].dropna(subset=["funding_spread_zscore"])
        sp = self._align_to_dataframe_tz(sp, dataframe)
        out = pd.merge_asof(
            dataframe.sort_values("date").reset_index(drop=True),
            sp.sort_values("date").reset_index(drop=True),
            on="date", direction="backward",
        )
        out["funding_spread_zscore"] = out["funding_spread_zscore"].fillna(0.0).astype(float)
        return out
'''
