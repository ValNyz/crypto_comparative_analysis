# =============================================================================
# FILE: lib/report/utils/dedup.py
# =============================================================================
"""Display-only dedup helpers (no impact on FDR / null-pool computation).

The strategy generator appends `_x{short_exit}` (and `_sl{N}`, `_r{N}`) to
signal names when those dimensions are swept without being in the name
template. For ranking displays, exit siblings often produce near-identical
results (the trail rule never fires). Stripping just the exit suffix lets
us collapse 3-9 redundant rows to a single representative without losing
the SL/ROI axes (which the user defines as part of strategy identity).
"""

from typing import Optional
import pandas as pd

from ...signals.registry import _short_exit_name


def strip_exit_suffix(signal_name: str, exit_config) -> str:
    """Remove the `_x{short}` exit suffix if present.

    Mirrors lib.signals.registry._short_exit_name. Idempotent — returns
    name unchanged when no suffix matches (already stripped, or exit_config
    not in the sweep).
    """
    if not exit_config or pd.isna(exit_config):
        return signal_name
    suf = f"_x{_short_exit_name(str(exit_config))}"
    if signal_name.endswith(suf):
        return signal_name[: -len(suf)]
    return signal_name


def add_signal_root(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `signal_root` column = signal name with exit suffix stripped.

    Idempotent. Returns the same df with the new column when `signal` and
    `exit_config` are both present, otherwise returns df unchanged.
    """
    if "signal_root" in df.columns:
        return df
    if "signal" not in df.columns:
        return df
    df = df.copy()
    df["signal_root"] = df.apply(
        lambda r: strip_exit_suffix(str(r["signal"]), r.get("exit_config")),
        axis=1,
    )
    return df


def dedup_for_display(
    df: pd.DataFrame,
    sort_cols=None,
    sort_ascending=None,
    keys: tuple = ("signal_root", "pair"),
) -> pd.DataFrame:
    """Sort then drop_duplicates on the given keys, keeping the best row.

    Default keys = (signal_root, pair) → collapses TF + exit + SL siblings,
    so each (core strategy, pair) combo shows once with its best variant
    (per the caller's sort criterion). Override `keys` for narrower views
    (e.g. ('signal_root',) when pair is already filtered).

    `sort_cols` accepts a single str or a list; `sort_ascending` is bool or
    list of bool (defaults to descending — caller usually wants "best first").
    """
    if not set(keys).issubset(df.columns):
        return df
    if sort_cols:
        if isinstance(sort_cols, str):
            sort_cols = [sort_cols]
        if sort_ascending is None:
            sort_ascending = [False] * len(sort_cols)
        elif isinstance(sort_ascending, bool):
            sort_ascending = [sort_ascending] * len(sort_cols)
        df = df.sort_values(sort_cols, ascending=sort_ascending)
    return df.drop_duplicates(subset=list(keys), keep="first")
