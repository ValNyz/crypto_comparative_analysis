# =============================================================================
# FILE: lib/utils/colors.py
# =============================================================================
"""Threshold-based coloring helpers for the live per-line backtest output.

Used solely in `runner._print_result` (Phase 2 live results) — recap
sections and Phase 3 summary stay neutral on purpose so log captures
remain clean for diff/grep.

Implementation uses `rich` to render ANSI codes via Console.capture, so
existing `print()` calls keep working without touching the rest of the
output pipeline. `rich` auto-detects TTY; honors NO_COLOR env var.
"""

import os

try:
    from rich.console import Console
    _CONSOLE = Console(
        highlight=False,
        soft_wrap=True,
        force_terminal=None,  # auto-detect
        no_color=os.environ.get("NO_COLOR") is not None,
    )
    _ENABLED = _CONSOLE.is_terminal
except ImportError:
    _CONSOLE = None
    _ENABLED = False


def _wrap(text: str, style: str) -> str:
    if not _ENABLED or _CONSOLE is None:
        return text
    with _CONSOLE.capture() as cap:
        _CONSOLE.print(f"[{style}]{text}[/]", end="", markup=True)
    return cap.get()


def color_sharpe(value, formatted: str) -> str:
    """Sharpe ≥ 2 → bold green, ≥ 1 → green, else neutral."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return formatted
    if v >= 2.0:
        return _wrap(formatted, "bold green")
    if v >= 1.0:
        return _wrap(formatted, "green")
    return formatted


def color_dd(value, formatted: str) -> str:
    """DD < 5% → bold green, < 10% → green, else neutral. Lower is better."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return formatted
    if v < 5.0:
        return _wrap(formatted, "bold green")
    if v < 10.0:
        return _wrap(formatted, "green")
    return formatted


def color_pvalue(value, formatted: str) -> str:
    """p ≤ 0.01 → bold green, ≤ 0.05 → green, else neutral."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return formatted
    if v <= 0.01:
        return _wrap(formatted, "bold green")
    if v <= 0.05:
        return _wrap(formatted, "green")
    return formatted
