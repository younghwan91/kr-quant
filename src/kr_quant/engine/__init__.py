"""kr_quant.engine — shared backtest primitives (metrics, panels, simulations).

A leaf package: imports numpy/pandas only and is imported *by* strategies, never
the reverse. Centralizes performance metrics and panel-construction conventions
so future experiments reuse them instead of re-deriving accounting logic.
"""

from __future__ import annotations

from .metrics import (
    ann_sharpe,
    cagr,
    max_drawdown,
    newey_west_t,
    paired_bootstrap,
    quantile_summary,
    regime_buckets,
    spearman,
    summarize_periods,
)
from .panels import (
    PANEL_CACHE,
    PanelCache,
    adv_panel,
    cached_panel_pivot,
    lookup_panel,
    panel_pivot,
    resolve_signal,
    yoy_panels,
)

__all__ = [
    # metrics
    "ann_sharpe",
    "cagr",
    "max_drawdown",
    "newey_west_t",
    "summarize_periods",
    "spearman",
    "quantile_summary",
    "paired_bootstrap",
    "regime_buckets",
    # panels
    "panel_pivot",
    "lookup_panel",
    "adv_panel",
    "yoy_panels",
    "resolve_signal",
    "cached_panel_pivot",
    "PanelCache",
    "PANEL_CACHE",
]
