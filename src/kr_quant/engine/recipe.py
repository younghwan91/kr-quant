"""Declarative experiment recipes for the backtest engine (Step 5).

An :class:`ExperimentConfig` names the simulation paradigm, the per-arm
simulation params, and the comparison params. :func:`run_recipe` dispatches to
the right simulation module. The point: a *new* experiment is defined by data (an
:class:`ExperimentConfig`), not by copying a simulation loop or re-deriving
accounting conventions.

Only the cross-sectional (rank-tilt) paradigm is wired today — the event-driven
breakout path was removed with the strategy it served. Re-adding a paradigm means
adding a ``_run_*`` branch here, not a new accounting loop in a strategy file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .panels import panel_pivot, resolve_signal
from .sim_crosssectional import rank_tilt_backtest


@dataclass
class ArmSpec:
    """One arm of an experiment — how to turn panels into a return series/summary.

    ``backtest_kwargs`` →
    :func:`kr_quant.engine.sim_crosssectional.rank_tilt_backtest`. The paradigm
    lives on :class:`ExperimentConfig.experiment_type`, not here — a per-arm
    ``kind`` field survived the event-driven removal with a single legal value
    and no reader, which is dead state that silently accepts any string.
    """

    name: str
    backtest_kwargs: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """A declarative experiment definition consumed by :func:`run_recipe`.

    ``experiment_type`` selects the simulation module. For ``"cross_sectional"``,
    ``signal_panel`` optionally supplies a precomputed signal (else the
    ``earnings`` YoY panel is used).
    """

    experiment_type: str        # "cross_sectional"
    arms: list[ArmSpec]
    adjust: bool = True
    signal_panel: pd.DataFrame | None = None  # cross_sectional precomputed signal


# --- Cross-sectional (rank-tilt) ----------------------------------------------


def _run_cross_sectional(config: ExperimentConfig, prices, earnings, shares=None):
    close = panel_pivot(prices, "close")
    tval = panel_pivot(prices, "trade_value")
    dates = list(close.columns)
    codes = list(close.index)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)
    sig, age = resolve_signal(earnings, config.signal_panel, codes, dates)

    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for spec in config.arms:
        _, summary = rank_tilt_backtest(C, V, sig, dates, age=age, **spec.backtest_kwargs)
        summaries[spec.name] = summary
        rows.append({"arm": spec.name, "n": summary["n"], "sharpe": summary["sharpe"],
                     "t_stat": summary["t_stat"], "cum_net": summary["cum_net"],
                     "hit_rate": summary["hit_rate"]})
    return pd.DataFrame(rows).set_index("arm"), summaries


def run_recipe(config: ExperimentConfig, prices, earnings, shares=None):
    """Dispatch an :class:`ExperimentConfig` to its simulation module.

    Returns ``(table, verdicts)``:
        - cross-sectional: per-arm summary table + ``{arm: summarize_periods(...)}``.
    """
    if config.experiment_type == "cross_sectional":
        return _run_cross_sectional(config, prices, earnings, shares)
    raise ValueError(f"unknown experiment_type: {config.experiment_type!r}")
