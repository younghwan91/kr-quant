"""Declarative experiment recipes for the backtest engine (Step 5).

An :class:`ExperimentConfig` names the paradigm (event-driven Minervini walk vs
cross-sectional rank-tilt), the per-arm simulation params, and the comparison
params. :func:`run_recipe` dispatches to the right simulation module and — for the
multi-arm event-driven case — hands the aligned monthly return series to
:func:`kr_quant.strategies.sepa_compare.compare_arms`. The point: a *new*
experiment is defined by data (an :class:`ExperimentConfig`), not by copying a
simulation loop or re-deriving accounting conventions.

Provenance: generalizes ``sepa_experiment.run_experiment`` / ``robustness_sweep``.
Strategy-level imports (``sepa_entries``, ``sepa_trades``, ``book_returns``,
``benchmark_returns``, ``compare_arms``) are lazy so ``engine`` stays a leaf.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .metrics import ann_sharpe, cagr
from .panels import build_panels, panel_pivot, resolve_signal
from .sim_crosssectional import rank_tilt_backtest


@dataclass
class ArmSpec:
    """One arm of an experiment — how to turn panels into a return series/summary.

    Event-driven (``kind="sepa"``): ``entry_kwargs`` → :func:`sepa_entries`,
    ``trade_kwargs`` → :func:`sepa_trades`, ``book_kwargs`` → :func:`book_returns`;
    ``universe`` selects the ``build_panels`` eligibility panel ("smallmid" /
    "largecap"); ``use_pe`` wires the P/E panel into the valuation exit.
    Benchmark (``kind="benchmark"``): cap-weighted index proxy from the cap panel.
    Cross-sectional (``kind="rank_tilt"``): ``backtest_kwargs`` →
    :func:`kr_quant.engine.sim_crosssectional.rank_tilt_backtest`.
    """

    name: str
    kind: str = "sepa"          # "sepa" | "benchmark" | "rank_tilt"
    universe: str = "smallmid"  # build_panels key ("smallmid" | "largecap")
    entry_kwargs: dict = field(default_factory=dict)
    trade_kwargs: dict = field(default_factory=dict)
    book_kwargs: dict = field(default_factory=dict)
    use_pe: bool = False
    backtest_kwargs: dict = field(default_factory=dict)  # cross-sectional (rank_tilt)


@dataclass
class ExperimentConfig:
    """A declarative experiment definition consumed by :func:`run_recipe`.

    ``experiment_type`` selects the simulation module. For ``"event_driven"``,
    ``compare=True`` runs the paired-bootstrap :func:`compare_arms` verdict (needs
    ``deployed``/``benchmark`` arm names); ``compare=False`` returns a bare per-arm
    Sharpe/CAGR/n_trades summary (the robustness-sweep path). For
    ``"cross_sectional"``, ``signal_panel`` optionally supplies a precomputed signal
    (else the ``earnings`` YoY panel is used).
    """

    experiment_type: str        # "event_driven" | "cross_sectional"
    arms: list[ArmSpec]
    deployed: str | None = None
    benchmark: str | None = None
    adjust: bool = True
    compare: bool = True        # event_driven: run compare_arms (else summary-only)
    boot_kwargs: dict = field(default_factory=dict)
    signal_panel: pd.DataFrame | None = None  # cross_sectional precomputed signal


# --- Event-driven (Minervini walk) --------------------------------------------


def _sepa_arm_returns(spec: ArmSpec, panels: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Return ``(monthly_book_returns, trades)`` for one event-driven arm."""
    from ..strategies.minervini_sepa import sepa_entries, sepa_trades
    from ..strategies.sepa_compare import book_returns

    px = panels["prices"]
    ent = sepa_entries(px, panels[spec.universe], panels["rs"], panels["code33"],
                       **spec.entry_kwargs)
    pe = panels["pe"] if spec.use_pe else None
    trades = sepa_trades(px, ent, pe_panel=pe, **spec.trade_kwargs)
    return book_returns(px, trades, **spec.book_kwargs), trades


def _run_event_driven(config: ExperimentConfig, prices, earnings, shares):
    from ..strategies.sepa_compare import benchmark_returns, compare_arms

    panels = build_panels(prices, earnings, shares, adjust=config.adjust)
    arm_returns: dict[str, pd.Series] = {}
    n_trades: dict[str, int] = {}
    for spec in config.arms:
        if spec.kind == "benchmark":
            arm_returns[spec.name] = benchmark_returns(panels["prices"], panels["cap"])
            n_trades[spec.name] = 0
        else:
            r, trades = _sepa_arm_returns(spec, panels)
            arm_returns[spec.name] = r
            n_trades[spec.name] = len(trades)

    # Align every arm on the union of months so the paired bootstrap is well-defined.
    months = sorted(set().union(*[r.index for r in arm_returns.values()]))
    idx = pd.Index(months, name="month")
    arm_returns = {k: r.reindex(idx).fillna(0.0) for k, r in arm_returns.items()}

    if config.compare:
        return compare_arms(arm_returns, deployed=config.deployed,
                            benchmark=config.benchmark, **config.boot_kwargs)
    # Summary-only (robustness sweep): per-arm Sharpe / CAGR / n_trades, no verdict.
    rows = [{"arm": name,
             "sharpe": ann_sharpe(r.to_numpy(float)),
             "cagr": cagr(r.to_numpy(float)),
             "n_trades": n_trades[name]}
            for name, r in arm_returns.items()]
    return pd.DataFrame(rows).set_index("arm"), {}


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
        - event-driven + ``compare``: ``compare_arms`` table + paired verdicts.
        - event-driven summary-only: per-arm Sharpe/CAGR/n_trades table, ``{}``.
        - cross-sectional: per-arm summary table + ``{arm: summarize_periods(...)}``.
    """
    if config.experiment_type == "event_driven":
        return _run_event_driven(config, prices, earnings, shares)
    if config.experiment_type == "cross_sectional":
        return _run_cross_sectional(config, prices, earnings, shares)
    raise ValueError(f"unknown experiment_type: {config.experiment_type!r}")


# --- Prebuilt recipe: the faithful 5-arm SEPA comparison ----------------------


def sepa_faithful_config(
    *,
    adjust: bool = True,
    n_concentrated: int = 6,
    n_diversified: int = 50,
    **boot_kwargs,
) -> ExperimentConfig:
    """The pre-registered faithful-SEPA 5-arm comparison as an :class:`ExperimentConfig`.

    Reproduces ``sepa_experiment.run_experiment``: arms A / A-diversified / A-noVCP
    (small-mid Minervini), B-shell (large-cap breakout shell), C-bench (cap-weighted
    index proxy). ``boot_kwargs`` forward to the paired bootstrap.
    """
    arms = [
        ArmSpec("A", universe="smallmid", use_pe=True,
                entry_kwargs={"use_vcp": True, "use_code33": True},
                book_kwargs={"n_slots": n_concentrated, "sized": True}),
        ArmSpec("A-diversified", universe="smallmid", use_pe=True,
                entry_kwargs={"use_vcp": True, "use_code33": True},
                book_kwargs={"n_slots": n_diversified, "sized": False}),
        ArmSpec("A-noVCP", universe="smallmid", use_pe=True,
                entry_kwargs={"use_vcp": False, "use_code33": True},
                book_kwargs={"n_slots": n_concentrated, "sized": True}),
        ArmSpec("B-shell", universe="largecap", use_pe=False,
                entry_kwargs={"use_vcp": False, "use_code33": False,
                              "use_base_count": False, "rs_min": 0.0},
                book_kwargs={"n_slots": n_diversified, "sized": False}),
        ArmSpec("C-bench", kind="benchmark"),
    ]
    return ExperimentConfig(
        experiment_type="event_driven", arms=arms,
        deployed="B-shell", benchmark="C-bench",
        adjust=adjust, compare=True, boot_kwargs=boot_kwargs)
