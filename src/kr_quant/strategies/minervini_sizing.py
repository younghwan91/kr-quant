"""Minervini position sizing — concentration, equity-risk, pilot, pyramiding.

Faithful arm A restores Minervini's concentrated, progressive sizing (the deployed
book inverted this to ≤50-name over-diversification): a handful of names, the best
weighted heaviest, sized so each trade risks a fixed slice of equity, entered as a
pilot then completed on proof, and pyramided into confirmed winners while holding
**total dollar risk constant** (``SEPA_FAITHFUL_DESIGN.md`` §5). Frozen defaults
follow the §사전등록 table.

Pure functions in → out.
"""

from __future__ import annotations

# Frozen SEPA hyperparameters (SEPA_FAITHFUL_DESIGN.md §사전등록 동결표).
N_NAMES = 6
TOP_WEIGHT = 0.25
REST_WEIGHT = 0.15
EQUITY_RISK = 0.0125
STOP_PCT = 0.05
POSITION_CAP = 0.25
PILOT_FRAC = 0.5
MAX_ADDS = 2
ADD_TRIGGER_R = 2.0


def concentration_weights(
    signals: dict[str, float],
    *,
    n: int = N_NAMES,
    top_w: float = TOP_WEIGHT,
    rest_w: float = REST_WEIGHT,
) -> dict[str, float]:
    """Top-``n`` names by signal: the best gets ``top_w``, the rest ``rest_w`` each.

    With the frozen 6 / 0.25 / 0.15 the weights sum to 1.0 (0.25 + 5×0.15). Fewer
    eligible names than ``n`` are all included (book may be under-invested).

    Args:
        signals: ``{code: signal_strength}`` (higher = stronger).
        n, top_w, rest_w: Concentration (count) and the top / non-top weights.

    Returns:
        ``{code: weight}`` for the selected top-``n`` names.
    """
    ranked = sorted(signals.items(), key=lambda kv: kv[1], reverse=True)[:n]
    if not ranked:
        return {}
    weights = {ranked[0][0]: top_w}
    for code, _ in ranked[1:]:
        weights[code] = rest_w
    return weights


def equity_risk_size(
    *,
    equity_risk: float = EQUITY_RISK,
    stop_pct: float = STOP_PCT,
    cap: float = POSITION_CAP,
) -> float:
    """Position weight from risk budget: ``equity_risk / stop_pct``, capped at ``cap``.

    Risking 1.25% of equity behind a 5% stop implies a 25% position — exactly the
    frozen cap. A wider stop shrinks the position (keeps the dollar risk fixed).
    """
    return float(min(equity_risk / stop_pct, cap))


def pilot_then_full(confirmed_r: float, *, pilot_frac: float = PILOT_FRAC) -> float:
    """Fraction of the target position to hold: ``pilot_frac`` until the trade proves
    itself (reaches +1R), then the full position. Minervini's progressive exposure."""
    return 1.0 if confirmed_r >= 1.0 else float(pilot_frac)


def pyramid_adds(
    new_base_rs: list[float],
    *,
    max_adds: int = MAX_ADDS,
    add_trigger_r: float = ADD_TRIGGER_R,
) -> int:
    """Number of add-and-reduce pyramids executed.

    Args:
        new_base_rs: The position's R-multiple profit at each successive new base.
        max_adds, add_trigger_r: Cap on adds and the R threshold to add (default +2R).

    Returns:
        Count of qualifying new bases (R ≥ ``add_trigger_r``), capped at ``max_adds``.
    """
    qualifying = sum(1 for r in new_base_rs if r >= add_trigger_r)
    return min(qualifying, max_adds)


def fixed_risk_stop(unit_entries: list[tuple[float, float]], initial_dollar_risk: float) -> float:
    """Raised stop that keeps total dollar risk equal to ``initial_dollar_risk`` after
    pyramiding. ``unit_entries`` = ``[(price, size), ...]`` across all units.

    Returns the stop ``s`` such that ``Σsize × (avg_entry − s) = initial_dollar_risk``
    — so adding units never enlarges the original risk (the core of add-and-reduce).
    """
    total_size = sum(size for _, size in unit_entries)
    avg_entry = sum(price * size for price, size in unit_entries) / total_size
    return float(avg_entry - initial_dollar_risk / total_size)
