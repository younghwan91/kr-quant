"""Minervini sell rules — the exit toolkit the deployed scanner dropped.

The deployed Minervini keeps only a fixed 5% stop + long hold; the real method has
a full sell discipline (``SEPA_FAITHFUL_DESIGN.md`` §3): tight/staggered stops,
selling into climactic strength, violation exits, a sell-half rule and a
break-even-or-better stop raise. These were rejected piecemeal on the large-cap
shell; the faithful arm A restores them as a coherent set.

Every function is pure and **as-of**: it judges only from the position's price path
up to the current bar (no future bars). Frozen defaults follow the §사전등록 table.
"""

from __future__ import annotations

import numpy as np

# Frozen SEPA hyperparameters (SEPA_FAITHFUL_DESIGN.md §사전등록 동결표).
HARD_STOP_PCT = 0.05
MAX_STOP_PCT = 0.10
STAGGERED_THIRDS = (-0.04, -0.06, -0.08)
CLIMAX_WINDOW = 10
CLIMAX_RUN_PCT = 0.25
UPDAYS_WINDOW = 15
UPDAYS_FRAC = 0.70
VIOLATION_MA = 20
VIOLATION_DOWN_DAYS = 3
SELL_HALF_R = 2.0


def hard_stop(entry: float, *, pct: float = HARD_STOP_PCT, max_pct: float = MAX_STOP_PCT) -> float:
    """Hard stop price = ``entry × (1 − min(pct, max_pct))`` (never worse than 10%)."""
    return float(entry * (1.0 - min(pct, max_pct)))


def staggered_stops(entry: float, *, thirds: tuple[float, ...] = STAGGERED_THIRDS) -> list[float]:
    """Split-position stop prices — thirds at −4/−6/−8% to survive volatility spikes."""
    return [float(entry * (1.0 + frac)) for frac in thirds]


def climax_run(
    closes: np.ndarray,
    *,
    window: int = CLIMAX_WINDOW,
    run_pct: float = CLIMAX_RUN_PCT,
    updays_window: int = UPDAYS_WINDOW,
    updays_frac: float = UPDAYS_FRAC,
) -> bool:
    """Sell-into-strength trigger: a climactic run just occurred (as of the last bar).

    True if either the price ran up at least ``run_pct`` over the last ``window``
    days, **or** at least ``updays_frac`` of the last ``updays_window`` days closed
    up — Minervini's two climax signatures (vertical spike / relentless up-days).
    """
    c = np.asarray(closes, float)
    c = c[np.isfinite(c)]
    if c.size >= window and c[-window] > 0:
        if c[-1] / c[-window] - 1.0 >= run_pct:
            return True
    if c.size > updays_window:
        diffs = np.diff(c[-(updays_window + 1):])
        if (diffs > 0).mean() >= updays_frac:
            return True
    return False


def violations(
    closes: np.ndarray,
    volumes: np.ndarray,
    *,
    ma_window: int = VIOLATION_MA,
    down_days: int = VIOLATION_DOWN_DAYS,
    vol_mult: float = 1.0,
) -> bool:
    """Sell-into-weakness trigger (as of the last bar): a breakdown occurred.

    True if the last close broke **below its ``ma_window``-day MA on above-average
    volume**, or the last ``down_days`` closes were strictly declining (successive
    lower lows) — Minervini's early violation signals.
    """
    c = np.asarray(closes, float)
    v = np.asarray(volumes, float)
    if c.size >= ma_window and np.all(np.isfinite(c[-ma_window:])):
        ma = float(np.mean(c[-ma_window:]))
        avg_vol = float(np.mean(v[-ma_window:])) if np.all(np.isfinite(v[-ma_window:])) else np.nan
        if c[-1] < ma and np.isfinite(avg_vol) and v[-1] > vol_mult * avg_vol:
            return True
    if c.size >= down_days:
        tail = c[-down_days:]
        if np.all(np.isfinite(tail)) and np.all(np.diff(tail) < 0):
            return True
    return False


def sell_half_level(entry: float, stop: float, *, r: float = SELL_HALF_R) -> float:
    """Price to sell half the position: ``entry + r × (entry − stop)`` (default 2R)."""
    return float(entry + r * (entry - stop))


def breakeven_plus_stop(entry: float, ma50: float, *, r_reached: bool) -> float | None:
    """Raised stop once the trade is ≥2R in profit: ``max(entry, ma50)`` (never let a
    real gain turn into a loss). ``None`` while the 2R threshold is not yet reached."""
    if not r_reached:
        return None
    return float(max(entry, ma50))


PE_EXPANSION_FACTOR = 2.5   # frozen: sell when P/E has expanded 2–3× since entry


def pe_expansion(pe_now: float, pe_entry: float, *, factor: float = PE_EXPANSION_FACTOR) -> bool:
    """Sell-into-strength trigger: P/E has expanded ≥ ``factor``× since the base start.

    Minervini exits when a leader's valuation multiple has run 2–3× from the trend's
    starting P/E (a late-base overheating signal). ``False`` when either P/E is
    missing or the entry P/E is non-positive (loss-making → ratio undefined)."""
    if not (np.isfinite(pe_now) and np.isfinite(pe_entry) and pe_entry > 0):
        return False
    return pe_now >= factor * pe_entry
