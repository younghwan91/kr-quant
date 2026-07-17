"""Minervini-style event-driven walk — the SEPA paradigm, generalized.

Enters on a signal fill and walks each sub-position forward independently under a
stop, selling into strength (climax / P/E expansion), out of weakness (violations),
culling the tennis-ball non-bouncers, banking half at +2R and raising the stop to
break-even-or-better. Extracted (logic-preserving) from
``strategies.minervini_sepa`` so future event-driven experiments reuse the exit
accounting instead of re-deriving it.

This is **not** a general-purpose event-driven simulator: ``position_walk``'s
parameters (``tennis``, ``sell_half_r``, ``pe_array``, ``breakeven``) are the
Minervini exit toolkit. The exit-rule primitives themselves stay in
``strategies.minervini_exits`` (a pure-numpy module, imported here — no cycle:
it depends on numpy only and never imports the engine).

Provenance (Step 4 of the backtest-engine migration — logic preserved verbatim):
    position_walk <- minervini_sepa.sepa_trades._walk (closure)
    trade_runner  <- minervini_sepa.sepa_trades (fill loop)

The 6 `_walk` closure invariants documented in ``docs/backtest-engine.md``
§"`position_walk` 불변식 6개" are preserved here; each is flagged with an inline
``# INVARIANT N`` comment at its site.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategies.minervini_exits import (
    breakeven_plus_stop,
    climax_run,
    hard_stop,
    pe_expansion,
    sell_half_level,
    violations,
)

STAGGERED_STOPS = (0.04, 0.06, 0.08)  # -4/-6/-8% tranche stops (frozen)


def position_walk(
    close: np.ndarray,        # per-code 1D array, full history
    open_: np.ndarray,        # per-code 1D array
    high: np.ndarray,         # per-code 1D array
    low: np.ndarray,          # per-code 1D array
    volume: np.ndarray,       # per-code 1D array
    ma50: np.ndarray,         # per-code 1D array (precomputed MA, explicit input)
    entry_idx: int,           # bar index of the fill (f0)
    entry_price: float,
    *,
    stop_price: float,        # ABSOLUTE stop price (NOT a percentage — see module note)
    ma_exit_window: int = 50,
    time_cap: int = 200,
    sell_half: bool = True,
    breakeven: bool = True,
    sell_half_r: float = 2.0,
    pe_array: np.ndarray | None = None,  # per-code 1D array (or None)
    tennis: bool = True,
    tennis_window: int = 10,
) -> tuple[float, float, int, str]:
    """One sub-position's forward walk under ``stop_price`` → (ret, price, exit_idx, reason).

    ``exit_idx`` is the integer bar index of the exit; the caller (``trade_runner``)
    maps it to the date label. ``stop_price`` is the ABSOLUTE stop level — the
    percentage→price conversion (via ``hard_stop``, with its 10% clamp) happens in
    ``trade_runner``, exactly as ``sepa_trades`` did, so the abstraction boundary is
    unchanged. ``ma50`` is precomputed once and shared across calls (never recomputed
    here).
    """
    C, OPN, H, L, V, PE = close, open_, high, low, volume, pe_array
    f0 = entry_idx
    entry = entry_price
    nD = C.shape[0]
    stop = stop_price
    # INVARIANT 3: half_target is computed ONCE at entry from the INITIAL stop and is
    # never recomputed — even after the stop is raised to break-even below.
    half_target = sell_half_level(entry, stop, r=sell_half_r)  # +2R (relative to this stop)
    half_ret = None
    pe_entry = PE[f0] if PE is not None else float("nan")
    made_new_high = False
    exit_price = exit_idx = reason = None
    for t in range(f0 + 1, min(f0 + time_cap + 1, nD)):
        if not np.isfinite(C[t]):
            continue
        if H[t] > entry:
            made_new_high = True
        # INVARIANT 1: exit-rule priority — stop first. A gap-through fills at the open.
        if L[t] <= stop:  # stop hit — gap-through fills at the open
            # INVARIANT 5: gap-through stop-fill = min(open, stop), written in the
            # original conditional form (algebraically min(OPN[t], stop)) for auditability.
            exit_price = float(min(OPN[t], stop) if OPN[t] < stop else stop)
            exit_idx, reason = t, "stop"
            break
        if PE is not None and pe_expansion(PE[t], pe_entry):  # valuation overheated
            exit_price, exit_idx, reason = float(C[t]), t, "pe_expansion"
            break
        # INVARIANT 4: tennis window counts CALENDAR bars from entry (t - f0), including
        # NaN-skipped ones above — not a finite-bar counter.
        if tennis and (t - f0) >= tennis_window and not made_new_high:
            exit_price, exit_idx, reason = float(C[t]), t, "tennis"
            break
        # INVARIANT 1: sell_half is a STATE MUTATION, not an exit — it sets half_ret,
        # raises the stop, and FALLS THROUGH (no break). A bar can trigger sell_half AND
        # then exit via violations/climax in the same iteration.
        if sell_half and half_ret is None and H[t] >= half_target:
            half_ret = half_target / entry - 1.0
            if breakeven:
                raised = breakeven_plus_stop(entry, float(ma50[t]), r_reached=True)
                stop = max(stop, raised)
        # INVARIANT 2: window asymmetry — violations() sees a RECENT window; climax_run()
        # sees the FULL price history from bar 0. climax_run is evaluated twice (condition
        # + reason disambiguation), matching the original; do not collapse to one call.
        window = C[max(0, t - ma_exit_window):t + 1]
        vol_w = V[max(0, t - ma_exit_window):t + 1]
        if violations(window, vol_w, ma_window=ma_exit_window) or climax_run(C[:t + 1]):
            exit_price, exit_idx = float(C[t]), t
            reason = "climax" if climax_run(C[:t + 1]) else "violation"
            break
    if exit_price is None:  # timed out — mark to last available bar (post-loop fallback)
        t = min(f0 + time_cap, nD - 1)
        exit_price, exit_idx, reason = float(C[t]), t, "time_cap"
    rem_ret = exit_price / entry - 1.0
    if half_ret is not None:                       # blend: half at 2R, half at final exit
        return 0.5 * half_ret + 0.5 * rem_ret, exit_price, exit_idx, f"{reason}+half"
    return rem_ret, exit_price, exit_idx, reason


def trade_runner(
    close: np.ndarray,        # codes x dates 2D array
    open_: np.ndarray,        # codes x dates 2D array
    high: np.ndarray,         # codes x dates 2D array
    low: np.ndarray,          # codes x dates 2D array
    volume: np.ndarray,       # codes x dates 2D array
    ma50: np.ndarray,         # dates x codes 2D array (precomputed, transposed as today)
    dates: list,              # date labels
    codes: list,              # code labels
    fills: pd.DataFrame,      # output of pivot_fills
    *,
    stop_pct: float = 0.05,   # percentage — converted to an absolute price HERE via hard_stop()
    ma_exit_window: int = 50,
    time_cap: int = 200,
    sell_half: bool = True,
    breakeven: bool = True,
    sell_half_r: float = 2.0,
    pe_array: np.ndarray | None = None,  # codes x dates 2D array (or None)
    tennis: bool = True,
    tennis_window: int = 10,
    staggered: bool = False,
    staggered_stops: tuple[float, ...] = STAGGERED_STOPS,
) -> pd.DataFrame:
    """Loop over filled entries, walk each, return the trades DataFrame.

    Mirrors ``sepa_trades``' fill loop exactly: the stop percentage → absolute price
    conversion happens here (``hard_stop``, incl. its 10% clamp), and ``ma50``/``pe``
    panels are precomputed once by the caller and sliced per code.
    """
    didx = {d: k for k, d in enumerate(dates)}
    cix = {c: k for k, c in enumerate(codes)}

    out: list[dict] = []
    for _, f in fills[fills["filled"]].iterrows():
        i = cix[f["code"]]
        f0 = didx[str(f["fill_date"])]
        entry = float(f["fill_price"])
        pe_i = pe_array[i] if pe_array is not None else None
        args = (close[i], open_[i], high[i], low[i], volume[i], ma50[:, i], f0, entry)
        kw = dict(
            ma_exit_window=ma_exit_window, time_cap=time_cap, sell_half=sell_half,
            breakeven=breakeven, sell_half_r=sell_half_r, pe_array=pe_i,
            tennis=tennis, tennis_window=tennis_window,
        )
        if staggered:  # 3 equal tranches at -4/-6/-8% → average their outcomes
            legs = [position_walk(*args, stop_price=hard_stop(entry, pct=p), **kw)
                    for p in staggered_stops]
            ret = float(np.mean([leg[0] for leg in legs]))
            # INVARIANT 6: exit_price/exit_date come from the LAST leg (widest 8% stop);
            # reason is the HARDCODED literal "staggered", NOT read from any leg.
            exit_price, exit_idx, reason = legs[-1][1], legs[-1][2], "staggered"
        else:
            ret, exit_price, exit_idx, reason = position_walk(
                *args, stop_price=hard_stop(entry, pct=stop_pct), **kw)
        out.append({
            "code": f["code"], "entry_date": f["fill_date"], "entry_price": entry,
            "exit_date": dates[exit_idx], "exit_price": exit_price, "ret": ret,
            "reason": reason,
            "score": f.get("score", float("nan")),  # RS score for concentration sizing
        })
    return pd.DataFrame(out)
