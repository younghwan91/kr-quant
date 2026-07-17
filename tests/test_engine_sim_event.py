"""Engine event-driven walk — generic coverage + the 6 `_walk` closure invariants.

The 6 invariant tests pin the subtle behaviors documented in
``docs/backtest-engine.md`` §"`position_walk` 불변식 6개"; each would be silently
broken by a naive reimplementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.engine.sim_eventdriven import position_walk, trade_runner


def _arr(vals) -> np.ndarray:
    return np.asarray(vals, float)


def _walk(close, *, open_=None, high=None, low=None, volume=None, ma50=None,
          entry_idx=0, entry_price=100.0, stop_price=95.0, **kw):
    """Call position_walk on per-code 1D arrays, deriving sane OHLCV defaults."""
    C = _arr(close)
    n = C.shape[0]
    OPN = _arr(open_) if open_ is not None else C.copy()
    H = _arr(high) if high is not None else C.copy()
    L = _arr(low) if low is not None else C.copy()
    V = _arr(volume) if volume is not None else np.full(n, 1000.0)
    M = _arr(ma50) if ma50 is not None else np.full(n, 100.0)
    return position_walk(C, OPN, H, L, V, M, entry_idx, entry_price,
                         stop_price=stop_price, **kw)


# --- Generic coverage ---------------------------------------------------------


def test_hard_stop_exit():
    # flat closes (no violation), low pierces the 95 stop on bar 3 → exit "stop".
    close = [100, 100, 100, 97, 97, 97]
    low = [100, 100, 100, 94, 97, 97]
    ret, px, idx, reason = _walk(close, low=low, tennis=False, sell_half=False)
    assert reason == "stop"
    assert idx == 3
    assert px == 95.0
    assert abs(ret - (-0.05)) < 1e-12


def test_time_cap_fallback():
    # flat, above stop, no exit rule fires → time_cap at f0+time_cap.
    close = [100.0] * 12
    ret, px, idx, reason = _walk(close, tennis=False, sell_half=False, time_cap=5)
    assert reason == "time_cap"
    assert idx == 5
    assert ret == 0.0


def test_sell_half_and_breakeven_blend():
    # +2R target = 110; high hits 111 → bank half at 110, then fall to break-even.
    close = [100, 106, 111, 100, 100, 100, 100]
    high = [100, 106, 111, 100, 100, 100, 100]
    low = [100, 106, 111, 99, 99, 99, 99]
    ret, px, idx, reason = _walk(close, high=high, low=low, ma50=np.full(7, 90.0),
                                 tennis=False, time_cap=10)
    assert reason.endswith("+half")
    # half banked at +10%, remainder near break-even (stop raised to entry 100).
    assert abs(ret - 0.05) < 0.06


def test_tennis_cull_generic():
    # drifts at 99 (never a new high) → culled after tennis_window.
    close = [100.0] + [99.0] * 15
    ret, px, idx, reason = _walk(close, stop_price=90.0, tennis=True, tennis_window=10)
    assert reason == "tennis"
    assert idx == 10


def test_pe_expansion_exit():
    close = [100, 101, 102, 103, 103, 103]
    pe = [10, 10, 30, 30, 30, 30]  # 3x from entry
    ret, px, idx, reason = _walk(close, stop_price=90.0, tennis=False,
                                 sell_half=False, pe_array=_arr(pe))
    assert reason == "pe_expansion"
    assert idx == 2


def test_violation_exit_three_down_days():
    # three strictly declining closes (101,100,99) complete the triple at bar 3.
    close = [100, 101, 100, 99, 98]
    ret, px, idx, reason = _walk(close, stop_price=80.0, tennis=False, sell_half=False)
    assert reason == "violation"
    assert idx == 3


def test_climax_exit_vertical_run():
    # +3%/bar → >25% over 10 bars → climax run (sell into strength).
    close = list(100 * 1.03 ** np.arange(14))
    ret, px, idx, reason = _walk(close, stop_price=50.0, tennis=False, sell_half=False,
                                 time_cap=50)
    assert reason == "climax"
    assert idx == 9  # full history reaches 10 bars (+25% run) at t=9


def test_no_exit_rides_to_time_cap():
    close = [100, 100.5, 101, 101.5, 102, 102.5]
    ret, px, idx, reason = _walk(close, stop_price=90.0, tennis=False, sell_half=False,
                                 time_cap=5)
    assert reason == "time_cap"
    assert ret > 0


def test_nan_bars_are_skipped():
    # a NaN bar mid-walk is skipped without exiting; walk continues past it.
    close = [100, 100, np.nan, 100, 100, 94, 94]
    low = [100, 100, np.nan, 100, 100, 94, 94]
    ret, px, idx, reason = _walk(close, low=low, stop_price=95.0, tennis=False,
                                 sell_half=False, time_cap=10)
    assert reason == "stop"
    assert idx == 5  # NaN at 2 did not trigger an exit


def test_gap_down_fills_at_open_generic():
    # gap-down open below stop → fill at the open, not the stop.
    close = [100, 100, 90]
    open_ = [100, 100, 90]
    low = [100, 100, 88]
    ret, px, idx, reason = _walk(close, open_=open_, low=low, stop_price=95.0,
                                 tennis=False, sell_half=False)
    assert reason == "stop"
    assert px == 90.0


def test_stop_price_is_absolute_not_percentage():
    # position_walk takes an ABSOLUTE stop price; a stop of 97 (not 0.03) fires at 97.
    # Open stays at 98 (above the stop) so the fill is at the stop, not a gap-through.
    close = [100, 96, 96]
    open_ = [100, 98, 98]
    low = [100, 96, 96]
    ret, px, idx, reason = _walk(close, open_=open_, low=low, stop_price=97.0,
                                 tennis=False, sell_half=False)
    assert reason == "stop"
    assert px == 97.0
    assert abs(ret - (-0.03)) < 1e-12


def test_staggered_blended_return_mean():
    # lows wick 96/94/92 on successive bars → mean(-4,-6,-8)% = -6%.
    codes, dates = ["X"], [f"d{k}" for k in range(20)]
    n = len(dates)
    close = np.full((1, n), 100.0)
    open_ = np.full((1, n), 100.0)
    high = np.full((1, n), 100.0)
    low = np.full((1, n), 100.0)
    low[0, 12], low[0, 13], low[0, 14] = 96.0, 94.0, 92.0
    vol = np.full((1, n), 1000.0)
    ma50 = np.full((n, 1), 100.0)
    fills = pd.DataFrame([{"code": "X", "fill_date": "d10", "fill_price": 100.0,
                           "filled": True, "score": np.nan}])
    out = trade_runner(close, open_, high, low, vol, ma50, dates, codes, fills,
                       time_cap=30, staggered=True, tennis=False, sell_half=False).iloc[0]
    assert out["reason"] == "staggered"
    assert abs(out["ret"] - (-0.06)) < 1e-9


# --- The 6 documented invariants ----------------------------------------------


def test_stop_fires_before_violation_on_same_bar():
    # INVARIANT 1: on a bar where BOTH the stop is hit AND violations() would fire,
    # the exit reason is "stop", not "violation". The declining triple (101,100,99)
    # completes exactly on bar 3, which is also the bar whose low pierces the 95 stop.
    close = [100, 101, 100, 99]
    low = [100, 101, 100, 94]  # only bar 3 pierces the 95 stop
    ret, px, idx, reason = _walk(close, low=low, stop_price=95.0, tennis=False,
                                 sell_half=False)
    assert reason == "stop"
    assert idx == 3
    # Sanity: the SAME bar independently satisfies violations() (proven by lifting
    # the stop out of reach → the same series then exits as "violation" on bar 3).
    ret2, px2, idx2, reason2 = _walk(close, low=low, stop_price=1.0, tennis=False,
                                     sell_half=False)
    assert reason2 == "violation"
    assert idx2 == 3


def test_climax_run_uses_full_history_not_recent_window():
    # INVARIANT 2: climax_run() sees the FULL history (C[:t+1]); violations() sees a
    # RECENT window. With a small ma_exit_window the recent window is too short for
    # climax_run to fire, but the full history triggers the vertical-run climax.
    close = list(100 * 1.03 ** np.arange(16))
    ret, px, idx, reason = _walk(close, stop_price=50.0, ma_exit_window=7,
                                 tennis=True, sell_half=False, time_cap=50)
    assert reason == "climax"
    assert idx == 9  # a naive same-(recent 8-bar)-window impl would ride to time_cap


def test_half_target_not_recomputed_after_stop_raise():
    # INVARIANT 3: half_target is frozen at entry (110 from stop 95). At the sell-half
    # bar the stop is raised to MA50=103; a naive recompute of half_target from the
    # raised stop would bank the half at a different level and change ret.
    close = [100, 105, 103]
    high = [100, 111, 103]   # bar 1 hits +2R (>=110) → bank half at 110
    low = [100, 104, 103]    # bar 2 low hits the raised stop (103)
    open_ = [100, 105, 103]
    ret, px, idx, reason = _walk(close, high=high, low=low, open_=open_,
                                 ma50=np.full(3, 103.0), stop_price=95.0,
                                 tennis=False, sell_half=True, breakeven=True, time_cap=10)
    assert reason == "stop+half"
    assert px == 103.0
    # correct (frozen half_target=110): 0.5*(110/100-1) + 0.5*(103/100-1) = 0.065
    assert abs(ret - 0.065) < 1e-12  # a recompute-from-raised-stop bug gives -0.015


def test_tennis_window_counts_calendar_bars_including_nan():
    # INVARIANT 4: the tennis window counts CALENDAR bars from entry, including
    # NaN-skipped ones. NaN bars at 3,4 must NOT delay the cull past f0+tennis_window.
    close = [100, 99, 99, np.nan, np.nan, 99, 99, 99, 99, 99, 99, 99, 99]
    high = [100, 99, 99, np.nan, np.nan, 99, 99, 99, 99, 99, 99, 99, 99]
    ret, px, idx, reason = _walk(close, high=high, stop_price=90.0, tennis=True,
                                 tennis_window=10, sell_half=False, time_cap=30)
    assert reason == "tennis"
    assert idx == 10  # calendar; a finite-bar counter would fire at 12 (2 NaNs later)


def test_gap_through_stop_fills_at_open():
    # INVARIANT 5: (a) gap-down open below stop → exit at open; (b) open above stop
    # but low pierces it → exit at the stop.
    a = _walk([100, 90], open_=[100, 90], low=[100, 88], stop_price=95.0,
              tennis=False, sell_half=False)
    assert a[3] == "stop" and a[1] == 90.0
    b = _walk([100, 98], open_=[100, 98], low=[100, 94], stop_price=95.0,
              tennis=False, sell_half=False)
    assert b[3] == "stop" and b[1] == 95.0


def test_staggered_reports_last_leg_metadata():
    # INVARIANT 6: exit_price/exit_date come from the LAST (8%) leg; ret is the mean
    # of all 3 legs; reason is the HARDCODED "staggered" literal even though the legs'
    # own reasons differ (4%/6% legs exit via "stop", the 8% leg via "time_cap").
    codes, dates = ["X"], [f"d{k}" for k in range(25)]
    n = len(dates)
    close = np.full((1, n), 100.0)
    open_ = np.full((1, n), 100.0)
    high = np.full((1, n), 100.0)
    low = np.full((1, n), 100.0)
    low[0, 13], low[0, 14] = 96.0, 94.0   # 4% leg (96) at d13, 6% leg (94) at d14
    # 8% leg (92) never hit → rides to time_cap at f0+time_cap = 10+6 = 16.
    vol = np.full((1, n), 1000.0)
    ma50 = np.full((n, 1), 100.0)
    fills = pd.DataFrame([{"code": "X", "fill_date": "d10", "fill_price": 100.0,
                           "filled": True, "score": np.nan}])
    out = trade_runner(close, open_, high, low, vol, ma50, dates, codes, fills,
                       time_cap=6, staggered=True, tennis=False, sell_half=False).iloc[0]
    assert out["reason"] == "staggered"                 # hardcoded literal, not a leg's
    assert out["exit_date"] == "d16"                    # 8% leg's time_cap bar
    assert out["exit_price"] == 100.0                   # 8% leg exit price
    assert abs(out["ret"] - np.mean([-0.04, -0.06, 0.0])) < 1e-12
