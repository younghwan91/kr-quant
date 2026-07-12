"""Pivot buy-stop fill — the faithful SEPA entry (no close look-ahead).

Minervini buys the **intraday break** of a VCP pivot, not the confirmed close.
Daily bars can't see intraday, so the honest proxy (``SEPA_FAITHFUL_DESIGN.md``
§2.2) is a buy-stop resting at the pivot: on the day *after* the as-of-t-1 signal,
if the bar's high reaches the pivot the order fills at ``max(open, pivot)`` — the
open when it gaps through, else the pivot itself. This never touches the signal
day's close (the deployed scanner's look-ahead) and is deliberately conservative.

Pure functions in → out; feed split-adjusted bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.panels import lookup_panel, panel_pivot
from ..engine.sim_eventdriven import trade_runner
from ..features.base_count import base_count_series
from ..features.vcp import detect_vcp

STAGGERED_STOPS = (0.04, 0.06, 0.08)  # −4/−6/−8% tranche stops (frozen)


def pivot_fill(open_next: float, high_next: float, pivot: float) -> float | None:
    """Buy-stop fill price for the day after the signal, or ``None`` if unfilled.

    Args:
        open_next: Next session's open.
        high_next: Next session's high.
        pivot: Resting buy-stop level (the VCP pivot; fixed as-of t-1).

    Returns:
        ``max(open_next, pivot)`` when ``high_next ≥ pivot`` (a gap-up fills at the
        open, otherwise at the pivot); ``None`` when the high never reached the
        pivot (no fill that day).
    """
    if not (np.isfinite(open_next) and np.isfinite(high_next) and np.isfinite(pivot)):
        return None
    if high_next >= pivot:
        return float(max(open_next, pivot))
    return None


def pivot_fills(entries: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`pivot_fill` to each signal at its **next** trading session.

    Args:
        entries: Rows with ``code``, ``date`` (signal date t, as-of t-1 confirmed),
            ``pivot`` (buy-stop level).
        prices: Long ``code``/``date``/``open``/``high`` (split-adjusted; abs'd).

    Returns:
        One row per entry: ``code``, ``date``, ``pivot``, ``fill_date`` (t+1),
        ``fill_price`` (NaN if unfilled), ``filled`` (bool). Entries with no next
        session (or unknown code) are ``filled=False``.
    """
    op = prices.pivot_table(index="code", columns="date", values="open", aggfunc="first").abs()
    hi = prices.pivot_table(index="code", columns="date", values="high", aggfunc="first").abs()
    dates = list(op.columns)
    didx = {d: i for i, d in enumerate(dates)}

    rows: list[dict] = []
    for _, e in entries.iterrows():
        code, sig_date, pivot = e["code"], str(e["date"]), float(e["pivot"])
        score = float(e["score"]) if "score" in entries.columns and pd.notna(e.get("score")) else float("nan")
        i = didx.get(sig_date)
        fill_date = fill_price = None
        if i is not None and i + 1 < len(dates) and code in op.index:
            nd = dates[i + 1]
            fp = pivot_fill(op.at[code, nd], hi.at[code, nd], pivot)
            if fp is not None:
                fill_date, fill_price = nd, fp
        rows.append({
            "code": code, "date": sig_date, "pivot": pivot, "score": score,
            "fill_date": fill_date,
            "fill_price": fill_price if fill_price is not None else float("nan"),
            "filled": fill_price is not None,
        })
    return pd.DataFrame(
        rows, columns=["code", "date", "pivot", "score", "fill_date", "fill_price", "filled"])


def sepa_entries(
    prices: pd.DataFrame,
    eligible_panel: pd.DataFrame,
    rs_panel: pd.DataFrame,
    code33: pd.DataFrame,
    *,
    rs_min: float = 70.0,
    use_vcp: bool = True,
    use_code33: bool = True,
    use_base_count: bool = True,
    start_index: int = 252,
    vcp_params: dict | None = None,
) -> pd.DataFrame:
    """Assemble faithful arm-A entry signals by AND-ing every SEPA gate (as-of t).

    Composes the Phase 0/1 components into one signal: for each (code, date) the
    name must (1) be in the small-mid PIT universe, (2) pass the 7-criterion trend
    template, (3) clear RS ≥ ``rs_min`` (the 8th criterion), (4) be in Code 33, and
    (5) sit in a VCP whose pivot is the buy-stop. Everything is judged **as-of the
    signal bar** (entry is the next session via :func:`pivot_fills`), so no
    look-ahead. Toggles support the decomposition arms: ``use_vcp=False`` = A₋VCP,
    ``use_code33=False`` drops the fundamental leg.

    Args:
        prices: Long ``code``/``date``/``open``/``high``/``low``/``close``/``volume``
            (split-adjusted; close/high/low abs'd).
        eligible_panel: ``smallmid_universe`` output (``code``/``date``/``eligible``).
        rs_panel: ``rs_rating_panel`` output (``code``/``date``/``rs_rating``).
        code33: ``code33_panel`` output (``code``/``date``/``is_code33``).
        rs_min: RS-rating floor (frozen 70).
        use_vcp, use_code33, use_base_count: Gate toggles (base_count keeps only
            1st/2nd-stage bases). Off for the shell / decomposition arms.
        start_index: First date index to signal from (trend-template warm-up).
        vcp_params: Optional overrides forwarded to :func:`detect_vcp` — for the VCP
            robustness sweep only (the verdict uses the frozen defaults).

    Returns:
        Long ``code``/``date``/``pivot`` of entry signals (one per qualifying bar).
    """
    close = panel_pivot(prices, "close")
    codes, dates = list(close.index), list(close.columns)
    C = close.to_numpy(float)
    H = panel_pivot(prices, "high").reindex(index=codes, columns=dates).to_numpy(float)
    L = panel_pivot(prices, "low").reindex(index=codes, columns=dates).to_numpy(float)
    V = prices.pivot_table(index="code", columns="date", values="volume", aggfunc="first").reindex(
        index=codes, columns=dates).to_numpy(float)
    elig = lookup_panel(eligible_panel.assign(eligible=eligible_panel["eligible"].astype(float)),
                        "eligible", codes, dates)
    rs = lookup_panel(rs_panel, "rs_rating", codes, dates)
    c33 = (
        lookup_panel(code33.assign(is_code33=code33["is_code33"].astype(float)), "is_code33", codes, dates)
        if use_code33 else None
    )
    nD = len(dates)

    # Precompute the trend-template rolling stats once (date × code), C-optimized —
    # replaces a per-(code, t) nanmean that was the O(codes·dates·window) hot path.
    # min_periods = window means a name with <window history is NaN (→ gate fails),
    # a hair stricter than the old nanmean-of-partial but faithful (SEPA needs 200d+).
    cT = pd.DataFrame(C.T)
    ma50 = cT.rolling(50, min_periods=50).mean().to_numpy()
    ma150 = cT.rolling(150, min_periods=150).mean().to_numpy()
    ma200 = cT.rolling(200, min_periods=200).mean().to_numpy()
    ma200_prev = np.full_like(ma200, np.nan)
    ma200_prev[20:] = ma200[:-20]                         # ma200 as of t-20
    hh = pd.DataFrame(H.T).rolling(252, min_periods=252).max().to_numpy()
    ll = pd.DataFrame(L.T).rolling(252, min_periods=252).min().to_numpy()
    vkw = vcp_params or {}

    rows: list[dict] = []
    for i, code in enumerate(codes):
        c = C[i]
        # base_count(c, t) re-scans c[:t+1] from scratch on every call — O(bars) per
        # call, O(bars²) total per code. Precompute the full per-bar stage sequence
        # once here instead (found 2026-07-12 as the sepa_entries hot-path bottleneck).
        base_early = base_count_series(c)["is_early"] if use_base_count else None
        for t in range(start_index, nD):
            if not np.isfinite(c[t]) or elig[i, t] != 1.0:
                continue
            if not (rs[i, t] >= rs_min):
                continue
            if use_code33 and c33[i, t] != 1.0:
                continue
            m50, m150, m200, m200p, h_, l_ = (
                ma50[t, i], ma150[t, i], ma200[t, i], ma200_prev[t, i], hh[t, i], ll[t, i])
            # 7-criterion trend template (RS is the 8th, gated above). The 52-week-high
            # criterion is a proximity/eligibility check ("Stage 2 uptrend, near highs"),
            # not the breakout trigger — the real breakout is the VCP pivot below (the
            # base's own last-contraction resistance), per docs/minervini.txt §2.
            if not (c[t] > m50 > m150 > m200 and m200 > m200p
                    and c[t] >= 1.25 * l_ and c[t] >= 0.75 * h_):
                continue
            # Base count: Minervini buys only 1st/2nd-stage bases (late bases fail more).
            if use_base_count and not base_early[t]:
                continue
            pivot = c[t]  # default breakout level; VCP refines it when enabled
            if use_vcp:
                vcp = detect_vcp(H[i], L[i], c, V[i], t, **vkw)
                if not vcp["is_vcp"]:
                    continue
                pivot = vcp["pivot"]
            # score = RS at signal, carried through for concentration sizing downstream.
            rows.append({"code": code, "date": dates[t], "pivot": float(pivot),
                         "score": float(rs[i, t])})
    return pd.DataFrame(rows, columns=["code", "date", "pivot", "score"])


def sepa_trades(
    prices: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    stop_pct: float = 0.05,
    ma_exit_window: int = 50,
    time_cap: int = 200,
    sell_half: bool = True,
    breakeven: bool = True,
    sell_half_r: float = 2.0,
    pe_panel: pd.DataFrame | None = None,
    tennis: bool = True,
    tennis_window: int = 10,
    staggered: bool = False,
) -> pd.DataFrame:
    """Per-trade forward walk from each filled entry, applying Minervini exits.

    Fills each signal via :func:`pivot_fills` (t+1 buy-stop), then walks forward and
    exits on the first of: hard stop (gap-through fills at the open), a violation
    (MA break on volume / lower lows), a climax run (sell into strength), or the
    ``time_cap``. When ``sell_half`` the position sells **half at +``sell_half_r``R**
    (the returned ``ret`` blends the half taken at the 2R target with the remainder
    at final exit); when ``breakeven`` the stop is raised to ``max(entry, MA50)``
    once 2R is reached (never let a real gain become a loss).

    Returns:
        Long DataFrame ``code``/``entry_date``/``entry_price``/``exit_date``/
        ``exit_price``/``ret``/``reason`` (``reason`` ``+half`` suffix when a half
        was banked at 2R).
    """
    # Entry logic (pivot fills) stays here; the forward walk + exit accounting is the
    # engine's Minervini-style event-driven sim. Panels are built once and handed over:
    # ma50 (for the break-even stop raise) and PE (for the valuation exit) are
    # precomputed here and passed in as explicit inputs — the engine never recomputes
    # them. stop_pct is converted to an absolute price inside trade_runner (via
    # hard_stop, incl. its 10% clamp), exactly as before.
    fills = pivot_fills(entries, prices)
    close = panel_pivot(prices, "close")
    codes, dates = list(close.index), list(close.columns)
    C = close.to_numpy(float)
    OPN = panel_pivot(prices, "open").reindex(index=codes, columns=dates).to_numpy(float)
    H = panel_pivot(prices, "high").reindex(index=codes, columns=dates).to_numpy(float)
    L = panel_pivot(prices, "low").reindex(index=codes, columns=dates).to_numpy(float)
    V = prices.pivot_table(index="code", columns="date", values="volume", aggfunc="first").reindex(
        index=codes, columns=dates).to_numpy(float)
    ma50 = pd.DataFrame(C.T).rolling(ma_exit_window, min_periods=1).mean().to_numpy()  # for breakeven
    PE = (
        pe_panel.pivot_table(index="code", columns="date", values="pe", aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
        if pe_panel is not None else None
    )
    return trade_runner(
        C, OPN, H, L, V, ma50, dates, codes, fills,
        stop_pct=stop_pct, ma_exit_window=ma_exit_window, time_cap=time_cap,
        sell_half=sell_half, breakeven=breakeven, sell_half_r=sell_half_r,
        pe_array=PE, tennis=tennis, tennis_window=tennis_window,
        staggered=staggered, staggered_stops=STAGGERED_STOPS)
