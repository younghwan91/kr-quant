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

from ..features.base_count import base_count_series
from ..features.vcp import detect_vcp
from .minervini_exits import (
    breakeven_plus_stop,
    climax_run,
    hard_stop,
    pe_expansion,
    sell_half_level,
    violations,
)

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


def _lookup(panel: pd.DataFrame, value: str, codes, dates) -> np.ndarray:
    """Reindex a long code/date/value panel to a codes×dates numpy array."""
    return (
        panel.pivot_table(index="code", columns="date", values=value, aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
    )


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
    close = _panel(prices, "close")
    codes, dates = list(close.index), list(close.columns)
    C = close.to_numpy(float)
    H = _panel(prices, "high").reindex(index=codes, columns=dates).to_numpy(float)
    L = _panel(prices, "low").reindex(index=codes, columns=dates).to_numpy(float)
    V = prices.pivot_table(index="code", columns="date", values="volume", aggfunc="first").reindex(
        index=codes, columns=dates).to_numpy(float)
    elig = _lookup(eligible_panel.assign(eligible=eligible_panel["eligible"].astype(float)),
                   "eligible", codes, dates)
    rs = _lookup(rs_panel, "rs_rating", codes, dates)
    c33 = (
        _lookup(code33.assign(is_code33=code33["is_code33"].astype(float)), "is_code33", codes, dates)
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
    fills = pivot_fills(entries, prices)
    close = _panel(prices, "close")
    codes, dates = list(close.index), list(close.columns)
    didx = {d: k for k, d in enumerate(dates)}
    C = close.to_numpy(float)
    OPN = _panel(prices, "open").reindex(index=codes, columns=dates).to_numpy(float)
    H = _panel(prices, "high").reindex(index=codes, columns=dates).to_numpy(float)
    L = _panel(prices, "low").reindex(index=codes, columns=dates).to_numpy(float)
    V = prices.pivot_table(index="code", columns="date", values="volume", aggfunc="first").reindex(
        index=codes, columns=dates).to_numpy(float)
    ma50 = pd.DataFrame(C.T).rolling(ma_exit_window, min_periods=1).mean().to_numpy()  # for breakeven
    PE = (
        pe_panel.pivot_table(index="code", columns="date", values="pe", aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
        if pe_panel is not None else None
    )
    cix = {c: k for k, c in enumerate(codes)}
    nD = len(dates)

    def _walk(i: int, f0: int, entry: float, stop0: float) -> tuple[float, float, str, str]:
        """One sub-position's forward walk under stop ``stop0`` → (ret, price, date, reason)."""
        stop = stop0
        half_target = sell_half_level(entry, stop, r=sell_half_r)  # +2R (relative to this stop)
        half_ret = None
        pe_entry = PE[i, f0] if PE is not None else float("nan")
        made_new_high = False
        exit_price = exit_date = reason = None
        for t in range(f0 + 1, min(f0 + time_cap + 1, nD)):
            if not np.isfinite(C[i, t]):
                continue
            if H[i, t] > entry:
                made_new_high = True
            if L[i, t] <= stop:  # stop hit — gap-through fills at the open
                exit_price = float(min(OPN[i, t], stop) if OPN[i, t] < stop else stop)
                exit_date, reason = dates[t], "stop"
                break
            if PE is not None and pe_expansion(PE[i, t], pe_entry):  # valuation overheated
                exit_price, exit_date, reason = float(C[i, t]), dates[t], "pe_expansion"
                break
            # Tennis-ball cull: a healthy leader bounces to a new high within a few days;
            # a "broken egg" that hasn't by tennis_window is dumped early.
            if tennis and (t - f0) >= tennis_window and not made_new_high:
                exit_price, exit_date, reason = float(C[i, t]), dates[t], "tennis"
                break
            # Sell half at +2R and raise the stop to break-even-or-better (max entry, MA50).
            if sell_half and half_ret is None and H[i, t] >= half_target:
                half_ret = half_target / entry - 1.0
                if breakeven:
                    raised = breakeven_plus_stop(entry, float(ma50[t, i]), r_reached=True)
                    stop = max(stop, raised)
            window = C[i, max(0, t - ma_exit_window):t + 1]
            vol_w = V[i, max(0, t - ma_exit_window):t + 1]
            if violations(window, vol_w, ma_window=ma_exit_window) or climax_run(C[i, :t + 1]):
                exit_price, exit_date = float(C[i, t]), dates[t]
                reason = "climax" if climax_run(C[i, :t + 1]) else "violation"
                break
        if exit_price is None:  # timed out — mark to last available bar
            t = min(f0 + time_cap, nD - 1)
            exit_price, exit_date, reason = float(C[i, t]), dates[t], "time_cap"
        rem_ret = exit_price / entry - 1.0
        if half_ret is not None:                       # blend: half at 2R, half at final exit
            return 0.5 * half_ret + 0.5 * rem_ret, exit_price, exit_date, f"{reason}+half"
        return rem_ret, exit_price, exit_date, reason

    out: list[dict] = []
    for _, f in fills[fills["filled"]].iterrows():
        i = cix[f["code"]]
        f0 = didx[str(f["fill_date"])]
        entry = float(f["fill_price"])
        if staggered:  # 3 equal tranches at −4/−6/−8% → average their outcomes
            legs = [_walk(i, f0, entry, hard_stop(entry, pct=p)) for p in STAGGERED_STOPS]
            ret = float(np.mean([leg[0] for leg in legs]))
            exit_price, exit_date, reason = legs[-1][1], legs[-1][2], "staggered"
        else:
            ret, exit_price, exit_date, reason = _walk(i, f0, entry, hard_stop(entry, pct=stop_pct))
        out.append({
            "code": f["code"], "entry_date": f["fill_date"], "entry_price": entry,
            "exit_date": exit_date, "exit_price": exit_price, "ret": ret, "reason": reason,
            "score": f.get("score", float("nan")),  # RS score for concentration sizing
        })
    return pd.DataFrame(out)


def _panel(prices: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot a long price frame to a code × date panel (abs — close/high/low signed)."""
    return prices.pivot_table(index="code", columns="date", values=value, aggfunc="first").abs()
