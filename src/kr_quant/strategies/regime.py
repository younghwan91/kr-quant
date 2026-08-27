"""Regime off-switch — a *risk overlay*, not an alpha.

The rank-tilt books in this package earn a cross-sectional edge; the deployable
long-only + inverse-hedge form still carries residual market exposure, and its
drawdowns cluster in market downtrends. This module scales an already-computed
book return series by a market-regime state (index above/below its own long
moving average), which is the discretionary "지수 역배열 = 비매매" rule made
mechanical.

⚠️ **Why the null matters more than the switch.** *Any* switch that cuts
time-in-market cuts drawdown. Beating always-on proves nothing. The discriminator
is :func:`rotation_null` — the same state sequence rotated in time, which holds
duty cycle, run lengths and switch count fixed and destroys only the *alignment*
with returns. If the real switch does not beat that, the effect is exposure
reduction, not timing.

Pure Series in -> Series out (no DB), unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MA_WINDOW = 200  # trading days — the preregistered long moving average


def market_index_level(daily_market_ret: pd.Series) -> pd.Series:
    """Compound a daily market return Series into an index level (starts at 1.0)."""
    r = daily_market_ret.dropna().astype(float).sort_index()
    return (1.0 + r).cumprod()


def ma_regime_state(
    level: pd.Series,
    *,
    window: int = MA_WINDOW,
    mid_window: int | None = None,
) -> pd.Series:
    """Risk exposure implied by the index level vs its own moving average.

    Binary (the preregistered form): ``1.0`` when the level is above its
    ``window``-day MA, else ``0.0``. Pass ``mid_window`` for the three-tier
    variant (sensitivity only): above both MAs -> 1.0, above one -> 0.5, else 0.0.

    Args:
        level: Date-indexed index level (see :func:`market_index_level`).
        window: Long MA lookback in observations (trading days).
        mid_window: Optional shorter MA enabling the 0.5 tier.

    Returns:
        Exposure Series aligned to ``level``; ``NaN`` during MA warm-up.
    """
    lv = level.astype(float)
    ma = lv.rolling(window, min_periods=window).mean()
    above = (lv > ma).astype(float).where(ma.notna())
    if mid_window is None:
        return above
    ma_mid = lv.rolling(mid_window, min_periods=mid_window).mean()
    above_mid = (lv > ma_mid).astype(float).where(ma_mid.notna())
    return ((above + above_mid) / 2.0).where(above.notna() & above_mid.notna())


def monthly_state(state_daily: pd.Series) -> pd.Series:
    """Month-end value of a daily state, indexed by ``PeriodIndex`` freq ``M``.

    The month-end reading is what a trader knows at the close of month ``t``;
    :func:`apply_switch` is responsible for lagging it onto month ``t+1``.
    """
    s = state_daily.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    df = s.rename("x").reset_index()
    df.columns = ["date", "x"]
    df["m"] = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    return df.groupby("m")["x"].last().sort_index()


def apply_switch(
    monthly_ret: pd.Series,
    state_monthly: pd.Series,
    *,
    switch_cost: float = 0.0034,
    lag: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """Scale a monthly book return by a lagged regime state, charging turnover.

    The state read at the close of month ``t`` is applied to month ``t+lag``'s
    return — so nothing from the return month enters the decision. Each change in
    applied exposure is charged ``switch_cost × |Δexposure|`` (one leg; going flat
    and back is therefore charged twice, once each way).

    Args:
        monthly_ret: Book return by month (``PeriodIndex`` freq ``M``).
        state_monthly: Exposure by month, *unlagged* (as of that month's close).
        switch_cost: One-way cost charged per unit of exposure changed.
        lag: Months to delay the state (1 = knowable at entry; 0 disables the lag
            and is a **lookahead** setting, provided only for leakage tests).

    Returns:
        ``(switched_ret, applied_exposure)`` — both on ``monthly_ret``'s index,
        restricted to months where the exposure is defined.
    """
    r = monthly_ret.astype(float).sort_index()
    e = state_monthly.astype(float).sort_index().shift(lag).reindex(r.index)
    keep = e.notna() & r.notna()
    r, e = r[keep], e[keep]
    if r.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    prev = e.shift(1).fillna(0.0)  # start flat: entering the first position costs
    cost = switch_cost * (e - prev).abs()
    return (e * r - cost), e


def rotation_null(
    monthly_ret: pd.Series,
    state_monthly: pd.Series,
    metric_fn,
    *,
    switch_cost: float = 0.0034,
    lag: int = 1,
) -> tuple[list[float], float]:
    """Null distribution from every circular rotation of the state sequence.

    The state is first restricted to the months actually evaluated (after the lag
    and the intersection with ``monthly_ret``), then rotated. Rotating a fixed
    vector is a permutation of the same elements, so duty cycle and run-length
    structure are preserved exactly and switch count up to the wrap point; only
    the alignment with returns is destroyed. So a real timing edge shows up as the
    un-rotated value beating the rotated ones; plain exposure reduction shows up
    as the rotations matching it.

    Rotating *before* restricting — as an earlier version did — lets a different
    subsequence enter the evaluation window on each rotation, so the very
    quantities the null is supposed to hold fixed drift instead.

    Args:
        monthly_ret: Book return by month.
        state_monthly: Exposure by month, unlagged.
        metric_fn: ``Series -> float`` scoring a switched return series.
        switch_cost: Passed to :func:`apply_switch`.
        lag: Passed to :func:`apply_switch`.

    Returns:
        ``(null_values, actual_value)`` — one null per non-trivial rotation.
    """
    # ⚠️ 상태를 **평가창으로 먼저 자른 뒤** 회전한다.
    # 이전 판본은 전체 state_monthly 를 회전하고 apply_switch 안에서 북 인덱스로
    # reindex 했다. 상태가 북보다 길면 회전마다 다른 부분수열이 평가창에 들어와
    # 듀티사이클·스위치 횟수가 **보존되지 않았다**(실측: 듀티 0.663 → 회전들은
    # 0.584~0.703, 스위치 51 → 45~59). 널이 보존해야 할 바로 그 양이 흔들리면
    # "노출 축소가 아니라 타이밍인가" 를 가릴 수 없다.
    # 먼저 자르면 회전은 같은 원소들의 순열이므로 duty·switch 가 정확히 불변이다.
    r = monthly_ret.astype(float).sort_index()
    e = state_monthly.astype(float).sort_index().shift(lag).reindex(r.index)
    keep = e.notna() & r.notna()
    if not keep.any():
        return [], float("nan")
    ret_w = r[keep]
    # 평가에 실제로 쓰인(lag 적용 후) 노출만 남긴다.
    used = e[keep]
    used.index = ret_w.index

    def _score(exposure: pd.Series) -> float:
        prev = exposure.shift(1).fillna(0.0)
        cost = switch_cost * (exposure - prev).abs()
        return float(metric_fn(exposure * ret_w - cost))

    actual = _score(used)
    vals = used.to_numpy()
    nulls: list[float] = []
    for k in range(1, len(vals)):
        nulls.append(_score(pd.Series(np.roll(vals, k), index=ret_w.index)))
    return nulls, actual


def percentile_of(value: float, nulls: list[float]) -> float:
    """Fraction of ``nulls`` strictly below ``value`` (0..1). NaN if no nulls."""
    if not nulls:
        return float("nan")
    return float(np.mean([n < value for n in nulls]))
