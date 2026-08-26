"""Cross-sectional rank-tilt simulation — the PEAD paradigm, generalized.

Rebalances a rank-weighted book on a fixed schedule and charges cost on measured
turnover. Extracted (byte-identical) from ``strategies.pead`` so future
cross-sectional experiments reuse the accounting (entry at ``t+1``, ADV floor,
turnover cost, borrow drag, long-only excess) instead of re-deriving it.

The inner loops operate on numpy ``code × date`` arrays; the strategy-level
wrappers in ``strategies.pead`` do the DataFrame→array conversion via
``engine.panels``. A ``dates`` label list is threaded through purely so the
returned ``periods`` frame carries the same ``date`` column as before.

Provenance (Step 3 of the backtest-engine migration — copied, logic preserved):
    rank_tilt_backtest        <- pead.pead_backtest (inner loop)
    staggered_tranche_backtest <- pead.staggered_backtest (inner loop)
    rank_ic                   <- pead.pead_rank_ic (inner loop)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import newey_west_t, summarize_periods


def rank_tilt_backtest(
    close: np.ndarray,
    trade_value: np.ndarray,
    signal: np.ndarray,
    dates: list,
    *,
    horizon: int = 40,
    adv_floor: float = 5000.0,
    adv_window: int = 20,
    cost_one_way: float = 0.0023,
    min_names: int = 30,
    start_index: int = 130,
    fresh_days: int = 0,
    long_only: bool = False,
    borrow_cost_annual: float = 0.0,
    top_n: int = 0,
    age: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Rank-weighted dollar-neutral (or long-only excess) book, net of measured cost.

    Args:
        close: ``code × date`` price panel (numpy, abs'd upstream).
        trade_value: ``code × date`` trade-value panel (same units as ``adv_floor``).
        signal: ``code × date`` signal panel (YoY or a precomputed blend).
        dates: date labels aligned to the panels' columns (for the ``date`` column).
        age: ``code × date`` filing-age panel; only consulted when ``fresh_days>0``.
            ``None`` → all-NaN (no fresh gate applies).

    See :func:`kr_quant.strategies.pead.pead_backtest` for the parameter semantics.
    """
    C = close
    V = trade_value
    yoy = signal
    if age is None:
        age = np.full_like(yoy, np.nan)
    nD = len(dates)

    adv = np.full_like(C, np.nan)
    for j in range(adv_window, nD):
        adv[:, j] = np.nanmean(V[:, j - adv_window:j], axis=1)

    def fwd(t: int, h: int) -> np.ndarray:
        return C[:, t + h] / C[:, t] - 1.0 if t + h < nD else np.full(C.shape[0], np.nan)

    rows: list[dict] = []
    prev_w = np.zeros(C.shape[0])
    t = start_index
    while t < nD - horizon - 1:
        sig = yoy[:, t].copy()
        if fresh_days > 0:
            sig = np.where(age[:, t] <= fresh_days, sig, np.nan)
        ok = np.isfinite(sig)
        ret = fwd(t + 1, horizon)  # t+1 entry avoids same-close look-ahead
        ok &= np.isfinite(ret)
        if adv_floor > 0:
            ok &= adv[:, t] >= adv_floor
        if ok.sum() < min_names:
            t += horizon
            continue
        idx = np.where(ok)[0]
        pct = pd.Series(sig[ok]).rank(pct=True).to_numpy()
        w = np.zeros(C.shape[0])
        if long_only:
            # Long the top, fully invested; alpha is EXCESS over the eligible
            # universe (the implementable form when shorting is barred).
            if top_n > 0:
                # Concentrated equal-weight top-N: fewer, bigger asymmetric bets —
                # low win-rate, high payoff-ratio (fat right tail from earnings drift).
                sel = idx[np.argsort(-sig[ok])[:top_n]]
                w[sel] = 1.0 / len(sel)
            else:
                lw = np.clip(pct - 0.5, 0, None)  # rank tilt across the book
                w[idx] = lw / lw.sum() if lw.sum() > 0 else 0.0
            bench = float(np.nanmean(ret[idx]))
            gross = float(np.nansum(w * np.nan_to_num(ret))) - bench
            short_gross = 0.0
        else:
            w[idx] = (pct - 0.5) / np.abs(pct - 0.5).sum()  # dollar-neutral, gross=1
            gross = float(np.nansum(w * np.nan_to_num(ret)))
            short_gross = float(np.abs(w[w < 0]).sum())  # ~0.5 for a neutral book
        turnover = float(np.abs(w - prev_w).sum())
        borrow = borrow_cost_annual * (horizon / 252.0) * short_gross
        rows.append({"date": dates[t], "gross": gross, "turnover": turnover,
                     "net": gross - turnover * cost_one_way - borrow})
        prev_w = w
        t += horizon

    periods = pd.DataFrame(rows)
    return periods, summarize_periods(periods, horizon)


def staggered_tranche_backtest(
    close: np.ndarray,
    trade_value: np.ndarray,
    signal: np.ndarray,
    dates: list,
    *,
    horizon: int = 60,
    step: int = 20,
    top_n: int = 40,
    adv_floor: float = 20000.0,
    adv_window: int = 20,
    start_index: int = 130,
    min_names: int = 20,
    cap_array: np.ndarray | None = None,
    cap_rank: tuple[int, int] | None = None,
    delisting_exit: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Long-only excess with staggered entry — the recommended real-money form.

    Args:
        close, trade_value, signal: ``code × date`` numpy panels.
        dates: date labels aligned to the panels' columns.
        cap_array: ``code × date`` market-cap panel (numpy), or ``None``.
        cap_rank: ``(lo, hi)`` cap-rank tier to restrict to, paired with ``cap_array``.
        delisting_exit: 보유 기간에 상장폐지된 종목의 손실을 반영할지.

            기본 ``False`` 는 기존 동작이다: 수익률이
            ``C[:, t+step] / C[:, t] - 1`` 이므로 폐지 후 가격이 NaN 이 되고,
            ``np.nanmean`` 이 그 종목을 **조용히 빼버린다**. 즉 폐지 종목은 살아
            있는 동안 상승에는 기여하고 죽을 때는 비용 없이 사라진다 — 북과
            벤치마크 양쪽을 낙관 쪽으로 왜곡한다.

            ``True`` 면 마지막 관측 종가(정리매매 종료가)로 청산한 것으로 본다.
            한국 시장은 상장폐지 전 정리매매(보통 7거래일)가 있어 보유자가 실제로
            그 가격대에 빠져나올 수 있으므로, 임의의 상수(-30% 등)를 씌우는 것보다
            데이터에 충실하다. 다만 정리매매 자체를 못 판 경우는 반영되지 않으므로
            이 역시 낙관 쪽 하한이다.

            기본값을 바꾸지 않는 이유: 발표된 수치와 ``tests/test_parity_*.py`` 가
            기존 동작에 고정돼 있다. 조용한 드리프트를 만들지 않고, 켠 값과 끈 값을
            나란히 보고하는 게 이 레포의 규율이다(GUARDRAILS §8).

    See :func:`kr_quant.strategies.pead.staggered_backtest` for the semantics.
    """
    C = close
    V = trade_value
    sig_m = signal
    nD = len(dates)
    adv = np.full_like(C, np.nan)
    for j in range(adv_window, nD):
        adv[:, j] = np.nanmean(V[:, j - adv_window:j], axis=1)
    n_tranches = max(1, horizon // step)
    capm = cap_array

    # 폐지 청산가: 각 시점까지의 마지막 관측 종가(행 방향 forward-fill).
    # delisting_exit 가 꺼져 있으면 만들지 않는다(메모리·시간 낭비 방지).
    C_ff = pd.DataFrame(C).ffill(axis=1).to_numpy(float) if delisting_exit else None

    # eligible/book 는 t 의 순수함수라 메모이즈해도 숫자가 안 바뀐다. 스태거링이
    # book(t - k*step) 로 지난 리밸런스일을 다시 부르므로, 캐시가 없으면 같은 날의
    # 전유니버스 스캔(+시총 argsort)이 트랜치 수만큼 반복된다.
    _elig_cache: dict[int, np.ndarray] = {}
    _book_cache: dict[int, np.ndarray | None] = {}

    def eligible(t: int) -> np.ndarray:
        cached = _elig_cache.get(t)
        if cached is not None:
            return cached
        ok = np.isfinite(sig_m[:, t]) & (adv[:, t] >= adv_floor)
        if capm is not None and cap_rank is not None:
            liq = np.where(ok & np.isfinite(capm[:, t]))[0]
            order = liq[np.argsort(-capm[liq, t])]  # descending market cap
            tier = order[cap_rank[0]:cap_rank[1]]
            mask = np.zeros(C.shape[0], bool)
            mask[tier] = True
            ok = ok & mask
        _elig_cache[t] = ok
        return ok

    def book(t: int) -> np.ndarray | None:
        if t in _book_cache:
            return _book_cache[t]
        ok = eligible(t)
        if ok.sum() < min_names:
            _book_cache[t] = None
            return None
        idx = np.where(ok)[0]
        out = idx[np.argsort(-sig_m[idx, t])[:top_n]]
        _book_cache[t] = out
        return out

    rows: list[dict] = []
    for t in range(start_index, nD - step - 1, step):
        uni = np.where(eligible(t))[0]
        if uni.size < min_names:
            continue
        ret = C[:, t + step] / C[:, t] - 1.0
        if C_ff is not None:
            # 진입 시점엔 가격이 있었는데 청산 시점에 없는 종목 = 보유 중 상장폐지.
            # 마지막 관측가로 청산 처리해 손실을 실현시킨다(안 그러면 nanmean 이 뺀다).
            gone = np.isfinite(C[:, t]) & ~np.isfinite(C[:, t + step])
            ret[gone] = C_ff[gone, t + step] / C[gone, t] - 1.0
        bench = float(np.nanmean(ret[uni]))
        tranche_excess, tranche_book = [], []
        for k in range(n_tranches):
            b = book(t - k * step)
            if b is not None:
                book_ret = float(np.nanmean(ret[b]))
                tranche_book.append(book_ret)
                tranche_excess.append(book_ret - bench)
        if tranche_excess:
            # book/bench 를 함께 남긴다 — 초과수익만 보면 전략이 좋아진 건지 벤치마크가
            # 나빠진 건지 구분이 안 된다. 생존편향처럼 유니버스 구성이 바뀌는 변경에서는
            # 그 구분이 결론 자체를 뒤집는다(research/logs/survivorship_bias/VERDICT.md).
            # 여기서 내보내지 않으면 호출부가 이 루프를 베껴 재현하게 되고, 실제로
            # 그렇게 만든 사본이 스태거링과 delisting_exit 를 빠뜨렸다.
            rows.append({"date": dates[t], "gross": float(np.mean(tranche_excess)),
                         "turnover": 1.0 / n_tranches, "net": float(np.mean(tranche_excess)),
                         "book": float(np.mean(tranche_book)), "bench": bench,
                         "n_universe": int(uni.size)})
    periods = pd.DataFrame(rows)
    return periods, summarize_periods(periods, step)


def rank_ic(
    close: np.ndarray,
    trade_value: np.ndarray,
    signal: np.ndarray,
    dates: list,
    *,
    horizon: int = 40,
    adv_floor: float = 5000.0,
    adv_window: int = 20,
    start_index: int = 130,
    fresh_days: int = 0,
    n_regimes: int = 4,
    age: np.ndarray | None = None,
) -> dict:
    """Daily cross-sectional rank-IC of the signal vs forward return.

    Args:
        close, trade_value, signal: ``code × date`` numpy panels.
        dates: date labels aligned to the panels' columns (for regime start/end).
        age: ``code × date`` filing-age panel; only consulted when ``fresh_days>0``.

    See :func:`kr_quant.strategies.pead.pead_rank_ic` for the semantics.
    """
    C = close
    V = trade_value
    yoy = signal
    if age is None:
        age = np.full_like(yoy, np.nan)
    nD = len(dates)
    adv = np.full_like(C, np.nan)
    for j in range(adv_window, nD):
        adv[:, j] = np.nanmean(V[:, j - adv_window:j], axis=1)

    ics: list[float] = []
    ic_dates: list[str] = []
    for t in range(start_index, nD - horizon - 1):
        sig = yoy[:, t].copy()
        if fresh_days > 0:
            sig = np.where(age[:, t] <= fresh_days, sig, np.nan)
        ok = np.isfinite(sig)
        if adv_floor > 0:
            ok &= adv[:, t] >= adv_floor
        ret = C[:, t + 1 + horizon] / C[:, t + 1] - 1.0
        ok &= np.isfinite(ret)
        if ok.sum() < 20:
            continue
        a = pd.Series(sig[ok]).rank().to_numpy()
        b = pd.Series(ret[ok]).rank().to_numpy()
        if a.std() > 0 and b.std() > 0:
            ics.append(float(np.corrcoef(a, b)[0, 1]))
            ic_dates.append(dates[t])

    ic = np.array(ics)
    mean_ic, nw_t = newey_west_t(ic, horizon)
    regimes: list[dict] = []
    if len(ic) >= n_regimes:
        b = len(ic) // n_regimes
        for k in range(n_regimes):
            s0 = k * b
            s1 = (k + 1) * b if k < n_regimes - 1 else len(ic)
            m, tt = newey_west_t(ic[s0:s1], horizon)
            regimes.append({"start": ic_dates[s0][:7], "end": ic_dates[s1 - 1][:7],
                            "ic_mean": m, "nw_t": tt})
    return {
        "ic_mean": mean_ic, "ic_nw_t": nw_t, "n_days": len(ic),
        "frac_positive": float((ic > 0).mean()) if len(ic) else float("nan"),
        "regimes": regimes,
    }
