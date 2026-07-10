"""Post-earnings-announcement drift (PEAD) — the project's one validated alpha.

Rank stocks cross-sectionally by year-over-year net-income growth (lookahead-safe
via :func:`kr_quant.features.fundamentals.earnings_yoy_panel`) and hold a
rank-weighted, dollar-neutral long/short book, rebalanced every ``horizon`` days.

Validated behaviour (DART earnings, ~812 liquid KR names, 2018–2026): net Sharpe
~0.8–1.0 after realistic cost at 30–60-day horizons, full-sample t≈2.2, positive
in every regime bucket, robust across horizons. Two properties are essential and
baked in here:

1. **Point-in-time liquidity floor** (``adv_floor``): trade only names whose
   *trailing* average daily value clears a floor at the rebalance date. Without
   it the small-cap tail injects noise and the signal goes net-negative; with it
   it is strongly net-positive.
2. **Actual-turnover cost**: the earnings signal only changes ~quarterly, so the
   book barely turns over. Costs are charged on *measured* turnover
   (``sum|w_t - w_{t-1}|``), not a full round-trip every rebalance — charging the
   latter (a ~6× over-charge on a slow signal) is what made earlier tests look
   unprofitable.

Pure DataFrame in → DataFrame out (no DB), so the backtest is unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _panel(prices: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot a long price frame to a code × date panel (abs — close is signed)."""
    return prices.pivot_table(index="code", columns="date", values=value, aggfunc="first").abs()


def _yoy_panels(earnings_panel, codes, dates):
    """Pivot the long earnings panel to code×date ``yoy`` and ``age_days`` arrays."""
    yoy = (
        earnings_panel.pivot_table(index="code", columns="date", values="yoy", aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
    )
    if "age_days" in earnings_panel.columns:
        age = (
            earnings_panel.pivot_table(index="code", columns="date", values="age_days", aggfunc="first")
            .reindex(index=codes, columns=dates).to_numpy(float)
        )
    else:
        age = np.full_like(yoy, np.nan)
    return yoy, age


def _resolve_signal(earnings_panel, signal_panel, codes, dates):
    """Return ``(sig, age)`` code×date arrays from the YoY panel or a precomputed
    ``signal_panel`` (long ``code``/``date``/``signal`` — e.g. a PEAD+value blend
    from :func:`kr_quant.features.fundamentals.blend_rank`). Freshness (``age``)
    only applies to the raw YoY path; it is ``NaN`` for a precomputed signal.
    """
    if signal_panel is not None:
        sig = (
            signal_panel.pivot_table(index="code", columns="date", values="signal", aggfunc="first")
            .reindex(index=codes, columns=dates).to_numpy(float)
        )
        return sig, np.full_like(sig, np.nan)
    return _yoy_panels(earnings_panel, codes, dates)


def pead_backtest(
    prices: pd.DataFrame,
    earnings_panel: pd.DataFrame,
    *,
    signal_panel: pd.DataFrame | None = None,
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
) -> tuple[pd.DataFrame, dict]:
    """Backtest the rank-weighted dollar-neutral PEAD book, net of measured cost.

    Args:
        prices: Long rows with ``code``, ``date``, ``close``, ``trade_value``
            (``trade_value`` in the same units as ``adv_floor`` — 백만원 in this
            project). Close may be signed (Kiwoom); it is abs'd.
        earnings_panel: Output of
            :func:`kr_quant.features.fundamentals.earnings_yoy_panel` — long rows
            with ``code``, ``date``, ``yoy``.
        horizon: Rebalance/holding period in trading days (30–60 is the validated
            sweet spot; drift is a multi-week effect).
        adv_floor: Minimum trailing ``adv_window``-day average ``trade_value`` a
            name must clear at the rebalance date (point-in-time liquidity, no
            look-ahead). ``0`` disables the floor.
        adv_window: Trailing window (days) for the ADV liquidity measure.
        cost_one_way: One-way transaction cost as a fraction (0.0023 = 0.23%).
        min_names: Skip a rebalance with fewer than this many eligible names.
        start_index: First date index to trade from (warm-up for ADV/earnings).
        fresh_days: If > 0, only trade names whose latest filing is at most this
            many calendar days old at the rebalance date — isolates the true
            post-announcement drift window instead of the standing-value tilt.
            ``0`` uses the standing YoY all quarter.
        long_only: If True, hold a fully-invested long-only rank tilt and report
            ``gross`` as EXCESS over the eligible universe mean — the form that is
            implementable when shorting is barred (KR short-sale bans, hard-to-
            borrow names). Validation: long-only excess is weak (Sharpe ~0.2);
            the alpha needs the short leg, so a shortable universe is preferred.
        borrow_cost_annual: Annual stock-borrow cost charged on the short book
            (dollar-neutral has ~0.5 short gross) — set e.g. 0.02 for KR large
            caps to make the L/S net honest. Ignored when ``long_only``.
        top_n: If > 0 (with ``long_only``), hold only the ``top_n`` highest-signal
            names equal-weighted — a concentrated book of fewer, bigger bets. At
            the position level this is a low win-rate (~45%), high payoff-ratio
            (~1.5–1.8) convex strategy: most bets lose small, a few earnings-drift
            winners run far (fat right tail). ``0`` uses the full rank-tilt book.

    Returns:
        ``(periods, summary)``. ``periods`` has one row per rebalance with
        ``date``, ``gross``, ``turnover``, ``net``. ``summary`` holds ``n``,
        ``sharpe`` (annualized, net), ``t_stat`` (full-sample net), ``mean_net``,
        ``hit_rate``, ``cum_net`` and ``avg_turnover``.
    """
    close = _panel(prices, "close")
    tval = _panel(prices, "trade_value")
    dates = list(close.columns)
    codes = list(close.index)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)
    nD = len(dates)

    adv = np.full_like(C, np.nan)
    for j in range(adv_window, nD):
        adv[:, j] = np.nanmean(V[:, j - adv_window:j], axis=1)

    yoy, age = _resolve_signal(earnings_panel, signal_panel, codes, dates)

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
    return periods, _summarize(periods, horizon)


def staggered_backtest(
    prices: pd.DataFrame,
    earnings_panel: pd.DataFrame,
    *,
    signal_panel: pd.DataFrame | None = None,
    horizon: int = 60,
    step: int = 20,
    top_n: int = 40,
    adv_floor: float = 20000.0,
    adv_window: int = 20,
    start_index: int = 130,
    min_names: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Long-only excess with **staggered entry** — the recommended real-money form.

    Instead of putting all capital in on one rebalance date (timing luck), enter
    ``horizon // step`` equal tranches offset by ``step`` days, each an equal-weight
    top-``top_n`` book held ``horizon`` days. Each ``step``-day period the portfolio
    return is the average of the active tranches' excess-over-universe returns.
    Empirically this roughly doubles the IR (~0.5 → ~1.0) versus a single
    non-overlapping schedule while preserving the asymmetric payoff (>1.25).

    Returns ``(periods, summary)`` with the same schema as :func:`pead_backtest`
    (``summary`` includes ``payoff_ratio``); ``turnover``/``best``/``worst`` are on
    the ``step``-period excess series.
    """
    close = _panel(prices, "close")
    tval = _panel(prices, "trade_value")
    dates = list(close.columns)
    codes = list(close.index)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)
    nD = len(dates)
    adv = np.full_like(C, np.nan)
    for j in range(adv_window, nD):
        adv[:, j] = np.nanmean(V[:, j - adv_window:j], axis=1)
    sig_m, _ = _resolve_signal(earnings_panel, signal_panel, codes, dates)
    n_tranches = max(1, horizon // step)

    def book(t: int) -> np.ndarray | None:
        s = sig_m[:, t]
        ok = np.isfinite(s) & (adv[:, t] >= adv_floor)
        if ok.sum() < min_names:
            return None
        idx = np.where(ok)[0]
        return idx[np.argsort(-s[ok])[:top_n]]

    rows: list[dict] = []
    for t in range(start_index, nD - step - 1):
        if (t - start_index) % step != 0:
            continue
        uni = np.where(np.isfinite(sig_m[:, t]) & (adv[:, t] >= adv_floor))[0]
        if uni.size < min_names:
            continue
        ret = C[:, t + step] / C[:, t] - 1.0
        bench = float(np.nanmean(ret[uni]))
        tranche_excess = []
        for k in range(n_tranches):
            b = book(t - k * step)
            if b is not None:
                tranche_excess.append(float(np.nanmean(ret[b])) - bench)
        if tranche_excess:
            rows.append({"date": dates[t], "gross": float(np.mean(tranche_excess)),
                         "turnover": 1.0 / n_tranches, "net": float(np.mean(tranche_excess))})
    periods = pd.DataFrame(rows)
    return periods, _summarize(periods, step)


def _newey_west_t(x: np.ndarray, lag: int) -> tuple[float, float]:
    """Mean and Newey-West (HAC) t-stat of ``x``, robust to serial correlation.

    Overlapping horizon returns are autocorrelated; a plain t overstates
    significance. The Bartlett-kernel HAC variance with ``lag`` corrects it.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lag + 2:
        return float("nan"), float("nan")
    mu = x.mean()
    d = x - mu
    var = (d @ d) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * ((d[k:] @ d[:-k]) / n)
    se = np.sqrt(var / n)
    return float(mu), float(mu / se) if se > 0 else float("nan")


def pead_rank_ic(
    prices: pd.DataFrame,
    earnings_panel: pd.DataFrame,
    *,
    signal_panel: pd.DataFrame | None = None,
    horizon: int = 40,
    adv_floor: float = 5000.0,
    adv_window: int = 20,
    start_index: int = 130,
    fresh_days: int = 0,
    n_regimes: int = 4,
) -> dict:
    """High-power confirmation: daily cross-sectional rank-IC of YoY vs forward return.

    Complements :func:`pead_backtest` (the tradeable, lower-power non-overlapping
    Sharpe/t) with the classic factor test: every trading day, Spearman-correlate
    the signal against the ``horizon``-day forward return across the eligible
    universe, then take the mean IC with a Newey-West t (lag = ``horizon``) that
    corrects for the overlap. Also splits the IC series into ``n_regimes`` equal
    time buckets to check the sign is persistent, not a single-regime artifact.

    Args:
        prices, earnings_panel, horizon, adv_floor, adv_window, start_index,
        fresh_days: As in :func:`pead_backtest`.
        n_regimes: Number of equal chronological buckets for the persistence check.

    Returns:
        ``{"ic_mean", "ic_nw_t", "n_days", "frac_positive", "regimes"}`` where
        ``regimes`` is a list of ``{"start", "end", "ic_mean", "nw_t"}`` dicts.
    """
    close = _panel(prices, "close")
    tval = _panel(prices, "trade_value")
    dates = list(close.columns)
    codes = list(close.index)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)
    nD = len(dates)
    adv = np.full_like(C, np.nan)
    for j in range(adv_window, nD):
        adv[:, j] = np.nanmean(V[:, j - adv_window:j], axis=1)
    yoy, age = _resolve_signal(earnings_panel, signal_panel, codes, dates)

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
    mean_ic, nw_t = _newey_west_t(ic, horizon)
    regimes: list[dict] = []
    if len(ic) >= n_regimes:
        b = len(ic) // n_regimes
        for k in range(n_regimes):
            s0 = k * b
            s1 = (k + 1) * b if k < n_regimes - 1 else len(ic)
            m, tt = _newey_west_t(ic[s0:s1], horizon)
            regimes.append({"start": ic_dates[s0][:7], "end": ic_dates[s1 - 1][:7],
                            "ic_mean": m, "nw_t": tt})
    return {
        "ic_mean": mean_ic, "ic_nw_t": nw_t, "n_days": len(ic),
        "frac_positive": float((ic > 0).mean()) if len(ic) else float("nan"),
        "regimes": regimes,
    }


def _summarize(periods: pd.DataFrame, horizon: int) -> dict:
    """Annualized net Sharpe, full-sample t-stat and cumulative return."""
    if periods.empty:
        return {"n": 0, "sharpe": float("nan"), "t_stat": float("nan"),
                "mean_net": float("nan"), "hit_rate": float("nan"),
                "cum_net": float("nan"), "avg_turnover": float("nan")}
    net = periods["net"].to_numpy()
    per_year = 252 / horizon
    std = net.std()
    ann = (1 + net.mean()) ** per_year - 1
    wins = net[net > 0]
    losses = net[net < 0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(-losses.mean()) if losses.size else 0.0
    return {
        "n": len(net),
        "sharpe": float(ann / (std * np.sqrt(per_year))) if std > 0 else float("nan"),
        "t_stat": float(net.mean() / (std / np.sqrt(len(net)))) if std > 0 else float("nan"),
        "mean_net": float(net.mean()),
        "hit_rate": float((net > 0).mean()),
        "cum_net": float((1 + net).prod() - 1),
        "avg_turnover": float(periods["turnover"].mean()),
        # payoff profile (a quant cares about this more than win rate): a low
        # hit_rate with payoff_ratio > 1 is an asymmetric, convex strategy.
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": float(avg_win / avg_loss) if avg_loss > 0 else float("nan"),
        "best": float(net.max()),
        "worst": float(net.min()),
    }


def market_cap_panel(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Long ``code``/``date``/``market_cap`` = |close| × shares-outstanding (as-of).

    Shares are sparse snapshots; a backward as-of join carries the latest known
    count forward (shares change slowly, so this is a fair market-cap proxy).
    """
    p = prices[["code", "date", "close"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    p["close"] = p["close"].abs()
    s = shares.rename(columns={shares.columns[-1]: "shares"})[["code", "date", "shares"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    p = p.sort_values("date")
    s = s.sort_values("date")
    mc = pd.merge_asof(p, s, on="date", by="code", direction="backward")
    mc["market_cap"] = mc["close"] * mc["shares"]
    mc["date"] = mc["date"].dt.strftime("%Y-%m-%d")
    return mc[["code", "date", "market_cap"]]


def main() -> int:
    """CLI (``kq-pead``): run the validated tradeable PEAD⊕value backtest.

    Loads prices + shares from the DB and DART earnings from a CSV (columns:
    code, period, avail_date, netinc, prior, yoy), builds the combined signal and
    reports the concentrated long-only large-cap book — the asymmetric, monetizable
    form (low win-rate, high payoff-ratio).
    """
    import argparse
    import pandas as _pd
    from ..storage import connect, default_db_path
    from ..features.fundamentals import combined_signal, earnings_yoy_panel

    ap = argparse.ArgumentParser(description="PEAD⊕가치 저회전 알파 백테스트 (실적 드리프트 + 가치)")
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--earnings-csv", required=True, help="DART 실적 CSV: code,period,avail_date,netinc,prior,yoy")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--adv-floor", type=float, default=20000, help="유동성 하한(백만원, 200억=공매도가능 대형주)")
    ap.add_argument("--value-weight", type=float, default=0.25)
    ap.add_argument("--top-n", type=int, default=5, help="집중 보유 종목수 (0=분산 랭크틸트)")
    ap.add_argument("--market-neutral", action="store_true", help="롱숏(기본은 롱온리)")
    ap.add_argument("--staggered", action="store_true",
                    help="스태거드 진입(권장 실전형태): 3트랜치 엇갈림으로 IR 개선")
    ap.add_argument("--step", type=int, default=20, help="스태거드 트랜치 간격(일)")
    ap.add_argument("--no-value", action="store_true",
                    help="가치(E/P) 결합 없이 순수 PEAD만 — 주식수/시총 데이터 불요, 롱온리 실전 권장")
    args = ap.parse_args()

    ea = _pd.read_csv(args.earnings_csv, dtype={"code": str, "avail_date": str, "period": str})
    con = connect(args.db)
    codes = sorted(ea["code"].unique())
    prices = _pd.read_sql_query(
        "SELECT code,date,close,trade_value FROM daily_bars WHERE code = ANY(%(c)s)",
        con, params={"c": codes})
    shares = None if args.no_value else _pd.read_sql_query(
        "SELECT code,date,shares_outstanding FROM shares_outstanding_history WHERE code = ANY(%(c)s)",
        con, params={"c": codes})
    con.close()
    prices["date"] = prices["date"].astype(str)
    dates = sorted(prices["date"].unique())
    yoy = earnings_yoy_panel(ea.dropna(subset=["yoy"]), dates)
    # Pure PEAD (--no-value) needs only earnings + prices; the value blend adds a
    # market-cap (shares) dependency for marginal benefit on the long-only form.
    sig = None if args.no_value else combined_signal(
        ea, market_cap_panel(prices, shares), dates, value_weight=args.value_weight)
    if args.staggered and not args.market_neutral:
        _, s = staggered_backtest(
            prices, yoy, signal_panel=sig, horizon=args.horizon, step=args.step,
            top_n=args.top_n or 40, adv_floor=args.adv_floor)
        form = f"스태거드 롱온리 top{args.top_n or 40} (step={args.step})"
    else:
        _, s = pead_backtest(
            prices, yoy, signal_panel=sig, horizon=args.horizon, adv_floor=args.adv_floor,
            cost_one_way=0.0018, long_only=not args.market_neutral, top_n=args.top_n,
            borrow_cost_annual=0.02 if args.market_neutral else 0.0)
        form = "롱숏(중립)" if args.market_neutral else (f"롱온리 집중 top{args.top_n}" if args.top_n else "롱온리 분산")
    strat = "순수 PEAD" if args.no_value else "PEAD⊕가치"
    print(f"{strat} | {form} | H={args.horizon} | 유동성>={args.adv_floor:.0f}백만 | 리밸런스 {s['n']}회")
    print(f"  누적수익 {s['cum_net']*100:+.0f}%  연Sharpe/IR {s['sharpe']:+.2f}  t {s['t_stat']:+.2f}")
    print(f"  기대값/기간 {s['mean_net']*100:+.2f}%  승률 {s['hit_rate']*100:.0f}%  "
          f"손익비 {s['payoff_ratio']:.2f} (평균이익 {s['avg_win']*100:+.1f}% / 평균손실 {s['avg_loss']*100:+.1f}%)")
    print(f"  최고기간 {s['best']*100:+.0f}%  최악기간 {s['worst']*100:+.0f}%  회전율 {s['avg_turnover']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
