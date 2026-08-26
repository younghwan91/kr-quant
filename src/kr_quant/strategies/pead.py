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

import pandas as pd

from ..engine.metrics import newey_west_t, summarize_periods
from ..engine.panels import panel_pivot, price_arrays, resolve_signal
from ..engine.sim_crosssectional import rank_ic, rank_tilt_backtest, staggered_tranche_backtest

# Backward-compat re-exports (the metric/panel logic now lives in the engine).
# `research/experiments/pead_refinement.py` and the engine parity tests still import these
# private names; keep them as thin aliases so nothing re-derives the accounting.
_panel = panel_pivot
_resolve_signal = resolve_signal
_summarize = summarize_periods
_newey_west_t = newey_west_t


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
    pa = price_arrays(prices)
    C, V, codes, dates = pa.C, pa.V, pa.codes, pa.dates
    yoy, age = resolve_signal(earnings_panel, signal_panel, codes, dates)
    return rank_tilt_backtest(
        C, V, yoy, dates, horizon=horizon, adv_floor=adv_floor, adv_window=adv_window,
        cost_one_way=cost_one_way, min_names=min_names, start_index=start_index,
        fresh_days=fresh_days, long_only=long_only, borrow_cost_annual=borrow_cost_annual,
        top_n=top_n, age=age)


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
    cap_panel: pd.DataFrame | None = None,
    cap_rank: tuple[int, int] | None = None,
    delisting_exit: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Long-only excess with **staggered entry** — the recommended real-money form.

    Set ``cap_panel`` (long code/date/market_cap) + ``cap_rank=(20, 60)`` to trade
    only the market-cap tier where the PEAD alpha actually lives (large-but-not-mega,
    ~21st–60th by cap). PEAD is structurally absent in the top-20 mega-caps
    (efficient/over-covered), so this is the honest recommended universe and the
    correct benchmark is the same tier equal-weight (not cap-weighted KOSPI).

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
    pa = price_arrays(prices)
    C, V, codes, dates = pa.C, pa.V, pa.codes, pa.dates
    sig_m, _ = resolve_signal(earnings_panel, signal_panel, codes, dates)
    capm = (
        cap_panel.pivot_table(index="code", columns="date", values="market_cap", aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
        if cap_panel is not None else None
    )
    return staggered_tranche_backtest(
        C, V, sig_m, dates, horizon=horizon, step=step, top_n=top_n, adv_floor=adv_floor,
        adv_window=adv_window, start_index=start_index, min_names=min_names,
        cap_array=capm, cap_rank=cap_rank, delisting_exit=delisting_exit)


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
    pa = price_arrays(prices)
    C, V, codes, dates = pa.C, pa.V, pa.codes, pa.dates
    yoy, age = resolve_signal(earnings_panel, signal_panel, codes, dates)
    return rank_ic(
        C, V, yoy, dates, horizon=horizon, adv_floor=adv_floor, adv_window=adv_window,
        start_index=start_index, fresh_days=fresh_days, n_regimes=n_regimes, age=age)


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


def recommend_holdings(
    prices: pd.DataFrame,
    earnings_panel: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    top_n: int = 10,
    adv_floor: float = 20000.0,
    adv_window: int = 20,
    cap_rank: tuple[int, int] = (20, 100),
    asof: str | None = None,
) -> pd.DataFrame:
    """Current recommended book: top-``top_n`` mid-large names by latest YoY.

    Turns the backtest into an operable tool — computes, as of ``asof`` (default:
    the latest price date), the eligible mid-large tier (``cap_rank`` by market
    cap among names clearing ``adv_floor`` trailing ADV) and returns the ``top_n``
    with the highest lookahead-safe earnings-YoY signal.

    Args:
        prices: Long ``code``/``date``/``close``/``trade_value``.
        earnings_panel: Output of ``earnings_yoy_panel`` (``code``/``date``/``yoy``).
        shares: ``code``/``date``/shares-outstanding (for the cap tier).
        top_n, adv_floor, adv_window, cap_rank: As in the backtest.
        asof: Date (``YYYY-MM-DD``) to build the book for; default = latest.

    Returns:
        DataFrame (``code``, ``yoy``, ``cap_rank``, ``adv``) of the recommended
        equal-weight book, best signal first.
    """
    pa = price_arrays(prices)
    C, V, codes, dates = pa.C, pa.V, pa.codes, pa.dates
    t = dates.index(asof) if asof else len(dates) - 1
    adv = V[:, max(0, t - adv_window):t].mean(axis=1)
    price_t = C[:, t]
    sh = shares.rename(columns={shares.columns[-1]: "sh"}).sort_values("date").groupby("code")["sh"].last()
    cap = pd.Series(price_t, index=codes) * sh.reindex(codes)
    yoy = (
        earnings_panel.pivot_table(index="code", columns="date", values="yoy", aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)[:, t]
    )
    df = pd.DataFrame({"code": codes, "yoy": yoy, "cap": cap.to_numpy(), "adv": adv}).dropna()
    df = df[df["adv"] >= adv_floor]
    df["cap_rank"] = df["cap"].rank(ascending=False)
    df = df[(df["cap_rank"] > cap_rank[0]) & (df["cap_rank"] <= cap_rank[1])]
    df = df.sort_values("yoy", ascending=False).head(top_n)
    return df[["code", "yoy", "cap_rank", "adv"]].reset_index(drop=True)


def main() -> int:
    """CLI (``kq-pead``): run the validated tradeable PEAD⊕value backtest.

    Loads prices + shares from the DB and DART earnings from a CSV (columns:
    code, period, avail_date, netinc, prior, yoy), builds the combined signal and
    reports the concentrated long-only large-cap book — the asymmetric, monetizable
    form (low win-rate, high payoff-ratio).
    """
    import argparse
    import pandas as _pd
    from ..storage import connect, read_prices, db_default
    from ..features.fundamentals import combined_signal, earnings_yoy_panel

    ap = argparse.ArgumentParser(description="PEAD⊕가치 저회전 알파 백테스트 (실적 드리프트 + 가치)")
    ap.add_argument("--db", default=db_default())
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
    # 분할조정 필수: 미조정 daily_bars로는 분할이 가짜 −68% 손실로 잡혀 검증된 알파가
    # 재현되지 않는다(Sharpe 0.42). daily_bars_adjusted는 weekly_price_adjust DAG가
    # 재생성한다 — 비어 있으면 그 DAG가 아직 안 돈 것.
    # 정문(read_prices)으로 전체를 읽고 CSV 종목으로 좁힌다. 좁히는 걸 SQL 에서 하면
    # 로딩 시점 생존편향 검사가 통째로 우회된다 — 검사는 DB 가 폐지 종목을 갖고 있는지를
    # 보는 것이라 WHERE 뒤에서는 의미가 없다.
    # ⚠️ 최종 유니버스는 이 CSV 가 정한다. CSV 가 생존자만 담고 있으면 여전히 생존편향이
    # 들어간다 — 그건 이 함수가 막을 수 없고, CSV 를 만든 쪽의 책임이다.
    prices = read_prices(con, cols=("code", "date", "close", "trade_value"))
    prices = prices[prices["code"].isin(codes)]
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
