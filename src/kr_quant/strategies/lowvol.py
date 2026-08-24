"""Low-volatility anomaly — the second cross-sectional factor (ported candidate).

Rank stocks by *trailing* return volatility and hold a rank-weighted, dollar-
neutral book long the calmest names (low realized vol) and short the most
volatile "lottery" names, rebalanced monthly. The signal is
:func:`kr_quant.features.volatility.lowvol_signal_panel` (``-vol``); the
accounting is the shared engine's rank-tilt simulator — the same one PEAD uses —
so nothing here re-derives entry/turnover/borrow logic.

Provenance: ported from scalp-it (``scalp_it.lowvol`` + factor-batch note #31).
In scalp-it's own frame the low-vol L/S cleared the pre-registered bar — 50억
universe net **+2.34%/month (t=3.66)**, 500억 net **+3.07% (t=2.77)**, driven by
long-calm / short-lottery, and it is strongly *inverse* to PEAD there (that
premise is re-examined in the combo work, :mod:`kr_quant.strategies.combo`).

⚠️ **Status.** This is a *ported candidate*, not yet re-adjudicated under this
repo's full pre-registration battery (``research/experiments/*_gate.py`` +
``prop_gate``). The project's standing claim — "the one alpha that cleared the
gate is PEAD" — is unchanged until low-vol is run through that battery here. What
this module gives you is the reusable, unit-tested backtest so that adjudication
can happen on kr-quant data. See ``docs/lowvol-strategy.md``.

Pure DataFrame in -> DataFrame out (no DB), so the backtest is unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.panels import lookup_panel, panel_pivot
from ..engine.sim_crosssectional import rank_ic, rank_tilt_backtest
from ..features.volatility import VOL_WINDOW, realized_vol_panel

# Deployment spec (scalp-it note #31 adoption) — the low-vol book's frozen rules.
DEPLOY_ADV_FLOOR = 5000.0   # trailing 20d avg trade value ≥ 5000 백만원 (50억)
HOLD_DAYS = 21              # monthly rebalance
N_DECILES = 10
MIN_NAMES = 30             # skip a rebalance below this many eligible names
COST_ONE_WAY = 0.0034      # 34bp one-way (68bp round-trip, note #31)
BORROW_ANNUAL = 0.03       # short-leg stock-borrow, 3%/yr


def _eligible_meta_exclusions(meta: pd.DataFrame) -> set[str]:
    """SPAC / REIT codes to drop (note #31 universe rule)."""
    if meta is None or meta.empty or "name" not in meta.columns:
        return set()
    mask = meta["name"].astype(str).str.contains("스팩|기업인수목적|리츠", na=False)
    return set(meta.loc[mask, "code"].astype(str))


def lowvol_backtest(
    prices: pd.DataFrame,
    *,
    vol_window: int = VOL_WINDOW,
    horizon: int = HOLD_DAYS,
    adv_floor: float = DEPLOY_ADV_FLOOR,
    adv_window: int = 20,
    cost_one_way: float = COST_ONE_WAY,
    min_names: int = MIN_NAMES,
    start_index: int = VOL_WINDOW + 25,
    long_only: bool = False,
    borrow_cost_annual: float = BORROW_ANNUAL,
    top_n: int = 0,
    signal_panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Backtest the rank-weighted low-vol book, net of measured cost.

    Delegates to :func:`kr_quant.engine.sim_crosssectional.rank_tilt_backtest`
    (entry at ``t+1``, ADV floor, cost on *measured* turnover, short-leg borrow)
    with the low-vol signal (``-vol``). Long-high-signal = long-low-vol; short-
    low-signal = short-high-vol, matching scalp-it's Q10-long / Q1-short spec in
    the continuous rank-tilt form.

    Args:
        prices: Long ``code``/``date``/``close``/``trade_value`` (백만원 units for
            ``trade_value``, matching ``adv_floor``).
        vol_window: Trailing window for the realized-vol signal (default 60d).
        horizon: Rebalance/holding period in trading days (21 = monthly, the spec).
        adv_floor: Point-in-time trailing-ADV liquidity floor (백만원). ``0`` off.
        long_only: Hold a long-only low-vol tilt and report ``gross`` as EXCESS
            over the eligible universe mean — the market-hedged form implementable
            without individual shorts (see :mod:`kr_quant.strategies.hedge`).
        borrow_cost_annual: Annual borrow on the short book (ignored if long_only).
        top_n: With ``long_only``, hold only the ``top_n`` calmest names EW.
        signal_panel: Optional precomputed long ``code``/``date``/``signal`` panel
            to use instead of recomputing ``-vol`` (e.g. a blended signal).

    Returns:
        ``(periods, summary)`` — same schema as
        :func:`kr_quant.strategies.pead.pead_backtest`.
    """
    close = panel_pivot(prices, "close")
    tval = panel_pivot(prices, "trade_value")
    dates = list(close.columns)
    codes = list(close.index)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)
    if signal_panel is None:
        vp = realized_vol_panel(prices, window=vol_window)
        vp["signal"] = -vp["vol"]
        sig = lookup_panel(vp[["code", "date", "signal"]], "signal", codes, dates)
    else:
        sig = lookup_panel(signal_panel, "signal", codes, dates)
    return rank_tilt_backtest(
        C, V, sig, dates, horizon=horizon, adv_floor=adv_floor, adv_window=adv_window,
        cost_one_way=cost_one_way, min_names=min_names, start_index=start_index,
        long_only=long_only, borrow_cost_annual=borrow_cost_annual, top_n=top_n)


def lowvol_rank_ic(
    prices: pd.DataFrame,
    *,
    vol_window: int = VOL_WINDOW,
    horizon: int = HOLD_DAYS,
    adv_floor: float = DEPLOY_ADV_FLOOR,
    adv_window: int = 20,
    start_index: int = VOL_WINDOW + 25,
    n_regimes: int = 4,
) -> dict:
    """High-power confirmation: daily cross-sectional rank-IC of ``-vol`` vs
    forward return, with a Newey-West t and a regime-persistence split.

    Mirrors :func:`kr_quant.strategies.pead.pead_rank_ic`; see it for semantics.
    """
    close = panel_pivot(prices, "close")
    tval = panel_pivot(prices, "trade_value")
    dates = list(close.columns)
    codes = list(close.index)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)
    vp = realized_vol_panel(prices, window=vol_window)
    vp["signal"] = -vp["vol"]
    sig = lookup_panel(vp[["code", "date", "signal"]], "signal", codes, dates)
    return rank_ic(
        C, V, sig, dates, horizon=horizon, adv_floor=adv_floor, adv_window=adv_window,
        start_index=start_index, n_regimes=n_regimes)


def select_lowvol_portfolio(
    prices: pd.DataFrame,
    meta: pd.DataFrame | None = None,
    *,
    asof: str | None = None,
    vol_window: int = VOL_WINDOW,
    adv_floor: float = DEPLOY_ADV_FLOOR,
    adv_window: int = 20,
    n_deciles: int = N_DECILES,
    min_names: int = N_DECILES,
) -> pd.DataFrame:
    """Current deployable book: decile Q10 (long, low-vol) / Q1 (short, high-vol).

    Turns the factor into an operable tool (mirroring
    :func:`kr_quant.strategies.pead.recommend_holdings`) — as of ``asof`` (default:
    latest date), among names clearing the ADV floor, ranks the cross-section by
    ``-vol`` into ``n_deciles`` deciles and returns the extreme-decile book,
    equal-weight within each leg. Ported from scalp-it ``select_lowvol_portfolio``.

    Args:
        prices: Long ``code``/``date``/``close``/``trade_value``.
        meta: Optional ``code``/``name``/``market``/``sector`` for display and the
            SPAC/REIT exclusion. ``None`` disables both.
        asof: Rebalance date (``YYYY-MM-DD``); default = latest price date.
        adv_floor: Trailing-ADV liquidity floor (백만원).
        n_deciles: Number of cross-sectional deciles (extreme two are the book).
        min_names: Minimum eligible names to form a book (else empty frame).

    Returns:
        DataFrame (``side``, ``code``, ``name``, ``market``, ``vol_ann``,
        ``close``, ``weight``) — ``side`` in {"long", "short"}, ``weight`` equal
        within each leg. Empty if the universe is too thin at ``asof``.
    """
    close = panel_pivot(prices, "close")
    tval = panel_pivot(prices, "trade_value")
    dates = list(close.columns)
    if not dates:
        return _empty_book()
    t = dates.index(asof) if asof else len(dates) - 1
    codes = list(close.index)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)

    lo = max(0, t - vol_window)
    rets = C[:, lo + 1:t + 1] / C[:, lo:t] - 1.0
    vol = np.nanstd(rets, axis=1, ddof=1) if rets.shape[1] > 1 else np.full(len(codes), np.nan)
    adv = np.nanmean(V[:, max(0, t - adv_window):t], axis=1)
    excl = _eligible_meta_exclusions(meta)

    rows = []
    for i, code in enumerate(codes):
        if code in excl:
            continue
        if not (np.isfinite(vol[i]) and vol[i] > 0 and np.isfinite(adv[i]) and adv[i] >= adv_floor):
            continue
        if not np.isfinite(C[i, t]):
            continue
        rows.append({"code": code, "vol": float(vol[i]), "close": float(abs(C[i, t]))})
    if len(rows) < max(min_names, n_deciles):
        return _empty_book()

    cur = pd.DataFrame(rows)
    r = (-cur["vol"]).rank(method="first")  # signal = -vol; top decile = low-vol
    cur["decile"] = pd.qcut(r, n_deciles, labels=range(1, n_deciles + 1)).astype(int)
    cur["vol_ann"] = cur["vol"] * np.sqrt(252)

    name_map, mkt_map = {}, {}
    if meta is not None and not meta.empty:
        name_map = dict(zip(meta["code"].astype(str), meta["name"]))
        if "market" in meta.columns:
            mkt_map = dict(zip(meta["code"].astype(str), meta["market"]))

    out = []
    legs = (("long", cur[cur["decile"] == n_deciles].sort_values("vol")),
            ("short", cur[cur["decile"] == 1].sort_values("vol", ascending=False)))
    for side, leg in legs:
        n = len(leg)
        weight = 1.0 / n if n else 0.0
        for x in leg.itertuples():
            out.append({
                "side": side, "code": x.code, "name": name_map.get(x.code, ""),
                "market": mkt_map.get(x.code, ""), "vol_ann": round(x.vol_ann, 4),
                "close": round(x.close, 1), "weight": round(weight, 4),
            })
    return pd.DataFrame(out)


def _empty_book() -> pd.DataFrame:
    return pd.DataFrame(columns=["side", "code", "name", "market", "vol_ann", "close", "weight"])
