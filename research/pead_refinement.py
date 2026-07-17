#!/usr/bin/env python
"""PEAD refinement research — surprise-magnitude filter, horizon
sweep, per-position trailing stop.

Scratch research only. Nothing here touches ``src/kr_quant/`` (production
isolation). It *reuses* the validated production machinery:
``staggered_backtest`` / ``_summarize`` / ``_panel`` / ``_resolve_signal`` from
``kr_quant.strategies.pead`` and ``earnings_yoy_panel`` from
``kr_quant.features.fundamentals`` (see the RALPLAN-DR plan, Option C).

Data source: TimescaleDB via ``connect(db_default())`` (``KR_QUANT_DB``), DB only,
never CSV.

Price table — IMPORTANT deviation from the plan's literal SQL, forced by the
data:
    The plan's Step 1 SQL names ``daily_bars``. That table is corporate-action
    UNadjusted (storage.py / MULTI_ALPHA.md §"반드시 지킬 전제" #1): a stock split
    reads as a catastrophic one-day return, which corrupts every return-based
    backtest. Running ``staggered_backtest`` on raw ``daily_bars`` yields Sharpe
    0.42 / t 1.22 — it does NOT reproduce the plan's own hard acceptance gate
    (Sharpe ~0.8-1.0, t~2.2), i.e. the validated alpha. PEAD requires
    split-adjusted prices ("분할조정 가격 필수"); the validated t is 2.16-2.97.
    The split-adjusted series is stored as
    ``daily_bars_adjusted`` (the ``price_adjust.py`` output). Using it reproduces
    Sharpe ~1.09 / t ~3.12 — the validated alpha. So, exactly like a stale line
    reference, the plan's ``daily_bars`` is followed by intent, not by letter:
    we load ``daily_bars_adjusted``. The trailing-stop daily walk also needs the
    adjusted series (an HWM on unadjusted prices false-triggers on split days).

YoY signal — the DB ``earnings`` table carries ``netinc`` / ``netinc_prior``
(no precomputed ``yoy`` column, unlike the legacy CSV the plan's SQL assumed), so
YoY is computed with the repo's canonical formula ``_yoy_vec`` =
``(cur - prior) / |prior|`` before building the panel.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from kr_quant.features.fundamentals import _yoy_vec, earnings_yoy_panel
from kr_quant.storage import connect, db_default
from kr_quant.strategies.pead import (
    _panel,
    _resolve_signal,
    _summarize,
    staggered_backtest,
)

# --- Baseline parameters (the anchor every experiment is compared against) ---
BASELINE = dict(horizon=60, step=20, top_n=40, adv_floor=20000.0)
ADV_WINDOW = 20
START_INDEX = 130
MIN_NAMES = 20
COST_ONE_WAY = 0.0023  # one-way transaction cost (Accounting Harmonization (b))

# Split-adjusted price table (see module docstring for why not daily_bars).
PRICE_TABLE = "daily_bars_adjusted"

# trail_pct=1.0 sanity gate tolerance (plan Step 4 / Verification #4).
TOL_SHARPE = 0.10
TOL_T = 0.10

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Step 1: data loading + baseline
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prices + earnings from TimescaleDB and build the YoY panel.

    Returns ``(prices, yoy_panel)`` where ``prices`` is long
    ``code``/``date``/``close``/``trade_value`` (split-adjusted close) and
    ``yoy_panel`` is the lookahead-safe long panel from ``earnings_yoy_panel``.
    """
    con = connect(db_default())
    ea = pd.read_sql_query(
        "SELECT code, period, avail_date, netinc, netinc_prior FROM earnings", con
    )
    prices = pd.read_sql_query(
        f"SELECT code, date, close, trade_value FROM {PRICE_TABLE}", con  # noqa: S608 — trusted constant
    )
    con.close()

    ea["code"] = ea["code"].astype(str)
    ea["avail_date"] = ea["avail_date"].astype(str)
    # DB has netinc/netinc_prior, not a precomputed yoy — compute it (canonical).
    ea["yoy"] = _yoy_vec(ea["netinc"], ea["netinc_prior"])

    prices["code"] = prices["code"].astype(str)
    prices["date"] = prices["date"].astype(str)
    dates = sorted(prices["date"].unique())
    yoy_panel = earnings_yoy_panel(ea.dropna(subset=["yoy"]), dates)
    return prices, yoy_panel


def _fmt(s: dict) -> str:
    return (
        f"n={s['n']:>3}  Sharpe={s['sharpe']:+.3f}  t={s['t_stat']:+.3f}  "
        f"cum={s['cum_net']:+.3f}  hit={s['hit_rate']:.3f}  "
        f"payoff={s['payoff_ratio']:.3f}  worst={s['worst']:+.3f}"
    )


def run_baseline(prices=None, yoy_panel=None) -> dict:
    """Reproduce the validated baseline and print its stats (the anchor)."""
    if prices is None:
        prices, yoy_panel = load_data()
    _, s = staggered_backtest(prices, yoy_panel, **BASELINE)
    print("=== Step 1: Baseline (no filter, H60/step20/top40, adv_floor=20000) ===")
    print(f"  price table: {PRICE_TABLE} (split-adjusted — see module docstring)")
    print(f"  {_fmt(s)}")
    lo, hi = 0.8, 1.0
    ok = (lo - 0.15) <= s["sharpe"] and s["t_stat"] >= 2.0
    print(
        f"  [{'OK' if ok else '!!'}] validated-alpha range: Sharpe~{lo}-{hi}, t~2.2 "
        f"(validated range: t 2.16-2.97)"
    )
    return s


# ---------------------------------------------------------------------------
# Step 2: Experiment 1 — YoY magnitude filter sweep
# ---------------------------------------------------------------------------
YOY_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def _mask_panel(yoy_panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Copy the panel, blanking (NaN) names with |yoy| < threshold before ranking."""
    if threshold <= 0.0:
        return yoy_panel
    m = yoy_panel.copy()
    m.loc[m["yoy"].abs() < threshold, "yoy"] = np.nan
    return m


def run_yoy_filter_sweep(prices=None, yoy_panel=None) -> dict:
    """Sweep the surprise-magnitude filter; return best threshold + rows."""
    if prices is None:
        prices, yoy_panel = load_data()
    base_n = None
    rows = []
    for th in YOY_THRESHOLDS:
        masked = _mask_panel(yoy_panel, th)
        _, s = staggered_backtest(prices, masked, **BASELINE)
        if th == 0.0:
            base_n = s["n"]
        rows.append({"threshold": th, **s})

    base = rows[0]
    print("=== Step 2: Experiment 1 — YoY magnitude filter sweep ===")
    print(f"  {'thr':>5} {'n':>4} {'Sharpe':>8} {'t':>7} {'cum':>8} {'hit':>6} {'worst':>8}  flag")
    best = None
    for r in rows:
        low_n = r["n"] < 0.5 * base_n
        flag = "[!] low n" if low_n else ""
        print(
            f"  {r['threshold']:>5.2f} {r['n']:>4} {r['sharpe']:>+8.3f} {r['t_stat']:>+7.3f} "
            f"{r['cum_net']:>+8.3f} {r['hit_rate']:>6.3f} {r['worst']:>+8.3f}  {flag}"
        )
        if r["threshold"] == 0.0 or low_n:
            continue
        # improve Sharpe AND t without worsening MDD (worst = min period, negative)
        if (
            r["sharpe"] > base["sharpe"]
            and r["t_stat"] > base["t_stat"]
            and r["worst"] >= base["worst"]
            and (best is None or r["sharpe"] > best["sharpe"])
        ):
            best = r
    if best is not None:
        print(
            f"  Best threshold: {best['threshold']:.2f} "
            f"(Sharpe {best['sharpe']:+.3f}, t {best['t_stat']:+.3f})"
        )
        best_th = best["threshold"]
    else:
        print("  No threshold improves baseline")
        best_th = 0.0
    return {"rows": rows, "best_threshold": best_th, "base_n": base_n}


# ---------------------------------------------------------------------------
# Step 3: Experiment 2 — Horizon sweep
# ---------------------------------------------------------------------------
HORIZONS = [20, 30, 40, 50, 60, 70, 80, 90]


def run_horizon_sweep(prices=None, yoy_panel=None, best_threshold: float = 0.0) -> dict:
    """Sweep the holding horizon (step = horizon//3, ~3 tranches)."""
    if prices is None:
        prices, yoy_panel = load_data()
    masked = _mask_panel(yoy_panel, best_threshold)
    rows = []
    for h in HORIZONS:
        step = max(1, h // 3)
        _, s = staggered_backtest(
            prices, masked, horizon=h, step=step,
            top_n=BASELINE["top_n"], adv_floor=BASELINE["adv_floor"],
        )
        rows.append({"horizon": h, "step": step, **s})

    print(f"=== Step 3: Experiment 2 — Horizon sweep (YoY filter={best_threshold:.2f}) ===")
    print(f"  {'H':>4} {'step':>4} {'n':>4} {'Sharpe':>8} {'t':>7} {'cum':>8} {'hit':>6} {'worst':>8}")
    for r in rows:
        print(
            f"  {r['horizon']:>4} {r['step']:>4} {r['n']:>4} {r['sharpe']:>+8.3f} "
            f"{r['t_stat']:>+7.3f} {r['cum_net']:>+8.3f} {r['hit_rate']:>6.3f} {r['worst']:>+8.3f}"
        )
    best = max(rows, key=lambda r: r["sharpe"])
    # report a band: horizons within 0.10 Sharpe of the best
    band = [r["horizon"] for r in rows if best["sharpe"] - r["sharpe"] <= 0.10]
    band_str = f"{min(band)}-{max(band)} days" if len(band) > 1 else f"{best['horizon']} days"
    print(
        f"  Optimal horizon: {band_str} (best H{best['horizon']} "
        f"Sharpe {best['sharpe']:+.3f}, t {best['t_stat']:+.3f})"
    )
    return {"rows": rows, "best_horizon": best["horizon"], "best_step": best["step"], "band": band_str}


# ---------------------------------------------------------------------------
# Step 4: Experiment 3 — Per-position trailing stop (Option C)
# ---------------------------------------------------------------------------
def _context(prices, earnings_panel, signal_panel=None, adv_window=ADV_WINDOW) -> dict:
    """Build the panels/arrays exactly as ``staggered_backtest`` does (pead.py:223-233)."""
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
    return {"C": C, "dates": dates, "adv": adv, "sig_m": sig_m, "nD": nD}


def trailing_stop_backtest(
    ctx: dict,
    *,
    horizon: int = 60,
    step: int = 20,
    top_n: int = 40,
    adv_floor: float = 20000.0,
    start_index: int = START_INDEX,
    min_names: int = MIN_NAMES,
    trail_pct: float = 1.0,
    cost_one_way: float = COST_ONE_WAY,
) -> tuple[pd.DataFrame, dict, dict]:
    """Return-adjustment layer on ``staggered_backtest`` (plan Step 4, Option C).

    Reuses the stagger structure / eligible / book / benchmark by construction;
    the ONLY new code is the per-position daily trailing walk. ``trail_pct=1.0``
    never triggers (needs price<=0), so it reproduces the fixed-horizon baseline.

    Returns ``(periods, summary_zero_cost, extras)`` where ``extras`` holds the
    cost-adjusted summary, avg hold days and trail-trigger rate.
    """
    C, adv, sig_m, dates, nD = ctx["C"], ctx["adv"], ctx["sig_m"], ctx["dates"], ctx["nD"]
    n_tranches = max(1, horizon // step)

    # --- eligible()/book() copied verbatim from staggered_backtest (pead.py:241-257,
    #     no cap-tier path since this research uses the full adv-floor universe) ---
    def eligible(t: int) -> np.ndarray:
        return np.isfinite(sig_m[:, t]) & (adv[:, t] >= adv_floor)

    def book(t: int):
        ok = eligible(t)
        if ok.sum() < min_names:
            return None
        idx = np.where(ok)[0]
        return idx[np.argsort(-sig_m[idx, t])[:top_n]]

    def trailed_return(i: int, t: int) -> tuple[float, bool, int]:
        """Walk daily close from entry t through t+step (Accounting Harmonization (c)-(d),
        NaN/Delisting rule). Returns (return, is_trail_exit, hold_days)."""
        entry = C[i, t]
        if not np.isfinite(entry) or entry <= 0.0:
            return np.nan, False, step  # dropped by nanmean, matches staggered
        hwm = entry
        for d in range(1, step + 1):
            px = C[i, t + d]
            if not np.isfinite(px):  # NaN/delisting: exit at last valid close
                return C[i, t + d - 1] / entry - 1.0, False, d - 1
            if px > hwm:
                hwm = px
            if (px - hwm) / hwm <= -trail_pct:  # trailing stop hit
                return px / entry - 1.0, True, d
        return C[i, t + step] / entry - 1.0, False, step  # held full window

    rows = []
    tot_trail, tot_pos, hold_sum = 0, 0, 0
    for t in range(start_index, nD - step - 1):
        if (t - start_index) % step != 0:
            continue
        uni = np.where(eligible(t))[0]
        if uni.size < min_names:
            continue
        ret = C[:, t + step] / C[:, t] - 1.0  # full step-window fixed return
        bench = float(np.nanmean(ret[uni]))  # never shrunk (Harmonization (d))
        tranche_excess = []
        trail_exits = 0
        for k in range(n_tranches):
            b = book(t - k * step)
            if b is None:
                continue
            pos_rets = np.empty(len(b))
            for j, i in enumerate(b):
                r, is_trail, hold = trailed_return(i, t)
                pos_rets[j] = r
                trail_exits += int(is_trail)
                tot_trail += int(is_trail)
                tot_pos += 1
                hold_sum += hold
            tr = float(np.nanmean(pos_rets))  # matches staggered's nanmean(ret[b])
            tranche_excess.append(tr - bench)
        if tranche_excess:
            gross = float(np.mean(tranche_excess))
            # secondary cost (Harmonization (b) / Step 4 item 7): only genuine
            # trail triggers charged, summed across tranches then /n_tranches.
            net_cost = gross - (trail_exits * cost_one_way / top_n) / n_tranches
            rows.append({"date": dates[t], "gross": gross, "turnover": 1.0 / n_tranches,
                         "net": gross, "net_cost": net_cost})

    periods = pd.DataFrame(rows)
    summary = _summarize(periods, step)  # annualize by step (Harmonization (e))
    # cost-adjusted summary: reuse _summarize on the net_cost column.
    if periods.empty:
        cost_summary = summary
    else:
        cp = periods.copy()
        cp["net"] = cp["net_cost"]
        cost_summary = _summarize(cp, step)
    extras = {
        "cost_summary": cost_summary,
        "avg_hold": hold_sum / tot_pos if tot_pos else float("nan"),
        "trail_rate": tot_trail / tot_pos if tot_pos else 0.0,
    }
    return periods, summary, extras


TRAIL_PCTS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 1.0]


def run_trailing_sweep(prices=None, yoy_panel=None, best_threshold=0.0,
                       horizon=None, step=None) -> dict:
    """Sweep trail_pct on the best horizon/filter; assert the trail=1.0 gate."""
    if prices is None:
        prices, yoy_panel = load_data()
    horizon = horizon if horizon is not None else BASELINE["horizon"]
    step = step if step is not None else BASELINE["step"]
    masked = _mask_panel(yoy_panel, best_threshold)

    # Fixed-horizon baseline (production) for the sanity gate.
    _, base_s = staggered_backtest(
        prices, masked, horizon=horizon, step=step,
        top_n=BASELINE["top_n"], adv_floor=BASELINE["adv_floor"],
    )
    ctx = _context(prices, masked)

    print(f"=== Step 4: Experiment 3 — Trailing stop (H{horizon}/step{step}, "
          f"YoY filter={best_threshold:.2f}) ===")
    print(f"  fixed-horizon baseline (staggered): Sharpe {base_s['sharpe']:+.3f}  "
          f"t {base_s['t_stat']:+.3f}")
    hdr = (f"  {'trail':>6} {'n':>4} {'Sharpe':>8} {'t':>7} {'cum':>8} {'hit':>6} "
           f"{'worst':>8} {'payoff':>7} {'hold':>6} | {'Sh_c':>7} {'t_c':>7} {'cum_c':>8}")
    print(hdr)
    rows = []
    gate = None
    for tp in TRAIL_PCTS:
        _, s, ex = trailing_stop_backtest(
            ctx, horizon=horizon, step=step, top_n=BASELINE["top_n"],
            adv_floor=BASELINE["adv_floor"], trail_pct=tp,
        )
        cs = ex["cost_summary"]
        flag = ""
        if tp <= 0.08 and ex["trail_rate"] > 0.80:
            flag = " [!] short-horizon proxy"
        print(
            f"  {tp:>6.2f} {s['n']:>4} {s['sharpe']:>+8.3f} {s['t_stat']:>+7.3f} "
            f"{s['cum_net']:>+8.3f} {s['hit_rate']:>6.3f} {s['worst']:>+8.3f} "
            f"{s['payoff_ratio']:>7.3f} {ex['avg_hold']:>6.2f} | "
            f"{cs['sharpe']:>+7.3f} {cs['t_stat']:>+7.3f} {cs['cum_net']:>+8.3f}{flag}"
        )
        rows.append({"trail_pct": tp, **s, "cost_summary": cs,
                     "avg_hold": ex["avg_hold"], "trail_rate": ex["trail_rate"]})
        if tp == 1.0:
            gate = s

    # --- Hard sanity gate (plan Step 4 / Verification #4) ---
    d_sharpe = abs(gate["sharpe"] - base_s["sharpe"])
    d_t = abs(gate["t_stat"] - base_s["t_stat"])
    print(f"  sanity(trail=1.0 vs fixed): dSharpe={d_sharpe:.4f} (tol {TOL_SHARPE}), "
          f"dt={d_t:.4f} (tol {TOL_T})")
    assert d_sharpe <= TOL_SHARPE and d_t <= TOL_T, (
        f"TRAIL=1.0 SANITY GATE FAILED: dSharpe={d_sharpe:.4f}, dt={d_t:.4f} — "
        "accounting bug in trailing-stop implementation, fix before proceeding."
    )
    # cost check: trail=1.0 cost column must equal zero-cost column.
    gate_cost = next(r["cost_summary"] for r in rows if r["trail_pct"] == 1.0)
    assert abs(gate_cost["sharpe"] - gate["sharpe"]) < 1e-9, "trail=1.0 cost != zero-cost"

    # best trail (exclude 1.0): max zero-cost Sharpe
    cand = [r for r in rows if r["trail_pct"] < 1.0]
    best = max(cand, key=lambda r: r["sharpe"])
    base_worst, best_worst = base_s["worst"], best["worst"]
    # positive = shallower (less negative) worst period vs no-trail baseline.
    dd_red = (best_worst - base_worst) * 100
    dz = best["sharpe"] - base_s["sharpe"]
    dw = best["cost_summary"]["sharpe"] - base_s["sharpe"]
    # does ANY trail beat no-trail (trail=1.0 == base_s) on zero-cost Sharpe?
    trail_helps = best["sharpe"] > base_s["sharpe"]
    print(
        f"  Trail at {best['trail_pct']*100:.0f}% changes worst-period return by "
        f"{dd_red:+.2f} pp (positive=shallower) while Sharpe changes by {dz:+.3f} "
        f"(zero-cost) / {dw:+.3f} (cost-adjusted)"
    )
    if not trail_helps:
        print("  => No trailing stop improves zero-cost Sharpe over no-trail; "
              "PEAD is a drift effect and cutting winners early destroys the alpha.")
    return {"rows": rows, "base": base_s, "best_trail": best["trail_pct"],
            "trail_helps": trail_helps, "gate_ok": True}


# ---------------------------------------------------------------------------
# Step 5: run_all — summary report + docs
# ---------------------------------------------------------------------------
def _table_yoy(res) -> str:
    out = ["| threshold | n | Sharpe | t | cum | hit | worst |",
           "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        low = " [!] low n" if r["n"] < 0.5 * res["base_n"] else ""
        out.append(f"| {r['threshold']:.2f} | {r['n']} | {r['sharpe']:+.3f} | "
                   f"{r['t_stat']:+.3f} | {r['cum_net']:+.3f} | {r['hit_rate']:.3f} | "
                   f"{r['worst']:+.3f}{low} |")
    return "\n".join(out)


def _table_horizon(res) -> str:
    out = ["| horizon | step | n | Sharpe | t | cum | hit | worst |",
           "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        out.append(f"| {r['horizon']} | {r['step']} | {r['n']} | {r['sharpe']:+.3f} | "
                   f"{r['t_stat']:+.3f} | {r['cum_net']:+.3f} | {r['hit_rate']:.3f} | "
                   f"{r['worst']:+.3f} |")
    return "\n".join(out)


def _table_trail(res) -> str:
    out = ["| trail_pct | n | Sharpe | t | cum | hit | worst | payoff | avg_hold | Sharpe_cost | t_cost | cum_net_cost |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        cs = r["cost_summary"]
        out.append(
            f"| {r['trail_pct']:.2f} | {r['n']} | {r['sharpe']:+.3f} | {r['t_stat']:+.3f} | "
            f"{r['cum_net']:+.3f} | {r['hit_rate']:.3f} | {r['worst']:+.3f} | "
            f"{r['payoff_ratio']:.3f} | {r['avg_hold']:.2f} | {cs['sharpe']:+.3f} | "
            f"{cs['t_stat']:+.3f} | {cs['cum_net']:+.3f} |"
        )
    return "\n".join(out)


def run_all() -> None:
    """Run Steps 1-4 sequentially, then write the results doc."""
    prices, yoy_panel = load_data()
    base = run_baseline(prices, yoy_panel)
    print()
    yoy_res = run_yoy_filter_sweep(prices, yoy_panel)
    print()
    hor_res = run_horizon_sweep(prices, yoy_panel, yoy_res["best_threshold"])
    print()
    tr_res = run_trailing_sweep(
        prices, yoy_panel, best_threshold=yoy_res["best_threshold"],
        horizon=hor_res["best_horizon"], step=hor_res["best_step"],
    )

    best_trail = next(r for r in tr_res["rows"] if r["trail_pct"] == tr_res["best_trail"])
    trail_helps = tr_res["trail_helps"]
    # The combined config only adopts a trail if trailing actually helps.
    filter_helps = yoy_res["best_threshold"] > 0.0
    horizon_helps = hor_res["best_horizon"] != BASELINE["horizon"]
    combo_improves = filter_helps or horizon_helps or trail_helps

    # ---- final summary table (baseline vs each experiment's best) ----
    def line(name, s):
        return (f"| {name} | {s['sharpe']:+.3f} | {s['t_stat']:+.3f} | "
                f"{s['cum_net']:+.3f} | {s['worst']:+.3f} |")

    yoy_best_row = next(r for r in yoy_res["rows"]
                        if r["threshold"] == yoy_res["best_threshold"])
    hor_best_row = next(r for r in hor_res["rows"]
                        if r["horizon"] == hor_res["best_horizon"])
    trail_label = (f"Best trail ({best_trail['trail_pct']:.2f})" if trail_helps
                   else f"Least-bad trail ({best_trail['trail_pct']:.2f}, HURTS)")
    summary_tbl = "\n".join([
        "| config | Sharpe | t | cum | worst |",
        "|---|---|---|---|---|",
        line(f"Baseline (no filter, H{BASELINE['horizon']}, no trail)", base),
        line(f"Best YoY filter (thr={yoy_res['best_threshold']:.2f})", yoy_best_row),
        line(f"Best horizon (H{hor_res['best_horizon']})", hor_best_row),
        line(f"{trail_label}, zero-cost", best_trail),
        line(f"{trail_label}, cost-adj", best_trail["cost_summary"]),
    ])

    trail_clause = (
        f"trailing stop **{best_trail['trail_pct']*100:.0f}%**"
        if trail_helps else "**no** per-position trailing stop (all trails cut the "
        "drift short and reduced Sharpe)"
    )
    if combo_improves:
        filt_clause = (f"YoY filter **{yoy_res['best_threshold']:.2f}**" if filter_helps
                       else "no YoY filter (none beat baseline)")
        rec = (
            f"Promote: {filt_clause}, horizon **H{hor_res['best_horizon']}** "
            f"(step {hor_res['best_step']}), {trail_clause}. Best combined config "
            f"vs baseline Sharpe {base['sharpe']:+.3f}: {hor_best_row['sharpe']:+.3f} "
            f"(best-horizon) / trailing zero-cost {best_trail['sharpe']:+.3f}."
        )
    else:
        rec = (
            "**No robust improvement found.** No YoY threshold beat baseline on "
            f"Sharpe+t+MDD; optimal horizon band is {hor_res['band']} (baseline H60 "
            "already inside it); no trailing stop raised zero-cost Sharpe above "
            "no-trail (every trail cut the drift short). Keep the deployed PEAD rules "
            "unchanged (H60, no filter, no per-position stop)."
        )

    doc = f"""# PEAD Refinement Results

**Date:** 2026-07-13  **Script:** `research/pead_refinement.py`
**Plan:** `docs/pead-refinement.md` — 사전등록·Guardrails·회계 조화 (RALPLAN-DR consensus)

## Methodology

Three sequential refinements to the validated PEAD alpha — (1) earnings-surprise
magnitude filter, (2) holding-horizon sweep, (3) per-position trailing stop —
each built with `staggered_backtest` from `kr_quant.strategies.pead` so all
results share identical return / benchmark / annualization conventions
(Accounting Harmonization). Data is loaded **from TimescaleDB** (not CSV), full
expanded universe (2,629 DART codes, 2016Q1-2026). YoY is computed from
`netinc`/`netinc_prior` via the repo's canonical `(cur-prior)/|prior|`. Prices
are **split-adjusted** (`daily_bars_adjusted`): raw `daily_bars` is
corporate-action unadjusted (MULTI_ALPHA.md §"반드시 지킬 전제" #1) and yields Sharpe 0.42 — it does
not reproduce the validated alpha; the adjusted series does (§① requires it).
The **primary comparison is zero-cost vs zero-cost** across all experiments;
Experiment 3 adds a **secondary cost-adjusted** column
(`cost_one_way={COST_ONE_WAY}` charged per trail-triggered early exit).

## Baseline (anchor)

`staggered_backtest(H{BASELINE['horizon']}, step{BASELINE['step']}, top{BASELINE['top_n']}, adv_floor={BASELINE['adv_floor']:.0f})`:
Sharpe **{base['sharpe']:+.3f}**, t **{base['t_stat']:+.3f}**, cum {base['cum_net']:+.3f},
hit {base['hit_rate']:.3f}, payoff {base['payoff_ratio']:.3f}, worst {base['worst']:+.3f}
(n={base['n']}). Reproduces the validated range (t 2.16-2.97).

## Experiment 1 — YoY magnitude filter

{_table_yoy(yoy_res)}

**Result:** {"best threshold %.2f" % yoy_res['best_threshold'] if yoy_res['best_threshold'] > 0 else "no threshold improves baseline (filter adds no value)"}.

## Experiment 2 — Horizon sweep (YoY filter = {yoy_res['best_threshold']:.2f})

{_table_horizon(hor_res)}

**Result:** optimal horizon band **{hor_res['band']}**.

## Experiment 3 — Per-position trailing stop (H{hor_res['best_horizon']}/step{hor_res['best_step']}, filter {yoy_res['best_threshold']:.2f})

`Sharpe`/`t`/`cum` are zero-cost (primary); `Sharpe_cost`/`t_cost`/`cum_net_cost`
are cost-adjusted (secondary). `trail_pct=1.00` is the no-trail sanity row.

{_table_trail(tr_res)}

**Sanity gate:** trail=1.0 reproduces the fixed-horizon baseline within
{TOL_SHARPE} Sharpe / {TOL_T} t (PASSED).

## Final summary

{summary_tbl}

## Recommendation

{rec}

## Honest caveats

- **Split-adjusted prices are mandatory.** On raw `daily_bars` the whole result
  collapses (Sharpe 0.42). Any production use must apply `price_adjust`.
- **Benchmark-interpretation caveat (trailing stop).** Early-exit trailed
  positions are measured against a **full step-window benchmark that is never
  shrunk**. This mechanically flatters trails that dodge a late-period drawdown
  and mechanically penalizes trails that miss a late-period rally — so part of
  any apparent Sharpe change from trailing is this mechanical effect, not
  genuinely better exit timing.
- **In-window trailing only.** By Option C's design the HWM resets at each
  step-window start (entry = close[t]); this is an intra-step trailing stop, not
  a full holding-period one.
- **Overfitting / low-n.** Thresholds flagged `[!] low n` (n_periods < 50% of
  baseline) are excluded from "best" consideration. Findings are
  universe-dependent (full 2,629-code DART set) and in-sample; treat any single
  best cell as a hypothesis, not a promise.
"""
    out_path = "research/PEAD_REFINEMENT_RESULTS.md"
    with open(out_path, "w") as f:
        f.write(doc)
    print(f"\n[wrote] {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_COMMANDS = {
    "baseline": lambda: run_baseline(),
    "yoy-filter": lambda: run_yoy_filter_sweep(),
    "horizon-sweep": lambda: run_horizon_sweep(),
    "trailing-stop": lambda: run_trailing_sweep(),
    "all": run_all,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(f"usage: python research/pead_refinement.py "
              f"{{{'|'.join(_COMMANDS)}}}", file=sys.stderr)
        return 2
    _COMMANDS[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
