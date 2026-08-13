#!/usr/bin/env python
"""Step -1 feasibility gate — do the 3 prop-swing setup families produce ENOUGH
independent swing entries per fold (and in an untouched final window) for a strict
5/6-fold expectancy bar to be *measurable*?

This is a CHEAP ENTRY-SIGNAL COUNT, not a backtest. No exits, no returns, no R —
just: how many distinct (code, entry_date) swing entries does each family fire,
sliced by the frozen walk-forward FOLDS test windows and by the untouched
2025-07-01 → 2026-07-01 confirmation window. If a family is signal-starved
(~<30/fold), the 5/6 bar is not measurable and it dies here before Step 0/1.

Thin runner (plan §"thin runner in research, calls library functions only"):
reuses ``kr_quant.engine.panels.panel_pivot`` for the code×date grid,
``kr_quant.validation.walkforward.FOLDS`` for the frozen folds, and
``kr_quant.features.fundamentals`` (``_yoy_vec`` / ``earnings_yoy_panel``) for the
PEAD signal — exactly as ``research/experiments/pead_refinement.py`` and
``research/signals/contrarian_retail.py`` do. Nothing in ``src/kr_quant/`` changes.

Data (MULTI_ALPHA.md §"반드시 지킬 전제" #1): SPLIT-ADJUSTED ``daily_bars_adjusted``
only — raw ``daily_bars`` reads splits as catastrophic returns and corrupts every
MA / high / breakout used here.

No-lookahead in the COUNT (Principle 4): every MA / high / RSI / ADV uses only
data through the signal day t (trailing windows, ``shift(1)`` where a *prior*
extreme is needed); the entry is counted at the signal day t. A real backtest
would enter at t+1 open — that lag does not change the entry COUNT.

CONSENSUS REVISIONS honored:
  R1 — the untouched 2025-H2→2026 window is counted separately (never a search knob).
  R2 — FOLDS test-windows for 2020/2021 sit INSIDE the entry<2022 TRAIN boundary;
       they are flagged non-clean-OOS and the effective clean-OOS fold count is
       reported honestly (do not claim 6 clean OOS folds).
  R5 — this whole file IS Step -1: a measurability verdict, not a strategy.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from prop_swing_common import dedup_gap, load_env_db, weekly_count

from kr_quant.engine.panels import panel_pivot
from kr_quant.features.fundamentals import _yoy_vec, earnings_yoy_panel
from kr_quant.storage import connect, db_default, read_earnings
from kr_quant.validation.walkforward import FOLDS

# --- Parameters -------------------------------------------------------------
PRICE_TABLE = "daily_bars_adjusted"       # split-adjusted (raw daily_bars forbidden)
ADV_WINDOW = 20                            # trailing days for ADV
ADV_FLOOR = 20000.0                        # trade_value units (~repo "liquid mid-large")
MIN_GAP = 10                               # min trading-day gap between same-code entries
TRAIN_HI = "2022-01-01"                    # global TRAIN boundary (entry<this = in-sample)
# Untouched final-confirmation window (R1) — never a search knob.
UNTOUCHED_LO, UNTOUCHED_HI = "2025-07-01", "2026-07-01"
# PEAD rebalance cadence (mirrors pead BASELINE step/start).
PEAD_START_INDEX = 130
PEAD_STEP = 20
PEAD_TOP_N = 3
OUT_DIR = "research/logs/prop_feasibility"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load split-adjusted OHLC + trade_value and earnings from TimescaleDB.

    Returns ``(prices, earnings)`` where ``prices`` is long code/date/open/high/
    low/close/trade_value and ``earnings`` carries the canonical ``yoy`` column.
    """
    load_env_db()
    con = connect(db_default())
    prices = pd.read_sql_query(
        f"SELECT code, date, open, high, low, close, trade_value FROM {PRICE_TABLE}",  # noqa: S608 — trusted constant
        con,
    )
    # 정정공시 버전 중 최신 1건만 — 그냥 SELECT 하면 (code, period)가 중복된다.
    ea = read_earnings(con, cols=("code", "period", "avail_date", "netinc", "netinc_prior"))
    con.close()
    prices["code"] = prices["code"].astype(str)
    prices["date"] = prices["date"].astype(str)
    ea["code"] = ea["code"].astype(str)
    ea["avail_date"] = ea["avail_date"].astype(str)
    ea["yoy"] = _yoy_vec(ea["netinc"], ea["netinc_prior"])
    return prices, ea


# ---------------------------------------------------------------------------
# Panel construction — everything is a date-indexed (rows=date, cols=code) frame
# so rolling windows run along the time axis with pandas' native (trailing,
# min_periods=window) semantics = no lookahead.
# ---------------------------------------------------------------------------
def _time_panels(prices: pd.DataFrame) -> dict:
    """Build date×code frames for close/high/low/trade_value + derived features."""
    close = panel_pivot(prices, "close").T          # index=date(sorted), cols=code
    high = panel_pivot(prices, "high").T.reindex_like(close)
    low = panel_pivot(prices, "low").T.reindex_like(close)
    tval = panel_pivot(prices, "trade_value").T.reindex_like(close)

    ma = {w: close.rolling(w).mean() for w in (20, 50, 150, 200)}
    hi252 = high.rolling(252).max()                 # 252-day high incl t
    prior_hi50 = high.rolling(50).max().shift(1)     # max high over [t-50, t-1]
    adv = tval.rolling(ADV_WINDOW).mean()

    # RSI(14), SMA-based (adequate for an entry count).
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss > 0, 100.0)             # all-gains window → RSI 100

    return {
        "close": close, "high": high, "low": low, "adv": adv,
        "ma20": ma[20], "ma50": ma[50], "ma150": ma[150], "ma200": ma[200],
        "hi252": hi252, "prior_hi50": prior_hi50, "rsi": rsi,
        "dates": list(close.index),
    }


def _entries_from_mask(mask: pd.DataFrame, adv: pd.DataFrame, floor: float) -> list[tuple[str, str]]:
    """(date, code) pairs where the boolean signal mask is True and ADV≥floor."""
    m = mask & (adv >= floor)
    out: list[tuple[str, str]] = []
    codes = np.asarray(m.columns)
    for d, row in zip(m.index, m.to_numpy(), strict=True):
        hit = codes[row]
        out.extend((str(d), str(c)) for c in hit)
    return out


# ---------------------------------------------------------------------------
# Family entry signals (simplified — entry only, no exits/returns)
# ---------------------------------------------------------------------------
def breakout_entries(p: dict) -> list[tuple[str, str]]:
    """Family 1 — Breakout/trend (Minervini-lite).

    close > MA50 > MA150 > MA200 (stage-2 trend template) AND close ≥ 0.90×(252-day
    high) AND a fresh 50-day-high breakout on the entry day (close > prior 50-day
    high). Trend-template + proximity + breakout trigger, all point-in-time.
    """
    c = p["close"]
    trend = (c > p["ma50"]) & (p["ma50"] > p["ma150"]) & (p["ma150"] > p["ma200"])
    near_high = c >= 0.90 * p["hi252"]
    breakout = c > p["prior_hi50"]
    mask = trend & near_high & breakout
    return _entries_from_mask(mask, p["adv"], ADV_FLOOR)


def pullback_entries(p: dict) -> list[tuple[str, str]]:
    """Family 2 — Pullback mean-reversion.

    close > MA50 > MA200 (confirmed uptrend) AND a pullback: low touches/undercuts
    MA20 OR RSI(14) < 35. Buy-the-dip-in-an-uptrend entry.
    """
    c = p["close"]
    uptrend = (c > p["ma50"]) & (p["ma50"] > p["ma200"])
    dip = (p["low"] <= p["ma20"]) | (p["rsi"] < 35.0)
    mask = uptrend & dip
    return _entries_from_mask(mask, p["adv"], ADV_FLOOR)


def pead_entries(prices: pd.DataFrame, earnings: pd.DataFrame, p: dict) -> list[tuple[str, str]]:
    """Family 3 — PEAD-concentrated.

    On each rebalance date (every PEAD_STEP trading days from PEAD_START_INDEX),
    among ADV-eligible names with a fresh (as-of) earnings YoY surprise, take the
    top-PEAD_TOP_N by surprise. Each pick = one entry. Reuses the validated
    lookahead-safe ``earnings_yoy_panel`` (avail_date-gated) exactly as
    pead_refinement does.
    """
    dates = p["dates"]
    yoy_panel = earnings_yoy_panel(earnings.dropna(subset=["yoy"]), dates)
    yoy = (yoy_panel.pivot_table(index="date", columns="code", values="yoy", aggfunc="first")
           .reindex(index=dates))
    adv = p["adv"]
    codes = np.asarray(yoy.columns)
    adv_al = adv.reindex(columns=yoy.columns)
    entries: list[tuple[str, str]] = []
    nD = len(dates)
    yv = yoy.to_numpy()
    av = adv_al.to_numpy()
    for t in range(PEAD_START_INDEX, nD, PEAD_STEP):
        elig = np.isfinite(yv[t]) & (av[t] >= ADV_FLOOR)
        idx = np.where(elig)[0]
        if idx.size == 0:
            continue
        top = idx[np.argsort(-yv[t][idx])[:PEAD_TOP_N]]
        entries.extend((str(dates[t]), str(codes[j])) for j in top)
    return entries


# ---------------------------------------------------------------------------
# Counting / reporting
# ---------------------------------------------------------------------------
def _fold_is_clean_oos(fold) -> bool:
    """A fold is clean-OOS only if its whole test window is at/after TRAIN_HI."""
    return fold.test_lo >= TRAIN_HI


def _count_window(entries: list[tuple[str, str]], lo: str, hi: str) -> int:
    return sum(1 for d, _ in entries if lo <= d < hi)


def analyze(name: str, entries: list[tuple[str, str]]) -> dict:
    """Fold/untouched/weekly breakdown + measurability verdict for one family."""
    per_fold = []
    for f in FOLDS:
        n = _count_window(entries, f.test_lo, f.test_hi)
        per_fold.append({"fold": f, "n": n, "clean_oos": _fold_is_clean_oos(f)})
    untouched_n = _count_window(entries, UNTOUCHED_LO, UNTOUCHED_HI)
    wk = weekly_count(entries)
    clean = [r for r in per_fold if r["clean_oos"]]
    clean_ns = [r["n"] for r in clean]
    # HUMAN-READ measurability: rule of thumb ≥30 entries per clean-OOS fold.
    min_clean = min(clean_ns) if clean_ns else 0
    median_clean = int(np.median(clean_ns)) if clean_ns else 0
    # Distinct entry DATES in the untouched window — an intra-date-clustered family
    # (e.g. top-N picked on one rebalance date) has far fewer INDEPENDENT time
    # points than raw entries, so this is the honest independence proxy.
    untouched_dates = len({d for d, _ in entries if UNTOUCHED_LO <= d < UNTOUCHED_HI})
    return {
        "name": name,
        "total": len(entries),
        "per_fold": per_fold,
        "untouched_n": untouched_n,
        "untouched_dates": untouched_dates,
        "weekly_median": float(wk.median()) if len(wk) else 0.0,
        "weekly_max": int(wk.max()) if len(wk) else 0,
        "clean_min": min_clean,
        "clean_median": median_clean,
        "clean_fold_count": len(clean),
    }


def _verdict_line(a: dict) -> str:
    """HUMAN-READ measurability tier. Rule of thumb: ≥30 INDEPENDENT entries per
    clean-OOS fold AND in the untouched window. ``untouched_dates`` (distinct entry
    dates) exposes intra-date clustering that inflates the raw entry count.
    """
    lo = min(a["clean_min"], a["untouched_n"])
    # Temporal sparsity: few distinct entry DATES in the untouched year means few
    # independent time points regardless of raw entry count (top-N-per-rebalance).
    # Same-day entries in DIFFERENT stocks are largely independent cross-sectionally,
    # so we only flag when the DATE count itself is low, not on any same-day overlap.
    date_sparse = a["untouched_dates"] < 30
    if lo < 30:
        which = []
        if a["clean_min"] < 30:
            which.append(f"clean-OOS fold min {a['clean_min']}")
        if a["untouched_n"] < 30:
            which.append(f"untouched {a['untouched_n']}")
        return (f"SIGNAL-STARVED — {', and '.join(which)} <30. A per-fold expectancy "
                "bar is NOT measurable; this family dies here.")
    if lo < 60 or date_sparse:
        note = ""
        if date_sparse:
            note = (f" NOTE: untouched entries land on only {a['untouched_dates']} "
                    "distinct dates (top-N picked per rebalance) — effective INDEPENDENT "
                    f"time points ≈ {a['untouched_dates']}/yr, so fold CIs are wide and "
                    "the 5/6 reading is temporally sparse.")
        return (f"MEASURABLE BUT THIN — min(clean fold, untouched) = {lo}, above the "
                "≥30 floor but with little margin; a 5/6 bar is testable yet fold-level "
                f"readings carry wide CIs.{note}")
    return (f"MEASURABLE — clean-OOS folds min {a['clean_min']} and untouched "
            f"{a['untouched_n']}, both comfortably ≥30 across many distinct dates. A "
            "5/6 expectancy bar is testable.")


def _fmt_report(results: list[dict], universe_med: int) -> str:
    lines = ["# Prop-Swing Feasibility — Step -1 Entry Counts", ""]
    lines.append(f"- Price table: `{PRICE_TABLE}` (split-adjusted)")
    lines.append(f"- Universe: ADV(20d) ≥ {ADV_FLOOR:.0f} trade_value units "
                 f"(median daily eligible names ≈ {universe_med})")
    lines.append(f"- De-dup: ≥{MIN_GAP} trading-day gap between same-code entries "
                 "(independent-swing proxy)")
    lines.append(f"- TRAIN boundary: entry < {TRAIN_HI} (R2). Untouched window (R1): "
                 f"{UNTOUCHED_LO} → {UNTOUCHED_HI}")
    lines.append("")

    # Per-fold table.
    lines.append("## Entries per walk-forward fold (FOLDS test windows)")
    lines.append("")
    hdr = "| family | " + " | ".join(
        f"{f.test_lo[:7]}→{f.test_hi[:7]}{'' if _fold_is_clean_oos(f) else ' *IN-TRAIN*'}"
        for f in FOLDS) + " | untouched 25H2→26 |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(FOLDS) + 2))
    for a in results:
        cells = " | ".join(str(r["n"]) for r in a["per_fold"])
        lines.append(f"| {a['name']} | {cells} | {a['untouched_n']} |")
    lines.append("")
    lines.append("`*IN-TRAIN*` = fold test-window falls inside the entry<2022 TRAIN "
                 "boundary (R2) → NOT clean OOS. Effective clean-OOS folds = "
                 f"{results[0]['clean_fold_count']} (test_lo ≥ {TRAIN_HI}).")
    lines.append("")

    # Concurrency table.
    lines.append("## Concurrency (entries per ISO week)")
    lines.append("")
    lines.append("| family | total entries | weekly median | weekly max |")
    lines.append("|---|---|---|---|")
    for a in results:
        lines.append(f"| {a['name']} | {a['total']} | {a['weekly_median']:.0f} | {a['weekly_max']} |")
    lines.append("")
    lines.append("With only 4–8 concurrent slots: a weekly median at/below ~4–8 means "
                 "signal supply is roughly matched to capacity (not a hard binding "
                 "constraint); a weekly max far above that means occasional clustering "
                 "the book cannot fully take (opportunity loss, not lookahead).")
    lines.append("")

    # Verdicts.
    lines.append("## Per-family measurability verdict (HUMAN-READ, not coded)")
    lines.append("")
    for a in results:
        lines.append(f"### {a['name']}")
        lines.append(f"- total {a['total']}, clean-OOS folds (n={a['clean_fold_count']}): "
                     f"min {a['clean_min']}, median {a['clean_median']}; "
                     f"untouched {a['untouched_n']} entries on {a['untouched_dates']} "
                     "distinct dates")
        lines.append(f"- **{_verdict_line(a)}**")
        lines.append("")
    return "\n".join(lines)


def run() -> None:
    prices, earnings = load_data()
    p = _time_panels(prices)
    date_pos = {str(d): i for i, d in enumerate(p["dates"])}

    # Median daily eligible universe size (transparency for the ADV floor choice).
    elig_counts = (p["adv"] >= ADV_FLOOR).sum(axis=1)
    universe_med = int(elig_counts[elig_counts > 0].median())

    families = {
        "breakout": dedup_gap(breakout_entries(p), date_pos, MIN_GAP),
        "pullback": dedup_gap(pullback_entries(p), date_pos, MIN_GAP),
        "pead_top3": dedup_gap(pead_entries(prices, earnings, p), date_pos, MIN_GAP),
    }
    results = [analyze(name, ent) for name, ent in families.items()]

    report = _fmt_report(results, universe_med)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "COUNTS.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    run()
