"""SEPA arm comparison harness — eval primitives + paired bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.strategies.sepa_compare import (
    _ann_sharpe,
    _cagr,
    _max_drawdown,
    book_returns,
    compare_arms,
    monthly_book_returns,
    paired_bootstrap,
    regime_buckets,
)

_MONTHS = [f"20{y:02d}-{m:02d}" for y in range(18, 24) for m in range(1, 13)]  # 72 months


def _series(mean: float, vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, vol, len(_MONTHS)), index=_MONTHS, name="ret")


def test_eval_primitives():
    r = np.full(12, 0.01)                       # +1%/mo, no vol
    assert abs(_cagr(r) - (1.01 ** 12 - 1)) < 1e-9
    assert np.isnan(_ann_sharpe(r))             # zero vol → undefined
    assert _max_drawdown(np.array([0.1, -0.5, 0.1])) < 0    # a drawdown exists


def test_monthly_book_returns_books_at_exit_month():
    trades = pd.DataFrame([
        {"exit_date": "2018-03-15", "ret": 0.10},
        {"exit_date": "2018-03-28", "ret": 0.20},   # same month → averaged
        {"exit_date": "2018-05-01", "ret": -0.05},
    ])
    s = monthly_book_returns(trades, _MONTHS)
    assert abs(s["2018-03"] - 0.15) < 1e-9      # mean(0.10, 0.20)
    assert s["2018-05"] == -0.05
    assert s["2018-04"] == 0.0                  # no exits → flat


def test_regime_buckets_sign_count():
    r = pd.Series([0.02] * 18 + [-0.01] * 18 + [0.03] * 18 + [0.01] * 18, index=_MONTHS)
    regs = regime_buckets(r, n=4)
    assert len(regs) == 4
    assert sum(x["positive"] for x in regs) == 3   # bucket 2 negative


def test_paired_bootstrap_detects_clear_winner():
    strong = _series(0.015, 0.03, seed=1)       # higher mean, same vol
    weak = _series(0.000, 0.03, seed=2)
    res = paired_bootstrap(strong, weak, n_boot=500, seed=0)
    assert res["d_sharpe_ci"][0] > 0            # CI excludes 0 → A clearly beats B
    assert res["prob_a_better_sharpe"] > 0.9


def test_paired_bootstrap_ties_include_zero():
    a = _series(0.005, 0.03, seed=3)
    b = _series(0.005, 0.03, seed=4)            # same distribution
    res = paired_bootstrap(a, b, n_boot=500, seed=0)
    lo, hi = res["d_sharpe_ci"]
    assert lo < 0 < hi                          # CI spans 0 → not a win


def _monthly_prices(paths: dict[str, list[float]], months: list[str]) -> pd.DataFrame:
    rows = []
    for code, closes in paths.items():
        for m, c in zip(months, closes):
            rows.append({"code": code, "date": f"{m}-15", "close": c})
    return pd.DataFrame(rows)


def test_book_returns_uses_exact_fill_price_within_a_single_month():
    # Regression guard for the 2026-07-12 stop-invariance bug: two trades that
    # enter/exit within the SAME calendar month but realize different prices
    # (e.g. a tighter vs looser stop) must produce different monthly returns —
    # previously book_returns ignored entry_price/exit_price entirely and marked
    # same-month trades to the market's monthly close, making them indistinguishable.
    months = ["2020-01", "2020-02", "2020-03"]
    prices = _monthly_prices({"TIGHT": [100.0, 100.0, 100.0], "LOOSE": [100.0, 100.0, 100.0]}, months)
    trades = pd.DataFrame([
        # Both "enter and exit in Feb" per (f,x), but realized very different prices.
        {"code": "TIGHT", "entry_date": "2020-02-15", "exit_date": "2020-02-15",
         "entry_price": 100.0, "exit_price": 96.0, "ret": -0.04},   # tight stop, small loss
        {"code": "LOOSE", "entry_date": "2020-02-15", "exit_date": "2020-02-15",
         "entry_price": 100.0, "exit_price": 92.0, "ret": -0.08},   # loose stop, bigger loss
    ])
    tight_only = book_returns(prices, trades[trades["code"] == "TIGHT"], n_slots=6, sized=False)
    loose_only = book_returns(prices, trades[trades["code"] == "LOOSE"], n_slots=6, sized=False)
    assert abs(tight_only["2020-02"] - (-0.04)) < 1e-9   # uses exit_price, not market's flat 0%
    assert abs(loose_only["2020-02"] - (-0.08)) < 1e-9
    assert tight_only["2020-02"] != loose_only["2020-02"]


def test_book_returns_falls_back_to_price_panel_when_prices_absent():
    # Without entry_price/exit_price columns, fall back to an exact-date close
    # lookup in the price panel (still precise, just sourced differently).
    prices = pd.DataFrame([
        {"code": "X", "date": "2020-01-15", "close": 100.0},
        {"code": "X", "date": "2020-02-10", "close": 90.0},   # exact exit-date close
        {"code": "X", "date": "2020-02-15", "close": 95.0},   # NOT the exit date — must be ignored
    ])
    trades = pd.DataFrame([{"code": "X", "entry_date": "2020-01-15", "exit_date": "2020-02-10"}])
    r = book_returns(prices, trades, n_slots=6, sized=False)
    assert abs(r["2020-02"] - (90.0 / 100.0 - 1.0)) < 1e-9   # uses exact exit-date close (90), not 95


def test_book_returns_concentration_tilts_to_high_score_winner():
    months = [f"2020-{i:02d}" for i in range(1, 8)]        # 7 months → 6 monthly returns
    prices = _monthly_prices(
        {"WIN": [100 * 1.1 ** k for k in range(7)],        # +10%/mo
         "LOSE": [100 * 0.9 ** k for k in range(7)]}, months)  # −10%/mo
    trades = pd.DataFrame([
        {"code": "WIN", "entry_date": "2020-01-15", "exit_date": "2020-07-15", "ret": 0.77, "score": 100.0},
        {"code": "LOSE", "entry_date": "2020-01-15", "exit_date": "2020-07-15", "ret": -0.47, "score": 10.0},
    ])
    sized = book_returns(prices, trades, n_slots=6, sized=True)
    equal = book_returns(prices, trades, n_slots=6, sized=False)
    assert not np.allclose(sized.to_numpy(), equal.to_numpy())
    assert equal.mean() == 0.0 or abs(equal.mean()) < 1e-9   # +10/−10 equal-weight ≈ flat
    assert sized.mean() > equal.mean()                        # concentration tilts to the winner
    # pilot: tilt grows once WIN is proven (+1R) and LOSE stays in pilot half.
    assert sized["2020-03"] > sized["2020-02"]


def test_book_returns_pyramid_scales_winners():
    months = [f"2020-{i:02d}" for i in range(1, 9)]        # 8 months
    prices = _monthly_prices(
        {"WIN": [100 * 1.1 ** k for k in range(8)],        # big winner (+10%/mo)
         "FLAT": [100.0] * 8}, months)
    trades = pd.DataFrame([
        {"code": "WIN", "entry_date": "2020-01-15", "exit_date": "2020-08-15", "ret": 0.9, "score": 100.0},
        {"code": "FLAT", "entry_date": "2020-01-15", "exit_date": "2020-08-15", "ret": 0.0, "score": 10.0},
    ])
    pyr = book_returns(prices, trades, n_slots=6, pyramid=True)
    nopyr = book_returns(prices, trades, n_slots=6, pyramid=False)
    assert pyr["2020-05"] > nopyr["2020-05"]   # winner amplified in later months
    assert pyr.mean() > nopyr.mean()


def test_book_returns_n_slots_caps_active_positions():
    months = [f"2020-{i:02d}" for i in range(1, 5)]
    prices = _monthly_prices(
        {"WIN": [100 * 1.1 ** k for k in range(4)],
         "LOSE": [100 * 0.9 ** k for k in range(4)]}, months)
    trades = pd.DataFrame([   # WIN listed first → earliest-entered kept when slots are scarce
        {"code": "WIN", "entry_date": "2020-01-15", "exit_date": "2020-04-15", "ret": 0.3, "score": 50.0},
        {"code": "LOSE", "entry_date": "2020-01-15", "exit_date": "2020-04-15", "ret": -0.3, "score": 50.0},
    ])
    book1 = book_returns(prices, trades, n_slots=1, sized=False)
    assert book1["2020-02"] > 0.09   # only WIN held (≈ +10%), not averaged with LOSE to 0


def test_book_returns_backward_compatible_without_score():
    # No score column → equal-weight, no sizing (old behavior).
    months = [f"2020-{i:02d}" for i in range(1, 5)]
    prices = _monthly_prices({"A": [100, 110, 121, 133], "B": [100, 90, 81, 73]}, months)
    trades = pd.DataFrame([
        {"code": "A", "entry_date": "2020-01-15", "exit_date": "2020-04-15", "ret": 0.33},
        {"code": "B", "entry_date": "2020-01-15", "exit_date": "2020-04-15", "ret": -0.27},
    ])
    b = book_returns(prices, trades, n_slots=6)   # sized default True but no score → equal
    assert abs(b["2020-02"]) < 1e-9               # +10/−10 equal-weight = 0


def test_compare_arms_table_and_verdicts():
    arms = {
        "A": _series(0.015, 0.03, seed=1),
        "A_noconc": _series(0.010, 0.03, seed=5),
        "B": _series(0.000, 0.03, seed=2),
        "C": _series(0.006, 0.03, seed=6),
    }
    table, verdicts = compare_arms(arms, deployed="B", benchmark="C", n_boot=400)
    assert set(table.index) == {"A", "A_noconc", "B", "C"}
    assert "vs_deployed" in verdicts["A"] and "vs_benchmark" in verdicts["A"]
    assert verdicts["A"]["beats_b_ci"] is True   # A clearly beats the shell B
    assert "B" not in verdicts and "C" not in verdicts   # reference arms not judged
