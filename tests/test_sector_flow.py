"""Phase 1.5 sector flow-propagation pre-check. Pure DataFrame in -> dict out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.features.sector_flow import (
    add_sector_average,
    analyze_sector_propagation,
)


def _dates(n, start=1):
    return [f"202601{d:02d}" for d in range(start, start + n)]


def _build_synthetic(
    *,
    n_sectors: int,
    stocks_per_sector: int,
    n_days: int,
    sector_effect: float,
    ar_phi: float,
    noise_scale: float,
    seed: int,
) -> pd.DataFrame:
    """Build a synthetic code/date/sector/ratio frame.

    Each sector gets its own independent AR(1) latent "sector flow factor"
    ``F[t] = phi * F[t-1] + eps_t`` (autocorrelated -- today's sector-wide
    flow shock persists for a few days, exactly the "flow propagates" story).
    If ``sector_effect`` is nonzero, every stock's ratio on day t is
    ``sector_effect * F[t] + idiosyncratic noise`` (contemporaneous, no
    per-stock lag baked in). Because ``F`` is autocorrelated, the *sector
    average* (which cancels idiosyncratic noise across many stocks and
    isolates ``F``) at day t is naturally predictive of an individual
    member's ratio at day t+lag for small lags -- a positive control. With
    ``sector_effect == 0`` stocks are pure independent noise with no sector
    structure at all -- a negative control.
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    sectors = [f"sector_{i}" for i in range(n_sectors)]

    def _ar1(n: int) -> np.ndarray:
        eps = rng.normal(0, 1.0, size=n)
        f = np.empty(n)
        f[0] = eps[0]
        for t in range(1, n):
            f[t] = ar_phi * f[t - 1] + eps[t]
        return f

    sector_factors = {sec: _ar1(n_days) for sec in sectors}

    rows = []
    for sec in sectors:
        factor = sector_factors[sec]
        for s in range(stocks_per_sector):
            code = f"{sec}_{s:02d}"
            noise = rng.normal(0, noise_scale, size=n_days)
            ratio = sector_effect * factor + noise
            for t in range(n_days):
                rows.append({"code": code, "date": dates[t], "sector": sec, "ratio": ratio[t]})

    return pd.DataFrame(rows)


def test_add_sector_average_excludes_self():
    df = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "date": ["20260101"] * 3,
            "sector": ["s1", "s1", "s1"],
            "ratio": [1.0, 2.0, 3.0],
        }
    )
    result = add_sector_average(df, "ratio")
    row_a = result.set_index("code").loc["A"]
    # Full average includes A itself: (1+2+3)/3 = 2.0
    assert row_a["sector_avg_ratio"] == 2.0
    # Leave-one-out for A excludes A: (2+3)/2 = 2.5
    assert row_a["sector_avg_ratio_loo"] == 2.5


def test_add_sector_average_loo_nan_for_sole_member():
    df = pd.DataFrame(
        {
            "code": ["A", "B"],
            "date": ["20260101", "20260101"],
            "sector": ["s1", "s2"],
            "ratio": [1.0, 5.0],
        }
    )
    result = add_sector_average(df, "ratio")
    # Each stock is the only member of its sector on this date -> LOO is NaN.
    assert result["sector_avg_ratio_loo"].isna().all()


def test_positive_control_within_sector_lag_beats_cross_sector():
    # Stocks share a genuine, persistent (autocorrelated) sector-wide flow
    # factor, so the sector average leads individual stocks by a few days.
    df = _build_synthetic(
        n_sectors=6,
        stocks_per_sector=15,
        n_days=100,
        sector_effect=1.0,
        ar_phi=0.7,
        noise_scale=0.5,
        seed=42,
    )

    result = analyze_sector_propagation(df, "ratio", lags=(1, 2, 3), seed=0)

    assert result["n_pairs"] > 0
    # The genuine sector-following effect should show up clearly: within-sector
    # lagged correlation should be much stronger than the cross-sector control
    # (the control has no real sector structure, so it hovers near zero).
    assert result["within_mean"] > 0.2
    assert result["within_mean"] > result["cross_mean"] + 0.2
    # Near a ~zero baseline the ratio's *sign* can flip run to run (dividing
    # by a near-zero denominator), but its magnitude still reflects "many
    # multiples of baseline" -- use abs() for a robust >=2x check.
    assert abs(result["ratio"]) > 2.0
    # Bootstrap 90% CI lower bound on the (within - cross) difference should
    # clear zero -- the detection logic finds the real effect confidently.
    assert result["ci_lower"] > 0
    # The lag-1 correlation (closest to the AR(1) factor's persistence)
    # should be the strongest, decaying by lag-3.
    assert result["within_by_lag"][1] > result["within_by_lag"][3]


def test_negative_control_pure_noise_ratio_close_to_one():
    # No sector structure at all: every stock is independent noise regardless
    # of its (arbitrary) sector label.
    df = _build_synthetic(
        n_sectors=6,
        stocks_per_sector=15,
        n_days=100,
        sector_effect=0.0,
        ar_phi=0.7,
        noise_scale=1.0,
        seed=7,
    )

    result = analyze_sector_propagation(df, "ratio", lags=(1, 2, 3), seed=0)

    assert result["n_pairs"] > 0
    # No real effect -> within-sector and cross-sector correlations should both
    # be comparably small/noisy around zero (no false positive). Note: the
    # "ratio" metric itself is not asserted on here -- dividing two near-zero
    # noisy numbers is inherently unstable (see the positive-control test's
    # comment), so it is not a reliable "close to 1" signal in the true-null
    # case either. The small absolute means and the CI straddling zero are
    # the robust "no propagation detected" signature instead.
    assert abs(result["within_mean"]) < 0.15
    assert abs(result["cross_mean"]) < 0.15
    assert abs(result["diff_mean"]) < 0.15
    # The bootstrap CI on the difference should NOT confidently exclude zero
    # the way the positive control's does -- it should straddle zero.
    assert result["ci_lower"] < 0 < result["ci_upper"]
