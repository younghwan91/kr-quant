# PEAD Refinement Results (HANDOFF #2)

**Date:** 2026-07-13  **Script:** `research/pead_refinement.py`
**Plan:** `.omc/plans/pead-refinement-plan.md` (RALPLAN-DR consensus)

## Methodology

Three sequential refinements to the validated PEAD alpha — (1) earnings-surprise
magnitude filter, (2) holding-horizon sweep, (3) per-position trailing stop —
each built with `staggered_backtest` from `kr_quant.strategies.pead` so all
results share identical return / benchmark / annualization conventions
(Accounting Harmonization). Data is loaded **from TimescaleDB** (not CSV), full
expanded universe (2,629 DART codes, 2016Q1-2026). YoY is computed from
`netinc`/`netinc_prior` via the repo's canonical `(cur-prior)/|prior|`. Prices
are **split-adjusted** (`daily_bars_adjusted`): raw `daily_bars` is
corporate-action unadjusted (HANDOFF caveat #1) and yields Sharpe 0.42 — it does
not reproduce the validated alpha; the adjusted series does (§① requires it).
The **primary comparison is zero-cost vs zero-cost** across all experiments;
Experiment 3 adds a **secondary cost-adjusted** column
(`cost_one_way=0.0023` charged per trail-triggered early exit).

## Baseline (anchor)

`staggered_backtest(H60, step20, top40, adv_floor=20000)`:
Sharpe **+1.087**, t **+3.123**, cum +1.172,
hit 0.628, payoff 1.251, worst -0.063
(n=113). Reproduces the validated range (HANDOFF §① t 2.16-2.97).

## Experiment 1 — YoY magnitude filter

| threshold | n | Sharpe | t | cum | hit | worst |
|---|---|---|---|---|---|---|
| 0.00 | 113 | +1.087 | +3.123 | +1.172 | 0.628 | -0.063 |
| 0.10 | 113 | +1.070 | +3.070 | +1.193 | 0.619 | -0.066 |
| 0.20 | 113 | +1.139 | +3.256 | +1.345 | 0.655 | -0.069 |
| 0.30 | 113 | +1.105 | +3.162 | +1.308 | 0.637 | -0.079 |
| 0.50 | 112 | +1.105 | +3.158 | +1.192 | 0.598 | -0.064 |
| 0.75 | 108 | +1.018 | +2.859 | +1.070 | 0.620 | -0.067 |
| 1.00 | 107 | +0.622 | +1.765 | +0.576 | 0.579 | -0.082 |

**Result:** no threshold improves baseline (filter adds no value).

## Experiment 2 — Horizon sweep (YoY filter = 0.00)

| horizon | step | n | Sharpe | t | cum | hit | worst |
|---|---|---|---|---|---|---|---|
| 20 | 6 | 379 | +0.966 | +2.785 | +1.064 | 0.567 | -0.040 |
| 30 | 10 | 227 | +0.677 | +1.971 | +0.677 | 0.577 | -0.123 |
| 40 | 13 | 175 | +0.634 | +1.860 | +0.536 | 0.543 | -0.054 |
| 50 | 16 | 142 | +0.829 | +2.409 | +0.805 | 0.577 | -0.078 |
| 60 | 20 | 113 | +1.087 | +3.123 | +1.172 | 0.628 | -0.063 |
| 70 | 23 | 99 | +1.028 | +2.973 | +1.072 | 0.687 | -0.075 |
| 80 | 26 | 87 | +0.879 | +2.552 | +0.797 | 0.644 | -0.073 |
| 90 | 30 | 75 | +0.955 | +2.763 | +0.861 | 0.667 | -0.101 |

**Result:** optimal horizon band **60-70 days**.

## Experiment 3 — Per-position trailing stop (H60/step20, filter 0.00)

`Sharpe`/`t`/`cum` are zero-cost (primary); `Sharpe_cost`/`t_cost`/`cum_net_cost`
are cost-adjusted (secondary). `trail_pct=1.00` is the no-trail sanity row.

| trail_pct | n | Sharpe | t | cum | hit | worst | payoff | avg_hold | Sharpe_cost | t_cost | cum_net_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 113 | +0.274 | +0.805 | +0.314 | 0.549 | -0.117 | 0.990 | 8.37 | +0.110 | +0.326 | +0.041 |
| 0.08 | 113 | +0.256 | +0.755 | +0.254 | 0.513 | -0.113 | 1.132 | 12.09 | +0.100 | +0.299 | +0.038 |
| 0.10 | 113 | +0.311 | +0.915 | +0.311 | 0.549 | -0.115 | 1.024 | 14.01 | +0.166 | +0.493 | +0.120 |
| 0.12 | 113 | +0.259 | +0.764 | +0.224 | 0.531 | -0.100 | 1.059 | 15.45 | +0.130 | +0.387 | +0.076 |
| 0.15 | 113 | +0.335 | +0.988 | +0.309 | 0.558 | -0.088 | 1.000 | 17.07 | +0.237 | +0.703 | +0.193 |
| 0.20 | 113 | +0.364 | +1.070 | +0.330 | 0.540 | -0.112 | 1.103 | 18.56 | +0.306 | +0.902 | +0.263 |
| 1.00 | 113 | +1.087 | +3.123 | +1.172 | 0.628 | -0.063 | 1.251 | 20.00 | +1.087 | +3.123 | +1.172 |

**Sanity gate:** trail=1.0 reproduces the fixed-horizon baseline within
0.1 Sharpe / 0.1 t (PASSED).

## Final summary

| config | Sharpe | t | cum | worst |
|---|---|---|---|---|
| Baseline (no filter, H60, no trail) | +1.087 | +3.123 | +1.172 | -0.063 |
| Best YoY filter (thr=0.00) | +1.087 | +3.123 | +1.172 | -0.063 |
| Best horizon (H60) | +1.087 | +3.123 | +1.172 | -0.063 |
| Least-bad trail (0.20, HURTS), zero-cost | +0.364 | +1.070 | +0.330 | -0.112 |
| Least-bad trail (0.20, HURTS), cost-adj | +0.306 | +0.902 | +0.263 | -0.114 |

## Recommendation

**No robust improvement found.** No YoY threshold beat baseline on Sharpe+t+MDD; optimal horizon band is 60-70 days (baseline H60 already inside it); no trailing stop raised zero-cost Sharpe above no-trail (every trail cut the drift short). Keep the deployed PEAD rules unchanged (H60, no filter, no per-position stop).

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
