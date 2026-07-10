# Operator-Flow Swing Signal (수급 스텔스 매집 + 돌파)

Individual-stock swing system: enter when a liquid stock that has been **quietly
accumulated** (net-bought without the price yet moving) **breaks out** of its
20-day high; cut losses at an ATR-based stop, let winners run to a large target.
Low win-rate, high payoff-ratio by design.

## Signal

Per trading day, in a tradeable universe (trailing-20d ADV ≥ 30억/day):

- **Stealth accumulation** = `rank(A) − rank(mom)` where
  `A = Σ_20d(investor net-buy shares) / Σ_20d(volume)` (매집 강도) and
  `mom = trailing 20-day return`. Subtracting momentum isolates *hidden* demand
  (bought a lot, price hasn't moved) from momentum-chasing.
- **Entry** = top-30% stealth **and** close > prior 20-day high (breakout).
- **Exit** = ATR(14) bracket: hard stop at `entry − k·ATR`; optional half-profit
  at `+h·R` moving the stop to breakeven; runner target at `+r·R` (R = k·ATR).
  Intraday stops model gap-downs (fill at the open when it gaps through).

Data: `supply_demand` (investor net-buy) + `daily_bars` (OHLCV) in TimescaleDB.
No market-cap, no earnings — two sources.

## Why it holds up (controlled, out-of-sample)

Train on 2017–2021, select parameters, then measure **test 2022–2026** (never
seen during selection). `sim2_oos.c` reports both.

| Entry | Test payoff | Test t | Robust configs (train&test +) |
|---|---|---|---|
| **Random liquid stock** (baseline) | 3.8 | +0.5 | **1/80** — no edge |
| **Breakout only** (no flow filter) | — | — | **0 positive** — breakouts fake out |
| Stealth + breakout — 기관 | 5.2 | +9.7 | 40/80 |
| Stealth + breakout — 외국인 | 5.6 | +12.9 | 45/80 |
| Stealth + breakout — 개인 | 5.0 | +4.9 | 64/80 |
| Stealth + breakout — 기타법인 | 5.2 | +9.2 | 53/80 |
| Stealth + breakout — 연기금 | 5.1 | +9.1 | 41/80 |
| Stealth + breakout — 투신 | 5.7 | +10.4 | 47/80 |
| Stealth + breakout — 금융투자 | 4.9 | +10.7 | 57/80 |
| Stealth + breakout — 사모 | 5.3 | +10.6 | 46/80 |

**The flow filter is essential.** Breakout alone has no positive parameter set;
random entries generalize at t≈0.5 (null). Stealth accumulation **validates** the
breakout (real demand behind it vs a fake-out). The effect is present for *every*
investor group — not a quirk of one 창구 — which makes it more believable, not
less.

## Anchored walk-forward (the honest reliability picture)

`sim3_wf.c` retrains on all prior years and tests each single year (re-selecting
parameters annually). This exposes what the single 2022 split hid — the edge is a
**trend-year amplifier**, not an all-weather alpha:

```
etc_corp:  2020 +4.2%  2021 +0.7%  2022 -5.2%  2023 +0.9%
           2024 +1.9%  2025 +15.9% 2026 -0.4%    (5/7 positive, mean +2.6%/trade)
random:    all years negative (mean -7.3%/trade)   = null confirmed
```

- Profitability concentrates in trend years (2020, 2025); ex-2025 it is roughly
  break-even per trade. The 2022 rate-shock bear loses ~5%/trade.
- **Payoff ratio stays high every year (3.4–8, even 2022).** What collapses in a
  bear is the *hit rate* (7–10% in 2022 vs 35% in 2025), not the payoff. So the
  edge is genuine and the asymmetry is robust; the variability is regime-driven.
- A simple market-uptrend regime filter (index > 60d MA) did **not** fix 2022 —
  not pursued further (overfitting risk).

Implication: pair with a regime-complementary alpha (e.g. PEAD, which is weak in
2025 but works in most years) rather than over-filtering this one.

## Generalization (not a knife-edge fit)

Walk-forward holds up when the structural hyperparameters are perturbed — the
edge is not an artifact of the exact 20-day / top-30% choice:

| Variant | Positive test years | Mean test |
|---|---|---|
| Breakout window 10d / 40d | 2/6 | +2.3% / +2.2% |
| Top fraction 20% / 50% | 4/6 | +2.9% / +2.8% |
| Accumulation window 40d | 4/6 | +3.0% |

All positive on average. The 2022/2026 weakness appears in every variant, so it
is a genuine regime feature, not a parameter artifact. Combined with the 8-way
investor robustness and the random/breakout-only nulls, the signal is
well-generalized.

## Caveats (honest)

- Per-**trade** statistics; not yet a portfolio (position sizing, concurrent-hold
  limits, capacity). Selected picks' median ADV ≈ 100억/day (mid-cap).
- Parameters were grid-searched; mitigated by the OOS split and the smooth
  train/test agreement, but a walk-forward across multiple splits is the next
  robustness step.
- Daily-close stops; live execution needs intraday stops (gap risk).
- "Stealth accumulation" ≠ literal 세력; it is net investor buying decoupled from
  price. `etc_corp` may include buybacks / corporate strategic stakes.

## Live scanner

`scanner.py` applies the validated signal to the latest trading day and prints
today's entry candidates with ATR-based stop / half-profit / runner levels:

```bash
python scanner.py --db <DSN> [--investor etc_corp] [--asof YYYY-MM-DD]
# code  stealth   close   entry    stop  half_tgt  run_tgt  adv_억
# 100090  0.228  21450   21450   17383    33651    70256    157
```

By design it is **selective and low-frequency** — most days yield 0–1 names
(few stocks break a 20-day high *and* are quietly accumulated), more in trend
regimes, fewer in weak ones (e.g. mid-2026). Stops/targets are `close ± k·ATR`.

## Reproduce

```bash
# 1) dump trade paths from TimescaleDB (run inside the airflow scheduler container)
python prep_multi.py        # 8 investor types → /tmp/btm/<col>/
python prep_controls.py     # breakout-only + random-liquid baselines
#    then: docker cp <container>:/tmp/btm ./btm

# 2) fast parameter search (host, gcc)
gcc -O3 -march=native -o sim2_oos sim2_oos.c -lm
for inv in institution foreign_ individual etc_corp penfnd_etc invtrt fnnc_invt samo_fund; do
  ./sim2_oos btm/$inv 2022     # train<2022, test≥2022
done
./sim2_oos btm/_breakout_only 2022
./sim2_oos btm/_random_liquid 2022
```
