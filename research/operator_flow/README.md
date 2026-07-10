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

## Caveats (honest)

- Per-**trade** statistics; not yet a portfolio (position sizing, concurrent-hold
  limits, capacity). Selected picks' median ADV ≈ 100억/day (mid-cap).
- Parameters were grid-searched; mitigated by the OOS split and the smooth
  train/test agreement, but a walk-forward across multiple splits is the next
  robustness step.
- Daily-close stops; live execution needs intraday stops (gap risk).
- "Stealth accumulation" ≠ literal 세력; it is net investor buying decoupled from
  price. `etc_corp` may include buybacks / corporate strategic stakes.

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
