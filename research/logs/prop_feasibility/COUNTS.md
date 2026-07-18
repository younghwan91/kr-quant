# Prop-Swing Feasibility — Step -1 Entry Counts

- Price table: `daily_bars_adjusted` (split-adjusted)
- Universe: ADV(20d) ≥ 20000 trade_value units (median daily eligible names ≈ 161)
- De-dup: ≥10 trading-day gap between same-code entries (independent-swing proxy)
- TRAIN boundary: entry < 2022-01-01 (R2). Untouched window (R1): 2025-07-01 → 2026-07-01

## Entries per walk-forward fold (FOLDS test windows)

| family | 2020-01→2021-01 *IN-TRAIN* | 2021-01→2022-01 *IN-TRAIN* | 2022-01→2023-01 | 2023-01→2024-01 | 2024-01→2025-01 | 2025-01→2027-01 | untouched 25H2→26 |
|---|---|---|---|---|---|---|---|
| breakout | 554 | 838 | 143 | 405 | 404 | 1283 | 1045 |
| pullback | 2042 | 2545 | 919 | 1466 | 1554 | 3388 | 2709 |
| pead_top3 | 39 | 36 | 39 | 36 | 36 | 57 | 36 |

`*IN-TRAIN*` = fold test-window falls inside the entry<2022 TRAIN boundary (R2) → NOT clean OOS. Effective clean-OOS folds = 4 (test_lo ≥ 2022-01-01).

## Concurrency (entries per ISO week)

| family | total entries | weekly median | weekly max |
|---|---|---|---|
| breakout | 4037 | 7 | 71 |
| pullback | 13651 | 23 | 167 |
| pead_top3 | 345 | 3 | 3 |

With only 4–8 concurrent slots: a weekly median at/below ~4–8 means signal supply is roughly matched to capacity (not a hard binding constraint); a weekly max far above that means occasional clustering the book cannot fully take (opportunity loss, not lookahead).

## Per-family measurability verdict (HUMAN-READ, not coded)

### breakout
- total 4037, clean-OOS folds (n=4): min 143, median 404; untouched 1045 entries on 219 distinct dates
- **MEASURABLE — clean-OOS folds min 143 and untouched 1045, both comfortably ≥30 across many distinct dates. A 5/6 expectancy bar is testable.**

### pullback
- total 13651, clean-OOS folds (n=4): min 919, median 1510; untouched 2709 entries on 242 distinct dates
- **MEASURABLE — clean-OOS folds min 919 and untouched 2709, both comfortably ≥30 across many distinct dates. A 5/6 expectancy bar is testable.**

### pead_top3
- total 345, clean-OOS folds (n=4): min 36, median 37; untouched 36 entries on 12 distinct dates
- **MEASURABLE BUT THIN — min(clean fold, untouched) = 36, above the ≥30 floor but with little margin; a 5/6 bar is testable yet fold-level readings carry wide CIs. NOTE: untouched entries land on only 12 distinct dates (top-N picked per rebalance) — effective INDEPENDENT time points ≈ 12/yr, so fold CIs are wide and the 5/6 reading is temporally sparse.**
