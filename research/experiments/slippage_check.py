#!/usr/bin/env python
"""Phase 0 슬리피지 리얼리티 체크 — 개미투매 급등주 엣지가 몇 bp 왕복비용에서 죽나.

MANDATORY GATE (repo-restructure.md Phase 0). 결정론 최적점이 아니라 **재현성**으로 판정:
한 경로의 평균이 아니라 (a) 2022 경계 no-lookahead OOS의 R-분포와 (b) 고정 6폴드
walk-forward의 폴드 일관성이 비용을 올릴 때 어디서 무너지나를 본다.

원칙(비협상):
  - R-멀티플 = 트레이드수익 ÷ 하드손절폭(stop=0.10). 개별 트레이드가 표본. 복리·포트프레이밍 없음.
  - no-lookahead: TRAIN = 진입일 < 2022-01-01, OOS = 진입일 ≥ 2022-01-01 앞으로 평가.
  - 선별 임계값 θ는 TRAIN(또는 각 폴드 TRAIN)에서만 학습 → TEST 그대로 적용.
  - fragility 1급: monster-share(상위5 트레이드 P&L 비중). 소수 꼬리 의존이면 통계신뢰 낮음.

비용은 _sim_core에서 진입/청산 선택에 **영향 없이** 각 트레이드 수익에서 상수로 차감되므로
(선택·손절·트레일은 gross rr로 판정), gross(cost=0) 시뮬을 1회만 돌리고 비용별로 사후 차감한다.
정확·효율 둘 다 만족.

실행: uv run python research/experiments/slippage_check.py
"""

from __future__ import annotations

import numpy as np

from kr_quant.validation.walkforward import FOLDS, fold_slices
from research.experiments.contrarian_selective import (
    MIN_SEL,
    PARAMS,
    TRAIN_HI,
    _load_env_db,
)

COSTS = (0.0046, 0.008, 0.010, 0.015)   # 왕복비용: 46 / 80 / 100 / 150 bp
STOP = PARAMS["stop"]                    # 0.10 — R 정규화 분모
FRAC = 0.30                             # 선별: 모멘텀 상위 30% (선별이 후보 엣지)


def _rsummary(R: np.ndarray) -> dict:
    """개별표본 R-분포 요약 — n·기대값R·%승·≥3R빈도·손익비·monster비중(상위5 P&L share).

    monster = 상위5 R의 합 / 전체 R 합. 양수 기대값이 소수 꼬리에 의존하면 100%에 근접.
    R = ret/stop 이므로 stop 스케일은 monster·payoff 비율에서 상쇄(부호·비율 불변).
    """
    R = R[np.isfinite(R)]
    if len(R) == 0:
        return dict(n=0, expR=float("nan"), win=float("nan"), tail3=float("nan"),
                    payoff=float("nan"), monster=float("nan"))
    wins, losses = R[R > 0], R[R < 0]
    payoff = (wins.mean() / -losses.mean()) if len(wins) and len(losses) else float("nan")
    tot = R.sum()
    top5 = np.sort(R)[::-1][:5].sum()
    monster = float(top5 / tot) if tot > 0 else float("nan")
    return dict(n=len(R), expR=float(R.mean()), win=float((R > 0).mean()),
                tail3=float((R >= 3.0).mean()), payoff=payoff, monster=monster)


def _base_fold_consistency(entry: np.ndarray, R: np.ndarray) -> tuple[int, int]:
    """고정 PARAMS를 6폴드 각 TEST에서 재최적화 없이 평가 → (기대값R>0 폴드, 유효폴드)."""
    pos = valid = 0
    for _tl, _th, sl, sh in FOLDS:
        te = (entry >= sl) & (entry < sh) & np.isfinite(R)
        if te.sum() < 1:
            continue
        valid += 1
        if R[te].mean() > 0:
            pos += 1
    return pos, valid


def _sel_fold_consistency(d: dict) -> tuple[int, int]:
    """선별(θ는 각 폴드 TRAIN서 학습) 6폴드 → (선별기대값R>0 폴드, 유효폴드).

    유효 = 선별표본 ≥ MIN_SEL 인 폴드(희소 폴드는 정직하게 무효 처리).
    """
    pos = valid = 0
    for f in FOLDS:
        # 라이브러리 fold_slices: TRAIN서 θ 학습(모멘텀 (1-FRAC) 분위) → TEST에 적용. no-lookahead.
        fs = fold_slices(d["entry"], d["mom"], d["ret"] / STOP, f, FRAC)
        if fs is None:
            continue
        _theta, _base, sel = fs
        if len(sel) < MIN_SEL:
            continue
        valid += 1
        if sel.mean() > 0:
            pos += 1
    return pos, valid


def main() -> int:
    _load_env_db()
    from research.contrarian_retail import load_data
    print("=== 데이터 로드 ===")
    prices, flow = load_data()

    from research.contrarian_retail import simulate_detailed
    cache: dict = {}
    # gross(cost=0) 상세 시뮬 1회 — 비용은 사후 차감(선택/청산에 영향 없음)
    d0 = simulate_detailed(prices, flow, _cache=cache, cost_roundtrip=0.0, **PARAMS)
    entry, mom, ret0 = d0["entry"], d0["mom"], d0["ret"]
    oos_mask = entry >= TRAIN_HI
    train_mask = (entry < TRAIN_HI) & np.isfinite(mom)
    theta_full = np.quantile(mom[train_mask], 1.0 - FRAC)  # 선별 θ: TRAIN서만 학습

    print(f"\n  설정: PARAMS={PARAMS}  선별=모멘텀 상위{FRAC:.0%}(θ는 TRAIN학습)")
    print(f"  gross 트레이드 총 {len(ret0)}건  (TRAIN<2022: {int(train_mask.sum())}, "
          f"OOS≥2022: {int(oos_mask.sum())})  선별 θ={theta_full:+.3f}")

    # === 전체-OOS R-분포 (2022 경계 no-lookahead) ===
    print("\n=== 전체 OOS(≥2022) R-분포 — 비용 스윕 ===")
    print(f"  {'변형':>6} {'cost':>6} | {'n':>4} {'expR':>7} {'%승':>5} {'≥3R':>5} "
          f"{'손익비':>6} {'monster':>8} | {'폴드일관 k/유효':>14}")
    verdict_rows = []
    for cost in COSTS:
        bp = int(round(cost * 1e4))
        R = (ret0 - cost) / STOP
        # base
        b = _rsummary(R[oos_mask])
        bpos, bval = _base_fold_consistency(entry, R)
        # selective (θ_full: TRAIN학습 → OOS 적용)
        sel_mask = oos_mask & (mom >= theta_full) & np.isfinite(mom)
        s = _rsummary(R[sel_mask])
        d_cost = {**d0, "ret": ret0 - cost}
        spos, sval = _sel_fold_consistency(d_cost)
        print(f"  {'base':>6} {bp:>4}bp | {b['n']:>4} {b['expR']:>+7.3f} {b['win']:>5.0%} "
              f"{b['tail3']:>5.0%} {b['payoff']:>6.2f} {b['monster']:>8.0%} | "
              f"{f'{bpos}/{bval}':>14}")
        print(f"  {'선별':>6} {bp:>4}bp | {s['n']:>4} {s['expR']:>+7.3f} {s['win']:>5.0%} "
              f"{s['tail3']:>5.0%} {s['payoff']:>6.2f} {s['monster']:>8.0%} | "
              f"{f'{spos}/{sval}':>14}")
        verdict_rows.append((bp, b, bpos, bval, s, spos, sval))

    # === 폴드별 상세 — 집계 양수가 한 레짐(2025~26) 아티팩트인지 폭로 ===
    print("\n=== 폴드별 base expR (재최적화 없음) — 46/100/150bp, IS=진입<2022 ===")
    print(f"  {'TEST창':>16} {'n':>5} {'46bp':>8} {'100bp':>8} {'150bp':>8}  구분")
    for tl, _th, sl, sh in FOLDS:
        te = (entry >= sl) & (entry < sh) & np.isfinite(ret0)
        vals = [((ret0[te] - c) / STOP).mean() if te.sum() else float("nan")
                for c in (0.0046, 0.010, 0.015)]
        tag = "IS " if sl < TRAIN_HI else "OOS"
        print(f"  {sl[:7] + '~' + sh[:7]:>16} {int(te.sum()):>5} "
              f"{vals[0]:>+8.3f} {vals[1]:>+8.3f} {vals[2]:>+8.3f}  {tag}")

    # === 판정 요약 ===
    print("\n=== 판정 요약 (cost bp × {OOS expR, 폴드일관 k/6, monster%}) ===")
    print(f"  {'cost':>6} | {'base_expR':>9} {'base_k/6':>8} {'base_mon':>8} | "
          f"{'sel_expR':>9} {'sel_k/n':>8} {'sel_mon':>8}")
    for bp, b, bpos, bval, s, spos, sval in verdict_rows:
        print(f"  {bp:>4}bp | {b['expR']:>+9.3f} {f'{bpos}/{bval}':>8} {b['monster']:>8.0%} | "
              f"{s['expR']:>+9.3f} {f'{spos}/{sval}':>8} {s['monster']:>8.0%}")

    # 엣지가 죽는 비용: OOS expR ≤ 0 이 되는 첫 비용, 또는 폴드일관이 과반 이하로
    def _dies_at(rows, exp_key, k_key, n_key):
        for bp, b, bpos, bval, s, spos, sval in rows:
            summ = (b, bpos, bval) if exp_key == "base" else (s, spos, sval)
            st, pos, val = summ
            if not (st["expR"] > 0) or val == 0 or pos <= val / 2:
                return bp
        return None

    base_die = _dies_at(verdict_rows, "base", None, None)
    sel_die = _dies_at(verdict_rows, "sel", None, None)
    print("\n  판독: expR>0 이고 폴드일관 k>유효/2 를 둘 다 만족하는 최대 비용까지 엣지 생존.")
    print(f"    base  → {'전 구간(≤150bp) 생존' if base_die is None else f'~{base_die}bp에서 사망'}")
    print(f"    선별  → {'전 구간(≤150bp) 생존' if sel_die is None else f'~{sel_die}bp에서 사망'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
