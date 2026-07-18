#!/usr/bin/env python
"""PEAD를 contrarian을 죽인 '분포 관문'에 태운다 — 개별 트레이드 = 표본 (apples-to-apples).

단위 원칙(사용자): 한 트레이드가 하나의 표본. 한 종목이 여러 리밸런스에서 뽑히면 각각
독립 표본. 그래야 분포가 형성된다. → 포트폴리오 기간집계(20일 북 평균)로 재지 않는다.

트레이드 정의: 매 리밸런스 t(step일 간격)에서 book(t)에 뽑힌 각 종목 i가 하나의 트레이드.
그 트레이드 수익 = 종목의 step일 전방수익 − 그날 유니버스 평균(초과). 진입일=dates[t].
같은 종목이 다음 리밸런스에 또 뽑히면 별개 표본. PEAD는 손절이 없으니 R-멀티플 대신 초과수익(%).

관문(전략종류 무관 killer 두 개):
  1. 폴드-재현성 — contrarian과 동일 walk-forward 테스트창별 개별트레이드 초과수익 부호. k/N.
  2. 분포 모양·집중도 — 왜도·기간양수율·상위 k트레이드 점유·꼬리제거 생존.
contrarian(개별트레이드): 왜도 +5.9, 상위5건 39~67%, 폴드 2/6 → 기각. PEAD는?

실행: python research/experiments/pead_gate.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kr_quant.validation.walkforward import FOLDS  # noqa: E402
from research.experiments.pead_refinement import (  # noqa: E402
    BASELINE,
    MIN_NAMES,
    START_INDEX,
    _context,
    load_data,
)


def extract_trades(ctx, *, horizon, step, top_n, adv_floor):
    """개별 트레이드 표본 추출: (진입일, 초과수익) — 매 리밸런스 × book 종목마다 하나."""
    C, adv, sig_m, dates, nD = ctx["C"], ctx["adv"], ctx["sig_m"], ctx["dates"], ctx["nD"]

    def eligible(t):
        return np.isfinite(sig_m[:, t]) & (adv[:, t] >= adv_floor)

    ent, exc = [], []
    for t in range(START_INDEX, nD - step - 1):
        if (t - START_INDEX) % step != 0:
            continue
        ok = eligible(t)
        uni = np.where(ok)[0]
        if uni.size < MIN_NAMES:
            continue
        ret = C[:, t + step] / C[:, t] - 1.0
        bench = float(np.nanmean(ret[uni]))
        bk = uni[np.argsort(-sig_m[uni, t])[:top_n]]
        for i in bk:
            r = ret[i]
            if np.isfinite(r):
                ent.append(str(dates[t])[:10])
                exc.append(r - bench)
    return np.array(ent), np.array(exc, float)


def _t(x):
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std() > 0 else float("nan")


def fold_reproducibility(ent, exc):
    print("\n=== 관문 1: 폴드-재현성 (개별트레이드, contrarian과 동일 테스트창) ===")
    print(f"  {'테스트창':>12} {'n':>5} {'평균초과':>9} {'t':>7} {'승률':>5} {'양수?':>5}")
    k = valid = 0
    for _tl, _th, lo, hi in FOLDS:
        m = (ent >= lo) & (ent < hi)
        r = exc[m]
        if len(r) < 5:
            continue
        valid += 1
        pos = r.mean() > 0
        k += pos
        print(f"  {lo[:7]+'~':>12} {len(r):>5} {r.mean():>+9.4f} {_t(r):>7.2f} "
              f"{np.mean(r > 0):>5.0%} {'●' if pos else '○':>5}")
    print(f"  → 양수 폴드 {k}/{valid}   (contrarian은 2/6이라 기각됨)")
    return k, valid


def distribution(exc):
    r = exc
    srt = np.sort(r)[::-1]
    print("\n=== 관문 2: 개별트레이드 분포 모양·집중도 ===")
    print(f"  n={len(r)}  평균초과={r.mean():+.4f}  승률={np.mean(r > 0):.0%}  "
          f"왜도={float(((r-r.mean())**3).mean()/r.std()**3):+.2f}  "
          f"최고={r.max():+.1%} 최저={r.min():+.1%}")
    pos_mass = r[r > 0].sum()
    for k in (5, 20, 50):
        share = srt[:k].sum() / pos_mass if pos_mass > 0 else float("nan")
        rem = np.delete(r, np.argsort(r)[::-1][:k])
        print(f"  상위 {k:>3}건이 양(+)수익의 {share:>4.0%}  |  제거후 평균 {rem.mean():+.4f} "
              f"({'생존' if rem.mean() > 0 else '붕괴'})")
    print("  → contrarian: 왜도 +5.9 / 상위5건이 39~67% / 손절-1R 왼꼬리. PEAD 대조하라.")


def main() -> int:
    if not os.environ.get("KR_QUANT_DB") and os.path.exists(".env"):
        for line in open(".env"):
            if line.startswith("KR_QUANT_DB"):
                os.environ["KR_QUANT_DB"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    print("=== 데이터 로드 (TimescaleDB, 분할조정) ===")
    prices, yoy = load_data()
    ctx = _context(prices, yoy)
    ent, exc = extract_trades(ctx, **BASELINE)
    print(f"개별 트레이드 표본: {len(exc)}건 (진입 {min(ent)}~{max(ent)})")

    k, valid = fold_reproducibility(ent, exc)
    distribution(exc)

    print("\n=== 판정 (리포터, 하드코딩 합격선 없음) ===")
    print(f"  폴드-재현 {k}/{valid} + 저왜도·저집중이면 → 분산형 진짜 알파(contrarian과 정반대 프로파일).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
