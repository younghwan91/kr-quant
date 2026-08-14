#!/usr/bin/env python
"""생존편향이 PEAD 성과를 얼마나 부풀렸는지 측정 (REPORTER).

2026-08-13~15 에 상장폐지 종목의 시세(460종목/459,847행)와 실적(364종목/7,283행)을
백필했다. 그 전까지 이 레포의 모든 백테스트는 **오늘까지 살아남은 회사만** 보고
성적을 쟀다 — GUARDRAILS §3 이 "가장 큰 숨은 인플레이터"로 지목한 편향이다.

이 스크립트는 같은 PEAD 베이스라인을 두 유니버스에서 돌려 그 차이를 숫자로 낸다:

    survivors  : 폐지 종목 제외 (= 백필 이전 상태의 재현)
    full       : 폐지 종목 포함 (= 지금)

**리포터지 판정기가 아니다**(GUARDRAILS §8). PASS/FAIL 도 임계값도 두지 않는다.
차이를 보고하고 해석은 사람이 한다.

⚠️ **이 측정의 한계 — 하한선이다, 정확한 보정치가 아니다.**
  1. 폐지 종목 460개 중 실적이 붙은 건 364개다. 나머지 96개는 DART corpCode 에
     없어 PEAD 신호가 안 생기고, 여전히 빠진 채로 남는다.
  2. 우리 시세 구간(2016-09~) 이전에 폐지된 회사는 애초에 대상이 아니다.
  3. 폐지 종목의 거래대금은 close*volume 근사치라(source='naver') ADV 문턱
     통과 여부가 실측치 기준과 미세하게 다를 수 있다.
  따라서 여기서 나오는 편향 크기는 **실제 편향의 하한**으로 읽어야 한다.

실행(DB 필요): docker start kr-quant-airflow-timescaledb-1 후
    uv run python research/experiments/survivorship_bias_gate.py
"""

from __future__ import annotations

import os
import sys

# 스크립트로 실행하면 sys.path[0]=이 파일 디렉터리뿐이라 레포 루트를 얹는다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prop_swing_common import load_env_db  # noqa: E402

from kr_quant.storage import connect, db_default  # noqa: E402
from kr_quant.strategies.pead import staggered_backtest  # noqa: E402
from research.experiments.pead_refinement import BASELINE, load_data  # noqa: E402


def delisted_codes(con) -> set[str]:
    """폐지 종목 코드 — 백필로 들어온 것(source='naver')이 판별 기준이다.

    delisted_stocks 마스터가 아니라 daily_bars.source 를 쓰는 이유: 마스터에는
    우리 시세 구간 밖의 폐지분과 채권·ELW 류가 섞여 있어, "이번 백필로 유니버스에
    새로 들어온 종목"과 일치하지 않는다.
    """
    with con.cursor() as cur:
        cur.execute("SELECT DISTINCT code FROM daily_bars WHERE source = 'naver'")
        return {r[0] for r in cur.fetchall()}


def _fmt(s: dict) -> str:
    return (f"n={s['n']:>4}  Sharpe={s['sharpe']:+.3f}  t={s['t_stat']:+.3f}  "
            f"cum={s['cum_net']:+.4f}  hit={s['hit_rate']:.3f}")


def decompose(prices, yoy, label: str) -> dict:
    """북 수익과 벤치마크 수익을 분리 보고.

    PEAD 는 **유니버스 대비 초과수익**이라, 유니버스 구성이 바뀌면 전략을 한 줄도
    안 건드려도 측정치가 움직인다. 합계(Sharpe)만 보면 그게 전략 변화인지 벤치마크
    변화인지 구분이 안 되므로 둘을 갈라 보고한다 — 이 실험의 결론이 나온 자리다.
    """
    import numpy as np

    from kr_quant.engine.panels import panel_pivot, resolve_signal
    from research.experiments.pead_refinement import MIN_NAMES, START_INDEX

    close = panel_pivot(prices, "close")
    tval = panel_pivot(prices, "trade_value")
    dates, codes = list(close.columns), list(close.index)
    C = close.to_numpy(float)
    V = tval.reindex(index=codes, columns=dates).to_numpy(float)
    sig, _ = resolve_signal(yoy, None, codes, dates)
    step, top_n, af = BASELINE["step"], BASELINE["top_n"], BASELINE["adv_floor"]

    adv = np.full_like(C, np.nan)
    for j in range(20, len(dates)):
        adv[:, j] = np.nanmean(V[:, j - 20:j], axis=1)

    books, benches, sizes, vanished, held = [], [], [], 0, 0
    for t in range(START_INDEX, len(dates) - step - 1):
        if (t - START_INDEX) % step:
            continue
        ok = np.isfinite(sig[:, t]) & (adv[:, t] >= af)
        if ok.sum() < MIN_NAMES:
            continue
        uni = np.where(ok)[0]
        ret = C[:, t + step] / C[:, t] - 1.0
        b = uni[np.argsort(-sig[uni, t])[:top_n]]
        entered = np.isfinite(C[b, t])
        held += int(entered.sum())
        vanished += int((entered & ~np.isfinite(C[b, t + step])).sum())
        books.append(np.nanmean(ret[b]))
        benches.append(np.nanmean(ret[uni]))
        sizes.append(uni.size)

    bk, bn = np.array(books), np.array(benches)
    out = {"book": bk.mean(), "bench": bn.mean(), "excess": (bk - bn).mean(),
           "uni": float(np.mean(sizes)), "vanished": vanished, "held": held}
    print(f"{label:24} 북={out['book']*100:+.3f}%  벤치={out['bench']*100:+.3f}%  "
          f"초과={out['excess']*100:+.3f}%  유니버스={out['uni']:.0f}종목  "
          f"보유중소멸={vanished}/{held}")
    return out


def main() -> int:
    load_env_db()
    con = connect(db_default())
    dl = delisted_codes(con)
    con.close()
    print(f"폐지 종목(백필분): {len(dl)}개\n")

    prices, yoy = load_data()
    print(f"전체 유니버스: 시세 {prices['code'].nunique()}종목 / "
          f"실적신호 {yoy['code'].nunique()}종목")

    in_data = dl & set(prices["code"].unique())
    with_sig = dl & set(yoy["code"].unique())
    print(f"  그중 폐지분: 시세 {len(in_data)}종목 / 실적신호 {len(with_sig)}종목\n")

    results = {}
    for label, keep in (("full (폐지 포함)", None), ("survivors (폐지 제외)", dl)):
        p, y = prices, yoy
        if keep is not None:
            p = prices[~prices["code"].isin(keep)]
            y = yoy[~yoy["code"].isin(keep)]
        _, s = staggered_backtest(p, y, **BASELINE)
        results[label] = s
        print(f"{label:24} {_fmt(s)}")

    a = results["survivors (폐지 제외)"]
    b = results["full (폐지 포함)"]
    print("\n=== 폐지 제외 -> 포함 ===")
    for k in ("sharpe", "t_stat", "cum_net", "hit_rate"):
        d = b[k] - a[k]
        rel = f" ({d / abs(a[k]) * 100:+.1f}%)" if a[k] else ""
        print(f"  {k:9} {a[k]:+.4f} -> {b[k]:+.4f}   차이 {d:+.4f}{rel}")

    print("\n=== 북/벤치 분해 (기간당 평균, step=20영업일) ===")
    df = decompose(prices, yoy, "full (폐지 포함)")
    ds = decompose(prices[~prices["code"].isin(dl)],
                   yoy[~yoy["code"].isin(dl)], "survivors (폐지 제외)")
    print(f"\n  북   차이 {(df['book'] - ds['book'])*100:+.3f}%p"
          f"   (폐지 종목을 담아서 생긴 전략 쪽 손실)")
    print(f"  벤치 차이 {(df['bench'] - ds['bench'])*100:+.3f}%p"
          f"   (유니버스가 실제로 얼마나 나빴는지)")
    print(f"  초과 차이 {(df['excess'] - ds['excess'])*100:+.3f}%p")
    print("\n※ 읽는 법: PEAD 는 '유니버스 대비 초과수익'이다. 생존편향은 전략보다"
          "\n  벤치마크를 더 크게 왜곡하므로, 폐지 종목을 빼면 초과수익이 *과소* 측정된다."
          "\n※ 보유중소멸 = 보유 기간에 가격이 끊긴 포지션. nanmean 이 조용히 빼므로"
          "\n  폐지 시점 손실이 반영되지 않는다 — 이 수가 크면 위 숫자는 낙관 편향이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
