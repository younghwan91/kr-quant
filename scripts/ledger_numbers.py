#!/usr/bin/env python
"""자금 원장 설계 문서 §3 의 실측을 **재현**한다.

DB 에 붙지 않는다 — 일일 리포트가 만든 ``payload.json`` 만 읽는다. 설계 판단
(무엇을 그리고 무엇을 안 그리나)이 이 숫자들 위에 서 있으므로, 숫자가 바뀌면
설계도 다시 봐야 한다. 그래서 "한 번 재보고 문서에 적기"가 아니라 스크립트로 둔다.

Run:  python scripts/ledger_numbers.py
      python scripts/ledger_numbers.py --dir <리포트 폴더>
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st

from kr_quant.tui.ledger_view import ACTOR_KEYS, load

NULL_SHIFTS = 20
SEED = 11


def _corr(a, b) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)


def _beta_resid(y, x):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((v - mx) ** 2 for v in x)
    b = 0.0 if vx <= 0 else sum((x[i] - mx) * (y[i] - my) for i in range(n)) / vx
    return [y[i] - my - b * (x[i] - mx) for i in range(n)]


def identity(d) -> None:
    """§3.1 — 4주체 합이 0 에서 얼마나 벗어나나.

    벗어나는 만큼이 **미분류 주체**(기타외국인 등) + 결측 + 종가환산 오차다.
    이걸 0 으로 밀어 넣지 않는 근거가 여기 있다.
    """
    F, n = d["flows"], len(d["dates"])
    tot = gross = tv = 0.0
    cell_res = cell_gross = 0.0
    daily = []
    for i in range(n):
        r = g = 0.0
        for m in d["markets"]:
            for s in d["sectors"]:
                c = F[m][s]
                cr = sum(c[k][i] for k in ACTOR_KEYS)
                cg = sum(abs(c[k][i]) for k in ACTOR_KEYS)
                cell_res += abs(cr)
                cell_gross += cg
                tv += c["tv"][i]
                r += cr
                g += cg
        daily.append(abs(r) / g * 100 if g else 0.0)
        tot += r
        gross += g
    daily.sort()
    # ⚠️ 잔차의 크기는 **분모를 정해야만** 말할 수 있다. 같은 −7,284억이
    # 거래대금으로 나누면 0.0076%, gross 순매수로는 0.0402%, 셀 단위로는 0.33% 다.
    # 한 숫자만 적으면 "닫힌다/안 닫힌다" 를 자 없이 주장하게 된다.
    print("## §3.1 회계 항등식 — 분모에 따라 자릿수 두 개가 달라진다")
    print(f"  기간 전체 4주체 순합      {tot:+,.0f}억")
    print(f"    ÷ 거래대금 {tv / 1e4:>8,.0f}조    = {abs(tot) / tv * 100:.4f}%"
          f"   (verify_report 의 자)")
    print(f"    ÷ gross 순매수 {gross / 1e4:>6,.0f}조 = {abs(tot) / gross * 100:.4f}%")
    print(f"  일별 전시장 잔차/gross    중앙값 {daily[n // 2]:.4f}%  "
          f"p90 {daily[int(n * 0.9)]:.4f}%  최대 {daily[-1]:.4f}%")
    print(f"  (시장,섹터,일) 셀 단위    {cell_res / cell_gross * 100:.4f}%")
    print()
    print("## §3.2 주체별 기간 순매수")
    for k in ACTOR_KEYS:
        v = sum(sum(F[m][s][k]) for m in d["markets"] for s in d["sectors"])
        print(f"  {k:>6}  {v:+12,.0f}억  ({v / 1e4:+.1f}조)")
    print(f"  {'잔여':>6}  {-tot:+12,.0f}억  ({-tot / 1e4:+.1f}조)")
    print()


def comovement(d) -> None:
    """§3.3 — 섹터쌍 상관 vs 순환이동 널.

    순수 노이즈면 음의 상관 쌍이 50% 다. 관측이 그보다 **낮으면** "섹터끼리 서로
    반대로 갔다(=로테이션)"는 그림이 우연보다도 드물다는 뜻이고, 그게 섹터→섹터
    화살표를 안 그리기로 한 직접 근거다.
    """
    F, secs, mkts = d["flows"], d["sectors"], d["markets"]
    n = len(d["dates"])
    cap = {s: sum(d["cap"].get(m, {}).get(s, 0.0) for m in mkts) for s in secs}
    nz = [s for s in secs if cap.get(s, 0) > 0]
    print("## §3.3 섹터쌍 동시성 — 관측 vs 널")
    print(f"  {'주체':<8}{'원계열 r':>10}{'음수쌍':>8}"
          f"{'β제거 r':>10}{'음수쌍':>8}{'널 음수쌍':>10}")
    for k in ACTOR_KEYS:
        ser = {s: [sum(F[m][s][k][i] for m in mkts) / cap[s] * 100 for i in range(n)]
               for s in nz}
        mkt = [sum(sum(F[m][s][k][i] for m in mkts) for s in nz)
               / sum(cap[s] for s in nz) * 100 for i in range(n)]
        res = {s: _beta_resid(ser[s], mkt) for s in nz}
        raw = [_corr(ser[a], ser[b]) for i, a in enumerate(nz) for b in nz[i + 1:]]
        dt = [_corr(res[a], res[b]) for i, a in enumerate(nz) for b in nz[i + 1:]]
        rnd = random.Random(SEED)
        nul = []
        for _ in range(NULL_SHIFTS):
            sh = {}
            for s in nz:
                t = rnd.randrange(n)
                sh[s] = res[s][t:] + res[s][:t]
            nul += [_corr(sh[a], sh[b]) for i, a in enumerate(nz) for b in nz[i + 1:]]
        f = lambda v: sum(1 for c in v if c < 0) / len(v)   # noqa: E731
        print(f"  {k:<8}{st.mean(raw):>+10.3f}{f(raw):>8.2f}"
              f"{st.mean(dt):>+10.3f}{f(dt):>8.2f}{f(nul):>10.2f}")
    print("  → 널은 0.50. 관측이 그보다 낮으면 로테이션은 우연보다 드물다.")
    print()


def spikes(d, sector: str = "전기/전자", win: int = 20) -> None:
    """§3.4 — 구간 합 뒤에 숨은 하루짜리 사건(GUARDRAILS §10 취약성의 시각화판)."""
    F, mkts, dates = d["flows"], d["markets"], d["dates"]
    n = len(dates)
    print(f"## §3.4 단일 사건 집중도 — 최근 {win}거래일 · {sector}")
    for k in ACTOR_KEYS:
        v = [sum(F[m][sector][k][i] for m in mkts) for i in range(n - win, n)]
        g = sum(abs(x) for x in v)
        i = max(range(win), key=lambda j: abs(v[j]))
        print(f"  {k:>6}  {win}일합 {sum(v):+11,.0f}억   "
              f"최대일 {dates[n - win + i]} {v[i]:+11,.0f}억  "
              f"({abs(v[i]) / g * 100:4.1f}% of gross)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="~/Documents/kr-quant-reports/latest")
    a = ap.parse_args()
    d = load(a.dir)
    print(f"# 자금 원장 실측 — {d['dates'][0]} ~ {d['dates'][-1]} "
          f"({len(d['dates'])}거래일, {d['n_names']}종목)\n")
    identity(d)
    comovement(d)
    spikes(d)


if __name__ == "__main__":
    main()
