#!/usr/bin/env python
"""섹터 수치 페이지 — 매매 판단용 숫자만. 해석·추천 없음.

`sector_flow.py` 가 만든 시계열 페이로드를 여러 구간(5·20·60·120거래일) × 시장
(전체·거래소·코스닥)으로 집계해 **정적 표**로 낸다. 탐색용 인터랙션은 뷰어 쪽에 있고
이 페이지는 숫자를 한눈에 훑고 옮겨적기 위한 것이다.

Run:  python scripts/sector_numbers.py --html numbers.html
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

WINDOWS = ((5, "5일"), (20, "20일"), (60, "60일"), (120, "120일"))


def load_payload(db: str, days: int) -> dict:
    """sector_flow 의 JSON 페이로드를 재사용한다 — 집계 로직을 두 벌 두지 않는다."""
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    subprocess.run([sys.executable, os.path.join(here, "sector_flow.py"),
                    "--db", db, "--days", str(days), "--json", tmp],
                   check=True, stdout=subprocess.DEVNULL)
    with open(tmp, encoding="utf-8") as f:
        payload = json.load(f)
    os.unlink(tmp)
    return payload


def agg(P: dict, markets: list[str], sec: str, i0: int, i1: int) -> dict:
    def s(key):
        t = 0.0
        for m in markets:
            a = P["flows"].get(m, {}).get(sec, {}).get(key)
            if a:
                t += sum(a[i0:i1 + 1])
        return t
    cap = sum(P["cap"].get(m, {}).get(sec, 0.0) for m in markets)
    # 지수 수익률과 같은 시장 집합의 시총 — 지수가 한쪽 시장에만 있는 섹터에서
    # 분자와 분모의 시장 집합이 어긋나는 것을 막는다(참고 컬럼).
    cap_idx = sum(P["cap"].get(m, {}).get(sec, 0.0) for m in markets
                  if P["iret"].get(m, {}).get(sec))
    inst = s("inst")
    # 지수 수익률 — 구간말 시총 가중(일별 시총은 2026년 이전에 없다)
    w = {m: P["cap"].get(m, {}).get(sec, 0.0) for m in markets
         if P["iret"].get(m, {}).get(sec)}
    acc, any_ = 1.0, False
    if sum(w.values()):
        for i in range(i0, i1 + 1):
            num = den = 0.0
            for m, wt in w.items():
                r = P["iret"][m][sec][i]
                if r is not None:
                    num += r * wt; den += wt; any_ = True
            acc *= 1 + (num / den if den else 0.0) / 100.0
    n_all = sum(P.get("n_by_sector", {}).get(m, {}).get(sec, 0) for m in markets)
    return {
        "sector": sec, "n_all": n_all, "thin": n_all < MIN_NAMES,
        "inst": inst, "forgn": s("forgn"), "indiv": s("indiv"),
        "etc": s("etc"), "tv": s("tv"), "cap": cap,
        "accel": (inst / cap * 100) if cap else 0.0,
        "cap_idx": cap_idx,
        "ret": ((acc - 1) * 100) if any_ else None,
    }


def drivers(P: dict, markets: list[str], sec: str, i0: int, i1: int, k: int = 3):
    rows = []
    for code, r in P["names"].items():
        if r["sector"] != sec or r["market"] not in markets:
            continue
        rows.append({"code": code, "name": r["name"],
                     "inst": sum(r["inst"][i0:i1 + 1]),
                     "tv": sum(r["tv"][i0:i1 + 1])})
    rows.sort(key=lambda x: -x["inst"])
    return {"buy": rows[:k], "sell": rows[-k:][::-1]}


def power(P: dict, markets: list[str], sec: str, i0: int, i1: int) -> "float | None":
    """일률 dW/dt — 구간을 반으로 갈라 W 의 변화를 낸다.

    W = F·d (가속도 × 수익률). U(포텐셜)는 상태량이라 "얼마나 눌렸나" 만 말하고
    언제 풀릴지는 말하지 않는다. dW/dt 는 유량이라 방향이 있다 — 음에서 양으로
    돌아서는 것이 힘과 운동이 정렬되는 순간이고, 물리에서 가속이 시작되는 지점이다.

    전반/후반 각각에서 W 를 구해 (W_후 − W_전) / (구간 절반의 거래일수) 로 낸다.
    차분이라 k 추정의 절편 편향이 상쇄된다.
    """
    mid = i0 + (i1 - i0) // 2
    if mid <= i0 or i1 <= mid:
        return None
    out = []
    for a, b in ((i0, mid), (mid + 1, i1)):
        g = agg(P, markets, sec, a, b)
        if g["ret"] is None or not g.get("cap_idx"):
            return None
        out.append(g["inst"] / g["cap_idx"] * 100 * g["ret"])   # W = F·d
    return (out[1] - out[0]) / max(1, (i1 - mid))


def release(P: dict, markets: list[str], sec: str, i0: int, i1: int) -> tuple:
    """풀림 속도 ẋ 와 풀림 가속도 ẍ — 미실현 변위 x 가 해소되는 동역학.

    x 는 "그 유입이면 갔어야 할 만큼에서 덜 간 폭"(압축)이다. x 가 줄어드는 것이
    풀리는 것이고, **줄어드는 속도가 빨라지는 것**이 ẍ 다. U 는 상태량이라 언제
    풀릴지 말하지 않고, dW/dt 는 이미 풀리는 중이어야 잡힌다. ẍ 는 그보다 이르다.

    구간을 셋으로 갈라 각 조각에서 x 를 구한 뒤 중앙차분한다:
        ẋ = (x₃ − x₁) / 2Δ           (부호 반전: x 가 줄면 풀리는 것이므로 −)
        ẍ = (x₃ − 2·x₂ + x₁) / Δ²    (같은 이유로 부호 반전)
    Δ 는 조각당 거래일수. 2차 차분이라 조각이 짧으면 노이즈가 크다 —
    짧은 창에서는 값이 흔들린다는 것을 알고 봐야 한다.
    """
    n = i1 - i0 + 1
    if n < 9:
        return None, None
    cut = n // 3
    segs = [(i0, i0 + cut - 1), (i0 + cut, i0 + 2 * cut - 1), (i0 + 2 * cut, i1)]
    xs = []
    for a, b in segs:
        g = agg(P, markets, sec, a, b)
        if g["ret"] is None or not g.get("cap_idx"):
            return None, None
        # 조각별 x 는 그 조각 안에서 다시 k 를 추정할 표본이 없으므로,
        # 블록 k·절편을 그대로 적용한다(호출부가 넣어준다).
        xs.append((g["inst"] / g["cap_idx"] * 100, g["ret"]))
    return xs, cut


#: 이 미만이면 "섹터" 로 읽지 않는다. 벤더 분류가 좁아 부동산 3 · 출판 2 처럼
#: 사실상 단일종목인 라벨이 있고, 그런 곳의 x·ẍ·G 는 개별 종목 이야기다.
MIN_NAMES = 15


#: 이 미만이면 "섹터" 로 읽지 않는다. 벤더 분류(stocks.sector)가 좁아 부동산 3 ·
#: 출판/매체복제 2 · 제조 8 처럼 사실상 단일종목인 라벨이 있다.
MIN_NAMES = 10


def build(P: dict) -> dict:
    N = len(P["dates"])
    out = {"asof": P["dates"][-1], "finalized": P["finalized"],
           "dates": [P["dates"][0], P["dates"][-1]], "blocks": {}}
    for win, wl in WINDOWS:
        if win > N:
            continue
        i0, i1 = N - win, N - 1
        for mkey, markets in (("전체", P["markets"]),
                              *[(m, [m]) for m in P["markets"]]):
            rows = [agg(P, markets, s, i0, i1) for s in P["sectors"]]
            # 포텐셜 에너지 U = ½·k·x²,  x = 미실현 변위 = 예상Δv − 실제Δv.
            # k 는 그 블록 횡단면에서 추정한 강성(가속도 1%p 당 %), 절편 포함 OLS.
            # 예상Δv = k·a + b 이므로 x 는 "그 유입이면 갔어야 할 만큼에서 덜 간 폭"이고,
            # U 는 그 변위에 실린 에너지다. x>0 = 압축(덜 감), x<0 = 신장(더 감).
            # ⚠️ k 는 추정치다 — t 를 같이 실어 블록별 신뢰도를 보이게 한다.
            # (이전 '미반응' 잔차 열은 내렸다 — 회귀 없이 −Δv 와 ρ 0.94~1.00 이었다.) 두 가지가 각각 치명적이었다:
            # (1) 자금흐름은 `stocks.sector` 바구니에서 재고 수익률은 KRX 업종지수에서
            #     쟀는데 두 바구니의 구성종목이 다르다(부동산: 3종목 중 시총 80%인
            #     자이에스앤디가 20일 +101% 인데 업종지수는 +6.0%).
            # (2) 설령 바구니가 같았어도 지표가 정보를 안 담는다 — 잔차와 −수익률의
            #     순위상관이 중앙값 0.94(20일 0.99, 일부 블록 1.000)이고 R² 는
            #     0.02~0.15 다. 가속도 항이 기여하지 않아 사실상 "많이 빠진 순"의
            #     재포장이었다. "돈은 들어왔는데 안 갔다" 는 이름이 거짓이 된다.
            # k 는 참고용으로만 계산해 남긴다(표에는 안 쓴다).
            # 원점 통과 회귀로 하면 절편이 없어 시장 전체 드리프트를 흡수할 데가 없고,
            # 그게 전부 잔차로 밀려 모든 섹터를 같은 방향으로 이동시킨다(실측: 지수평균
            # +16.3% 구간에서 24개 중 23개가 음수, −11.2% 구간에서 24개 중 22개가 양수).
            # 부호가 섹터 정보가 아니라 장세를 따라가면 지표로 쓸 수 없다.
            # 양변을 평균 대비 편차로 놓으면 시장 드리프트가 상쇄되고, 남는 것은
            # "다른 섹터 대비 돈은 더 받았는데 덜 갔나" 라는 상대 진술이다.
            use = [r for r in rows if r["ret"] is not None]
            if use:
                abar = sum(r["accel"] for r in use) / len(use)
                rbar = sum(r["ret"] for r in use) / len(use)
                sxy = sum((r["accel"] - abar) * (r["ret"] - rbar) for r in use)
                sxx = sum((r["accel"] - abar) ** 2 for r in use)
            else:
                abar = rbar = sxy = sxx = 0.0
            slope = sxy / sxx if sxx else 0.0
            # 강성 k 와 절편 b, 그리고 t
            use = [r for r in rows if r["ret"] is not None and r.get("cap_idx")]
            n = len(use)
            if n > 2:
                xs = [r["inst"] / r["cap_idx"] * 100 for r in use]
                ys = [r["ret"] for r in use]
                mx = sum(xs) / n
                my = sum(ys) / n
                sxx = sum((v - mx) ** 2 for v in xs)
                sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
                slope = sxy / sxx if sxx else 0.0
                icpt = my - slope * mx
                sse = sum(((b - my) - slope * (a - mx)) ** 2 for a, b in zip(xs, ys))
                se = (sse / (n - 2) / sxx) ** 0.5 if sxx else float("nan")
                tstat = slope / se if se else float("nan")
            else:
                slope = icpt = tstat = 0.0
            for r in rows:
                if r["ret"] is not None and r.get("cap_idx"):
                    a = r["inst"] / r["cap_idx"] * 100
                    r["a_idx"] = a
                    r["exp"] = slope * a + icpt          # 예상 Δv (%)
                    r["x"] = r["exp"] - r["ret"]         # 미실현 변위 (%p)
                    r["U"] = 0.5 * slope * r["x"] ** 2   # 포텐셜 에너지 ½kx²
                else:
                    r["a_idx"] = r["exp"] = r["x"] = r["U"] = None
                r["P"] = power(P, markets, r["sector"], i0, i1)
                segs, cut = release(P, markets, r["sector"], i0, i1)
                if segs and slope:
                    # 조각별 미실현 변위 x = (k·a + b) − Δv, 블록 k·절편 적용
                    xv = [(slope * a + icpt) - dv for a, dv in segs]
                    # x 가 줄어드는 것이 "풀림" 이므로 부호를 뒤집는다
                    r["xdot"] = -(xv[2] - xv[0]) / (2 * cut)
                    r["xddot"] = -(xv[2] - 2 * xv[1] + xv[0]) / (cut ** 2)
                else:
                    r["xdot"] = r["xddot"] = None
                r["top"] = drivers(P, markets, r["sector"], i0, i1)
            # ── 성장 점수 G ──────────────────────────────────────────────
            # 물리로: 아직 눌려 있고(x>0), 계속 밀리고 있고(a>0), 막 풀리기
            # 시작한(ẍ>0) 상태. 세 조건은 **부호 검정**으로 걸러 통과 여부를 내고,
            # 순서는 세 양의 **횡단면 순위 평균**으로 매긴다.
            #
            # 순위를 쓰는 이유: a 는 %p(0~5), x 는 %p(±50), ẍ 는 1e-2 스케일이라
            # 그대로 더하면 x 가 전부 지배한다. 곱으로 묶으면 하나가 0 근처일 때
            # 전체가 죽어 불안정하다. 순위는 스케일에 불변이다.
            #
            # ⚠️ x 는 −Δv 와 강하게 얽혀 있다(이 저장소 실측 ρ 0.94~1.00). 따라서
            # G 는 순수 수급 신호가 아니라 **"많이 빠졌고 + 돈은 들어오고 + 반등이
            # 시작된"** 평균회귀형 화면이다. 그렇게 읽어야 한다.
            # 얇은 섹터는 순위 산정에서 뺀다 — 3종목짜리가 24개 섹터의 분위를
            # 밀어내면 나머지 순위까지 왜곡된다. 값은 계산해 두되 G 는 안 준다.
            scored = [r for r in rows if r.get("xddot") is not None
                      and r.get("a_idx") is not None and not r["thin"]]
            if len(scored) > 1:
                def _rank(key):
                    vals = [r[key] for r in scored]
                    order = sorted(range(len(vals)), key=lambda i: vals[i])
                    out_ = [0.0] * len(vals)
                    for pos, i in enumerate(order):
                        out_[i] = pos / (len(vals) - 1)
                    return out_
                ra, rx, rd = _rank("a_idx"), _rank("x"), _rank("xddot")
                for r, A, X, Dd in zip(scored, ra, rx, rd):
                    r["G"] = (A + X + Dd) / 3.0
                    r["G_pass"] = bool(r["a_idx"] > 0 and r["x"] > 0 and r["xddot"] > 0)
            for r in rows:
                r.setdefault("G", None)
                r.setdefault("G_pass", False)

            out["blocks"][f"{win}|{mkey}"] = {
                "from": P["dates"][i0], "to": P["dates"][i1],
                "k": round(slope, 3), "b": round(icpt, 3),
                "t": round(tstat, 2), "rows": rows,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("KR_QUANT_DB", ""))
    ap.add_argument("--days", type=int, default=260)
    ap.add_argument("--html", required=True)
    a = ap.parse_args()
    P = load_payload(a.db, a.days)
    data = build(P)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "templates", "sector_numbers.html"), encoding="utf-8") as f:
        html = f.read()
    with open(a.html, "w", encoding="utf-8") as f:
        f.write(html.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    print(f"wrote {a.html}  (asof {data['asof']}, blocks {len(data['blocks'])})")


if __name__ == "__main__":
    main()
