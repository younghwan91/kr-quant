#!/usr/bin/env python
"""일일 리포트 검증 — 수식·데이터·계산을 한 번에 점검한다.

층을 나눠서 본다. 층마다 실패 원인이 다르고 대응도 다르기 때문이다.

  A. 수식   — 표에 실린 값들이 정의된 항등식을 만족하는가(전정밀도 재적합 기준)
  B. 데이터 — DB 에서 들어온 것이 온전한가(기준일·커버리지·항등식·중복·단위)
  C. 계산   — 표의 구조(열 정의 = 셀 개수)와 값이 정합한가
  D. 부류   — 계산에 들어가는 두 값이 같은 집합·시점·파라미터에서 나오는가
  E. 화면   — **그려진 글자**가 열 정의가 내는 값과 같은가(폭 80·120·200)
  F. 독립   — **DB 원자료에서 처음부터 다시 만든 값**과 페이로드가 같은가

A~D 는 전부 *파일 안의 값끼리* 정합한지만 본다. 그래서 producer 가 임펄스를
통째로 잘못 계산해도(수량을 금액으로 착각, `flu_rt` 의 bp 단위를 놓침 — 둘 다
이 저장소에서 실제로 사고를 냈다) 전부 초록이다. 파생층이 그 틀린 값 위에서
일관될 뿐이기 때문이다. **F 가 그 구멍을 막는다** — `scripts/` 의 함수를 하나도
부르지 않고 SQL 로 다시 집계한다. 같은 버그를 두 번 통과시키지 않으려면 경로가
달라야 한다. (E·F 는 두 번째 검증기를 만들지 않고 여기에 합쳤다 — 검증기가
둘이면 배치가 하나만 돌리고 나머지는 조용히 썩는다. F 만 DB 를 요구한다.)

⚠️ **반올림된 값으로 검산하지 않는다.** payload 의 k 는 소수 3자리라, 그걸로
U=½kx² 를 검산하면 1e-3 오차가 나서 멀쩡한 수식이 FAIL 로 뜬다(실제로 한 번
밟았다). 원자료에서 전정밀도로 다시 적합해 비교한다.

⚠️ **상대오차로 비교하지 않는다.** 섹터 합계가 0 근처인 칸(섬유/의류 20일
+2.0억)에서 상대오차는 무의미하게 커진다. 허용오차는 페이로드가 반올림하는
자릿수에서 유도한 **절대값**이다(`TOL_FLOW`·`TOL_CAP`·`TOL_RET`).

Run:  python scripts/verify_report.py --dir ~/Documents/kr-quant-reports/2026-08-27
      python scripts/verify_report.py --dir <폴더> --db-check     # B·F 층까지
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

TOL_EXACT = 1e-9      # 항등식 — 기계정밀도
TOL_FIT = 1e-8        # 재적합 경유 — 누적 부동소수 오차 여유
FAILS: list[str] = []
CHECKS = 0


def chk(layer: str, name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {layer} · {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(f"{layer}/{name}")


def ols(xs, ys):
    """절편 포함 단순회귀 — (기울기, 절편, t, R²)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    k = sxy / sxx if sxx else 0.0
    b = my - k * mx
    res = [y - (k * x + b) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in res)
    sst = sum((y - my) ** 2 for y in ys)
    se = math.sqrt(sse / (n - 2) / sxx) if sxx and n > 2 else float("nan")
    return k, b, (k / se if se else float("nan")), (1 - sse / sst if sst else float("nan"))


# ─────────────────────────────────────────────── A. 수식

def layer_a(D: dict) -> None:
    print("\nA. 수식 — 표의 값이 정의된 항등식을 만족하는가")
    worst = {}

    def upd(key, v):
        worst[key] = max(worst.get(key, 0.0), abs(v))

    blocks = 0
    for key, B in D["blocks"].items():
        rows = [r for r in B["rows"] if r.get("exp") is not None]
        if len(rows) < 8:
            continue
        blocks += 1
        xs = [r["a_idx"] for r in rows]
        ys = [r["ret"] for r in rows]
        k, b, _t, _r2 = ols(xs, ys)                 # 전정밀도 재적합

        upd("a = 임펄스/시총×100", max(r["a_idx"] - r["inst"] / r["cap_idx"] * 100 for r in rows))
        upd("예상Δv = k·a + b", max(r["exp"] - (k * r["a_idx"] + b) for r in rows))
        upd("x = 예상 − 실제", max(r["x"] - (r["exp"] - r["ret"]) for r in rows))
        upd("x = −(OLS 잔차)", max(r["x"] + (r["ret"] - (k * r["a_idx"] + b)) for r in rows))
        # U 는 **k>0 인 블록에만** 실린다(k≤0 이면 |x| 의 순감소 변환이라 뜻을
        # 잃는다 — `layer_a2` 가 그 규약을 따로 검사한다). 여기서는 실린 것만 본다.
        if all(r["U"] is not None for r in rows):
            upd("U = ½·k·x²", max(r["U"] - 0.5 * k * r["x"] ** 2 for r in rows))
        res = [y - (k * x + b) for x, y in zip(xs, ys)]
        upd("OLS: Σ잔차 = 0", sum(res) / len(res))
        upd("OLS: 잔차 ⊥ a", sum(r * x for r, x in zip(res, xs)) / max(1.0, sum(abs(v) for v in xs)))
        upd("k 반올림 편차(표시용)", k - B["k"])

        # U 의 부호 규약: k>0 이면 U≥0
        if k > 0 and any(r["U"] is not None and r["U"] < -TOL_EXACT for r in rows):
            chk("A", f"{key} k>0 인데 U<0", False)

        # G 는 0~1 순위평균, 통과는 세 부호의 논리곱
        for r in rows:
            if r.get("G") is not None and not (-TOL_EXACT <= r["G"] <= 1 + TOL_EXACT):
                chk("A", f"{key} G 범위 이탈", False, f"{r['sector']} G={r['G']}")
            if r.get("G") is not None:
                want = bool(r["a_idx"] > 0 and r["x"] > 0 and r["xddot"] > 0) if r.get("xddot") is not None else False
                if bool(r.get("G_pass")) != want:
                    chk("A", f"{key} G_pass 불일치", False, r["sector"])
        # 얇은 섹터는 G 를 받지 않아야
        for r in B["rows"]:
            if r.get("thin") and r.get("G") is not None:
                chk("A", f"{key} 얇은 섹터에 G 부여", False, r["sector"])

    for name, v in worst.items():
        tol = 5e-4 if "반올림" in name else TOL_FIT
        chk("A", name, v < tol, f"최대오차 {v:.2e}")
    chk("A", "검사한 블록 수", blocks >= 8, f"{blocks}개")

    # 중앙차분 공식이 해석적으로 맞는가 — x=t² 에서 ẋ=2t, ẍ=2
    f = [0.0, 1.0, 4.0]
    chk("A", "1차 중앙차분 (x=t², t=1 → 2)", abs((f[2] - f[0]) / 2 - 2) < TOL_EXACT)
    chk("A", "2차 중앙차분 (x=t² → 2)", abs((f[2] - 2 * f[1] + f[0]) - 2) < TOL_EXACT)

    layer_a2(D)


def _spearman(a: list, b: list) -> float:
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db_ = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db_) if da and db_ else float("nan")


def layer_a2(D: dict) -> None:
    """A 층 보강 — **원 검사가 안 보던 열들.**

    A 층은 오래도록 `a·exp·x·U` 네 개의 항등식만 봤다. 그건 회귀 한 줄에서
    파생되는 값들이라 서로 맞을 수밖에 없고, 실제로 화면에서 가장 많은 칸을
    차지하는 `1년[%ile]`·`추이[8]`·`풀림`·`G`·`종합 G` 는 **한 번도 재계산되지
    않았다.** 여기서 그 빈 곳을 메운다.
    """
    worst = {}

    def upd(key, v):
        worst[key] = max(worst.get(key, 0.0), abs(v))

    # ① 포텐셜의 **부호 규약** — k ≤ 0 인 블록에서는 U 가 없어야 한다.
    #
    # U = ½kx² 의 뜻은 "|미실현| 이 클수록 크다" 하나인데, 그건 k>0 에서만
    # 성립한다. k<0 이면 |x| 의 순**감소** 변환이라 화면이 "포텐셜 큰 순" 이라
    # 말하면서 정확히 반대로 줄세운다(실측: 5일|코스닥 k=−0.587, 22개 섹터
    # 전부 U<0, ρ(|x|,U) = −1.0000). 도움말이 "4개 구간 전부 Spearman 1.0000"
    # 이라 적었던 것은 **시장=전체에서만 잰 값**이었다 — 실측 주장이 실측
    # 범위를 넘어 일반화된 자리다.
    bad_sign, bad_rho = [], []
    for key, B in D["blocks"].items():
        rows = [r for r in B["rows"] if r.get("x") is not None]
        if len(rows) < 3:
            continue
        us = [r.get("U") for r in rows]
        # k 는 표시용 반올림값이므로 부호 판정에만 쓴다(0 근처는 아래 ρ 로 잡힌다).
        if B["k"] <= 0:
            if any(u is not None for u in us):
                bad_sign.append(f"{key} k={B['k']} 인데 U 가 있다")
            continue
        if any(u is None for u in us):
            bad_sign.append(f"{key} k={B['k']}>0 인데 U 가 없다")
            continue
        if any(u < -TOL_EXACT for u in us):
            bad_sign.append(f"{key} k>0 인데 U<0")
        rho = _spearman([abs(r["x"]) for r in rows], us)
        if not (rho > 0.9999):
            bad_rho.append(f"{key} ρ(|x|,U)={rho:+.4f}")
    chk("A", "포텐셜: k≤0 인 블록에는 U 가 없다", not bad_sign, "; ".join(bad_sign[:3]))
    chk("A", "포텐셜: U 가 있는 블록은 |미실현| 과 Spearman +1",
        not bad_rho, "; ".join(bad_rho[:3]))

    # ② Σ미실현 = 0 — OLS 잔차의 부호 반전이므로 블록마다 정확히 0 이어야 한다.
    bad = []
    for key, B in D["blocks"].items():
        xs = [r["x"] for r in B["rows"] if r.get("x") is not None]
        if not xs:
            continue
        scale = max(abs(v) for v in xs) * len(xs)
        upd("Σ미실현 = 0", sum(xs) / max(scale, 1e-12))
        if abs(sum(xs)) > 1e-9 * max(scale, 1.0):
            bad.append(f"{key} Σx={sum(xs):.3e}")
    chk("A", "Σ미실현 = 0 (전 블록)", not bad, "; ".join(bad[:3]))

    # ③ 1년 백분위 — 정의역 [0,100] 이고 **주체마다 다른 값**이어야 한다.
    #    (한 행에 두 주체의 숫자가 섞이는 사고를 이 저장소가 이미 냈다.)
    bad, same = [], 0
    for key, B in D["blocks"].items():
        for r in B["rows"]:
            p = r.get("pct1y") or {}
            for a, v in p.items():
                if not (0.0 <= v <= 100.0):
                    bad.append(f"{key}/{r['sector']}/{a}={v}")
            if len(set(p.values())) == 1 and len(p) > 1:
                same += 1
    chk("A", "1년 백분위 ∈ [0,100]", not bad, "; ".join(bad[:3]))
    chk("A", "1년 백분위가 주체마다 갈리는가", same < 0.5 * 27 * len(D["blocks"]),
        f"4주체가 모두 같은 행 {same}개")

    # ④ 추이[8] — **조각 합이 그 주체의 임펄스와 같다**는 약속. 스파크라인의
    #    오른쪽 끝이 임펄스 열의 숫자로 이어진다는 말이 여기서 나온다.
    bad = []
    for key, B in D["blocks"].items():
        win = int(key.split("|")[0])
        segs = min(8, win)
        for r in B["rows"]:
            for a, arr in (r.get("spark") or {}).items():
                if len(arr) != segs:
                    bad.append(f"{key}/{r['sector']}/{a} 조각 {len(arr)} != {segs}")
                    continue
                # 조각은 0.1억으로 반올림된다 — 허용오차는 거기서 나온다.
                d = abs(sum(arr) - (r.get(a) or 0.0))
                upd("추이 Σ조각 − 임펄스 [억]", d)
                if d > 0.05 * segs + 1e-6:
                    bad.append(f"{key}/{r['sector']}/{a} Σ={sum(arr):.2f} vs {r.get(a)}")
    chk("A", "추이[8]: 조각 수 = min(8,창) · Σ조각 = 임펄스", not bad, "; ".join(bad[:3]))

    # ⑤ G — 세 순위의 평균이고, 각 순위는 pos/(N−1) 이다. 값이 [0,1] 인지만
    #    보던 것을 **순위 자체를 다시 매겨** 대조한다.
    bad = []
    for key, B in D["blocks"].items():
        scored = [r for r in B["rows"] if r.get("G") is not None]
        if len(scored) < 2:
            continue
        def _rank(k2):
            vals = [r[k2] for r in scored]
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            o = [0.0] * len(vals)
            for pos, i in enumerate(order):
                o[i] = pos / (len(vals) - 1)
            return o
        ra, rx, rd = _rank("a_idx"), _rank("x"), _rank("xddot")
        for r, A, X, Dd in zip(scored, ra, rx, rd):
            d = abs(r["G"] - (A + X + Dd) / 3.0)
            upd("G = 세 순위 평균", d)
            if d > TOL_FIT:
                bad.append(f"{key}/{r['sector']} G={r['G']} vs {(A+X+Dd)/3:.6f}")
    chk("A", "G = rank(a)·rank(x)·rank(ẍ) 의 평균", not bad, "; ".join(bad[:3]))

    # ⑥ 종합 G = 창별 G 의 **산술평균**, 통과[구간] = pass_n/seen.
    bad = []
    for mkey, C in (D.get("combined") or {}).items():
        wins = C.get("windows") or []
        for r in C["rows"]:
            per = {str(k2): v for k2, v in (r.get("per") or {}).items()}
            got = {}
            for w in wins:
                b = D["blocks"].get(f"{w}|{mkey}")
                row = next((q for q in (b or {}).get("rows", [])
                            if q["sector"] == r["sector"]), None)
                if row and row.get("G") is not None:
                    got[str(w)] = round(row["G"], 3)
            if per != got:
                bad.append(f"{mkey}/{r['sector']} per {per} vs {got}")
                continue
            if r.get("seen") != len(got):
                bad.append(f"{mkey}/{r['sector']} seen {r.get('seen')} vs {len(got)}")
            want = sum(got.values()) / len(got) if got else None
            if (r.get("G") is None) != (want is None):
                bad.append(f"{mkey}/{r['sector']} G {r.get('G')} vs {want}")
            elif want is not None:
                upd("종합 G = 창별 G 평균", r["G"] - want)
                if abs(r["G"] - want) > TOL_FIT:
                    bad.append(f"{mkey}/{r['sector']} G {r['G']} vs {want}")
    chk("A", "종합 G = G 가 나온 창의 등가중 평균", not bad, "; ".join(bad[:3]))

    for name, v in worst.items():
        print(f"       · {name} 최대오차 {v:.2e}")


# ─────────────────────────────────────────────── B. 데이터

def layer_b(payload: dict, db: str | None) -> None:
    print("\nB. 데이터 — DB 에서 들어온 것이 온전한가")
    dates = payload["dates"]
    chk("B", "거래일 수", len(dates) >= 60, f"{len(dates)}일")
    chk("B", "날짜 오름차순·중복 없음", dates == sorted(set(dates)))
    chk("B", "기준일 확정 플래그 존재", isinstance(payload.get("finalized"), bool),
        f"finalized={payload.get('finalized')}")

    n = len(dates)
    secs, mkts = payload["sectors"], payload["markets"]
    lens = {len(payload["flows"][m][s][k])
            for m in mkts for s in secs for k in ("inst", "forgn", "indiv", "etc", "tv")}
    chk("B", "모든 유량 배열 길이 = 거래일 수", lens == {n}, f"{lens}")

    # 4주체 항등식 — 합이 0 에 가까워야(기타법인 포함)
    tot = {k: sum(sum(payload["flows"][m][s][k]) for m in mkts for s in secs)
           for k in ("inst", "forgn", "indiv", "etc")}
    tv = sum(sum(payload["flows"][m][s]["tv"]) for m in mkts for s in secs)
    imb = sum(tot.values())
    chk("B", "4주체 순매수 합 ≈ 0", abs(imb) < 0.01 * tv,
        f"합 {imb:+,.0f}억 / 거래대금 {tv:,.0f}억 = {imb/tv*100:+.3f}%")

    # 시총·업종지수 커버리지
    cap_n = sum(1 for m in mkts for s in secs if payload["cap"].get(m, {}).get(s))
    idx_n = sum(1 for m in mkts for s in secs if payload["iret"].get(m, {}).get(s))
    chk("B", "섹터 시총 커버리지", cap_n >= len(secs), f"{cap_n}/{len(secs)*len(mkts)}")
    chk("B", "업종지수 커버리지", idx_n >= len(secs) * 0.6, f"{idx_n}/{len(secs)*len(mkts)}")

    # 종목
    nm = payload["names"]
    covered = {r["sector"] for r in nm.values()}
    chk("B", "종목이 전 섹터를 덮는가", covered >= set(secs) - {"(미분류)"},
        f"{len(covered)}/{len(secs)} 섹터, {len(nm)}종목")
    # 종목은 일별 배열이 아니라 **구간별 집계**를 싣는다(전 종목을 실으면서 바뀜).
    wins = {w for r in nm.values() for w in (r.get("win") or {})}
    chk("B", "종목 구간 집계 존재", bool(wins), f"구간 {sorted(wins, key=int)}")
    # 화면이 그리는 종목 값 — **새 열을 추가하면 여기에 이름을 적어라.**
    # producer 가 조용히 빠뜨리면 화면은 `—` 로 예쁘게 뜨고 아무도 모른다.
    # `invtrt`·`penfnd_etc` 는 기관 세부(투신·연기금), `ret` 은 구간 수익률로
    # 상대수익 열의 분자다.
    NAME_WIN_KEYS = {"inst", "forgn", "indiv", "etc", "tv",
                     "invtrt", "penfnd_etc", "ret"}
    lack = sorted({k for r in nm.values() for v in (r.get("win") or {}).values()
                   for k in NAME_WIN_KEYS - set(v)})
    chk("B", "종목 집계에 화면이 그리는 값이 다 있는가", not lack, f"없는 키 {lack}")
    # 투신·연기금이 **실제로 값을 갖는가.** 키만 있고 전부 0 이면 열이 두 줄
    # 늘고 정보는 0 이다(`natn` 이 정확히 그 모양이라 INST_DETAIL 에서 뺐다).
    for k in ("invtrt", "penfnd_etc"):
        live = sum(1 for r in nm.values()
                   if any((v.get(k) or 0) for v in (r.get("win") or {}).values()))
        chk("B", f"{k} 가 0 이 아닌 종목 수", live >= len(nm) * 0.2,
            f"{live}/{len(nm)}종목")
    # 종목 수익률은 등락률의 기하누적이라 −100% 아래로 못 내려간다.
    bad_ret = [f"{c}/{w}={(v or {}).get('ret'):.1f}"
               for c, r in nm.items() for w, v in (r.get("win") or {}).items()
               if (v or {}).get("ret") is not None
               and not (-100.0 <= (v or {}).get("ret") < 1e6)]
    chk("B", "종목 구간수익률이 −100% 아래로 안 내려가는가", not bad_ret,
        "; ".join(bad_ret[:3]))
    # 섹터 종목 수가 n_by_sector 와 맞는가 — 전 종목을 싣는다는 주장의 검사다.
    from collections import Counter
    cnt = Counter((r["market"], r["sector"]) for r in nm.values())
    bad = [f"{m}/{s}: names {cnt.get((m,s),0)} vs n_by_sector {v}"
           for m, d2 in payload.get("n_by_sector", {}).items()
           for s, v in d2.items() if cnt.get((m, s), 0) != v]
    chk("B", "섹터별 종목 수 = n_by_sector", not bad, "; ".join(bad[:2]))

    if not db:
        print("       (--db-check 없음 — DB 대조는 건너뛴다)")
        return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from kr_quant.storage import connect  # noqa: PLC0415
    con = connect(db)
    cur = con.cursor()
    cur.execute("SELECT max(date) FROM supply_demand")
    db_last = str(cur.fetchone()[0])
    chk("B", "페이로드 기준일 = DB 최신일", db_last == dates[-1], f"DB {db_last} / 리포트 {dates[-1]}")
    for tbl in ("supply_demand", "daily_bars_adjusted"):
        cur.execute(f"SELECT count(*) FROM (SELECT code,date FROM {tbl} "  # noqa: S608
                    f"GROUP BY code,date HAVING count(*)>1) q")
        chk("B", f"{tbl} (code,date) 중복", cur.fetchone()[0] == 0)
    cur.execute("SELECT count(*) FROM shares_outstanding_history WHERE shares_outstanding<=0")
    chk("B", "주식수 ≤ 0 행", cur.fetchone()[0] == 0)
    cur.execute("SELECT count(*) FROM daily_bars_adjusted WHERE close<=0 OR low>close OR high<close")
    chk("B", "OHLC 불변식 위반", cur.fetchone()[0] == 0)
    # 단위 sanity — 수량×종가 ≈ 보고된 거래대금
    cur.execute("""SELECT count(*) FILTER (WHERE err<0.05), count(*) FROM (
        SELECT abs(s.acc_trde_qty*abs(s.close)/1e6 / nullif(b.trade_value,0) - 1) err
        FROM supply_demand s JOIN daily_bars b ON b.code=s.code AND b.date=s.date
        WHERE s.date=(SELECT max(date) FROM supply_demand) AND b.trade_value>0) q""")
    good, total = cur.fetchone()
    chk("B", "수량×종가 ≈ 거래대금 (5% 이내)", total and good / total > 0.9,
        f"{good}/{total} = {good/total*100:.1f}%" if total else "표본 없음")
    con.close()


# ─────────────────────────────────────────────── C. 계산

def layer_d(D: dict, P: dict, html: str) -> None:
    """D. **부류 검사** — 오늘 나온 버그들이 전부 같은 모양이었다:
    *계산에 들어가는 두 값이 서로 다른 집합·시점·파라미터에서 나오는데 아무도
    검사하지 않는다.* 개별 증상이 아니라 그 부류를 막는다.

    실제 사례(전부 실측으로 확인됨):
      · 분모 시점 불일치 — 구간말이 1년 전인데 오늘 시총으로 나눔(중앙값 1.33배)
      · 분모 집합 불일치 — 수익률은 지수 있는 시장만, 시총은 모든 시장(겹침 0인 섹터 2개)
      · 표 헤더 ≠ 셀 개수 — 같은 실수가 두 번(열이 한 칸씩 밀림)
      · 라벨 ≠ 실제 계산 — 민감도 config 는 바뀌는데 시뮬레이션은 동일
    """
    print("\nD. 부류 — 두 값이 같은 집합·시점에서 나오는가")

    # ⓪ 화면이 그리는 키를 producer 가 실제로 싣는가.
    #
    # 이 저장소의 서명 같은 실패 모드다: producer 가 새 키를 조용히 빠뜨려도
    # 화면은 `—` 로 예쁘게 뜨고 아무도 모른다. 열이 있는데 전부 결측인 것과
    # 열이 아예 없는 것은 화면에서 구분되지 않는다.
    #
    # 새 열을 추가하면 **여기에 키를 등록하라.** 등록을 잊으면 그 열은
    # 조용히 빈 채로 매일 나온다.
    RENDERED_KEYS = ("inst", "forgn", "indiv", "etc",     # 주체 토글이 쓰는 넷
                     "accel", "pct1y", "spark", "x", "U", "P", "xddot",
                     "G", "ret", "cap", "n_all", "top")
    for bk, b in (D.get("blocks") or {}).items():
        rows = b.get("rows") or []
        if not rows:
            continue
        missing = [k for k in RENDERED_KEYS
                   if not any(k in r for r in rows)]
        chk("D", f"{bk} 화면이 그리는 키가 페이로드에 있는가",
            not missing, f"없는 키 {missing}" if missing else "")
        break        # 블록 구조는 동일하다 — 하나만 봐도 producer 누락은 잡힌다

    # ① 분모의 **시점**: 월별 시총이 구간말 월을 덮는가
    cbm = P.get("cap_by_month", {})
    if cbm:
        months = {m for mk in cbm.values() for sec in mk.values() for m in sec}
        want = {d[:7] for d in P["dates"]}
        chk("D", "월별 시총이 페이로드 전 구간의 월을 덮는가", want <= months,
            f"부족한 월 {sorted(want - months)[:3]}")
    else:
        chk("D", "월별 시총 계열 존재", False, "cap_by_month 없음 — 임의 구간 분모가 틀어진다")

    # ② 분모의 **집합**: 지수가 있는 시장과 시총 집계 시장이 같은가
    bad = []
    for m in P["markets"]:
        for sec in P["sectors"]:
            has_idx = bool(P["iret"].get(m, {}).get(sec))
            has_cap = bool(P["cap"].get(m, {}).get(sec))
            if has_idx and not has_cap:
                bad.append(f"{m}/{sec}")
    chk("D", "지수 있는 (시장,섹터)에 시총도 있는가", not bad, ", ".join(bad[:3]))

    # ③ **라벨 ≠ 계산**: 표에 실린 a 가 그 행의 임펄스/시총과 실제로 일치하는가
    #    (라벨만 바뀌고 값이 안 바뀌는 부류를 잡는다)
    mismatch = 0
    for B in D["blocks"].values():
        for r in B["rows"]:
            if r.get("a_idx") is None or not r.get("cap_idx"):
                continue
            if abs(r["a_idx"] - r["inst"] / r["cap_idx"] * 100) > TOL_EXACT:
                mismatch += 1
    chk("D", "표의 가속도 = 그 행의 임펄스/시총", mismatch == 0, f"{mismatch}건")

    # ④ **블록 간 값이 실제로 달라야 한다** — 창을 바꿨는데 값이 완전히 같으면
    #    파라미터가 계산에 도달하지 않는다는 신호다(민감도 격자가 그 모양이었다).
    keys = sorted(D["blocks"])
    same = []
    for i in range(len(keys) - 1):
        a, b = D["blocks"][keys[i]], D["blocks"][keys[i + 1]]
        if keys[i].split("|")[1] != keys[i + 1].split("|")[1]:
            continue
        va = [r.get("inst") for r in a["rows"]]
        vb = [r.get("inst") for r in b["rows"]]
        if va == vb:
            same.append(f"{keys[i]} == {keys[i+1]}")
    chk("D", "창이 다르면 값도 다른가(파라미터가 계산에 도달하는가)", not same,
        ", ".join(same[:2]))

    # ⑤ **개수의 집합**: 표가 적는 종목 수 = 그 줄에서 Enter 를 눌렀을 때 나오는
    #    목록의 길이. 이 층은 "분자와 분모가 같은 집합인가" 를 오래 봤는데
    #    **개수는 안 봤다.** 실측(수정 전): 표가 "전기/전자 387" 이라 적고 목록은
    #    386 개였다 — 두 달 전 수급이 끊긴 종목이 `stocks` 에 남아 있어서다.
    #    더 나쁜 건 `MIN_NAMES` 판정이 이 수로 이뤄져 **죽은 이름 하나가 얇은
    #    섹터 경고를 지운다**는 것이다(코스닥/종이/목재 10 vs 실제 9).
    names = D.get("names") or {}
    mkts_all = P["markets"]
    bad, flips = [], []
    for bk, B in (D.get("blocks") or {}).items():
        win, mk = bk.split("|")
        sel = mkts_all if mk == "전체" else [mk]
        listed: dict = {}
        for nm in names.values():
            if nm.get("market") in sel and (nm.get("win") or {}).get(win):
                listed[nm.get("sector")] = listed.get(nm.get("sector"), 0) + 1
        for r in B["rows"]:
            got = listed.get(r["sector"], 0)
            if got != r.get("n_all"):
                bad.append(f"{bk}/{r['sector']} 표 {r.get('n_all')} vs 목록 {got}")
                if (r.get("n_all", 0) < 10) != (got < 10):
                    flips.append(f"{bk}/{r['sector']}")
    chk("D", "표의 종목[수] = 드릴다운 목록 길이", not bad,
        f"{len(bad)}건 · ~마커 뒤집힘 {len(flips)}건  " + "; ".join(bad[:2]))

    # ⑥ **분자의 집합**: 구간말 시총이 없는(=상장이 끊긴) 종목의 순매수가
    #    임펄스에 얼마나 섞였나. 분모(섹터 시총)에는 그들이 없으므로 가속도가
    #    그만큼 오염된다. 실측(2026-08-28): 5·20일 창은 0.000%p, 60일 최대
    #    0.029%p, 120일 최대 0.142%p — 작지만 0 은 아니다. 임펄스는 "실제로
    #    흘러간 돈" 이라 빼지 않고, 대신 **오염 크기를 검사로 묶어둔다.**
    dead = {c for c, nm in names.items() if nm.get("cap") is None}
    worst_pol, where = 0.0, ""
    for bk, B in (D.get("blocks") or {}).items():
        win, mk = bk.split("|")
        sel = mkts_all if mk == "전체" else [mk]
        for r in B["rows"]:
            if not r.get("cap_idx"):
                continue
            s = 0.0
            for c in dead:
                nm = names[c]
                if nm.get("sector") != r["sector"] or nm.get("market") not in sel:
                    continue
                w = (nm.get("win") or {}).get(win)
                if w:
                    s += w.get("inst") or 0.0
            pol = abs(s / r["cap_idx"] * 100)
            if pol > worst_pol:
                worst_pol, where = pol, f"{bk}/{r['sector']}"
    chk("D", "상장이 끊긴 종목이 가속도를 오염시키는 폭 < 1%p", worst_pol < 1.0,
        f"최대 {worst_pol:.5f}%p ({where}) · 해당 종목 {len(dead)}개")


# ─────────────────────────────────────────────── E. 화면

def layer_e(D: dict) -> None:
    """E. **화면** — 페이로드가 맞아도 렌더가 틀릴 수 있다.

    A~D 는 전부 파일 안의 숫자만 본다. 그런데 이 저장소가 실제로 낸 사고 중
    여럿은 **그 숫자가 화면에 도달하는 길**에서 났다: 열 정의와 셀 개수가
    어긋나 한 칸씩 밀리고, 한글이 두 칸인 걸 잊어 줄이 접히고, 검사는 폭 80
    으로 묻는데 화면은 79 로 그렸다(`view_width` 가 그래서 생겼다).

    그래서 렌더가 낸 **문자열을 다시 파싱해** 열 정의가 내는 값과 대조한다.
    DB 가 없어도 돌고, 리포트 파일 하나만 있으면 된다.

    ⚠️ 종목명·섹터명은 폭에 맞춰 `pad` 가 자른다(설계다). **숫자 열은 하나도
    잘리면 안 된다** — `-1,360` 이 `-1` 로 보이는 것이 이 검사의 표적이다.
    """
    print("\nE. 화면 — 그려진 글자가 값과 같은가")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "src"))
    try:
        from kr_quant.tui import flow_view as FV  # noqa: PLC0415
    except ImportError as e:                      # pragma: no cover
        chk("E", "flow_view 를 읽을 수 있는가", False, str(e))
        return

    TEXT_COLS = {"섹터", "종목", "순매수상위[억]", "순매도상위[억]", ""}
    bad_w, bad_v, seen = [], [], 0

    def cells(line: str, widths: list[int], i: int) -> str:
        a, w = FV.span_at(widths, i)
        cur, buf = 0, []
        for ch in line:
            if a <= cur < a + w:
                buf.append(ch)
            cur += FV.cell_width(ch)
        return "".join(buf).strip()

    for term in (80, 120, 200):
        width = FV.view_width(term)
        for wi in range(len(FV.WINDOWS)):
            for mi in range(3):
                st = FV.State(D)
                st.wi = wi
                if mi >= len(st.markets):
                    continue
                st.mi = mi
                seen += 1
                lines, _thin, _nh = FV.table_lines(st, width, 50)
                for ln in lines:
                    if FV.cell_len(ln) != width:
                        bad_w.append(f"표 {term}/{FV.WINDOWS[wi]}/{st.market} "
                                     f"{FV.cell_len(ln)} != {width}")
                cols = FV._fit(FV.table_cols(st, width), width)
                widths = [c.width for c in cols]
                for ln, r in zip(lines[1:], st.rows()):
                    for i, c in enumerate(cols):
                        want = c.fn(r, st).strip()
                        if c.header in TEXT_COLS:
                            continue
                        if cells(ln, widths, i) != want:
                            bad_v.append(f"{term}/{FV.WINDOWS[wi]}/{st.market} "
                                         f"{c.header!r} {cells(ln, widths, i)!r} != {want!r}")
                rows = st.rows()
                for ri in ({0, len(rows) // 2, len(rows) - 1} if rows else set()):
                    st.row = ri
                    nl, _ = FV.names_lines(st, width)
                    for ln in nl:
                        if FV.cell_len(ln) != width:
                            bad_w.append(f"종목 {term}/{FV.WINDOWS[wi]}/{st.market} "
                                         f"{FV.cell_len(ln)} != {width}")
                    ncols = FV._fit(FV.names_cols(), width)
                    nwid = [c.width for c in ncols]
                    for ln, t in zip(nl[2:], st.names()):
                        for i, c in enumerate(ncols):
                            if c.header in TEXT_COLS:
                                continue
                            want = c.fn(t, st).strip()
                            if cells(ln, nwid, i) != want:
                                bad_v.append(f"{term}/종목 {c.header!r} "
                                             f"{cells(ln, nwid, i)!r} != {want!r}")
                for ln in FV.header_lines(st, width) + FV.detail_lines(st, width):
                    if FV.cell_len(ln) != width:
                        bad_w.append(f"헤더/상세 {term} {FV.cell_len(ln)} != {width}")
    chk("E", "모든 줄의 표시 폭 = view_width(터미널−1)", not bad_w,
        f"{len(bad_w)}건  " + "; ".join(bad_w[:2]))
    chk("E", "숫자 열: 그려진 글자 = 열 정의가 내는 값 (폭 80·120·200)", not bad_v,
        f"{len(bad_v)}건  " + "; ".join(bad_v[:2]))
    chk("E", "검사한 (폭,창,시장) 조합", seen >= 30, f"{seen}개")

    # 종목 목록의 누적[%] — 단조증가하고 100 에서 끝난다.
    bad = []
    for wi in range(len(FV.WINDOWS)):
        st = FV.State(D)
        st.wi = wi
        for ri in range(len(st.rows())):
            st.row = ri
            arr = [t["cum"] for t in st.names() if t.get("cum") is not None]
            if not arr:
                continue
            if any(b < a - 1e-9 for a, b in zip(arr, arr[1:])):
                bad.append(f"{FV.WINDOWS[wi]}/{ri} 단조증가 아님")
            if abs(arr[-1] - 100.0) > 1e-6:
                bad.append(f"{FV.WINDOWS[wi]}/{ri} 끝값 {arr[-1]}")
    chk("E", "종목 누적[%] 이 단조증가하고 100 에서 끝나는가", not bad,
        "; ".join(bad[:3]))

    # 상대수익의 기준선이 **섹터 표의 그 수익률**과 같은 값인가.
    bad = []
    for wi, w in enumerate(FV.WINDOWS):
        if w == "종합":
            continue
        st = FV.State(D)
        st.wi = wi
        for ri, r in enumerate(st.rows()):
            st.row = ri
            sret = r.get("ret")
            for t in st.names()[:5]:
                if t.get("rrel") is None or t.get("ret") is None:
                    continue
                if abs((t["ret"] - sret) - t["rrel"]) > TOL_EXACT:
                    bad.append(f"{w}/{r['sector']}/{t['code']}")
    chk("E", "상대수익 = 종목 수익률 − 섹터 표의 수익률", not bad, "; ".join(bad[:3]))


def layer_c(D: dict, html: str) -> None:
    print("\nC. 계산 — 표의 구조와 값이 정합한가")
    # ⚠️ 표가 둘 이상이므로 `rows.forEach` 로 찾으면 안 된다 — 종합 표(ctbl)가
    # 먼저 잡혀 본표(tbl)의 열 수와 어긋난다(실제로 한 번 오탐이 났다).
    # 각 표의 getElementById(...) 부터 t.append(tb) 까지를 구간으로 잡는다.
    def _cells(table_id: str) -> int:
        m = re.search(
            r'getElementById\("' + table_id + r'"\)(.*?)\n\s*t\.append\(tb\)',
            html, re.S)
        return len(re.findall(r"x\.append\(", m.group(1))) if m else -1

    cols = re.search(r"const COLS=\[(.*?)\];", html, re.S)
    n_cols = len(re.findall(r'^\s*\["', cols.group(1), re.M)) if cols else -1
    chk("C", "본표: 열 정의 수 = 셀 append 수", n_cols == _cells("tbl"),
        f"COLS {n_cols} vs 셀 {_cells('tbl')}")

    # 종합 표는 열 수가 창 수만큼 가변이다. 고정 셀과 루프 셀을 나눠 센 뒤,
    # 헤더 배열의 고정 항목 수(섹터·종목·종합G + 통과창)와 맞는지 본다.
    m = re.search(r'getElementById\("ctbl"\)(.*?)\n\s*t\.append\(tb\)', html, re.S)
    if m:
        body_c = m.group(1)
        loop = re.findall(r"windows\.forEach\([^)]*=>\s*x\.append\(", body_c)
        fixed = len(re.findall(r"^\s*x\.append\(", body_c, re.M))
        hdr = re.search(r'\[("섹터".*?)\]\.forEach', body_c, re.S)
        # 헤더 배열 안의 `...C.windows.map(w=>w+"일 G")` 는 **루프 라벨**이므로
        # 고정 항목에서 뺀다 — 안 빼면 그 안의 따옴표 문자열까지 세어 1 이 더 나온다.
        hdr_txt = re.sub(r"\.\.\.[^,]*?\.map\([^)]*\)", "", hdr.group(1)) if hdr else ""
        n_hdr_fixed = len(re.findall(r'"[^"]+"', hdr_txt)) if hdr else -1
        wins = len(next(iter(D.get("combined", {}).values()), {}).get("windows", []))
        chk("C", "종합표: 고정 헤더 = 고정 셀", n_hdr_fixed == fixed,
            f"헤더 고정 {n_hdr_fixed} · 셀 고정 {fixed}")
        chk("C", "종합표: 창 루프 정확히 1개", len(loop) == 1, f"창 {wins}개")
    chk("C", "템플릿 자리표시자 없음", "__DATA__" not in html)

    for key, B in D["blocks"].items():
        rows = B["rows"]
        chk("C", f"{key} 섹터 중복 없음",
            len({r["sector"] for r in rows}) == len(rows))
        bad = [r["sector"] for r in rows
               if any(isinstance(r.get(k), float) and not math.isfinite(r[k])
                      for k in ("inst", "accel", "ret", "exp", "x", "U", "P", "xdot", "xddot", "G"))]
        chk("C", f"{key} Inf/NaN 없음", not bad, ", ".join(bad[:3]))
        break_after = True
        if break_after:
            pass
    # 전 블록 Inf/NaN 스캔(위는 첫 블록만 상세, 여기서 전수)
    allbad = [(key, r["sector"]) for key, B in D["blocks"].items() for r in B["rows"]
              if any(isinstance(r.get(k), float) and not math.isfinite(r[k])
                     for k in ("inst", "accel", "ret", "exp", "x", "U", "P", "xdot", "xddot", "G"))]
    chk("C", "전 블록 Inf/NaN 없음", not allbad, f"{len(allbad)}건")


# ─────────────────────────────────────────────── F. 독립 재계산

#: 페이로드가 반올림하는 자릿수에서 나오는 허용오차.
#: 유량·거래대금은 0.01억(round(...,2)), 시총·가중은 0.1억, 수익률은 0.001%.
#: **상대오차를 쓰지 않는다** — 섹터 합계가 0 근처인 칸(섬유/의류 +2.0억)에서
#: 상대오차는 무의미하게 커진다. 반올림 폭에서 유도한 절대오차가 맞는 축이다.
TOL_FLOW = 0.011
TOL_CAP = 0.11
TOL_RET = 0.0011


def layer_f(P: dict, db: str | None) -> None:
    """F. **독립 재계산** — DB 원자료에서 페이로드를 처음부터 다시 만든다.

    A~E 는 전부 *페이로드 안의 값끼리* 정합한지를 본다. 그래서 `sector_flow.py`
    가 임펄스를 통째로 잘못 계산해도(예: 수량을 금액으로 착각, `flu_rt` 의 bp
    단위를 놓침 — 둘 다 이 저장소에서 실제로 사고를 낸 함정이다) A~E 는 전부
    초록이다. 파생층이 그 틀린 값 위에서 일관될 뿐이기 때문이다.

    그래서 여기서는 **`scripts/` 의 함수를 하나도 부르지 않고** SQL 로 다시
    집계해 대조한다. 같은 버그를 두 번 통과시키지 않으려면 경로가 달라야 한다.

    ⚠️ 두 함정을 재현 쪽에서도 명시한다.
      · `supply_demand` 의 순매매는 **수량**이다 — 금액은 그날 종가를 곱해 만든다.
      · `flu_rt` 는 **bp**(등락률×100)다 — 100 으로 나눠야 퍼센트다.
    """
    print("\nF. 독립 재계산 — DB 원자료에서 다시 만든 값과 같은가")
    if not db:
        print("       (--db-check 없음 — 건너뛴다)")
        return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "src"))
    from kr_quant.storage import connect  # noqa: PLC0415
    con = connect(db)
    pg = con.__class__.__module__.startswith("psycopg2")
    ph = "%s" if pg else "?"
    cur = con.cursor()
    dates = P["dates"]
    d0, d1 = dates[0], dates[-1]
    n = len(dates)
    SEC = "coalesce(nullif(st.sector,''),'(미분류)')"
    SRC = f"s.source={ph} AND s.date BETWEEN {ph} AND {ph}"
    args = ("kiwoom", d0, d1)

    def cmp_abs(name, pairs, tol):
        """(라벨, 페이로드값, 재계산값) 을 절대오차로 본다."""
        bad, worst = [], 0.0
        for label, x, y in pairs:
            if x is None or y is None:
                if (x is None) != (y is None):
                    bad.append(f"{label} {x} vs {y}")
                continue
            worst = max(worst, abs(x - y))
            if abs(x - y) > tol:
                bad.append(f"{label} {x} vs {y}")
        chk("F", name, not bad,
            f"{len(bad)}/{len(pairs)} 불일치 · 최대오차 {worst:.3e}"
            + ("  " + "; ".join(bad[:2]) if bad else ""))

    # ── ① 유량 (시장,섹터,날짜) × 5값 ──
    ACT = (("indiv", "individual"), ("forgn", "foreign_"),
           ("inst", "institution"), ("etc", "etc_corp"))
    sel = ", ".join([f"sum(s.{c} * 1.0 * abs(s.close))" for _k, c in ACT]
                    + ["sum(s.acc_trde_qty * 1.0 * abs(s.close))"])
    cur.execute(f"SELECT st.market, {SEC}, s.date, {sel} "  # noqa: S608
                f"FROM supply_demand s JOIN stocks st ON st.code=s.code "
                f"WHERE {SRC} GROUP BY 1,2,3", args)
    keys = [k for k, _ in ACT] + ["tv"]
    got = {}
    for row in cur.fetchall():
        for j, k in enumerate(keys):
            got[(row[0], row[1], str(row[2]), k)] = float(row[3 + j]) / 1e8
    pairs = []
    for m in P["markets"]:
        for s in P["sectors"]:
            for k in keys:
                arr = P["flows"][m][s][k]
                for i, v in enumerate(arr):
                    pairs.append((f"{m}/{s}/{k}/{dates[i]}", v,
                                  got.get((m, s, dates[i], k), 0.0)))
    cmp_abs("유량 일별 (시장×섹터×5값×거래일)", pairs, TOL_FLOW)

    # ── ② 기관 세부 ──
    det = sorted({k for m in P["detail"] for s in P["detail"][m]
                  for k in P["detail"][m][s]})
    if det:
        sel = ", ".join(f"sum(s.{c} * 1.0 * abs(s.close))" for c in det)
        cur.execute(f"SELECT st.market, {SEC}, s.date, {sel} "  # noqa: S608
                    f"FROM supply_demand s JOIN stocks st ON st.code=s.code "
                    f"WHERE {SRC} GROUP BY 1,2,3", args)
        got = {}
        for row in cur.fetchall():
            for j, k in enumerate(det):
                got[(row[0], row[1], str(row[2]), k)] = float(row[3 + j]) / 1e8
        pairs = []
        for m in P["markets"]:
            for s in P["sectors"]:
                for k in det:
                    for i, v in enumerate(P["detail"][m][s][k]):
                        pairs.append((f"{m}/{s}/{k}/{dates[i]}", v,
                                      got.get((m, s, dates[i], k), 0.0)))
        cmp_abs("기관 세부 일별", pairs, TOL_FLOW)

    # ── ③ 섹터 시총(구간말) — daily_bars.close × 상장주식수(asof) ──
    #    ⚠️ `supply_demand.close` 가 아니다. 이 DB 에서 둘은 같은 계열이 아니다
    #    (storage.market_cap_asof_bulk 가 실측으로 확인한 함정).
    CAP = ("b.close * 1.0 * (SELECT h.shares_outstanding "
           "FROM shares_outstanding_history h "
           "WHERE h.code=q.code AND h.date<=b.date ORDER BY h.date DESC LIMIT 1)")
    cur.execute(f"SELECT st.market, {SEC}, sum({CAP}) "   # noqa: S608
                f"FROM (SELECT DISTINCT s.code FROM supply_demand s "
                f"      JOIN stocks st2 ON st2.code=s.code WHERE {SRC}) q "
                f"JOIN stocks st ON st.code=q.code "
                f"JOIN daily_bars b ON b.code=q.code AND b.date={ph} "
                f"GROUP BY 1,2", (*args, d1))
    got = {(r[0], r[1]): float(r[2]) / 1e8 for r in cur.fetchall()}
    pairs = [(f"{m}/{s}", P["cap"].get(m, {}).get(s), got.get((m, s)))
             for m in P["markets"] for s in P["sectors"]
             if P["cap"].get(m, {}).get(s) is not None or (m, s) in got]
    cmp_abs("섹터 시총 (구간말)", pairs, TOL_CAP)

    # ── ④ 섹터 일별 수익률 — **전일** 시총 가중. flu_rt 는 bp 다. ──
    lag = ("lag(cap) OVER (PARTITION BY code ORDER BY date)" if pg
           else "lag(cap) OVER (PARTITION BY code ORDER BY date)")
    cur.execute(f"""
      WITH cd AS (
        SELECT s.code, s.date, s.flu_rt,
               b.close * (SELECT h.shares_outstanding
                          FROM shares_outstanding_history h
                          WHERE h.code=s.code AND h.date<=s.date
                          ORDER BY h.date DESC LIMIT 1) AS cap
        FROM supply_demand s
        JOIN stocks st ON st.code=s.code
        LEFT JOIN daily_bars b ON b.code=s.code AND b.date=s.date
        WHERE {SRC}),
           w AS (SELECT code, date, flu_rt, {lag} AS cap_lag FROM cd)
      SELECT st.market, {SEC}, w.date,
             sum(w.flu_rt/100.0 * w.cap_lag/1e8), sum(w.cap_lag/1e8)
      FROM w JOIN stocks st ON st.code=w.code
      WHERE w.cap_lag IS NOT NULL
      GROUP BY 1,2,3""", args)  # noqa: S608
    gr, gw = {}, {}
    for m, s, dt, num, den in cur.fetchall():
        gr[(m, s, str(dt))] = float(num) / float(den) if den else 0.0
        gw[(m, s, str(dt))] = float(den)
    pairs_r, pairs_w = [], []
    for m in P["markets"]:
        for s in P["sectors"]:
            for i, d in enumerate(dates):
                pairs_r.append((f"{m}/{s}/{d}", P["ret"][m][s][i], gr.get((m, s, d), 0.0)))
                pairs_w.append((f"{m}/{s}/{d}", P["retw"][m][s][i], gw.get((m, s, d), 0.0)))
    cmp_abs("섹터 일별 수익률 (전일 시총 가중)", pairs_r, TOL_RET)
    cmp_abs("섹터 일별 가중합", pairs_w, TOL_CAP)

    # ── ⑤ 종목 창별 집계 ──
    wins = sorted({int(w) for r in P["names"].values() for w in (r.get("win") or {})})
    WK = [("inst", "institution"), ("forgn", "foreign_"), ("indiv", "individual"),
          ("etc", "etc_corp"), ("tv", "acc_trde_qty"),
          ("invtrt", "invtrt"), ("penfnd_etc", "penfnd_etc")]
    for win in wins:
        if win > n:
            continue
        start = dates[n - win]
        sel = ", ".join(f"sum(s.{c} * 1.0 * abs(s.close))" for _k, c in WK)
        cur.execute(f"SELECT s.code, {sel} FROM supply_demand s "  # noqa: S608
                    f"JOIN stocks st ON st.code=s.code WHERE {SRC} GROUP BY 1",
                    ("kiwoom", start, d1))
        got = {r[0]: [float(v) / 1e8 for v in r[1:]] for r in cur.fetchall()}
        pairs = []
        for c, r in P["names"].items():
            w = (r.get("win") or {}).get(str(win))
            if not w:
                continue
            g = got.get(c)
            for j, (k, _col) in enumerate(WK):
                pairs.append((f"{c}/{win}/{k}", w.get(k),
                              None if g is None else g[j]))
        cmp_abs(f"종목 {win}일 집계", pairs, TOL_CAP)

    # 종목 구간 수익률 — 일별 등락률의 **기하** 누적(결측일은 0).
    cur.execute(f"SELECT s.code, s.date, s.flu_rt FROM supply_demand s "  # noqa: S608
                f"JOIN stocks st ON st.code=s.code WHERE {SRC}", args)
    fr = {}
    for c, dt, v in cur.fetchall():
        fr[(c, str(dt))] = float(v) / 100.0
    pairs = []
    for win in wins:
        if win > n:
            continue
        span = dates[n - win:]
        for c, r in P["names"].items():
            w = (r.get("win") or {}).get(str(win))
            if not w or w.get("ret") is None:
                continue
            acc = 1.0
            for d in span:
                acc *= 1.0 + fr.get((c, d), 0.0) / 100.0
            pairs.append((f"{c}/{win}/ret", w["ret"], (acc - 1.0) * 100.0))
    cmp_abs("종목 구간 수익률 (기하누적)", pairs, 0.011)

    # ── ⑥ 유니버스의 **폭** — 이 화면이 못 보는 종목이 몇 개인가. ──
    #
    # 이 리포트는 개인·기관세부가 키움 소스에만 있어 `sources=("kiwoom",)` 로
    # 유니버스를 좁히고, `stocks`(현재 상장 마스터)와 INNER JOIN 한다. 그래서
    # 구간 안에 거래가 있었지만 지금은 마스터에 없는 종목이 통째로 빠진다.
    # 이건 심사 성적이 아니라 **관측 화면의 한계**라 실패로 만들지 않는다 —
    # 다만 조용히 두지 않는다. `docs/GUARDRAILS.md` §4.1 이 다루는 생존편향의
    # 거울상이고, 폭이 갑자기 커지면 수집 쪽에 문제가 생겼다는 신호다.
    cur.execute(f"SELECT count(DISTINCT s.code) FROM supply_demand s "  # noqa: S608
                f"WHERE s.date BETWEEN {ph} AND {ph} AND s.code NOT IN "
                f"(SELECT code FROM stocks)", (d0, d1))
    outside = cur.fetchone()[0]
    inside = len(P["names"])
    chk("F", "유니버스 밖(마스터에 없는) 종목 비중 < 20%",
        outside < 0.2 * max(inside, 1),
        f"밖 {outside}종목 / 안 {inside}종목 = {outside/max(inside,1)*100:.1f}%")
    con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="리포트 폴더(numbers.html·payload.json)")
    ap.add_argument("--db-check", action="store_true")
    ap.add_argument("--db", default=os.environ.get("KR_QUANT_DB"))
    a = ap.parse_args()

    d = os.path.expanduser(a.dir)
    html = open(os.path.join(d, "numbers.html"), encoding="utf-8").read()
    D = json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1))
    payload = json.load(open(os.path.join(d, "payload.json"), encoding="utf-8"))

    print(f"검증 대상: {d}")
    print(f"  기준일 {payload['dates'][-1]} · 거래일 {len(payload['dates'])} · "
          f"섹터 {len(payload['sectors'])} · 블록 {len(D['blocks'])}")
    layer_a(D)
    layer_b(payload, a.db if a.db_check else None)
    layer_c(D, html)
    layer_d(D, payload, html)
    layer_e(D)
    if a.db_check:
        layer_f(payload, a.db)

    print(f"\n검사 {CHECKS}건 · 실패 {len(FAILS)}건")
    if FAILS:
        print("실패 목록: " + ", ".join(FAILS))
        return 1
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
