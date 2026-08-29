#!/usr/bin/env python
"""일일 리포트 검증 — 수식·데이터·계산을 한 번에 점검한다.

세 층을 나눠서 본다. 층마다 실패 원인이 다르고 대응도 다르기 때문이다.

  A. 수식   — 표에 실린 값들이 정의된 항등식을 만족하는가(전정밀도 재적합 기준)
  B. 데이터 — DB 에서 들어온 것이 온전한가(기준일·커버리지·항등식·중복·단위)
  C. 계산   — 페이로드가 독립 재계산과 일치하는가, 표의 열이 정합한가

⚠️ **반올림된 값으로 검산하지 않는다.** payload 의 k 는 소수 3자리라, 그걸로
U=½kx² 를 검산하면 1e-3 오차가 나서 멀쩡한 수식이 FAIL 로 뜬다(실제로 한 번
밟았다). 원자료에서 전정밀도로 다시 적합해 비교한다.

Run:  python scripts/verify_report.py --dir ~/Documents/kr-quant-reports/2026-08-27
      python scripts/verify_report.py --dir <폴더> --db-check     # B 층까지
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
        upd("U = ½·k·x²", max(r["U"] - 0.5 * k * r["x"] ** 2 for r in rows))
        res = [y - (k * x + b) for x, y in zip(xs, ys)]
        upd("OLS: Σ잔차 = 0", sum(res) / len(res))
        upd("OLS: 잔차 ⊥ a", sum(r * x for r, x in zip(res, xs)) / max(1.0, sum(abs(v) for v in xs)))
        upd("k 반올림 편차(표시용)", k - B["k"])

        # U 의 부호 규약: k>0 이면 U≥0
        if k > 0 and any(r["U"] < -TOL_EXACT for r in rows):
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

    print(f"\n검사 {CHECKS}건 · 실패 {len(FAILS)}건")
    if FAILS:
        print("실패 목록: " + ", ".join(FAILS))
        return 1
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
