"""섹터 자금흐름 TUI 의 **순수 렌더 로직** — curses 를 import 하지 않는다.

화면 그리기(curses)와 무엇을 그릴지(여기)를 나눈다. 이 파일은 데이터와 상태를
받아 문자열 행 목록을 돌려주므로 단위 테스트가 된다. curses 를 섞으면 렌더가
터미널 없이는 검증 불가능해지고, 열 정렬이 어긋나도 아무도 못 잡는다 —
이 저장소는 그 실수를 이미 두 번 했다(표 헤더와 셀 개수 불일치).
"""

from __future__ import annotations

import unicodedata

WINDOWS = ("5", "20", "60", "120", "종합")
# 종목 목록은 **절대 순매수 금액** 순이다 — 섹터 합계가 금액의 합이므로 기여도는
# 금액으로만 정의된다. 시총 대비는 참고 열. (시총 대비로 줄세웠더니 스팩이 1위가
# 됐고, 그걸 막으려 유동성 하한을 넣었더니 작은 섹터가 통째로 비었다.)
ACTORS = (("inst", "기관"), ("forgn", "외국인"), ("indiv", "개인"), ("etc", "기타법인"))
SORTS = (("G", "성장"), ("inst", "임펄스"), ("accel", "가속"), ("ret", "수익률"),
         ("x", "미실현"), ("n_all", "종목수"))


def fmt_amt(v) -> str:
    if v is None:
        return "—"
    return ("+" if v >= 0 else "-") + f"{abs(v):,.0f}"


def fmt_pct(v, nd: int = 2) -> str:
    if v is None:
        return "—"
    return ("+" if v >= 0 else "-") + f"{abs(v):.{nd}f}"


def cell_width(ch: str) -> int:
    """문자 하나의 터미널 표시 폭.

    ``ord(c) > 0x1100`` 같은 어림은 쓰지 않는다 — U+2212(−, 마이너스)가 그 범위에
    들어가 2칸으로 세어졌고, 음수 행이 한 칸씩 밀렸다. 유니코드 표준
    East Asian Width 가 'W'(Wide)·'F'(Fullwidth)인 것만 2칸이다.
    """
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _w(text: str) -> int:
    """표시 폭 — 한글·전각은 두 칸."""
    return sum(cell_width(c) for c in text)


def pad(text: str, width: int, right: bool = False) -> str:
    """표시 폭 기준 패딩(한글 2칸). 넘치면 자른다."""
    out = ""
    used = 0
    for c in text:
        cw = cell_width(c)
        if used + cw > width:
            break
        out += c
        used += cw
    space = " " * (width - used)
    return (space + out) if right else (out + space)


class State:
    """화면 상태 — 어떤 창·시장·주체·정렬로 무엇을 선택했나."""

    def __init__(self, data: dict):
        self.d = data
        self.markets = ["전체"] + list(
            {k.split("|")[1] for k in data["blocks"]} - {"전체"})
        self.wi = WINDOWS.index("20") if "20" in WINDOWS else 0
        self.mi = 0
        self.ai = 0
        self.si = 0
        self.row = 0
        self.drill = False      # 종목 목록 화면인가
        self.drow = 0           # 종목 목록에서 선택된 행

    # --- 현재 선택 ---
    @property
    def window(self) -> str:
        return WINDOWS[self.wi]

    @property
    def market(self) -> str:
        return self.markets[self.mi]

    @property
    def actor(self) -> str:
        return ACTORS[self.ai][0]

    @property
    def sort_key(self) -> str:
        return SORTS[self.si][0]

    def cycle(self, what: str, step: int = 1) -> None:
        if what == "w":
            self.wi = (self.wi + step) % len(WINDOWS)
        elif what == "m":
            self.mi = (self.mi + step) % len(self.markets)
        elif what == "a":
            self.ai = (self.ai + step) % len(ACTORS)
        elif what == "s":
            self.si = (self.si + step) % len(SORTS)
        self.row = 0

    # --- 데이터 ---
    def rows(self) -> list[dict]:
        if self.window == "종합":
            c = self.d.get("combined", {}).get(self.market)
            return list(c["rows"]) if c else []
        b = self.d["blocks"].get(f"{self.window}|{self.market}")
        if not b:
            return []
        rows = list(b["rows"])
        key = self.sort_key
        def val(r):
            v = r.get(key)
            return (v is None, -(v if v is not None else 0))
        rows.sort(key=val)
        return rows

    def block_meta(self) -> str:
        if self.window == "종합":
            c = self.d.get("combined", {}).get(self.market)
            return f"창 {'·'.join(str(w) for w in c['windows'])}일 등가중" if c else ""
        b = self.d["blocks"].get(f"{self.window}|{self.market}")
        return f"{b['from']} ~ {b['to']} · k={b['k']} (t={b['t']})" if b else ""


    def selected(self) -> dict | None:
        rows = self.rows()
        return rows[min(self.row, len(rows) - 1)] if rows else None

    def names(self) -> list[dict]:
        """선택 섹터의 **전 종목** — 시총 대비 순매수(%p) 큰 순.

        페이로드가 프리셋 구간별 집계를 싣는다. 이전엔 (시장,섹터)당 12개만 미리
        뽑아 실어서 "그 12개 안에서의 순위"를 보여줬고, 20일 기준 각 섹터 상위 3 중
        32% 가 화면에 없었다.
        """
        r = self.selected()
        if not r:
            return []
        sec = r.get("sector")
        win = self.window if self.window != "종합" else "20"
        mkts = ([self.market] if self.market != "전체" else self.markets[1:])
        out = []
        for code, nm in (self.d.get("names") or {}).items():
            if nm.get("sector") != sec or nm.get("market") not in mkts:
                continue
            w = (nm.get("win") or {}).get(win)
            if not w:
                continue
            cap = nm.get("cap")
            out.append({"code": code, "name": nm.get("name", "—"),
                        "inst": w.get("inst"), "tv": w.get("tv"), "cap": cap,
                        "a": (w["inst"] / cap * 100) if cap else None})
        out.sort(key=lambda t: -(t["inst"] or 0))       # 절대 기여도 순
        return out


def header_lines(st: State, width: int) -> list[str]:
    d = st.d
    chip = "확정" if d.get("finalized") else "장중·미확정"
    l1 = f" 섹터 자금 흐름 · {d['asof']} {chip}"
    l2 = (f" 구간[{st.window}] 시장[{st.market}] 주체[{ACTORS[st.ai][1]}]"
          f" 정렬[{SORTS[st.si][1]}]")
    return [pad(l1, width), pad(l2 + "  " + st.block_meta(), width)]


def table_lines(st: State, width: int, height: int) -> tuple[list[str], list[bool], int]:
    """(행 문자열, 얇은섹터 여부, 헤더 줄 수). 폭에 따라 열을 줄인다."""
    wide = width >= 100
    if st.window == "종합":
        cols = [("섹터", 11, False), ("종목", 4, True), ("G", 5, True), ("", 1, False)]
        wins = (st.d.get("combined", {}).get(st.market) or {}).get("windows", [])
        for w in wins:
            cols.append((f"{w}일", 6, True))
        cols.append(("통과", 6, True))
    else:
        cols = [("섹터", 11, False), ("종목", 4, True), ("G", 5, True), ("", 1, False),
                ("임펄스", 10, True), ("가속", 7, True), ("수익률", 8, True)]
        if wide:
            cols += [("미실현", 8, True), ("포텐셜", 8, True), ("견인주", 22, False)]

    head = " ".join(pad(c[0], c[1], c[2]) for c in cols)
    out = [pad(head, width)]
    thin = [False]

    for r in st.rows():
        cells = [pad(r.get("sector", "—"), 11),
                 pad(str(r.get("n_all", "—")), 4, True),
                 pad(f"{r['G']:.2f}" if r.get("G") is not None else "—", 5, True),
                 # 마커는 **별도 1칸 열**이다. 값에 붙이면 ● 가 2칸이라 열이 밀린다.
                 pad("*" if r.get("G_pass") else "", 1)]
        if st.window == "종합":
            per = r.get("per", {})
            for w in wins:
                v = per.get(str(w), per.get(w))
                cells.append(pad(f"{v:.2f}" if v is not None else "—", 6, True))
            cells.append(pad(f"{r.get('pass_n',0)}/{r.get('seen',0)}", 6, True))
        else:
            cells += [pad(fmt_amt(r.get(st.actor if st.actor != 'inst' else 'inst')), 10, True),
                      pad(fmt_pct(r.get("accel")), 7, True),
                      pad(fmt_pct(r.get("ret")), 8, True)]
            if wide:
                top = (r.get("top") or {}).get("buy") or []
                names = " ".join(t["name"] for t in top[:2])
                cells += [pad(fmt_pct(r.get("x"), 1), 8, True),
                          pad(f"{r['U']:.0f}" if r.get("U") is not None else "—", 8, True),
                          pad(names, 22)]
        out.append(pad(" ".join(cells), width))
        thin.append(bool(r.get("thin")))
    return out, thin, 1


def detail_lines(st: State, width: int) -> list[str]:
    rows = st.rows()
    if not rows:
        return [pad(" (데이터 없음)", width)]
    r = rows[min(st.row, len(rows) - 1)]
    top = r.get("top") or {}
    def side(key, label):
        arr = top.get(key) or []
        if not arr:
            return f" {label}: —"
        parts = []
        for t in arr[:3]:
            a = t.get("a")
            parts.append(f"{t['name']} {fmt_pct(a)}%p" if a is not None
                         else f"{t['name']} {fmt_amt(t['inst'])}")
        return f" {label}: " + " · ".join(parts)
    n = (r.get("top") or {}).get("n", 0)
    return [pad(f" {r.get('sector','—')} · 종목 {n}개 · Enter 로 전체", width),
            pad(side("buy", "기관 순매수 상위"), width),
            pad(side("sell", "기관 순매도 상위"), width)]


def names_lines(st: State, width: int) -> tuple[list[str], int]:
    """종목 목록 화면 — (행, 헤더 줄 수)."""
    r = st.selected()
    if not r:
        return [pad(" (섹터를 고르라)", width)], 1
    names = st.names()
    title = f" {r.get('sector','—')} · 종목 {len(names)}개 · {st.window}일 기준"
    cols = [("종목", 14, False), ("코드", 7, False),
            ("순매수", 11, True), ("시총대비", 9, True), ("거래대금", 11, True)]
    head = " ".join(pad(c[0], c[1], c[2]) for c in cols)
    out = [pad(title, width), pad(head, width)]
    for t in names:
        a = t.get("a")
        out.append(pad(" ".join([
            pad(t.get("name", "—"), 14),
            pad(t.get("code", ""), 7),
            pad(fmt_amt(t.get("inst")), 11, True),
            pad((fmt_pct(a) + "%p") if a is not None else "—", 9, True),
            pad(fmt_amt(t.get("tv")).replace("+", ""), 11, True),
        ]), width))
    if not names:
        out.append(pad("  (이 시장·섹터에 대표종목이 없다)", width))
    return out, 2


FOOTER = " w:구간  m:시장  a:주체  s:정렬  ↑↓:섹터  Enter:종목  q:종료"
FOOTER_DRILL = " ↑↓:종목  Esc/←:돌아가기  q:종료"
