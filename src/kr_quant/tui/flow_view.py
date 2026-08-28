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
MARKET_ORDER = ("거래소", "코스닥")

#: k·b 가 기관 유입에 회귀해 나온 값들. 다른 주체로는 재계산할 수 없다.
INST_ONLY = ("exp", "x", "U", "P", "xdot", "xddot", "G", "G_pass")

ACTORS = (("inst", "기관"), ("forgn", "외국인"), ("indiv", "개인"), ("etc", "기타법인"))
SORTS = (("G", "성장"), ("flow", "임펄스"), ("accel", "가속"), ("ret", "수익률"),
         ("x", "미실현"), ("U", "포텐셜"), ("P", "dW/dt"), ("xddot", "풀림"),
         ("tv", "거래대금"), ("cap_idx", "시총"), ("n_all", "종목수"))
#: 종목 목록의 정렬. 기본은 **절대 순매수** — 섹터 합계가 금액의 합이므로
#: "누가 이 섹터를 움직였나" 는 금액으로만 정의된다. 나머지는 다른 질문에 답한다.
NAME_SORTS = (("flow", "순매수"), ("a", "시총대비"), ("cap", "시총"),
              ("tv", "거래대금"), ("name", "종목명"))


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
        seen = {k.split("|")[1] for k in data["blocks"]} - {"전체"}
        # set 순서는 PYTHONHASHSEED 마다 다르다. "m 두 번 = 코스닥" 손버릇이
        # 어느 날 조용히 깨지므로 고정 순서로 둔다.
        self.markets = ["전체"] + [m for m in MARKET_ORDER if m in seen] + \
            sorted(seen - set(MARKET_ORDER))
        self.wi = WINDOWS.index("20") if "20" in WINDOWS else 0
        self.mi = 0
        self.ai = 0
        self.si = 0
        self.row = 0
        self.drill = False      # 종목 목록 화면인가
        self.drow = 0           # 종목 목록에서 선택된 행
        self.nsi = 0            # 종목 정렬
        self.help = False       # 도움말 화면인가
        self.hrow = 0           # 도움말 스크롤

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

    @property
    def name_sort(self) -> str:
        return NAME_SORTS[self.nsi][0]

    def cycle(self, what: str, step: int = 1) -> None:
        if what == "w":
            self.wi = (self.wi + step) % len(WINDOWS)
        elif what == "m":
            self.mi = (self.mi + step) % len(self.markets)
        elif what == "a":
            self.ai = (self.ai + step) % len(ACTORS)
        elif what == "s":
            self.si = (self.si + step) % len(SORTS)
        elif what == "ns":
            self.nsi = (self.nsi + step) % len(NAME_SORTS)
            self.drow = 0
            return
        self.row = 0

    # --- 데이터 ---
    def _project(self, r: dict) -> dict:
        """행을 **선택된 주체 기준으로** 다시 쓴다.

        예전엔 임펄스 열만 주체를 따르고 가속·견인주·종목목록은 기관 값이
        그대로 남아, 한 행 안에 두 주체의 숫자가 섞였다. 정렬키도 문자열
        "inst" 라 외국인 화면에서 임펄스 열이 단조가 아니었다.

        k·b 는 기관 유입에 회귀한 것이라 x·U·P·ẍ·G 는 주체를 바꿔도
        재계산할 수 없다. 지우지 않고 남기되 헤더에 (기관) 을 붙여
        출처를 분리한다 — 값이 틀린 게 아니라 다른 주체의 값이다.
        """
        a = self.actor
        flow = r.get(a)
        cap = r.get("cap")
        out = dict(r)
        out["flow"] = flow
        out["accel"] = (flow / cap * 100) if (cap and flow is not None) else None
        out["top"] = self._leads().get(r.get("sector"))
        return out

    def _leads(self) -> dict:
        """섹터 → 그 주체의 순매수/순매도 1위. 표의 견인주가 종목 목록과
        같은 집합에서 나오도록 `names` 에서 직접 뽑는다(페이로드의 top 은
        기관 전용이고 집계 대상도 미묘하게 달랐다)."""
        win = self.window if self.window != "종합" else "20"
        ck = (win, self.market, self.actor)
        if getattr(self, "_lead_ck", None) == ck:
            return self._lead_v
        mkts = [self.market] if self.market != "전체" else self.markets[1:]
        best: dict = {}
        for code, nm in (self.d.get("names") or {}).items():
            if nm.get("market") not in mkts:
                continue
            w = (nm.get("win") or {}).get(win)
            if not w:
                continue
            v = w.get(self.actor)
            if v is None:
                continue
            best.setdefault(nm.get("sector"), []).append(
                {"code": code, "name": nm.get("name", "—"), "flow": v})
        out = {}
        for sec, arr in best.items():
            arr.sort(key=lambda t: -t["flow"])
            out[sec] = {"buy": arr[:3], "sell": arr[::-1][:3], "n": len(arr)}
        self._lead_ck, self._lead_v = ck, out
        return out

    #: 정렬키가 그 블록에서 전멸했을 때 대신 쓸 키 — 앞에서부터 값이 있는 것.
    SORT_FALLBACK = ("accel", "flow", "ret")

    def effective_sort(self, rows: list[dict]) -> tuple[str, bool]:
        """실제로 줄세우는 데 쓸 키와, 그게 폴백인지.

        5일 창은 G·풀림이 27/27 전부 결측이다. 예전엔 그대로 정렬해서 전 행이
        동률이 되어 페이로드 원순서(가나다)로 남았는데, 헤더는 그 열을
        하이라이트하며 "이걸로 줄세웠다"고 말했다. 정렬이 없는데 있는 척한 것이다.
        """
        key = self.sort_key
        if any(r.get(key) is not None for r in rows):
            return key, False
        for alt in self.SORT_FALLBACK:
            if alt != key and any(r.get(alt) is not None for r in rows):
                return alt, True
        return key, False

    def rows(self) -> list[dict]:
        if self.window == "종합":
            c = self.d.get("combined", {}).get(self.market)
            if not c:
                return []
            leads = self._leads()
            return [dict(r, top=leads.get(r.get("sector"))) for r in c["rows"]]
        b = self.d["blocks"].get(f"{self.window}|{self.market}")
        if not b:
            return []
        rows = [self._project(r) for r in b["rows"]]
        key, _fell = self.effective_sort(rows)
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
                        "flow": w.get(self.actor), "tv": w.get("tv"), "cap": cap,
                        "a": ((w.get(self.actor) or 0) / cap * 100) if cap else None})
        key = self.name_sort
        if key == "name":
            out.sort(key=lambda t: t["name"])
        else:
            out.sort(key=lambda t: -(t.get(key) if t.get(key) is not None else -1e18))
        return out


def header_lines(st: State, width: int) -> list[str]:
    d = st.d
    chip = "확정" if d.get("finalized") else "장중·미확정"
    l1 = f" 섹터 자금 흐름 · {d['asof']} {chip}"
    label, note = SORTS[st.si][1], ""
    if st.window != "종합":
        key, fell = st.effective_sort(st.rows())
        if fell:
            note = f"  ※ {label} 은 이 창에 값이 없다"
            label = dict(SORTS).get(key, key)
    l2 = (f" 구간[{st.window}] 시장[{st.market}] 주체[{ACTORS[st.ai][1]}]"
          f" 정렬[{label}]{note}")
    if st.actor != "inst" and st.window != "종합":
        l2 += "  ※ 미실현·포텐셜·dW/dt·풀림·G 는 기관 기준"
    return [pad(l1, width), pad(l2 + "  " + st.block_meta(), width)]


#: 정렬 키 → 그 열의 헤더 이름. 하이라이트할 열을 찾는 데 쓴다.
SORT_COL = {"G": "G[0~1]", "flow": "임펄스[억]", "accel": "가속[%p]",
            "ret": "수익률[%]",
            "x": "미실현[%p]", "U": "포텐셜[½kx²]", "P": "dW/dt[%p/일]",
            "xddot": "풀림[%p/일²]", "n_all": "종목[수]"}   # 거래대금·시총은 표에 열이 없어 하이라이트 대상이 아니다
NAME_SORT_COL = {"flow": "순매수[억]", "a": "시총대비[%p]", "cap": "시총[억]",
                 "tv": "거래대금[억]", "name": "종목"}


def table_cols(st: State, width: int) -> list[tuple[str, int, bool]]:
    """섹터 표의 열 정의 — 렌더와 하이라이트가 **같은 정의**를 봐야 어긋나지 않는다."""
    wide = width >= 100
    if st.window == "종합":
        cols = [("섹터", 11, False), ("종목[수]", 8, True), ("G[0~1]", 7, True),
                ("", 1, False)]
        wins = (st.d.get("combined", {}).get(st.market) or {}).get("windows", [])
        for w in wins:
            cols.append((f"{w}일[G]", 8, True))
        cols.append(("통과[창]", 8, True))
        return cols
    cols = [("섹터", 11, False), ("종목[수]", 8, True), ("G[0~1]", 7, True),
            ("", 1, False),
            ("임펄스[억]", 12, True), ("가속[%p]", 9, True), ("수익률[%]", 10, True)]
    if wide:
        cols += [("미실현[%p]", 10, True), ("포텐셜[½kx²]", 12, True)]
    # 아주 넓은 터미널에서는 동역학 열까지 보여준다 — 정렬은 되는데 값을 못 보는
    # 상태를 없앤다(dW/dt·풀림으로 줄세워놓고 그 숫자가 화면에 없으면 읽을 수 없다).
    if width >= 132:
        cols += [("dW/dt[%p/일]", 12, True), ("풀림[%p/일²]", 12, True)]
    # 이름만 잘라 넣으면 한글 길이가 제각각이라 줄마다 다르게 잘려 보인다.
    # **이름 + 금액**을 한 덩어리로 넣고 칸을 고정하면 모양이 일정하다.
    if width >= 150:
        cols += [("순매수상위[억]", 23, False), ("순매도상위[억]", 23, False)]
    elif wide:
        cols += [("순매수상위[억]", 23, False)]
    return cols


def _fit(cols: list[tuple[str, int, bool]], width: int) -> list[tuple[str, int, bool]]:
    """폭에 **온전히** 들어가는 열까지만 남긴다. 첫 열은 잘려도 남긴다
    (섹터 이름은 잘려도 뜻이 남지만, 숫자는 잘리면 다른 값이 된다)."""
    out, used = [], 0
    for c in cols:
        need = c[1] + (1 if out else 0)
        if out and used + need > width:
            break
        out.append(c)
        used += need
    return out


def col_span(cols: list[tuple[str, int, bool]], header: str) -> tuple[int, int] | None:
    """열 헤더의 (시작 표시칸, 폭). 열 사이 공백 1칸을 더해가며 센다."""
    cell = 0
    for name, w, _r in cols:
        if name == header:
            return cell, w
        cell += w + 1
    return None


def table_lines(st: State, width: int, height: int) -> tuple[list[str], list[bool], int]:
    """(행 문자열, 얇은섹터 여부, 헤더 줄 수). 폭에 따라 열을 줄인다."""
    cols = table_cols(st, width)
    wide = width >= 100
    wins = (st.d.get("combined", {}).get(st.market) or {}).get("windows", [])

    # 폭이 모자라면 **열 경계에서** 떨어뜨린다. 줄을 통째로 잘라내면 숫자가
    # 자릿수 중간에서 끊겨 -1,360 이 -1 로 보인다 — 안 보이는 것보다 나쁘다.
    cols = _fit(cols, width)
    ncol = len(cols)
    head = " ".join(pad(c[0], c[1], c[2]) for c in cols)
    out = [pad(head, width)]
    thin = [False]

    for r in st.rows():
        cells = [pad(r.get("sector", "—"), 11),
                 pad(str(r.get("n_all", "—")), 8, True),
                 pad(f"{r['G']:.2f}" if r.get("G") is not None else "—", 7, True),
                 # 마커는 **별도 1칸 열**이다. 값에 붙이면 ● 가 2칸이라 열이 밀린다.
                 # 얇은 섹터(~)는 글자로도 표시한다 — 색에만 실으면 무색
                 # 터미널·색맹에서 경고가 통째로 사라진다(파랑은 검은 바탕에서
                 # 8색 중 대비가 가장 나쁘고, A_DIM 을 무시하는 터미널도 많다).
                 pad("~" if r.get("thin") else ("*" if r.get("G_pass") else ""), 1)]
        if st.window == "종합":
            per = r.get("per", {})
            for w in wins:
                v = per.get(str(w), per.get(w))
                cells.append(pad(f"{v:.2f}" if v is not None else "—", 8, True))
            cells.append(pad(f"{r.get('pass_n',0)}/{r.get('seen',0)}", 8, True))
        else:
            cells += [pad(fmt_amt(r.get("flow")), 12, True),
                      pad(fmt_pct(r.get("accel")), 9, True),
                      pad(fmt_pct(r.get("ret")), 10, True)]
            if wide:
                cells += [pad(fmt_pct(r.get("x"), 1), 10, True),
                          pad(f"{r['U']:.0f}" if r.get("U") is not None else "—",
                              12, True)]
            if width >= 132:
                cells += [pad(fmt_pct(r.get("P"), 3), 12, True),
                          pad(fmt_pct(r.get("xddot"), 3), 12, True)]
            top = r.get("top") or {}

            def _lead(side: str) -> str:
                """이름 13칸 + 금액 7칸(우측정렬). 이름 길이가 제각각이라
                그냥 이어붙이면 금액이 줄마다 다른 칸에 떨어진다."""
                arr = (top or {}).get(side) or []
                t = arr[0] if arr else None
                if not t:
                    return pad("—", 13) + " " + pad("", 9, right=True)
                return (pad(t["name"], 13) + " "
                        + pad(fmt_amt(t["flow"]) + "억", 9, right=True))

            if wide:
                cells.append(pad(_lead("buy"), 23))
            if width >= 150:
                cells.append(pad(_lead("sell"), 23))
        out.append(pad(" ".join(cells[:ncol]), width))
        thin.append(bool(r.get("thin")))
    return out, thin, 1


def sort_span(st: State, width: int) -> tuple[int, int] | None:
    """지금 정렬 중인 열의 (시작 표시칸, 폭). 종합 화면·이름정렬은 None."""
    if st.window == "종합":
        return None
    key, _fell = st.effective_sort(st.rows())
    header = SORT_COL.get(key)
    return col_span(_fit(table_cols(st, width), width), header) if header else None


def names_cols() -> list[tuple[str, int, bool]]:
    return [("종목", 14, False), ("코드", 7, False),
            ("순매수[억]", 13, True), ("시총대비[%p]", 12, True),
            ("시총[억]", 13, True), ("거래대금[억]", 13, True)]


def name_sort_span(st: State) -> tuple[int, int] | None:
    """종목 목록에서 지금 정렬 중인 열의 (시작 표시칸, 폭)."""
    header = NAME_SORT_COL.get(st.name_sort)
    return col_span(names_cols(), header) if header else None


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
        # 정렬이 **금액** 순이므로 금액을 보여준다. 시총대비(%p)를 보이면
        # 표시값과 순서가 어긋나 보인다(현대 +1.34%p 가 대우 +1.40%p 위에 오는 식).
        parts = [f"{t['name']} {fmt_amt(t['flow'])}억" for t in arr[:3]]
        return f" {label}: " + " · ".join(parts)
    n = (r.get("top") or {}).get("n", 0)
    return [pad(f" {r.get('sector','—')} · 종목 {n}개 · Enter 로 전체", width),
            pad(side("buy", "순매수 상위"), width),
            pad(side("sell", "순매도 상위"), width)]


def names_lines(st: State, width: int) -> tuple[list[str], int]:
    """종목 목록 화면 — (행, 헤더 줄 수)."""
    r = st.selected()
    if not r:
        return [pad(" (섹터를 고르라)", width)], 1
    names = st.names()
    title = (f" {r.get('sector','—')} · 종목 {len(names)}개 · {st.window}일 기준"
             f" · 정렬[{NAME_SORTS[st.nsi][1]}]")
    cols = _fit(names_cols(), width)
    ncol = len(cols)
    head = " ".join(pad(c[0], c[1], c[2]) for c in cols)
    out = [pad(title, width), pad(head, width)]
    for t in names:
        a = t.get("a")
        out.append(pad(" ".join([
            pad(t.get("name", "—"), 14),
            pad(t.get("code", ""), 7),
            pad(fmt_amt(t.get("flow")), 13, True),
            pad(fmt_pct(a) if a is not None else "—", 12, True),
            pad(fmt_amt(t.get("cap")).replace("+", "") if t.get("cap") else "—",
                13, True),
            pad(fmt_amt(t.get("tv")).replace("+", ""), 13, True),
        ][:ncol]), width))
    if not names:
        out.append(pad("  (이 시장·섹터에 대표종목이 없다)", width))
    return out, 2


#: 색을 입힐 구간을 찾는 정규식. 렌더는 문자열만 만들고, 어디를 무슨 색으로
#: 칠할지는 이 함수가 (시작칸, 길이, 역할) 로 낸다 — curses 를 여기 들이지 않는다.
_NUM = __import__("re").compile(r"[+-][\d,]+(?:\.\d+)?%?p?")


def color_spans(line: str) -> list[tuple[int, int, str]]:
    """한 줄에서 색칠할 구간 — [(시작 표시칸, 표시폭, 역할)].

    역할: ``up``(양수) · ``down``(음수) · ``mark``(통과 표시).
    문자 인덱스가 아니라 **표시 칸**을 낸다 — 한글이 2칸이라 curses 의 addstr
    좌표와 문자 인덱스가 다르다(이 저장소가 이미 밟은 함정).
    """
    spans = []
    # 문자 인덱스 → 표시 칸 매핑
    cell_at, cell = [], 0
    for ch in line:
        cell_at.append(cell)
        cell += cell_width(ch)
    cell_at.append(cell)
    for m in _NUM.finditer(line):
        role = "up" if line[m.start()] == "+" else "down"
        a, b = cell_at[m.start()], cell_at[m.end()]
        spans.append((a, b - a, role))
    for i, ch in enumerate(line):
        if ch == "*":
            spans.append((cell_at[i], 1, "mark"))
    return spans


#: 도움말 — 열의 뜻과 계산식. 화면에서 읽는 사람이 "이 숫자가 뭐냐" 를 물을 자리에
#: 답을 둔다. 한계도 같이 적는다(추정치·상대지표·유동성 함정).
HELP = [
    ("", "── 섹터 표 ──"),
    ("섹터", "벤더 분류(stocks.sector) 27개. KRX 업종 분류와 다르다."),
    ("종목[수]", "그 (시장,섹터)의 상장 종목 수. 10개 미만은 회색 — 벤더 분류가 좁아"),
    ("", "        사실상 단일종목인 라벨이 있다(부동산 3, 출판/매체복제 2)."),
    ("G[0~1]", "성장 점수 0~1. 세 순위의 평균 — 힘(가속도)·압축(미실현)·풀림(ẍ)."),
    ("", "        * 는 세 조건을 다 만족(a>0 · x>0 · ẍ>0). 검증된 적 없는 탐색 지표다."),
    ("임펄스[억]", "구간 누적 순매수 [억원] = Σ(순매매 수량 × 그날 종가)."),
    ("", "        DB 는 수량만 주므로 금액은 종가 환산 근사다(참값은 VWAP 가중)."),
    ("가속[%p]", "임펄스 ÷ 구간말 섹터 시총 × 100 [%p]. 물리로 a = F/m."),
    ("", "        금액만 보면 대형 섹터가 늘 이기므로 쏠림은 이쪽이 정직하다."),
    ("수익률[%]", "그 섹터 **자체 바구니**의 구간 수익률 [%], 전일 시총 가중."),
    ("", "        KRX 업종지수가 아니다 — 구성종목이 달라 분자·분모가 어긋난다."),
    ("예상Δv", "k × 가속 + b. k·b 는 그 창 27개 섹터의 횡단면 회귀(절편 포함 OLS)."),
    ("미실현[%p]", "예상Δv − 실제 수익률 [%p]. + 면 덜 갔고(눌림), − 면 이미 더 갔다."),
    ("", "        OLS 잔차의 부호 반전이라 27개 합이 0 이다 — **상대** 지표다."),
    ("포텐셜[½kx²]", "½·k·x². **x 의 제곱이라 부호가 없다** — 이걸로 정렬하면 '많이 눌린 것'과"),
    ("", "        '이미 많이 간 것'이 같이 위로 온다. 미실현을 같이 봐야 방향이 갈린다."),
    ("dW/dt[%p/일]", "구간을 반으로 갈라 W=가속×수익률 의 변화. 힘과 운동이 정렬되는가."),
    ("", "        폭 132칸 이상에서 보인다. 좁으면 정렬만 되고 값은 안 보인다."),
    ("풀림[%p/일²]", "미실현 x 가 해소되는 **가속**(ẍ). 구간을 셋으로 갈라 중앙차분한다."),
    ("", "        2차 차분이라 짧은 창(조각 6~7일)에서는 값이 흔들린다."),
    ("순매수상위[억]", "그 섹터에서 기관이 **가장 많이 산** 종목과 그 금액[억원]."),
    ("순매도상위[억]", "**가장 많이 판** 종목. 둘은 같은 목록(금액순)의 위/아래 끝이다."),
    ("", "        표는 1개씩, 하단 패널은 3개씩 — 같은 목록이고 시총과 무관하다."),
    ("", "        순매도상위는 폭 150칸 이상에서 보인다. Enter 로 전 종목."),
    ("N일[G]", "종합 화면에서 그 창의 G. 20일G·60일G·120일G 로 뜬다."),
    ("통과[창]", "세 조건(a>0·x>0·ẍ>0)을 만족한 창 수 / 전체 창 수. 예 2/3."),
    ("", ""),
    ("", "── 종목 목록 (Enter) ──"),
    ("순매수[억]", "그 구간 기관 순매수 [억원]. **기본 정렬** — 섹터 합계가 금액의"),
    ("", "        합이므로 '누가 이 섹터를 움직였나' 는 금액으로만 정의된다."),
    ("시총대비[%p]", "순매수 ÷ 그 종목 시총 × 100 [%p]. 시총 작은 스팩이 위로 올라온다."),
    ("시총[억]", "그 종목의 구간말 시가총액 [억원]. 시총대비의 분모다."),
    ("거래대금[억]", "그 구간 거래대금 [억원]."),
    ("", ""),
    ("", "── 알아둘 것 ──"),
    ("", "· k 는 추정치다. 창마다 다르다(5일 15.9 · 20일 14.7 · 60일 9.2 · 120일 9.0)."),
    ("", "· 위 관계는 **동시기**다. 사면 오른다는 것이지 미래를 예측하지 않는다."),
    ("", "  예측 가설은 따로 검정해 기각됐다(research/logs/inst_flow_accel)."),
    ("", "· 값 0 은 '관망' 이 아니라 '0 또는 미보고' 다 — 수집기가 파싱 실패를 0 으로 준다."),
    ("", "· 코스닥 단독은 관계가 약하다(R² 0.00~0.25). 거래소가 전체를 끈다."),
]


def help_lines(width: int, offset: int, height: int) -> tuple[list[str], int]:
    """도움말 화면 — (행, 전체 줄 수). offset 부터 height 줄을 낸다."""
    out = [pad(" 열의 뜻 — ↑↓ 스크롤 · q·Esc·?·Enter 로 닫기", width)]
    body = []
    for name, desc in HELP:
        # ⚠️ f"{name:>9}" 는 **문자 폭**이라 한글 라벨(임펄스=6칸)과 ASCII(G=1칸)가
        # 어긋난다. pad 는 표시 칸으로 맞춘다 — 이 저장소가 세 번 밟은 함정이다.
        body.append(pad((pad(name, 10, right=True) + "  " + desc) if name
                        else ("   " + desc), width))
    view = body[offset:offset + max(height - 1, 1)]
    return out + view, len(body)


FOOTER = " w:구간 m:시장 a:주체 s:정렬 ↑↓:섹터 Enter:종목 ?:도움말 q:종료"
FOOTER_DRILL = " ↑↓:종목  s:정렬  ?:도움말  Esc/←:돌아가기  q:종료"
