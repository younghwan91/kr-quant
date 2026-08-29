"""섹터 자금흐름 TUI 의 **순수 렌더 로직** — curses 를 import 하지 않는다.

화면 그리기(curses)와 무엇을 그릴지(여기)를 나눈다. 이 파일은 데이터와 상태를
받아 문자열 행 목록을 돌려주므로 단위 테스트가 된다. curses 를 섞으면 렌더가
터미널 없이는 검증 불가능해지고, 열 정렬이 어긋나도 아무도 못 잡는다 —
이 저장소는 그 실수를 이미 두 번 했다(표 헤더와 셀 개수 불일치).
"""

from __future__ import annotations

import unicodedata
from collections import namedtuple

WINDOWS = ("5", "20", "60", "120", "종합")
# 종목 목록은 **절대 순매수 금액** 순이다 — 섹터 합계가 금액의 합이므로 기여도는
# 금액으로만 정의된다. 시총 대비는 참고 열. (시총 대비로 줄세웠더니 스팩이 1위가
# 됐고, 그걸 막으려 유동성 하한을 넣었더니 작은 섹터가 통째로 비었다.)
MARKET_ORDER = ("거래소", "코스닥")

#: k·b 가 기관 유입에 회귀해 나온 값들. 다른 주체로는 재계산할 수 없다.
INST_ONLY = ("exp", "x", "U", "P", "xdot", "xddot", "G", "G_pass")

ACTORS = (("inst", "기관"), ("forgn", "외국인"), ("indiv", "개인"), ("etc", "기타법인"))
#: 화면 정렬. **기본은 가속** — 규모로 정규화된 유입이라 대형 섹터가 늘 이기지
#: 않는다. 예전 기본은 G 였는데, G 는 세 순위의 평균인 **검증 안 된 탐색 점수**라
#: 화면 순서를 지배할 근거가 없다(20일 기본 화면 1위가 임펄스 +14억짜리 섹터였고,
#: 기관이 2.4조 판 전기/전자는 24위라 스크롤해야 보였다). G 는 보조로 남는다.
#: 순서는 **열 순서와 같다** — 정렬을 돌릴 때 눈이 왼쪽에서 오른쪽으로 따라간다.
SORTS = (("accel", "가속"), ("flow", "임펄스"), ("pct", "1년%"), ("ret", "수익률"),
         ("x", "미실현"), ("xddot", "풀림"), ("G", "성장"), ("U", "포텐셜"),
         ("P", "dW/dt"), ("tv", "거래대금"), ("cap_idx", "시총"), ("n_all", "종목수"))
#: 종목 목록의 정렬. 기본은 **절대 순매수** — 섹터 합계가 금액의 합이므로
#: "누가 이 섹터를 움직였나" 는 금액으로만 정의된다. 나머지는 다른 질문에 답한다.
#: (`flow` 는 **절댓값** 순이다. 부호순으로 두면 판 종목이 목록 맨 끝으로 밀려
#: 누적 기여율이 뜻을 잃는다 — 섹터를 움직인 건 산 쪽과 판 쪽 둘 다다.)
NAME_SORTS = (("flow", "순매수"), ("part", "참여율"), ("a", "시총대비"),
              ("cap", "시총"), ("tv", "거래대금"), ("name", "종목명"))

#: 누적 기여율의 가로줄을 그을 지점. 실측상 섹터 흐름의 80% 를 설명하는 종목이
#: 평균 6.2개인데 종목행의 51% 가 +0(|순매수|<0.5억) 이다 — 어디까지가 "그 섹터를
#: 움직인 종목" 인지 눈으로 끊어준다.
CUM_CUT = 80.0


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
        # 정렬 역순. 없을 때 "순매도 상위" 로 가는 유일한 길이 G(맨 끝으로) 였는데
        # 그 키가 화면 어디에도 안 적혀 있었다 — 발견 불가능한 유일 경로.
        self.rev = False        # 섹터 표 역순(오름차순)
        self.nrev = False       # 종목 목록 역순

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
        # 1년 백분위·구간 모양도 **같은 주체**에서 꺼낸다. 오늘 고친 버그가 정확히
        # "한 행 안에 두 주체의 숫자가 섞이는" 것이었다 — 새 열에서 반복하지 않는다.
        out["pct"] = (r.get("pct1y") or {}).get(a)
        out["spark"] = (r.get("spark") or {}).get(a)
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
        """화면에 나갈 행 — 선택 상태별로 캐시한다.

        렌더가 열 정의를 순회하면서 셀마다 횡단면 스케일(미실현 막대의 최댓값)을
        묻기 때문에, 캐시가 없으면 한 프레임에 rows() 가 수백 번 돈다.
        """
        # 캐시 키에는 **정렬 결과를 바꾸는 상태가 전부** 들어가야 한다.
        # rev 를 빠뜨렸더니 r 을 눌러도 캐시가 옛 순서를 돌려줬다 — 캐시와
        # 역순이 각각은 옳은데 합쳐서 깨진 자리다.
        ck = (self.wi, self.mi, self.ai, self.si, self.rev)
        if getattr(self, "_rows_ck", None) == ck:
            return self._rows_v
        out = self._rows_uncached()
        self._rows_ck, self._rows_v = ck, out
        return out

    def _rows_uncached(self) -> list[dict]:
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
            # 결측은 역순에서도 **맨 뒤**다 — '값이 없는 것'은 작은 값이 아니다.
            return (v is None, (v if self.rev else -v) if v is not None else 0)
        rows.sort(key=val)
        return rows

    def xmax(self) -> float:
        """이 화면에서 |미실현| 의 최댓값 — 발산 막대의 눈금.

        x 는 OLS 잔차의 부호 반전이라 27개 합이 0 인 **상대** 지표다. 절대 눈금이
        없으므로 막대도 화면 안 최댓값에 맞춘다. 창·시장을 바꾸면 눈금이 바뀐다 —
        막대끼리는 비교되지만 화면끼리는 비교되지 않는다.
        """
        vals = [abs(r["x"]) for r in self.rows() if r.get("x") is not None]
        return max(vals) if vals else 0.0

    def block_meta(self) -> str:
        if self.window == "종합":
            c = self.d.get("combined", {}).get(self.market)
            return f"구간 {'·'.join(str(w) for w in c['windows'])}일 등가중" if c else ""
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
            v, tv = w.get(self.actor), w.get("tv")
            out.append({"code": code, "name": nm.get("name", "—"),
                        "flow": v, "tv": tv, "cap": cap,
                        # 참여율 — 그 종목 거래대금 중 이 주체의 순매수가 차지한 몫.
                        # 거래대금 자체는 "얼마나 붐볐나" 일 뿐이고, 판단에 직결되는
                        # 것은 "그 거래의 몇 %가 한 방향이었나" 다.
                        "part": (v / tv * 100) if (tv and v is not None) else None,
                        "a": ((v or 0) / cap * 100) if cap else None})
        key = self.name_sort
        if key == "name":
            out.sort(key=lambda t: t["name"], reverse=self.nrev)
        elif key == "flow":
            # **절댓값** 순. 부호순이면 판 종목이 목록 끝으로 밀려, 화면 앞쪽은
            # 산 종목 몇 개 + 0 의 벌판이 된다(종목행의 51%가 +0 이다).
            # 결측은 어느 방향에서도 맨 뒤 — '값이 없는 것'은 작은 값이 아니다.
            out.sort(key=lambda t: (t["flow"] is None,
                                    abs(t["flow"]) * (1 if self.nrev else -1)
                                    if t["flow"] is not None else 0))
        else:
            out.sort(key=lambda t: (t.get(key) is None,
                                    (t.get(key) * (1 if self.nrev else -1))
                                    if t.get(key) is not None else 0))
        self._attach_cum(out)
        return out

    def _attach_cum(self, rows: list[dict]) -> None:
        """누적 기여율 — |순매수| 의 화면 순서 누적 몫(%).

        **순매수 정렬에서만** 채운다. 기여도는 |금액| 으로만 정의되는데, 시총·종목명
        순서로 누적하면 아무 뜻 없는 톱니가 되고, 그런데도 숫자가 단조증가라서
        읽는 사람은 뜻이 있다고 믿는다. 뜻이 없는 칸은 비워 두는 편이 정직하다.
        """
        tot = sum(abs(t["flow"]) for t in rows if t.get("flow") is not None)
        run, cut_done = 0.0, False
        for t in rows:
            if self.name_sort != "flow" or not tot:
                t["cum"] = None
                t["cut"] = False
                continue
            run += abs(t.get("flow") or 0.0)
            t["cum"] = run / tot * 100
            # 가로줄은 80% 를 **처음 넘긴 행** 하나에만 긋는다.
            t["cut"] = not cut_done and t["cum"] >= CUM_CUT
            cut_done = cut_done or t["cut"]


def header_lines(st: State, width: int) -> list[str]:
    d = st.d
    chip = "확정" if d.get("finalized") else "장중·미확정"
    l1 = f" 섹터 자금 흐름 · {d['asof']} {chip}"
    label, note = SORTS[st.si][1], ""
    if st.window != "종합":
        key, fell = st.effective_sort(st.rows())
        if fell:
            note = f"  ※ 이 구간에 {label} 값이 없다"
            label = dict(SORTS).get(key, key)
    # 종합 화면은 정렬 자체가 없다(페이로드 순서) — 없는 방향을 표시하면 또 거짓말이다.
    arrow = "" if st.window == "종합" else ("▲" if st.rev else "▼")
    l2 = (f" 구간[{st.window}] 시장[{st.market}] 주체[{ACTORS[st.ai][1]}]"
          f" 정렬[{label}{arrow}]{note}")
    if st.actor != "inst" and st.window != "종합":
        l2 += "  ※ 미실현·포텐셜·dW/dt·풀림·G 는 기관 기준"
    return [pad(l1, width), pad(l2 + "  " + st.block_meta(), width)]


#: 표의 한 열. **헤더·폭·정렬·값 추출을 한 자리에** 둔다.
#:
#: 예전엔 `table_cols()`(헤더) 와 `table_lines()`(셀) 가 같은 사실을 두 번 적었다 —
#: 폭 상수(11·8·7·1·12·9·10…)가 양쪽에 각각 있었고, 폭 임계값(>=100·>=132·>=150)도
#: 양쪽에 있었고, 심지어 분기 모양이 서로 달랐다(한쪽은 if/elif, 다른 쪽은 if 두 개).
#: 결과가 같았던 것은 우연이고, 어긋나기 직전이었다. 이제 렌더는 이 정의를
#: **순회만** 하므로 "헤더와 셀이 어긋나는" 버그가 표현 불가능하다 —
#: docs/GUARDRAILS.md §0 의 "위험한 기능을 코드에서 제거" 와 같은 처방이다.
Col = namedtuple("Col", "header width right fn")      # fn(r, st) -> str


#: 스파크라인·발산 막대에 쓰는 글자는 **브라유**(U+28xx) 다.
#:
#: East Asian Width 가 'N'(Narrow) 이라 어떤 로케일에서도 1칸이다. 블록 문자
#: ``▁▂▃▄▅▆▇█`` 는 'A'(Ambiguous) 라 한글 로케일 터미널이 2칸으로 그릴 수 있고,
#: 그러면 그 행만 통째로 밀린다. tmux 실측(en_US.UTF-8·ko_KR.utf8, cursor_x)
#: 으로는 둘 다 1칸이었지만, **표준이 '애매' 라고 말하는 글자에 표 정렬을 걸
#: 이유가 없다.** 이 저장소는 폭 계산 함정을 이미 세 번 밟았다.
#: 값: +2 강한 유입 · +1 유입 · 0 · -1 유출 · -2 강한 유출.
SPARK = {2: "⠛", 1: "⠒", 0: "⠀", -1: "⠤", -2: "⣤"}
BAR_FILL = "⣿"       # 꽉 찬 브라유 — 막대 몸통
BAR_AXIS = "|"            # 0 기준선. ASCII 라 폭이 애매할 여지가 없다


def spark(vals, cells: int = 8) -> str:
    """구간을 비겹침 조각으로 나눈 합의 모양. 오른쪽 끝이 **가장 최근**이다.

    조각 합을 다 더하면 임펄스 열의 그 숫자가 된다 — 같은 사실의 시간축 전개다.
    눈금은 **그 행 안에서** 최댓값에 맞춘다(행끼리 높이는 비교되지 않는다).
    "지금 들어오는 중인가, 이미 끝났나" 만 답하는 열이기 때문이다.
    """
    vals = [v for v in (vals or [])]
    if not vals:
        return "—"
    got = [abs(v) for v in vals if v is not None]
    mx = max(got) if got else 0.0
    out = ""
    for v in vals:
        if v is None or not mx or v == 0:
            lv = 0
        else:
            lv = 2 if abs(v) > mx / 2 else 1
            lv = lv if v > 0 else -lv
        out += SPARK[lv]
    # 조각이 cells 보다 적으면(5일 창) **왼쪽**을 비운다 — 오른쪽 끝이 최근이라는
    # 약속을 깨지 않기 위해서다.
    return pad(out, cells, right=True)


def xbar(x, mx: float, half: int = 4) -> str:
    """미실현 x 의 0 기준 발산 막대 — 부호와 크기를 한 도형에서 읽는다.

    포텐셜 U = ½kx² 는 k 가 블록당 상수라 |x| 의 순증가 변환이다(4개 창 전부
    Spearman 1.0000, 실측). 즉 U 열은 x 열의 **크기만** 다시 적은 것이고, 잃는
    것은 부호다. 막대는 그 크기를 부호와 함께 보여주므로 겹침이 정보 손실로
    바뀌지 않는다. U 는 지우지 않고 넓은 폭 전용 열 + 정렬 옵션으로 남긴다.
    """
    if x is None:
        return "—"
    if not mx or x == 0:
        lv = 0
    else:
        q = abs(x) / mx * half
        lv = min(half, int(q) + (1 if q > int(q) else 0))
    left = (" " * (half - lv) + BAR_FILL * lv) if x < 0 else " " * half
    right = (BAR_FILL * lv + " " * (half - lv)) if x > 0 else " " * half
    return left + BAR_AXIS + right


def _num(key, nd=2):
    return lambda r, st: fmt_pct(r.get(key), nd)


#: 정렬 키 → 그 열의 헤더 이름. 하이라이트할 열을 찾는 데 쓴다.
SORT_COL = {"G": "G[0~1]", "flow": "임펄스[억]", "accel": "가속[%p]",
            "ret": "수익률[%]", "pct": "1년[%ile]",
            "x": "미실현[%p]", "U": "포텐셜[½kx²]", "P": "dW/dt[%p/일]",
            "xddot": "풀림[%p/일²]", "n_all": "종목[수]"}   # 거래대금·시총은 표에 열이 없어 하이라이트 대상이 아니다
NAME_SORT_COL = {"flow": "순매수[억]", "part": "참여율[%]", "a": "시총대비[%p]",
                 "cap": "시총[억]", "tv": "거래대금[억]", "name": "종목"}


def _lead(side: str):
    """이름 13칸 + 금액 9칸(우측정렬). 이름 길이가 제각각이라 그냥 이어붙이면
    금액이 줄마다 다른 칸에 떨어진다."""
    def fn(r, st):
        arr = (r.get("top") or {}).get(side) or []
        t = arr[0] if arr else None
        if not t:
            return pad("—", 13) + " " + pad("", 9, right=True)
        return pad(t["name"], 13) + " " + pad(fmt_amt(t["flow"]) + "억", 9, right=True)
    return fn


#: 섹터 표의 열 — **왼쪽부터 중요한 순서**이고, 폭이 되는 데까지 `_fit` 이 자른다.
#:
#: 순서는 물리 서사를 따른다: 힘(가속·임펄스) → 그 힘이 1년 안에서 어느 정도인가
#: (1년%·추이) → 운동(수익률) → 차이(미실현) → 해소(풀림) → 요약(G) → 누가(주도주).
#: 예전엔 결론인 G 가 자기 입력(가속·미실현·풀림)보다 **왼쪽**에 있었고, 판단
#: 변수가 아니라 데이터 품질 주석인 종목수가 2번 자리를 차지했다.
#: 종목수는 오른쪽 끝으로 보냈다 — 얇은 섹터 경고는 이미 `~` 마커가 한다.
_TABLE_COLS = (
    Col("섹터", 13, False, lambda r, st: r.get("sector", "—")),
    # 마커는 **별도 1칸 열**이다. 값에 붙이면 ● 가 2칸이라 열이 밀린다.
    # 얇은 섹터(~)는 글자로도 표시한다 — 색에만 실으면 무색 터미널·색맹에서
    # 경고가 통째로 사라진다.
    Col("", 1, False,
        lambda r, st: "~" if r.get("thin") else ("*" if r.get("G_pass") else "")),
    Col("가속[%p]", 9, True, _num("accel")),
    Col("임펄스[억]", 12, True, lambda r, st: fmt_amt(r.get("flow"))),
    Col("1년[%ile]", 9, True,
        lambda r, st: "—" if r.get("pct") is None else f"{r['pct']:.0f}"),
    Col("추이[8]", 8, False, lambda r, st: spark(r.get("spark"))),
    Col("수익률[%]", 10, True, _num("ret")),
    Col("미실현[%p]", 10, False, lambda r, st: xbar(r.get("x"), st.xmax())),
    Col("풀림[%p/일²]", 12, True, _num("xddot", 3)),
    Col("G[0~1]", 7, True,
        lambda r, st: f"{r['G']:.2f}" if r.get("G") is not None else "—"),
    # 이름만 잘라 넣으면 한글 길이가 제각각이라 줄마다 다르게 잘려 보인다.
    # **이름 + 금액**을 한 덩어리로 넣고 칸을 고정하면 모양이 일정하다.
    Col("순매수상위[억]", 23, False, _lead("buy")),
    Col("순매도상위[억]", 23, False, _lead("sell")),
    Col("포텐셜[½kx²]", 12, True,
        lambda r, st: f"{r['U']:.0f}" if r.get("U") is not None else "—"),
    Col("dW/dt[%p/일]", 12, True, _num("P", 3)),
    Col("종목[수]", 8, True, lambda r, st: str(r.get("n_all", "—"))),
)


def _per_win(w):
    def fn(r, st):
        per = r.get("per", {})
        v = per.get(str(w), per.get(w))
        return f"{v:.2f}" if v is not None else "—"
    return fn


def table_cols(st: State, width: int) -> list[Col]:
    """섹터 표의 열 정의 — 렌더·하이라이트·검사가 **모두 이걸 본다**.

    폭에 따라 여기서 열을 빼지 않는다. 순서가 곧 우선순위이고, 자르는 일은
    `_fit` 하나가 한다 — 폭 임계값이 두 군데에 있으면 언젠가 어긋난다.
    """
    if st.window != "종합":
        return list(_TABLE_COLS)
    cols = [_TABLE_COLS[0], _TABLE_COLS[1],
            Col("G[0~1]", 7, True,
                lambda r, st_: f"{r['G']:.2f}" if r.get("G") is not None else "—")]
    wins = (st.d.get("combined", {}).get(st.market) or {}).get("windows", [])
    for w in wins:
        cols.append(Col(f"{w}일[G]", 8, True, _per_win(w)))
    cols.append(Col("통과[구간]", 10, True,
                    lambda r, st_: f"{r.get('pass_n', 0)}/{r.get('seen', 0)}"))
    cols.append(_TABLE_COLS[-1])                      # 종목[수]
    return cols


def _fit(cols: list[Col], width: int) -> list[Col]:
    """폭에 **온전히** 들어가는 열까지만 남긴다. 첫 열은 잘려도 남긴다
    (섹터 이름은 잘려도 뜻이 남지만, 숫자는 잘리면 다른 값이 된다)."""
    out, used = [], 0
    for c in cols:
        need = c.width + (1 if out else 0)
        if out and used + need > width:
            break
        out.append(c)
        used += need
    return out


def col_span(cols: list[Col], header: str) -> tuple[int, int] | None:
    """열 헤더의 (시작 표시칸, 폭). 열 사이 공백 1칸을 더해가며 센다."""
    cell = 0
    for c in cols:
        if c.header == header:
            return cell, c.width
        cell += c.width + 1
    return None


def _render(cols: list[Col], r: dict, st: State, width: int) -> str:
    return pad(" ".join(pad(c.fn(r, st), c.width, c.right) for c in cols), width)


def table_lines(st: State, width: int, height: int) -> tuple[list[str], list[bool], int]:
    """(행 문자열, 얇은섹터 여부, 헤더 줄 수). 폭에 따라 열을 줄인다.

    폭이 모자라면 **열 경계에서** 떨어뜨린다. 줄을 통째로 잘라내면 숫자가
    자릿수 중간에서 끊겨 -1,360 이 -1 로 보인다 — 안 보이는 것보다 나쁘다.
    """
    cols = _fit(table_cols(st, width), width)
    head = pad(" ".join(pad(c.header, c.width, c.right) for c in cols), width)
    out, thin = [head], [False]
    for r in st.rows():
        out.append(_render(cols, r, st, width))
        thin.append(bool(r.get("thin")))
    return out, thin, 1


def sort_span(st: State, width: int) -> tuple[int, int] | None:
    """지금 정렬 중인 열의 (시작 표시칸, 폭). 종합 화면·이름정렬은 None."""
    if st.window == "종합":
        return None
    key, _fell = st.effective_sort(st.rows())
    header = SORT_COL.get(key)
    return col_span(_fit(table_cols(st, width), width), header) if header else None


#: 종목 목록의 열. 누적 기여율이 순매수 바로 옆에 붙고, 80% 를 처음 넘긴 행에
#: 가로 마커가 선다 — 그 위가 "이 섹터를 움직인 종목", 아래는 스크롤할 0 이다.
_NAME_COLS = (
    Col("종목", 14, False, lambda t, st: t.get("name", "—")),
    Col("코드", 7, False, lambda t, st: t.get("code", "")),
    Col("순매수[억]", 13, True, lambda t, st: fmt_amt(t.get("flow"))),
    Col("누적[%]", 8, True,
        lambda t, st: "—" if t.get("cum") is None else f"{t['cum']:.0f}"),
    Col("", 1, False, lambda t, st: "-" if t.get("cut") else ""),
    Col("참여율[%]", 9, True, lambda t, st: fmt_pct(t.get("part"), 1)),
    Col("시총대비[%p]", 12, True,
        lambda t, st: fmt_pct(t["a"]) if t.get("a") is not None else "—"),
    Col("시총[억]", 13, True,
        lambda t, st: fmt_amt(t["cap"]).replace("+", "") if t.get("cap") else "—"),
    Col("거래대금[억]", 13, True,
        lambda t, st: fmt_amt(t.get("tv")).replace("+", "")),
)


def names_cols() -> list[Col]:
    return list(_NAME_COLS)


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
    # 표에서 막대로 바뀐 미실현과, 넓은 폭에서만 보이는 포텐셜의 **정확한 값**을
    # 여기 둔다. 도형은 순서를, 숫자는 크기를 말한다.
    x, u = r.get("x"), r.get("U")
    exact = "" if x is None else f" · 미실현 {fmt_pct(x, 1)}%p"
    exact += "" if u is None else f"(포텐셜 {u:.0f})"
    return [pad(f" {r.get('sector','—')} · 종목 {n}개 · Enter 로 전체{exact}", width),
            pad(side("buy", "순매수 상위"), width),
            pad(side("sell", "순매도 상위"), width)]


def names_lines(st: State, width: int) -> tuple[list[str], int]:
    """종목 목록 화면 — (행, 헤더 줄 수).

    ⚠️ 본문 행은 `st.names()` 와 **1:1** 이어야 한다. 80% 가로줄을 별도 행으로
    끼워 넣으면 화면 선택(`st.drow`)이 그 아래 전 종목에서 한 칸씩 어긋난다 —
    그래서 줄은 행 안의 1칸 마커 열로 긋는다.
    """
    r = st.selected()
    if not r:
        return [pad(" (섹터를 고르라)", width)], 1
    names = st.names()
    title = (f" {r.get('sector','—')} · 종목 {len(names)}개 · {st.window}일 기준"
             f" · 정렬[{NAME_SORTS[st.nsi][1]}]")
    cols = _fit(names_cols(), width)
    head = pad(" ".join(pad(c.header, c.width, c.right) for c in cols), width)
    out = [pad(title, width), head]
    for t in names:
        out.append(_render(cols, t, st, width))
    if not names:
        out.append(pad("  (이 시장·섹터에 종목이 없다)", width))
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
    # 발산 막대 — 기준선 왼쪽은 음수, 오른쪽은 양수. 막대에는 부호 문자가 없어
    # 위 정규식이 못 잡는다. 색까지 빠지면 왼쪽/오른쪽만으로 부호를 읽어야 한다.
    i = 0
    while i < len(line):
        if line[i] != BAR_FILL:
            i += 1
            continue
        j = i
        while j < len(line) and line[j] == BAR_FILL:
            j += 1
        if j < len(line) and line[j] == BAR_AXIS:
            spans.append((cell_at[i], cell_at[j] - cell_at[i], "down"))
        elif i > 0 and line[i - 1] == BAR_AXIS:
            spans.append((cell_at[i], cell_at[j] - cell_at[i], "up"))
        i = j
    return spans


#: 도움말 — 열의 뜻과 계산식. 화면에서 읽는 사람이 "이 숫자가 뭐냐" 를 물을 자리에
#: 답을 둔다. 한계도 같이 적는다(추정치·상대지표·유동성 함정).
HELP = [
    ("", "── 키 ──"),
    ("↑↓ j k", "한 줄 이동. g·Home 처음 · G·End 끝 · PgUp/PgDn 한 화면씩."),
    ("Enter l →", "그 섹터의 전 종목 보기(드릴다운)."),
    ("h ← Esc", "드릴다운에서 나가기. h 는 vim 의 '왼쪽' 이라 l 의 반대다."),
    ("w W", "구간 5·20·60·120·종합. 대문자는 역방향. 드릴다운에서도 듣는다."),
    ("m M", "시장 전체·거래소·코스닥. 대문자는 역방향."),
    ("a A", "주체 기관·외국인·개인·기타법인. 대문자는 역방향."),
    ("s S", "정렬 열 바꾸기. 대문자는 역방향. 드릴다운에서는 종목 정렬."),
    ("r", "정렬 역순 토글. ▼ 내림차순(큰 것 먼저) · ▲ 오름차순 —"),
    ("", "        **순매도 상위**(가장 많이 판 쪽)는 이걸 켜야 위로 온다."),
    ("? F1", "이 도움말. q·Esc·?·Enter 로 닫는다."),
    ("q", "종료. 단 도움말 안에서는 **닫기만** 한다(한 번 더 눌러야 종료)."),
    ("", ""),
    ("", "── 섹터 표 ──"),
    ("섹터", "벤더 분류(stocks.sector) 27개. KRX 업종 분류와 다르다."),
    ("종목[수]", "그 (시장,섹터)의 상장 종목 수. 10개 미만은 **~ 마커**와 회색 — 벤더"),
    ("", "        분류가 좁아 사실상 단일종목인 라벨이 있다(부동산 3, 출판/매체복제 2)."),
    ("", "        색이 없는 터미널에서도 남도록 마커를 글자로 둔다. 오른쪽 끝 열이다."),
    ("G[0~1]", "성장 점수 0~1. 세 순위의 평균 — 힘(가속도)·압축(미실현)·풀림(ẍ)."),
    ("", "        * 는 세 조건을 다 만족(a>0 · x>0 · ẍ>0). 검증된 적 없는 탐색 지표다."),
    ("임펄스[억]", "구간 누적 순매수 [억원] = Σ(순매매 수량 × 그날 종가)."),
    ("", "        DB 는 수량만 주므로 금액은 종가 환산 근사다(참값은 VWAP 가중)."),
    ("가속[%p]", "임펄스 ÷ 구간말 섹터 시총 × 100 [%p]. 물리로 a = F/m."),
    ("", "        금액만 보면 대형 섹터가 늘 이기므로 쏠림은 이쪽이 정직하다."),
    ("1년[%ile]", "그 주체의 롤링 N일 순매수 합이 **최근 1년(260거래일) 분포에서 몇 등**인가"),
    ("", "        [0~100]. 100=1년 최대 매수 · 0=최대 매도 · 50=평범. 동률은 중간순위."),
    ("", "        z 를 안 쓴다 — 구간이 겹쳐 자기상관이 크고 260일에 비겹침 20일 구간은"),
    ("", "        13개뿐이라 z 는 정밀도를 과장한다. 이 표에서 **유일한 시계열 맥락**이다"),
    ("", "        (나머지 열은 전부 27개 섹터 사이의 상대순위다)."),
    ("추이[8]", "구간을 8개 **비겹침** 조각으로 나눈 순매수 합의 모양. 오른쪽이 최근이고"),
    ("", "        8칸을 다 더하면 임펄스 열의 그 숫자다. 높이는 **그 행 안에서만** 비교된다"),
    ("", "        — 행끼리 크기를 견주려면 임펄스를 봐라. 짧은 구간은 왼쪽이 빈다."),
    ("수익률[%]", "그 섹터 **자체 바구니**의 구간 수익률 [%], 전일 시총 가중."),
    ("", "        KRX 업종지수가 아니다 — 구성종목이 달라 분자·분모가 어긋난다."),
    ("예상Δv", "k × 가속 + b. k·b 는 그 구간 27개 섹터의 횡단면 회귀(절편 포함 OLS)."),
    ("미실현[%p]", "예상Δv − 실제 수익률 [%p]. 0 기준 **발산 막대** — 기준선 오른쪽이"),
    ("", "        +(덜 갔다·눌림), 왼쪽이 −(이미 더 갔다). 길이는 화면 안 |최댓값| 대비라"),
    ("", "        절대 크기가 아니다. 정확한 값은 하단 상세 패널 첫 줄에 있다."),
    ("", "        OLS 잔차의 부호 반전이라 27개 합이 0 이다 — **상대** 지표다."),
    ("포텐셜[½kx²]", "½·k·x². k 가 블록당 상수라 **|미실현| 의 순증가 변환**이다 — 4개 구간"),
    ("", "        전부 Spearman 1.0000(실측). 즉 미실현 막대와 같은 크기를 부호 없이"),
    ("", "        다시 적은 값이라 정보가 겹친다. 넓은 폭에서만 보이고 정렬 옵션으로 남겼다."),
    ("dW/dt[%p/일]", "구간을 반으로 갈라 W=가속×수익률 의 변화. 힘과 운동이 정렬되는가."),
    ("", "        오른쪽 끝 열이라 넓은 폭에서만 보인다. 열은 폭이 모자라면 **경계에서**"),
    ("", "        통째로 빠진다 — 숫자가 자릿수 중간에서 잘려 다른 값처럼 보이지 않게."),
    ("풀림[%p/일²]", "미실현 x 가 해소되는 **가속**(ẍ). 구간을 셋으로 갈라 중앙차분한다."),
    ("", "        2차 차분이라 짧은 구간(조각 6~7일)에서는 값이 흔들린다."),
    ("순매수상위[억]", "그 섹터에서 기관이 **가장 많이 산** 종목과 그 금액[억원]."),
    ("순매도상위[억]", "**가장 많이 판** 종목. 둘은 같은 목록(금액순)의 위/아래 끝이다."),
    ("", "        표는 1개씩, 하단 패널은 3개씩 — 같은 목록이고 시총과 무관하다."),
    ("", "        순매도상위는 화면이 더 넓어야 보인다. Enter 로 전 종목."),
    ("N일[G]", "종합 화면에서 그 구간의 G. 20일G·60일G·120일G 로 뜬다."),
    ("통과[구간]", "세 조건(a>0·x>0·ẍ>0)을 만족한 구간 수 / 전체 구간 수. 예 2/3."),
    ("", ""),
    ("", "── 종목 목록 (Enter) ──"),
    ("순매수[억]", "그 구간 **선택된 주체**의 순매수 [억원]. 기본 정렬은 이것의 **절댓값**"),
    ("", "        순 — 많이 산 종목과 많이 판 종목이 같이 위로 온다."),
    ("누적[%]", "|순매수| 큰 순으로 훑을 때의 **누적 기여율**. '-' 마커가 80% 지점이다 —"),
    ("", "        그 위가 이 섹터를 움직인 종목이고 아래는 사실상 0 이다."),
    ("", "        실측: 전기/전자 386종목 중 **7개**가 80% 를 설명한다."),
    ("", "        순매수 정렬에서만 채운다 — 시총·이름 순의 누적은 뜻이 없다."),
    ("참여율[%]", "순매수 ÷ 그 종목 거래대금 × 100 [%]. '얼마나 붐볐나' 가 아니라"),
    ("", "        '그 거래의 몇 %가 한 방향이었나' 다. 시총대비와 달리 분모가 유동성이라"),
    ("", "        **살 수 있는 종목인가** 를 같이 말해준다."),
    ("시총대비[%p]", "순매수 ÷ 그 종목 시총 × 100 [%p]. 시총 작은 스팩이 위로 올라온다."),
    ("시총[억]", "그 종목의 구간말 시가총액 [억원]. 시총대비의 분모다."),
    ("거래대금[억]", "그 구간 거래대금 [억원]."),
    ("", ""),
    ("", "── 알아둘 것 ──"),
    ("", "· k 는 추정치다. 구간마다 다르다(5일 15.9·20일 14.7·60일 9.2·120일 9.0)."),
    ("", "· 위 관계는 **동시기**다. 사면 오른다는 것이지 미래를 예측하지 않는다."),
    ("", "  예측 가설은 따로 검정해 기각됐다(research/logs/inst_flow_accel)."),
    ("", "· 값 0 은 '관망' 이 아니라 '0 또는 미보고' 다 — 수집기가 파싱 실패를 0 으로 준다."),
    ("", "· 코스닥 단독은 관계가 약하다(R² 0.00~0.25). 거래소가 전체를 끌고 간다."),
]


#: 도움말 제목 — 넓은 것부터. 폭에 안 들어가면 다음 단계로 내려간다.
HELP_TITLE_TIERS = (
    "키와 열의 뜻 — ↑↓/PgDn 스크롤 · q·Esc·?·Enter 로 닫기 (종료는 닫은 뒤 q 를 한 번 더)",
    "키와 열의 뜻 — ↑↓ 스크롤 · q 로 닫기(종료는 한 번 더)",
    "키와 열의 뜻 — q 로 닫기(종료는 한 번 더)",
    "q 로 닫기(종료는 한 번 더)",
    "q 닫기",
)


def cell_len(text: str) -> int:
    """표시 칸 수. 문자 수가 아니다."""
    return sum(cell_width(c) for c in text)


def help_lines(width: int, offset: int, height: int) -> tuple[list[str], int]:
    """도움말 화면 — (행, 전체 줄 수). offset 부터 height 줄을 낸다."""
    # 제목도 푸터처럼 **폭에 맞춰 단계별로** 줄인다. 한 줄 고정이면 85칸짜리
    # 문장이 80칸(SSH 기본)에서 단어 중간에 잘린다 — 푸터에 단계를 만든 바로
    # 그 이유가 이 줄에도 그대로 적용된다.
    title = next(t for t in HELP_TITLE_TIERS if cell_len(t) + 1 <= width)
    out = [pad(" " + title, width)]
    body = []
    for name, desc in HELP:
        # ⚠️ f"{name:>9}" 는 **문자 폭**이라 한글 라벨(임펄스=6칸)과 ASCII(G=1칸)가
        # 어긋난다. pad 는 표시 칸으로 맞춘다 — 이 저장소가 세 번 밟은 함정이다.
        #
        # **강조** 는 소스에서 눈에 띄라고 쓴 표기지 화면에 나갈 글자가 아니다.
        # 힌트바는 떼는데 여기는 안 떼서 `**닫기만**` 이 그대로 보였다.
        desc = desc.replace("**", "")
        body.append(pad((pad(name, 10, right=True) + "  " + desc) if name
                        else ("   " + desc), width))
    view = body[offset:offset + max(height - 1, 1)]
    return out + view, len(body)


#: 푸터는 폭에 맞춰 **단계별로** 줄인다. 예전엔 한 줄 고정이라 좁은 터미널에서
#: 잘렸고(무엇이 잘렸는지도 몰랐고), 넓은 터미널에서는 절반이 비어 있는데도
#: 대문자 역방향·g/G·PgUp/PgDn·r 이 화면 어디에도 안 적혀 있었다.
FOOTER_TIERS = (
    " w/W:구간 m/M:시장 a/A:주체 s/S:정렬 r:역순 ↑↓:섹터 g/G:처음/끝"
    " PgUp/PgDn:쪽 Enter/l:종목 ?:도움말 q:종료",
    " w:구간 m:시장 a:주체 s:정렬 r:역순 ↑↓:섹터 Enter:종목 ?:도움말 q:종료",
    " w m a s:바꾸기 r:역순 Enter:종목 ?:전체 키 q:종료",
    " w m a s r:바꾸기 Enter:종목 ?:키 q:종료",
    " ?:키 q:종료",
)
FOOTER_DRILL_TIERS = (
    " ↑↓:종목 s/S:정렬 r:역순 g/G:처음/끝 PgUp/PgDn:쪽 w/W:구간 m:시장 a:주체"
    " h/←/Esc:돌아가기 ?:도움말 q:종료",
    " ↑↓:종목 s:정렬 r:역순 w:구간 m:시장 a:주체 h/←:돌아가기 ?:도움말 q:종료",
    " s:정렬 r:역순 w m a:바꾸기 h:돌아가기 ?:전체 키 q:종료",
    " s r w m a:바꾸기 h:뒤로 ?:키 q:종료",
    " ?:키 h:뒤로 q:종료",
)
#: 예전 이름 — 중간 단계가 기본이다.
FOOTER = FOOTER_TIERS[1]
FOOTER_DRILL = FOOTER_DRILL_TIERS[1]


def footer_line(width: int, drill: bool = False) -> str:
    """폭에 **온전히** 들어가는 가장 자세한 푸터.

    어느 단계에서도 ``?`` 는 남긴다 — 줄어든 푸터가 "여기가 전부" 로 읽히면
    안 되기 때문이다. 나머지 키는 ``?`` 뒤에 전부 적혀 있다.
    """
    tiers = FOOTER_DRILL_TIERS if drill else FOOTER_TIERS
    for t in tiers:
        if _w(t) <= width:
            return t
    return tiers[-1]
