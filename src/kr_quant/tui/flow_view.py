"""섹터 자금흐름 TUI 의 **순수 렌더 로직** — curses 를 import 하지 않는다.

화면 그리기(curses)와 무엇을 그릴지(여기)를 나눈다. 이 파일은 데이터와 상태를
받아 문자열 행 목록을 돌려주므로 단위 테스트가 된다. curses 를 섞으면 렌더가
터미널 없이는 검증 불가능해지고, 열 정렬이 어긋나도 아무도 못 잡는다 —
이 저장소는 그 실수를 이미 두 번 했다(표 헤더와 셀 개수 불일치).
"""

from __future__ import annotations

import os
import unicodedata
from collections import namedtuple

#: East Asian Width 가 'A'(Ambiguous) 인 글자를 **두 칸으로 센다**.
#:
#: 이 화면들은 `·` `—` `↑` `↓` `→` `×` `÷` `²` `½` `Δ` `Σ` `β` `▲` `▼` `※` `≠`
#: `…` `─` 같은 'A' 글자를 쓴다. 유니코드는 이 글자들의 폭을 **정하지 않았고**,
#: 터미널이 정한다 — 대부분 1칸이지만 한국어권에서 흔한 "ambiguous=wide" 설정
#: (PuTTY 의 "Treat CJK ambiguous chars as wide", iTerm2·mintty 의 같은 옵션)
#: 에서는 2칸으로 그린다. 그러면 `cell_width` 가 1로 센 칸이 2칸을 먹어 **그 줄
#: 오른쪽이 통째로 밀리고**, `pad(..., width)` 로 폭에 딱 맞춘 줄은 넘쳐서 다음
#: 줄로 접힌다(화면 전체가 어긋난다).
#:
#: 글자를 ASCII 로 바꾸는 대신 **세는 규칙을 터미널에 맞추는** 쪽을 골랐다.
#: 이유는 세 가지다. (i) 'A' 글자가 31종·440여 곳이라 일부만 바꾸면 나머지가
#: 그대로 밀린다 — 부분 치환은 문제를 줄일 뿐 없애지 못한다. (ii) `Σ` `Δ` `β`
#: `÷` 처럼 뜻을 지고 있어 ASCII 로 옮기면 길어지거나 읽기 나빠지는 것이 있다.
#: (iii) 폭 계산이 전부 `cell_width` 한 곳을 지나므로, 여기만 고치면 표·푸터·
#: 도움말·원장까지 한 번에 맞는다(원장 블록 문자 ``▁▂▃█▒▓▌`` 도 'A' 다).
#:
#: 켜지 않은 기본값에서는 예전과 **한 글자도 다르게 세지 않는다.** 실측: 폭 80
#: 에서 그려지는 14,473 줄이 기준과 바이트 단위로 같다. 켜면 그 터미널에서
#: 폭을 넘던 6,144 줄(42.5%, 최대 120칸)이 0 이 된다.
#:
#: ⚠️ **아직 남은 것** — 켜면 줄이 밀리지는 않지만 내용이 좁아진다. `²` `½` 가
#: 든 헤더(`풀림[%p/일²]`·`포텐셜[½kx²]`)는 폭이 딱 맞게 잡혀 있어 한두 칸
#: 잘리고, 도움말·힌트바는 한 단계 짧은 문구로 내려간다. 원장의 막대·히트맵은
#: 블록 문자가 1칸이라고 **가정하고 칸을 세므로**(`heat_cell`·`signed_bar`)
#: 모드를 켜면 그림이 어긋난다 — 그건 이 커밋이 안 건드린 자리다.
#: 기본값이 꺼짐이라 오늘 아무도 이걸 밟지 않지만, 켜는 사람은 알아야 한다.
AMBIGUOUS_WIDE = os.environ.get("KQ_AMBIGUOUS_WIDE", "").strip().lower() \
    not in ("", "0", "false", "no", "off")

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

    'A'(Ambiguous) 는 터미널이 정한다 — ``AMBIGUOUS_WIDE`` 주석을 보라.
    """
    eaw = unicodedata.east_asian_width(ch)
    if eaw in ("W", "F"):
        return 2
    return 2 if (AMBIGUOUS_WIDE and eaw == "A") else 1


def cell_len(text: str) -> int:
    """표시 칸 수 — 한글·전각은 두 칸. 문자 수가 아니다."""
    return sum(cell_width(c) for c in text)


def tier_for(tiers: tuple[str, ...], width: int) -> str:
    """폭에 **온전히** 들어가는 가장 자세한 단계. 하나도 안 들어가면 마지막 단계.

    긴 것부터 짧은 것까지 손으로 쓴 문구를 늘어놓고 폭에 맞는 것을 고르는 기법은
    이 화면들이 이미 네 군데에서 쓴다(푸터·드릴다운 푸터·도움말 제목·원장 배너,
    그리고 종합 힌트바). 네 벌이 각자 구현돼 있었고 이미 갈라져 있었다 —
    푸터는 못 맞으면 마지막 단계로 내려갔는데, 도움말 제목은 ``next()`` 를
    기본값 없이 써서 폭 6 이하에서 ``StopIteration`` 으로 **TUI 를 통째로
    죽였다**(실측: 폭 1·5·6 크래시, 7 부터 정상). 같은 사실을 네 번 적으면
    이렇게 갈라진다.

    폭을 넘겨 잘라내지 않는 이유는 푸터에 적힌 그대로다 — 잘린 안내문은
    "여기가 전부" 로 읽힌다. 줄이려면 **더 짧게 쓴 문장**으로 바꿔야 한다.

    호출자가 앞에 공백 따위를 덧붙인다면 그만큼 뺀 폭을 넘겨라.
    """
    return next((t for t in tiers if cell_len(t) <= width), tiers[-1])


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
#: 누적 경로의 높이 4단계. 브라유는 4행 × 2열이라 **아래에서부터 채우면**
#: 진짜 막대가 된다 — 세로 위치로 부호를 흉내내던 예전 글자표(⠛⠒⠀⠤⣤)보다
#: 읽기 쉽다. 그쪽은 0 이 빈칸이라 "값이 0" 과 "조각이 없다" 가 구분되지 않았고,
#: 기준선이 없어 위/아래를 잡을 데가 없었으며, 크기 단계도 둘뿐이었다.
#:
#: East Asian Width 가 'N'(Narrow) 이라 어떤 로케일에서도 1칸이다. 블록 문자
#: ``▁▂▃▄▅▆▇█`` 는 'A'(Ambiguous) 라 한글 로케일 터미널이 2칸으로 그릴 수 있고,
#: 그러면 그 행만 통째로 밀린다. 이 저장소는 폭 계산 함정을 이미 세 번 밟았다.
SPARK = ("⣀", "⣤", "⣶", "⣿")
SPARK_EMPTY = "⠀"      # 조각이 없는 자리(짧은 구간의 왼쪽). 값 0 과 구분된다.


def spark(vals, cells: int = 8) -> str:
    """구간 동안 **순매수가 누적된 경로**. 오른쪽 끝이 구간 끝이다.

    예전에는 조각별 순매수를 부호 있는 글자로 그렸다. 그건 "이 조각에 얼마가
    들어왔나" 라 조각마다 오르내려서, 정작 묻고 싶은 **"지금 들어오는 중인가,
    이미 끝났나"** 가 모양으로 안 드러났다. 누적으로 바꾸면 그게 기울기가 된다 —
    오른쪽으로 **올라가면 계속 들어오는 중**, 내려가면 빠져나가는 중, 평평하면 멈췄다.

    높이는 **그 행 안에서** 경로의 최저~최고를 4단계에 편다(0 도 범위에 넣는다).
    행끼리 높이는 비교되지 않는다 — 크기는 임펄스 열이 말한다.
    """
    vals = [v for v in (vals or []) if v is not None]
    if not vals:
        return pad("—", cells, right=True)
    cum, t = [], 0.0
    for v in vals:
        t += v
        cum.append(t)
    lo, hi = min(0.0, *cum), max(0.0, *cum)
    span = hi - lo
    out = ""
    for c in cum:
        # span 이 0 이면(전 구간 0) 전부 최저단계 — 평평한 바닥이 곧 "아무 일 없음".
        lv = 0 if not span else min(3, int((c - lo) / span * 3 + 0.5))
        out += SPARK[lv]
    # 조각이 cells 보다 적으면(짧은 구간) **왼쪽**을 비운다 — 오른쪽 끝이
    # 구간 끝이라는 약속을 깨지 않기 위해서다.
    return SPARK_EMPTY * max(0, cells - len(out)) + out


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
    Col("미실현[%p]", 10, True, _num("x", 1)),
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


def fit_widths(widths: list[int], total: int) -> int:
    """``total`` 칸에 **온전히** 들어가는 열 **개수**. 열 사이 공백 1칸을 센다.

    첫 열은 잘려도 남긴다 — 섹터 이름은 잘려도 뜻이 남지만, 숫자는 잘리면 다른
    값이 된다(-1,360 이 -1 로 보인다).

    열 표현이 아니라 **폭 목록**만 받는다. 흐름 화면의 열은 ``Col`` 이고 원장
    화면의 열은 3-튜플이라 같은 규칙이 두 벌 살고 있었다 — 둘을 잇는 검사가
    하나도 없어서, 한쪽을 고치면 다른 쪽이 조용히 옛 규칙을 유지하고 같은 앱의
    두 화면이 열을 다르게 자른다. 폭만 받으면 양쪽이 같이 쓸 수 있다.
    """
    n, used = 0, 0
    for w in widths:
        need = w + (1 if n else 0)
        if n and used + need > total:
            break
        n += 1
        used += need
    return n


def span_at(widths: list[int], i: int) -> tuple[int, int]:
    """``i`` 번째 열의 (시작 표시칸, 폭). 열 사이 공백 1칸을 더해가며 센다."""
    return sum(w + 1 for w in widths[:i]), widths[i]


def _fit(cols: list[Col], width: int) -> list[Col]:
    """폭에 **온전히** 들어가는 열까지만 남긴다."""
    return cols[:fit_widths([c.width for c in cols], width)]


def col_span(cols: list[Col], header: str) -> tuple[int, int] | None:
    """열 헤더의 (시작 표시칸, 폭). 같은 헤더가 둘이면 **앞의 것**."""
    widths = [c.width for c in cols]
    for i, c in enumerate(cols):
        if c.header == header:
            return span_at(widths, i)
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
            pad(_actors_line(r, width), width),
            pad(side("buy", "순매수 상위"), width),
            pad(side("sell", "순매도 상위"), width)]


#: 상세 패널의 4주체 줄 — 넓은 것부터. 폭이 모자라면 뒤에서 잘라 낸다.
_ACTORS_TIERS = (
    ("개인", "외국인", "기관", "기타법인"),
    ("개인", "외국인", "기관"),
    ("개인", "기관"),
)
_ACTOR_KEY = dict((ko, k) for k, ko in ACTORS)


def _actors_line(r: dict, width: int) -> str:
    """선택 섹터의 **4주체 순매수 분해**를 한 줄에 — 표에 없는 나머지 셋까지.

    화면이 한 번에 한 주체만 보여주므로 "기관이 팔았다" 까지는 알아도
    **누가 받았는지**는 앱을 바꿔야 알 수 있었다(`kq-ledger` 의 원장 화면).
    주식은 누가 사면 누가 판 것이고 4주체 합은 0 에 닫히므로, 그 답은
    같은 페이로드 안에 이미 있다 — 화면을 바꿀 이유가 없다.

    라벨은 안 붙인다. ``개인 -69 · 외국인 -1 · 기관 +14 · 기타법인 +56 [억]``
    은 무엇인지 딱 봐도 읽히고, 붙어 있던 ``반대편:`` 은 **틀린 이름이었다** —
    이 줄은 선택한 주체까지 포함해 넷을 다 적는데(위 예의 `기관 +14` 는 바로
    위 표에서 보던 그 값이다), "반대편" 은 선택 주체를 뺀 나머지를 뜻한다.
    라벨을 떼면 폭도 6칸 벌어 좁은 화면에서 `기타법인` 이 늦게 잘린다.

    잔여(= −Σ4주체)는 여기 안 적는다. 원장에는 그 열이 있지만, 다섯째 주체까지
    말하려면 설명이 붙어야 하고 그건 원장이 할 일이다.
    """
    for names in _ACTORS_TIERS:
        parts = []
        for ko in names:
            v = r.get(_ACTOR_KEY[ko])
            parts.append(f"{ko} {fmt_amt(v)}" if v is not None else f"{ko} —")
        # 앞의 공백 한 칸은 다른 패널 줄과 들여쓰기를 맞추는 것이다.
        line = " " + " · ".join(parts) + " [억]"
        if cell_len(line) + 1 <= width:
            return line
    return " —"


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
    ("1년[%ile]", "그 주체의 롤링 N일 순매수 합, **최근 1년(260거래일) 분포에서 몇 등**인가"),
    ("", "        [0~100]. 100=1년 최대 매수 · 0=최대 매도 · 50=평범. 동률은 중간순위."),
    ("", "        z 를 안 쓴다 — 구간이 겹쳐 자기상관이 크고 260일에 비겹침 20일 구간은"),
    ("", "        13개뿐이라 z 는 정밀도를 과장한다. 이 표에서 **유일한 시계열 맥락**이다"),
    ("", "        (나머지 열은 전부 27개 섹터 사이의 상대순위다)."),
    ("추이[8]", "구간 동안 순매수가 **누적된 경로**. 왼쪽이 구간 시작, 오른쪽이 끝이다."),
    ("", "        오른쪽으로 **올라가면 계속 들어오는 중** · 내려가면 빠져나가는 중 ·"),
    ("", "        평평하면 멈췄다. 임펄스가 같아도 앞에서 다 들어온 것과 지금"),
    ("", "        들어오는 것은 다른 이야기인데, 숫자 하나로는 그게 안 보인다."),
    ("", "        높이는 **그 행 안에서** 경로의 최저~최고를 4단계(⣀⣤⣶⣿)에 편 것이라"),
    ("", "        행끼리 비교되지 않는다 — 크기는 임펄스 열이 말한다."),
    ("", "        짧은 구간(5일)은 조각이 모자라 왼쪽이 빈다(⠀)."),
    ("수익률[%]", "그 섹터 **자체 바구니**의 구간 수익률 [%], 전일 시총 가중."),
    ("", "        KRX 업종지수가 아니다 — 구성종목이 달라 분자·분모가 어긋난다."),
    ("예상Δv", "k × 가속 + b. k·b 는 그 구간 27개 섹터의 횡단면 회귀(절편 포함 OLS)."),
    ("미실현[%p]", "예상Δv − 실제 수익률 [%p]. + 면 덜 갔고(눌림), − 면 이미 더 갔다."),
    ("", "        OLS 잔차의 부호 반전이라 27개 합이 0 이다 — **상대** 지표다."),
    ("", "        OLS 잔차의 부호 반전이라 27개 합이 0 이다 — **상대** 지표다."),
    ("포텐셜[½kx²]", "½·k·x². k 가 블록당 상수라 **|미실현| 의 순증가 변환**이다 — 4개 구간"),
    ("", "        전부 Spearman 1.0000(실측). 즉 미실현 막대와 같은 크기를"),
    ("", "        부호 없이 다시 적은 값이라 정보가 겹친다. 넓은 폭에서만 보이고"),
    ("", "        정렬 옵션으로 남겼다."),
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
    ("", "        '그 거래의 몇 %가 한 방향이었나' 다. 시총대비와 달리 분모가"),
    ("", "        유동성이라 **살 수 있는 종목인가** 를 같이 말해준다."),
    ("시총대비[%p]", "순매수 ÷ 그 종목 시총 × 100 [%p]. 시총 작은 스팩이 위로 올라온다."),
    ("시총[억]", "그 종목의 구간말 시가총액 [억원]. 시총대비의 분모다."),
    ("거래대금[억]", "그 구간 거래대금 [억원]."),
    ("", ""),
    ("", "── 알아둘 것 ──"),
    ("", "· k 는 추정치다. 구간마다 다르다(5일 15.9·20일 14.7·60일 9.2·120일 9.0)."),
    ("", "· 위 관계는 **동시기**다. 사면 오른다는 것이지 미래를 예측하지 않는다."),
    ("", "  예측 가설은 따로 검정해 기각됐다(research/logs/inst_flow_accel)."),
    ("", "· 값 0 은 '관망' 이 아니라 '0 또는 미보고' 다 — 수집기가 파싱 실패를"),
    ("", "  0 으로 준다."),
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


def help_desc(entries: list[tuple[str, str]], name: str) -> str:
    """도움말 목록에서 항목 하나의 설명. 없으면 빈 문자열.

    ``**강조**`` 는 소스 표기라 여기서 뗀다 — 한 줄 힌트로 쓰는 쪽은 별표를
    통과 마커(``*``)와 헷갈리기만 한다.
    """
    for n, desc in entries:
        if n == name:
            return desc.strip().replace("**", "")
    return ""


#: 힌트바 전용 **짧은 설명** — 열 이름 → 한 줄. 없으면 :data:`HELP` 로 떨어진다.
#:
#: 왜 두 벌인가: 같은 문장을 길이 제약이 **정반대**인 두 자리에 쓰고 있었다.
#: 도움말은 86줄 모달이라 비유·유래·주의사항이 다 들어가도 되지만, 힌트바는
#: 한 줄이라 그 뒤가 `…` 로 잘려나간다 — 그리고 잘린 설명은 설명이 아니다
#: (:func:`hint_line` 독스트링이 스스로 인정하는 문제였다).
#:
#: 실제로 가속 정렬에서 화면에 늘 떠 있던 줄이 이랬다:
#: ``정렬 가속[%p]▼ · 임펄스 ÷ 구간말 섹터 시총 × 100 [%p]. 물리로 a = F/m.``
#: 값을 읽는 데 필요한 것은 "임펄스 ÷ 시총" 까지고, 물리 비유는 한 줄짜리
#: 힌트바에서 아무 일도 안 한다. 게다가 정확하지도 않다 — 여기 `a` 는 자금
#: 축의 0차 비율(F/m)이고 `ẍ`(풀림)는 가격 갭의 2차 시간미분인데, 둘 다
#: "가속" 이라 불러서 읽는 사람을 헷갈리게 했다. 비유는 도움말에 남는다.
#:
#: 규칙: **그 숫자를 읽는 데 필요한 것만** — 분자·분모와 단위까지. 비유·유래·
#: 한계는 `?` 의 일이다. 길이는 폭 80 에서 `…` 로 잘리지 않는 선을 지킨다
#: (검사가 전 열 × 폭 80 을 돌며 확인한다).
HINT_DESC = {
    # 섹터 표
    "가속[%p]": "임펄스 ÷ 구간말 섹터 시총 × 100 [%p]",
    "임펄스[억]": "구간 누적 순매수 [억원] — 수량 × 그날 종가",
    "1년[%ile]": "이 순매수가 최근 1년 분포에서 몇 등 [0~100]",
    "추이[8]": "구간 동안 순매수가 누적된 경로. 왼쪽이 시작",
    "수익률[%]": "그 섹터 바구니의 구간 수익률 [%], 시총 가중",
    "미실현[%p]": "예상Δv − 실제 수익률 [%p]. + 면 덜 갔다",
    "풀림[%p/일²]": "미실현이 해소되는 2차 변화 [%p/일²]",
    "G[0~1]": "가속·미실현·풀림 세 순위의 평균 [0~1]",
    "포텐셜[½kx²]": "½·k·x² — |미실현| 을 부호 없이 다시 적은 값",
    "dW/dt[%p/일]": "전·후반 W=가속×수익률 의 변화 [%p/일]",
    "종목[수]": "그 (시장,섹터)의 상장 종목 수. ~ 는 10개 미만",
    # 종목 목록(드릴다운)
    "종목": "종목명 가나다순",
    "순매수[억]": "선택 주체의 구간 순매수 [억원], 절댓값 순",
    "누적[%]": "|순매수| 큰 순으로 훑을 때의 누적 기여율. - 가 80%",
    "참여율[%]": "순매수 ÷ 그 종목 거래대금 × 100 [%]",
    "시총대비[%p]": "순매수 ÷ 그 종목 시총 × 100 [%p]",
    # 이 둘은 **두 화면이 같은 헤더를 쓴다** — 섹터 표에서는 섹터 합계고
    # 드릴다운에서는 종목 값이다. 어느 쪽에서도 참인 말로 적는다.
    "시총[억]": "구간말 시가총액 [억원] — 가속·시총대비의 분모",
    "거래대금[억]": "그 구간 거래대금 [억원]",
}


def hint_desc(header: str) -> str:
    """힌트바 한 줄 설명 — 짧은 것이 있으면 그것, 없으면 도움말의 긴 설명.

    폴백을 남기는 이유: 열이 하나 늘었는데 여기 안 적히면 힌트바가 **비어**
    버린다. 긴 설명은 잘리기라도 하지만 빈 줄은 아무 말도 안 한다.
    """
    return HINT_DESC.get(header) or help_desc(HELP, header)


def hint_line(head: str, desc: str, width: int) -> str:
    """항상 보이는 한 줄 힌트 — ``head · desc`` 를 폭에 **맞춰서** 낸다.

    설명이 자리를 못 채우면 열 이름만 남긴다. 한두 글자로 잘린 설명은 설명이
    아니다. ``kq-flow`` 와 ``kq-ledger`` 가 같은 자리에 같은 줄을 그리므로
    자르기 규칙도 한 곳에 둔다 — 예전에 flow 쪽만 폭을 안 봐서 폭 40 에서
    78칸짜리 줄이 나갔고, 그 실수를 원장에서 다시 하지 않으려면 함수가 하나여야 한다.
    """
    room = width - cell_len(head) - 4          # " · " 세 칸 + 여유 한 칸
    if room < 2:
        return pad(head, width)
    if cell_len(desc) > room:
        cut, used = "", 0
        for ch in desc:
            w2 = cell_width(ch)
            if used + w2 > room - 1:
                break
            cut += ch
            used += w2
        desc = cut.rstrip() + "…"
    return head + " · " + desc


def is_section(line: str) -> bool:
    """도움말에서 **구역 제목 줄**인가 — ``── 키 ──`` ``── 섹터 표 ──``.

    색을 입히는 쪽(`flow_app`)이 문자열을 다시 뜯어 맞히면 문구를 고칠 때
    조용히 어긋난다. 줄을 만드는 쪽이 판정을 진다 — 이 저장소가 헤더와 셀,
    히트맵과 색에서 이미 두 번 밟은 자리다.
    """
    return line.strip().startswith("──")


def help_lines(width: int, offset: int, height: int,
               entries: list[tuple[str, str]] | None = None,
               title_tiers: tuple[str, ...] = HELP_TITLE_TIERS,
               label_w: int = 10) -> tuple[list[str], int]:
    """도움말 화면 — (행, 전체 줄 수). offset 부터 height 줄을 낸다.

    ``entries`` 와 ``title_tiers`` 를 받는 이유: **원장도 같은 도움말이 필요하다.**
    ``kq-flow`` 와 ``kq-ledger`` 는 같은 제품이고, 도움말이 두 벌이면 렌더 규칙
    (라벨 폭 · 이어쓰기 들여쓰기 · ``**`` 떼기 · 스크롤 하한)이 반드시 갈라진다.
    이 저장소는 오늘 폭 단계 고르기가 네 벌로 갈라져 있던 걸 :func:`tier_for`
    하나로 합쳤다 — 같은 실수를 도움말에서 반복하지 않는다. **내용만** 다르다.
    """
    # 제목도 푸터처럼 **폭에 맞춰 단계별로** 줄인다. 한 줄 고정이면 85칸짜리
    # 문장이 80칸(SSH 기본)에서 단어 중간에 잘린다 — 푸터에 단계를 만든 바로
    # 그 이유가 이 줄에도 그대로 적용된다.
    title = tier_for(title_tiers, width - 1)      # 앞의 공백 한 칸
    out = [pad(" " + title, width)]
    body = []
    for name, desc in (HELP if entries is None else entries):
        # ⚠️ f"{name:>9}" 는 **문자 폭**이라 한글 라벨(임펄스=6칸)과 ASCII(G=1칸)가
        # 어긋난다. pad 는 표시 칸으로 맞춘다 — 이 저장소가 세 번 밟은 함정이다.
        #
        # **강조** 는 소스에서 눈에 띄라고 쓴 표기지 화면에 나갈 글자가 아니다.
        # 힌트바는 떼는데 여기는 안 떼서 `**닫기만**` 이 그대로 보였다.
        desc = desc.replace("**", "")
        if not name:
            body.append(pad("   " + desc, width))
        elif cell_len(name) > label_w:
            # 라벨이 열보다 길면 **자기 줄에 온전히** 둔다. `pad` 로 자르면
            # `순매수상위[억]` 이 `순매수상위` 가 되어 표 헤더와 이름이 안 맞고,
            # 그러면 도움말이 어느 열 설명인지 알려주지 못한다. 열을 넓히면
            # 설명이 오른쪽으로 밀려 폭 80(SSH 기본)을 넘는다 — 줄을 하나 쓰는
            # 편이 싸다. (flow 에서 7개가 이렇게 잘려 있었다.)
            body.append(pad(name, width))
            body.append(pad(" " * (label_w + 2) + desc, width))
        else:
            body.append(pad(pad(name, label_w, right=True) + "  " + desc, width))
    view = body[offset:offset + max(height - 1, 1)]
    return out + view, len(body)


#: 푸터는 폭에 맞춰 **단계별로** 줄인다. 예전엔 한 줄 고정이라 좁은 터미널에서
#: 잘렸고(무엇이 잘렸는지도 몰랐고), 넓은 터미널에서는 절반이 비어 있는데도
#: 대문자 역방향·g/G·PgUp/PgDn·r 이 화면 어디에도 안 적혀 있었다.
#:
#: 그 뒤 반대로 기울었다 — ``w/W m/M a/A s/S`` 처럼 **한 키의 두 방향을 다 적으니**
#: 푸터가 길어져서 정작 무슨 키가 있는지가 안 읽혔다. 이제 푸터는 **소문자 한
#: 벌만** 적는다. 대문자 역방향·``G``·``l``·``←``·``Esc`` 는 **여전히 듣고**,
#: :data:`HELP` 의 키 절에 그대로 적혀 있다 — 화면 어디에도 안 적혀 있으면
#: 없는 기능이라는 원칙은 그대로다. 바뀐 것은 **어느 화면에 적히느냐** 뿐이다.
FOOTER_TIERS = (
    " w:구간 m:시장 a:주체 s:정렬 r:역순 ↑↓:섹터 g:처음"
    " PgUp/PgDn:쪽 Enter:종목 ?:도움말 q:종료",
    " w:구간 m:시장 a:주체 s:정렬 r:역순 ↑↓:섹터 Enter:종목 ?:도움말 q:종료",
    " w m a s:바꾸기 r:역순 Enter:종목 ?:전체 키 q:종료",
    " w m a s r:바꾸기 Enter:종목 ?:키 q:종료",
    " ?:키 q:종료",
)
FOOTER_DRILL_TIERS = (
    " ↑↓:종목 s:정렬 r:역순 g:처음 PgUp/PgDn:쪽 w:구간 m:시장 a:주체"
    " h:돌아가기 ?:도움말 q:종료",
    " ↑↓:종목 s:정렬 r:역순 w:구간 m:시장 a:주체 h:돌아가기 ?:도움말 q:종료",
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
    return tier_for(FOOTER_DRILL_TIERS if drill else FOOTER_TIERS, width)
