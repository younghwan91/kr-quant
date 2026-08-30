"""섹터 자금흐름 TUI — SSH 에서 바로 보는 화면.

렌더 로직은 :mod:`kr_quant.tui.flow_view` 에 있고 여기는 **화면 그리기와 키 입력만**
한다. 표준 라이브러리 curses 만 쓰므로 새 의존성이 없다.

데이터는 일일 리포트가 이미 만들어 둔 ``numbers.html`` 안의 JSON 을 읽는다 —
DB 에 접속하지 않으므로 즉시 뜨고, 화면과 표가 **같은 숫자**를 보게 된다.

키 처리(`handle_key`)와 세로 배치(`layout`)는 **curses 를 타지 않는 순수 함수**다.
pty 검사만으로는 "통과했는데 아무것도 안 한" 검사가 반복해서 나왔다 — 키 하나가
무엇을 했는지는 상태를 직접 보고 판정하는 게 정직하다.

Run:  kq-flow                       # ~/Documents/kr-quant-reports/latest
      kq-flow --dir <리포트 폴더>
"""

from __future__ import annotations

import argparse
import curses
import json
import locale
import os
import re
from collections import namedtuple

from kr_quant.tui.flow_view import (
    HELP_FOOT_TIERS, NAME_SORT_COL, all_lines, NAME_SORTS, SORT_COL, SORTS,
    State, cell_len as view_cell_len, color_spans,
    detail_lines, detail_title_span, footer_line, header_lines, help_lines,
    hint_desc, hint_line,
    is_section, name_sort_span, names_lines, sort_span, table_lines, tier_for,
    view_width)

DEFAULT_DIR = "~/Documents/kr-quant-reports/latest"


def load(report_dir: str) -> dict:
    path = os.path.join(os.path.expanduser(report_dir), "numbers.html")
    if not os.path.exists(path):
        raise SystemExit(f"리포트를 찾을 수 없다: {path}\n"
                         f"  먼저 scripts/daily_report.sh 를 돌리거나 --dir 로 지정하라.")
    html = open(path, encoding="utf-8").read()
    m = re.search(r"const D = (\{.*?\});\n", html, re.S)
    if not m:
        raise SystemExit(f"{path} 에서 데이터를 못 읽었다 — 리포트 형식이 바뀌었나?")
    # 아래 두 갈래는 예전엔 curses 안에서 생 트레이스백으로 터졌다. 방어가 행
    # 수준(.get)에만 있고 문서 수준에는 없었다.
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} 의 JSON 이 깨졌다 ({e.lineno}행 {e.colno}칸): {e.msg}\n"
                         f"  리포트를 다시 생성하라 — scripts/daily_report.sh") from None
    missing = [k for k in ("asof", "blocks") if not d.get(k)]
    if missing:
        raise SystemExit(f"{path} 의 데이터에 없는 키: {', '.join(missing)} — "
                         f"리포트 형식이 바뀌었나?\n  있는 키: {sorted(d)}")
    return d


# --- 색 -------------------------------------------------------------------
#
# 블룸버그 단말 배색 — **검은 바탕에 앰버**가 골격이고, 값만 부호색으로 튄다.
# 뼈대(헤더 띠·제목·힌트바)는 앰버, 본문은 밝은 회색, 비활성·얇은 섹터는
# 어두운 회색이다. 화면 대부분이 무채색이라 부호색 몇 군데가 눈에 걸린다.
#
# **부호-색 대응은 뒤집지 않는다.** 블룸버그는 상승이 녹색이지만 이 화면은
# 한국 시장 도구고, 이 저장소는 여태 상승·매수를 빨강으로 그려 왔다. 색조만
# 팔레트로 옮기고 방향은 그대로 둔다 — 방향을 바꾸면 화면을 읽던 사람의
# 눈이 먼저 틀린다(그건 배색 취향이 아니라 오독이다).
#
# 정렬 표시는 **헤더만** 한다. 한때 열 전체에 배경을 깔았는데, 배경 위에서
# 부호색이 묻혀 읽기 어려웠다. curses 는 색쌍 단위로만 칠해져 "배경만 바꾸기"
# 가 없으므로 (전경,배경) 조합을 다 만들어야 했고, 그렇게까지 해도 대비가
# 안 나왔다.
#
# 색으로만 전하는 정보는 **새로 만들지 않는다** — 얇은 섹터의 `~`, G 통과의
# `*` 는 글자로 남아 있고 부호는 `+`/`-` 로 적혀 있다. 색은 이미 적힌 것을
# 빨리 찾게 할 뿐이라, 8색 터미널이나 NO_COLOR 에서도 사라지는 뜻이 없다.
C_AMBER, C_UP, C_DOWN, C_HEAD, C_DIM, C_SEL, C_MARK, C_SORT, C_BODY = range(1, 10)

#: 256색 팔레트. xterm 색번호다 — 앰버 214 · 본문 회색 252 · 어두운 회색 240 ·
#: 상승 빨강 203 · 하락 청록 80 · 선택행 바탕 238.
RICH_BG, RICH_AMBER, RICH_BODY = 16, 214, 252
RICH_UP, RICH_DOWN, RICH_DIM = 203, 80, 240
RICH_MARK, RICH_SEL_BG, RICH_SEL_FG = 220, 238, 253
RICH_SORT_BG, RICH_SORT_FG = 24, 231

#: 색을 쓸 수 있는 터미널인가. curses.window 에는 속성을 붙일 수 없어서
#: (`scr._colored = ...` 는 AttributeError) 모듈 수준으로 둔다.
_COLORED = False
#: 256색 팔레트를 실제로 받아냈는가. 8색으로 내려가면 회색 톤이 없어
#: "어두운 회색" 을 `A_DIM` 으로 흉내내야 한다.
_RICH = False


def _dim_attr():
    """비활성 행의 속성 — 8색에서는 어두운 회색이 없어 `A_DIM` 으로 대신한다."""
    return curses.color_pair(C_DIM) | (0 if _RICH else curses.A_DIM)


def _sel_attr():
    """선택행 — 256색이면 **은은한 바탕**, 8색이면 반전.

    반전은 그 줄만 하얗게 타서 표 전체의 리듬을 끊는다. 다만 8색에는 검정과
    흰색 사이의 회색이 없어서 흉내낼 수단이 없다 — 그 터미널에서는 선택이
    보이는 쪽(반전)이 예쁜 쪽보다 낫다.
    """
    return curses.color_pair(C_SEL) | curses.A_BOLD


def _init_colors(scr=None) -> bool:
    global _COLORED, _RICH
    _COLORED = _RICH = False
    # NO_COLOR 표준(no-color.org): **비어 있지 않은** 값으로 설정돼 있으면 색을
    # 쓰지 않는다. 안 보던 시절엔 리다이렉트·로그 캡처에서 색이 그대로 나왔다.
    if os.environ.get("NO_COLOR"):
        return False
    if not curses.has_colors():
        return False
    curses.start_color()
    rich = getattr(curses, "COLORS", 0) >= 256
    if rich:
        try:
            # 바탕을 터미널 기본값(-1)이 아니라 **검정**으로 박는다. 이 배색의
            # 정체성이 검은 바탕이라, 밝은 테마 터미널에서 앰버만 얹으면
            # 대비가 무너진다(노란 글씨가 흰 종이 위에 뜬다).
            curses.init_pair(C_BODY, RICH_BODY, RICH_BG)
            curses.init_pair(C_AMBER, RICH_AMBER, RICH_BG)
            curses.init_pair(C_UP, RICH_UP, RICH_BG)
            curses.init_pair(C_DOWN, RICH_DOWN, RICH_BG)
            curses.init_pair(C_DIM, RICH_DIM, RICH_BG)
            curses.init_pair(C_MARK, RICH_MARK, RICH_BG)
            curses.init_pair(C_HEAD, RICH_BG, RICH_AMBER)
            curses.init_pair(C_SEL, RICH_SEL_FG, RICH_SEL_BG)
            curses.init_pair(C_SORT, RICH_SORT_FG, RICH_SORT_BG)
        except curses.error:
            rich = False        # 256색이라 말해놓고 못 받는 터미널이 있다
    if not rich:
        # 8색 대체. 앰버 자리는 노랑, 본문은 흰색, 어두운 회색은 `A_DIM` 이다.
        bg = curses.COLOR_BLACK
        curses.init_pair(C_BODY, curses.COLOR_WHITE, bg)
        curses.init_pair(C_AMBER, curses.COLOR_YELLOW, bg)
        curses.init_pair(C_UP, curses.COLOR_RED, bg)
        curses.init_pair(C_DOWN, curses.COLOR_CYAN, bg)
        curses.init_pair(C_DIM, curses.COLOR_WHITE, bg)
        curses.init_pair(C_MARK, curses.COLOR_YELLOW, bg)
        curses.init_pair(C_HEAD, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        curses.init_pair(C_SEL, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(C_SORT, curses.COLOR_WHITE, curses.COLOR_BLUE)
    # 안 쓴 칸까지 검게 — 배경이 반쪽이면 검은 바탕이 아니라 얼룩이 된다.
    if scr is not None:
        try:
            scr.bkgd(" ", curses.color_pair(C_BODY))
        except curses.error:
            pass
    _COLORED, _RICH = True, rich
    return True


def _put(scr, y: int, line: str, base, colored: bool, selected: bool = False) -> None:
    """한 줄 그리기 — 숫자 구간만 부호색으로 덧칠한다.

    선택행은 덧칠하지 않는다. curses 의 `chgat` 은 색쌍을 통째로 갈아치우므로
    부호색을 얹는 순간 그 칸만 선택 바탕이 벗겨져 강조 띠에 구멍이 뚫린다
    (8색에서 반전이던 시절에도 반전 위의 색은 읽기 어려웠다). 부호는 `+`/`-`
    글자로 그 줄에 그대로 남아 있으므로 사라지는 정보는 없다.
    """
    if y < 0:
        return
    try:
        scr.addstr(y, 0, line, base)
    except curses.error:
        return
    if not colored or selected:
        return
    for start, width, role in color_spans(line):
        attr = {"up": curses.color_pair(C_UP),
                "down": curses.color_pair(C_DOWN),
                "mark": curses.color_pair(C_MARK) | curses.A_BOLD}[role]
        try:
            scr.chgat(y, start, width, attr | (base & curses.A_BOLD))
        except curses.error:
            pass


def _hl_span(scr, y: int, span, attr) -> None:
    """뷰가 낸 (시작 표시칸, 폭) 구간에 속성을 덧칠한다.

    좌표를 **여기서 계산하지 않는다** — 어느 칸이 무엇인지는 문자열을 만든
    쪽만 안다(`col_span`·`name_sort_span`·`detail_title_span`).
    """
    if not span:
        return
    start, width = span
    try:
        scr.chgat(y, start, width, attr)
    except curses.error:
        pass


def _hl_sort(scr, y: int, span, col: bool) -> None:
    """정렬 중인 열 헤더를 덧칠한다 — 어떤 기준으로 줄세웠는지 한눈에 보이게."""
    _hl_span(scr, y, span,
             (curses.color_pair(C_SORT) if col else curses.A_UNDERLINE) | curses.A_BOLD)


# --- 세로 배치 -------------------------------------------------------------

#: 상세 패널을 접는 높이. 이 아래에서는 패널(3줄)이 표를 통째로 밀어냈다 —
#: 6줄 터미널에서 헤더 위에 패널이 겹쳐 그려져 표가 사라졌고 설명도 없었다.
DETAIL_MIN_H = 10

#: 표와 상세 패널 사이 **빈 줄**. 패널이 표 마지막 행에 딱 붙어 있어서 어디까지가
#: 표인지 눈이 못 끊었다 — 패널의 첫 줄(섹터 이름)이 표의 28번째 행처럼 읽혔다.
DETAIL_GAP = 1

#: 여백을 넣고도 표에 남아야 할 최소 행 수. 여백은 **가장 먼저 포기하는** 것이다 —
#: 폭에서 :func:`~kr_quant.tui.flow_view.tier_for` 가 문구를 단계적으로 줄이듯,
#: 높이에서도 없어도 되는 것부터 뺀다. 빈 줄 하나 때문에 패널 줄이나 표가 잘리면
#: 읽기 좋아지려던 것이 읽을 것을 없앤 셈이다.
GAP_MIN_ROWS = 3


#: 세로 배치 — 이름으로 꺼낼 수 있어야 필드가 하나 늘 때 호출자가 조용히
#: 어긋나지 않는다(예전 5-튜플에 여백이 끼어들면서 자리번호가 밀렸다).
Layout = namedtuple("Layout", "head rows gap detail hint_y foot_y")


def layout(h: int, drill: bool = False) -> Layout:
    """세로 배치 — (맨 위 헤더 줄 수, 보이는 표 행 수, 표·패널 사이 여백,
    상세 패널 줄 수, 힌트바 y, 푸터 y).

    그리는 쪽과 PgUp/PgDn 이 **같은 함수**를 봐야 어긋나지 않는다. 예전엔
    PgDn 이 화면 높이와 무관하게 10줄 고정이라 200x50 에서 반 페이지도 안 갔다.
    힌트바 y 가 음수면 그 줄을 그릴 자리가 없다는 뜻이다.

    자리가 모자랄 때 버리는 순서는 **여백 → 맨 위 헤더 → (그 아래로는 패널)** 이다.
    """
    foot_y = max(h - 1, 0)
    hint_y = h - 2 if h >= 5 else -1
    limit = hint_y if hint_y >= 0 else foot_y      # 표가 쓸 수 있는 y 의 끝(배타)
    nhead = 2 if drill else 1                      # 표 자체의 머리 줄(제목·열이름)
    # 상세는 우선순위 순서로 그린다(제목 → 반대편 → 순매수상위 → 순매도상위).
    # 한 줄이 더 필요해진 것은 **4주체 줄** 때문이다 — "기관이 팔았다" 다음에
    # 오는 질문이 "그럼 누가 받았지" 라, 그 답이 화면을 바꾸지 않고 여기 있다.
    # 자리가 모자라면 뒤에서부터 잘린다.
    if drill or h < DETAIL_MIN_H:
        detail, want_gap = 0, 0
    else:
        detail = 4 if h >= DETAIL_MIN_H + 1 else 3
        want_gap = DETAIL_GAP
    for head in (2, 1, 0):
        # 같은 헤더 높이 안에서 **여백을 먼저** 내려놓는다. 헤더를 먼저 버리면
        # 빈 줄 하나 지키려고 날짜·구간 줄이 사라진다.
        for gap in (want_gap, 0):
            rows = limit - head - nhead - gap - detail
            if rows >= (GAP_MIN_ROWS if gap else 1):
                return Layout(head, rows, gap, detail, hint_y, foot_y)
    return Layout(min(max(limit, 0), 2), 0, 0, 0, hint_y, foot_y)


# --- 힌트바 ---------------------------------------------------------------

#: 표에 열이 없어 `SORT_COL` 이 안 다루는 정렬키 → 도움말의 항목 이름.
_EXTRA_COL = {"tv": "거래대금[억]", "cap_idx": "시총[억]"}


def _help_desc(header: str) -> str:
    """힌트바에 쓸 열 설명 한 줄 — 짧은 것이 있으면 그것(``flow_view.hint_desc``).

    도움말의 긴 설명을 그대로 쓰던 시절엔 뒤가 `…` 로 잘려나갔고, 잘리지 않는
    자리에서는 비유(``물리로 a = F/m``)가 한 줄을 잡아먹었다. 고르는 규칙은
    문장을 가진 쪽(`flow_view`)에 둔다."""
    return hint_desc(header)


#: 종합 화면의 힌트바 — 넓은 것부터. 이 화면은 정렬이 **없어서** 열 설명 대신
#: "여기엔 정렬이 없다" 를 말한다. 어느 단계에서도 ``?`` 는 남긴다 — 줄어든 안내가
#: "여기가 전부" 로 읽히면 안 되기 때문이다(푸터와 같은 규칙).
HINT_COMBINED_TIERS = (
    " 종합 화면은 구간별 G 를 나란히 볼 뿐, 정렬·역순이 없다 · ? 로 열 설명",
    " 종합은 구간별 G 를 나란히 본다. 정렬·역순이 없다 · ? 로 설명",
    " 종합 — 구간별 G. 정렬·역순 없다 · ? 로 설명",
    " 종합 — 정렬·역순 없다 · ? 로 설명",
    " 종합 · 정렬 없다 ?",
)

#: 전 종목 화면의 힌트바 — 이 화면도 정렬이 **없다**(곱 내림차순 고정). 예전엔
#: 섹터 표의 힌트를 그대로 그려서 ``정렬 가속[%p]▼(내림차순, r 로 뒤집기)`` 라고
#: 적혀 있었다. 표는 곱 순인데 힌트는 가속으로 줄세웠다고 말했고, `r` 은 이 화면에서
#: 안 듣는다 — 화면이 자기가 무엇을 하는지 틀리게 말한 것이다. 종합 화면이 같은
#: 이유로 자기 힌트를 가진다.
HINT_ALL_TIERS = (
    " 전 종목 — 섹터선정 × 종목선정 의 곱 내림차순 고정. 정렬·역순이 없다 · ? 로 열 설명",
    " 전 종목 — 곱(섹터선정×종목선정) 내림차순 고정. 정렬·역순 없다 · ? 로 설명",
    " 전 종목 — 곱 내림차순 고정. 정렬·역순 없다 · ? 로 설명",
    " 전 종목 — 곱순 고정. 정렬 없다 · ? 로 설명",
    " 곱순 고정 · 정렬 없다 ?",
)


def hint_text(st: State, width: int = 200) -> str:
    """푸터 위 **항상 보이는 한 줄** — 지금 정렬 중인 열의 뜻.

    도움말이 전면 모달이라 "이 숫자가 뭐냐" 를 물은 사람이 **그 숫자를 보면서**
    답을 읽을 수 없었다. 41줄 모달은 한계·주의사항용으로 남기고, 지금 줄세운
    열 한 줄은 표 옆에 늘 둔다.
    """
    if st.drill:
        key = st.name_sort
        header = NAME_SORT_COL.get(key) or dict(NAME_SORTS).get(key, key)
        arrow = "▲" if st.nrev else "▼"
    elif st.allv:
        # 전 종목은 곱 내림차순 **고정**이다 — 없는 정렬을 있는 척하지 않는다.
        return tier_for(HINT_ALL_TIERS, width)
    elif st.window == "종합":
        # 종합은 페이로드 순서 그대로다 — 없는 정렬을 있는 척하지 않는다.
        return tier_for(HINT_COMBINED_TIERS, width)
    else:
        key, _fell = st.effective_sort(st.rows())
        header = (SORT_COL.get(key) or _EXTRA_COL.get(key)
                  or dict(SORTS).get(key, key))
        arrow = "▲" if st.rev else "▼"
    desc = _help_desc(header) or "? 로 설명"
    order = "오름차순" if arrow == "▲" else "내림차순"
    # 좁은 화면에서는 방향 풀이를 접는다 — 잘려나갈 자리는 열 설명에 준다.
    tag = f"({order}, r 로 뒤집기)" if width >= 90 else ""
    # 자르기는 `flow_view.hint_line` 하나가 한다 — 원장 힌트바도 같은 함수다.
    return hint_line(f" 정렬 {header}{arrow}{tag}", desc, width)


def _draw(scr, st: State) -> None:
    scr.erase()
    h, w = scr.getmaxyx()
    # curses 는 오른쪽 아래 칸에 쓰면 스크롤을 유발해 터진다 — 마지막 칸은 비운다.
    # 그 −1 은 뷰의 열 배치 검사도 알아야 하므로 규칙은 flow_view 에 하나만 둔다.
    w = view_width(w)

    col = _COLORED

    if st.help:
        lines, total = help_lines(w, st.hrow, h - 1)
        for i, line in enumerate(lines):
            if col and i == 0:
                base = curses.color_pair(C_HEAD) | curses.A_BOLD
            elif col and is_section(line):
                # 구역 제목은 앰버 — 86줄을 스크롤하는 화면에서 "여기서부터
                # 다른 이야기" 를 색이 먼저 말한다. 판정은 `flow_view` 가 진다.
                base = curses.color_pair(C_AMBER) | curses.A_BOLD
            else:
                base = curses.color_pair(C_BODY) if col else curses.A_NORMAL
            _put(scr, i, line, base, False)
        _put(scr, h - 1, help_foot(st.hrow, h - 2, total, w),
             curses.color_pair(C_HEAD) if col else curses.A_REVERSE, False)
        scr.refresh()
        return

    head, rows_avail, gap, dh, hint_y, foot_y = layout(h, st.drill)
    hdr = header_lines(st, w)
    for i, line in enumerate(hdr[:head]):
        base = (curses.color_pair(C_HEAD) | curses.A_BOLD if col and i == 0
                else (curses.color_pair(C_AMBER) if col else curses.A_BOLD))
        _put(scr, i, line, base, col and i > 0)
    top = head

    if st.allv and not st.drill:
        lines, nhead = all_lines(st, w)
        body = lines[nhead:]
        first = max(0, min(st.arow - rows_avail // 2, len(body) - rows_avail))
        first = max(first, 0)
        view = lines[:nhead] + body[first:first + rows_avail]
        for i, line in enumerate(view):
            if top + i >= (hint_y if hint_y >= 0 else foot_y):
                break
            sel = i >= nhead and (first + i - nhead) == st.arow
            if i < nhead:
                base = curses.color_pair(C_HEAD) | curses.A_BOLD if col else curses.A_REVERSE
                _put(scr, top + i, line, base, False)
            else:
                base = curses.color_pair(C_BODY) if col else curses.A_NORMAL
                _put(scr, top + i, line, base, col, sel)
        _draw_hint_and_footer(scr, st, w, hint_y, foot_y, col)
        # 다른 세 갈래는 전부 여기서 refresh 한다. 이 갈래만 빠져 있었다 —
        # `get_wch` 가 암묵적으로 refresh 해 줘서 **우연히** 보였을 뿐이다.
        scr.refresh()
        return
    if st.drill:
        lines, nhead = names_lines(st, w)
        nspan = name_sort_span(st) if col else None
        body = lines[nhead:]
        first = max(0, min(st.drow - rows_avail // 2, len(body) - rows_avail))
        first = max(first, 0)
        view = lines[:nhead] + body[first:first + rows_avail]
        for i, line in enumerate(view):
            if top + i >= (hint_y if hint_y >= 0 else foot_y):
                break
            sel = i >= nhead and (first + i - nhead) == st.drow
            if i == 0:
                base = curses.color_pair(C_AMBER) | curses.A_BOLD if col else curses.A_BOLD
            elif i == 1:
                base = curses.color_pair(C_HEAD) if col else curses.A_REVERSE
            elif sel:
                base = _sel_attr() if col else curses.A_REVERSE
            else:
                base = curses.color_pair(C_BODY) if col else curses.A_NORMAL
            _put(scr, top + i, line, base, col and i > 1, sel)
            if i == 1:
                _hl_sort(scr, top + i, nspan, col)
        _draw_hint_and_footer(scr, st, w, hint_y, foot_y, col)
        scr.refresh()
        return

    lines, thin, nhead = table_lines(st, w, rows_avail + 1)
    total = len(lines) - nhead
    first = max(0, min(st.row - rows_avail // 2, total - rows_avail))
    first = max(first, 0)

    if rows_avail:
        _put(scr, top, lines[0],
             curses.color_pair(C_HEAD) if col else curses.A_REVERSE, False)
        _hl_sort(scr, top, sort_span(st, w), col)
    shown = 0
    for j in range(rows_avail):
        idx = first + j
        if idx >= total:
            break
        sel = idx == st.row
        if sel:
            base = _sel_attr() if col else curses.A_REVERSE
        elif thin[idx + nhead]:
            base = _dim_attr() if col else curses.A_DIM
        else:
            base = curses.color_pair(C_BODY) if col else curses.A_NORMAL
        # 정렬 표시는 **헤더만** — 본문에 배경을 깔면 부호색이 묻혀 읽기 어렵다.
        _put(scr, top + 1 + j, lines[idx + nhead], base,
             col and not thin[idx + nhead], sel)
        shown += 1

    # 상세 패널은 **표 바로 아래**에 붙인다. 바닥 고정이던 시절 200x50 에서는
    # 27개 섹터를 다 그리고도 표와 패널 사이에 빈 줄이 14줄 남았다.
    # 붙이되 **한 줄은 띄운다**(`gap`) — 붙여 놓으니 패널 첫 줄이 표의 다음 행처럼
    # 읽혔다. 그 한 줄은 자리가 모자라면 `layout` 이 가장 먼저 도로 가져간다.
    if dh:
        dtop = min(top + 1 + shown + gap, (hint_y if hint_y >= 0 else foot_y) - dh)
        for i, line in enumerate(detail_lines(st, w)[:dh]):
            base = curses.color_pair(C_BODY) if col else curses.A_NORMAL
            _put(scr, dtop + i, line, base, col and i > 0)
        # 첫 줄에서 **섹터 이름만** 앰버로 세운다 — 그 줄에서 "지금 무엇을 보고
        # 있는가" 를 말하는 건 이름 하나뿐인데 부속 정보에 묻혀 있었다. 좌표는
        # 뷰가 낸다(`detail_title_span`) — 앱이 문구를 다시 뜯으면 어긋난다.
        # 무색 터미널에서는 굵게만 — 글자는 안 바꾸므로 정보가 사라지지 않는다.
        _hl_span(scr, dtop, detail_title_span(st, w),
                 (curses.color_pair(C_AMBER) if col else 0) | curses.A_BOLD)
    _draw_hint_and_footer(scr, st, w, hint_y, foot_y, col)
    scr.refresh()


def help_foot(hrow: int, shown: int, total: int, width: int) -> str:
    """도움말 모달의 마지막 줄 — 키 안내 + **어디까지 읽었나**. 정확히 ``width`` 칸.

    ⚠️ **위치 표시가 먼저 자리를 잡는다.** 예전엔 이 줄이 한 줄 고정 문자열이라
    키 안내가 폭을 다 먹고 위치가 뒤에서 잘렸다 — 폭 50 에서 ``1-20 / 200`` 이
    ``1-20 / 20`` 으로 보였다. **잘린 숫자는 다른 숫자다**(전체 200줄이 20줄로
    읽힌다). 잘린 안내문은 "여기가 전부" 로 읽히지만 잘린 숫자는 아예 거짓말이라,
    둘 중 양보하는 쪽은 안내문이어야 한다.

    앱 안에 박아 두면 폭을 넘겨 검사할 수가 없어서 순수 함수로 뺀다 — 원장은
    이미 같은 모양이다(``ledger_view.help_screen``).
    """
    more = f" {hrow + 1}-{min(hrow + shown, total)} / {total}"
    room = max(0, width - view_cell_len(more))
    return pad_footer(tier_for(HELP_FOOT_TIERS, room) + more, width)


def _draw_hint_and_footer(scr, st: State, w: int, hint_y: int, foot_y: int,
                          col: bool) -> None:
    if hint_y >= 0:
        _put(scr, hint_y, pad_footer(hint_text(st, w), w),
             curses.color_pair(C_AMBER) if col else curses.A_DIM, False)
    # 어느 화면이 어느 푸터를 쓰는지의 판정은 **뷰 한 곳**이다(`footer_tiers`).
    # 여기서 `if` 로 고르던 시절, 화면이 하나 늘었는데 그 `if` 를 안 늘려서
    # 전 종목 화면이 섹터 표의 푸터를 그렸다 — 안 듣는 키를 광고하면서.
    _put(scr, foot_y, pad_footer(footer_line(w, st.drill, st.allv), w),
         curses.color_pair(C_HEAD) if col else curses.A_REVERSE, False)


def pad_footer(text: str, width: int) -> str:
    """푸터를 폭에 맞춘다 — 반전 배경이 줄 끝까지 이어지게."""
    from kr_quant.tui.flow_view import pad
    return pad(text, width)


# --- 키 처리 ---------------------------------------------------------------

def _keep_selection(st: State, change) -> None:
    """섹터와 종목 커서를 유지한 채 상태를 바꾼다.

    드릴다운에서 "이 섹터를 5일로도 볼까" 를 하려면 예전엔 Esc → W → Enter 로
    나갔다 들어와야 했고, 그때 종목 커서를 잃었다. `cycle` 은 `row=0` 으로
    되돌리므로 바꾸기 **전에** 무엇을 보고 있었는지 적어 두고 다시 찾는다.
    """
    sec = (st.selected() or {}).get("sector")
    names = st.names()
    code = names[st.drow].get("code") if 0 <= st.drow < len(names) else None
    change()
    if sec is not None:
        for i, r in enumerate(st.rows()):
            if r.get("sector") == sec:
                st.row = i
                break
    st.drow = 0
    if code is not None:
        for i, t in enumerate(st.names()):
            if t.get("code") == code:
                st.drow = i
                break


def _toggle_rev(st: State) -> None:
    if st.drill:
        st.nrev = not st.nrev
    else:
        st.rev = not st.rev


# --- 한글 입력 상태의 키 ---------------------------------------------------
#
# 한영을 켜 둔 채 `w` 를 누르면 터미널에는 `ㅈ` 이 들어온다. 예전엔 그 순간
# 아무 일도 안 일어났다 — 화면을 보다가 정렬 하나 바꾸려면 한영을 껐다 켜야
# 했다. 두벌식은 자판 **자리** 대응이라 `w`→`ㅈ` 는 기계적이고, 그 대응만
# 넣으면 한글 상태에서도 같은 키가 같은 일을 한다.
#
# ⚠️ **대문자 역방향은 절반이 원리적으로 불가능하다.** 두벌식에서 Shift 로
# 다른 글자가 나오는 자음은 `ㅂㅈㄷㄱㅅ`(→`ㅃㅉㄸㄲㅆ`)뿐이다. 그래서
# `W`(=`ㅉ`)·`R`(=`ㄲ`)은 구분되지만 `A`·`S`·`G`·`M` 은 Shift 를 눌러도
# `ㅁ`·`ㄴ`·`ㅎ`·`ㅡ` 그대로라 소문자와 **같은 글자**가 온다. 터미널에 도착한
# 뒤에는 정보가 이미 없으므로 앱이 할 수 있는 일이 없다. 구분 안 되는 것은
# 소문자 동작으로 떨어뜨린다 — 나중에 "한글에서 A 가 왜 안 되냐" 를 묻는
# 사람은 이 주석이 답이다. (정렬 역순은 `r`(`ㄱ`)이 있고, 목록 끝은 `End` 가
# 있다. 즉 한글 상태에서 못 하는 것은 **역방향 순회**뿐이다.)
#
# 실측(pty 에 UTF-8 을 그대로 흘려 넣어 확인): `getch` 는 `ㅈ` 을 227·133·136
# 세 바이트로 쪼개 준다 — 그래서 `ord("w")` 와 비교하는 구조로는 **절대**
# 안 잡힌다. `get_wch` 는 `'ㅈ'`(U+3148) 한 글자로 준다. 그래서 입력 루프만
# `get_wch` 로 옮기고(:func:`_read_key`), `handle_key` 는 여전히 int 를 받는다 —
# 키 하나가 무엇을 했는지 보는 검사 수십 개가 `ord("w")` 를 넘기고 있고,
# 그 규약을 흔드는 것보다 **경계에서 정규화**하는 편이 변경면적이 작다.
#
# 호환 자모(U+31xx)와 첫가끝 자모(U+11xx)를 **둘 다** 받는다. 리눅스 ibus 는
# 앞의 것을 보내지만(실측) IME·터미널 조합마다 다르고, 표 하나 더 적는 값이
# 싸다. 완성형 음절(`자`)은 안 건드린다 — 자음+모음이 조합된 것이라 어느
# 키를 누른 것인지가 한 가지로 정해지지 않는다.
JAMO_TO_ASCII = {
    # 구분되는 것 — Shift 로 쌍자음이 나오는 자리
    "ㅂ": "q", "ㅃ": "q", "ㅈ": "w", "ㅉ": "W", "ㄱ": "r", "ㄲ": "R",
    "ㅅ": "t", "ㅆ": "t",
    # 구분 안 되는 것 — 소문자 동작으로 떨어진다
    "ㅁ": "a", "ㄴ": "s", "ㅎ": "g", "ㅡ": "m",
    # vim 이동
    "ㅗ": "h", "ㅓ": "j", "ㅏ": "k", "ㅣ": "l",
    # 첫가끝 자모(U+11xx) — 같은 자리를 다른 코드로 보내는 IME 대비
    "\u1107": "q", "\u1108": "q", "\u110c": "w", "\u110d": "W",
    "\u1100": "r", "\u1101": "R", "\u1106": "a", "\u1102": "s",
    "\u1109": "t", "\u110a": "t",
    "\u1112": "g", "\u1173": "m", "\u1169": "h", "\u1165": "j",
    "\u1161": "k", "\u1175": "l",
}


def normalize_key(k) -> int:
    """`get_wch` 가 준 것을 `handle_key` 가 아는 **int** 로.

    특수키는 이미 int 로 온다(방향키·PgUp·F1). 글자는 한 글자 문자열로 오는데,
    한글 자모면 두벌식에서 같은 자리인 ASCII 키로 바꾼다. 나머지는 그대로
    `ord` 다 — 모르는 글자는 `handle_key` 가 어느 분기에도 안 걸려 무시한다.
    """
    if isinstance(k, int):
        return k
    if not k:
        return -1
    return ord(JAMO_TO_ASCII.get(k, k)) if len(k) == 1 else -1


def _read_key(scr) -> int:
    """키 하나 — 한글 자모까지 온전한 한 글자로 받아 int 로 정규화한다.

    `get_wch` 는 `getch` 의 ERR(-1) 자리에 **예외**(`curses.error`)를 쓴다.
    터미널이 사라졌을 때(SSH 끊김·창 닫힘) 그걸 아무 분기에도 안 태우면
    draw→읽기→실패 로 100% CPU 를 태우며 영원히 돈다 — 끊긴 세션마다 서버에
    좀비가 쌓인다. 그래서 예외를 예전과 **같은 뜻의 -1** 로 되돌린다.
    """
    try:
        return normalize_key(scr.get_wch())
    except curses.error:
        return -1


def handle_key(st: State, k: int, page: int = 10, help_page: int = 10) -> bool:
    """키 하나를 상태에 적용한다. 계속 돌면 True, 종료면 False.

    curses 를 부르지 않는 **순수 함수**다 — 터미널 없이 키 조합을 그대로 태워
    "무엇이 달라졌나" 로 판정할 수 있다. pty 검사만 있던 시절, 통과하면서
    아무것도 확인 못 하는 검사가 이 저장소에서 세 번 나왔다.
    """
    page = max(page, 1)
    help_page = max(help_page, 1)
    # 터미널이 사라지면(SSH 끊김·창 닫힘) getch 가 ERR 을 **즉시** 돌려준다.
    # 이걸 아무 분기에도 안 태우면 draw→getch→-1 로 100% CPU 를 태우며
    # 영원히 돈다 — 끊긴 세션마다 서버에 좀비가 쌓인다. less·htop 처럼 나간다.
    if k == -1:
        return False
    # 리사이즈는 "아무 키" 가 아니다. 그냥 통과시키면 도움말이 소리 없이 닫힌다.
    if k == curses.KEY_RESIZE:
        return True

    if st.help:
        total = help_lines(80, 0, 10**6)[1]
        # 하한이 total-5 이던 시절 끝까지 내리면 화면 아래가 비었다.
        bottom = max(total - help_page, 0)
        if k in (curses.KEY_DOWN, ord("j")):
            st.hrow = min(st.hrow + 1, bottom)
        elif k in (curses.KEY_UP, ord("k")):
            st.hrow = max(st.hrow - 1, 0)
        elif k in (curses.KEY_NPAGE, ord(" ")):
            st.hrow = min(st.hrow + help_page, bottom)
        elif k == curses.KEY_PPAGE:
            st.hrow = max(st.hrow - help_page, 0)
        elif k in (ord("g"), curses.KEY_HOME):
            st.hrow = 0
        elif k in (ord("G"), curses.KEY_END):
            st.hrow = bottom
        elif k in (ord("q"), 27, ord("?"), curses.KEY_ENTER, 10, 13):
            # q 는 **닫기**다(종료 아님). less 관례와는 어긋나지만, 키를 배우러
            # 연 화면에서 q 가 앱을 끝내면 확인 절차 없이 화면이 사라진다.
            # 대신 도움말 첫 줄과 푸터가 "종료는 닫은 뒤 q 를 한 번 더" 라고
            # 밝힌다 — 관례를 어기는 쪽은 화면에 적어야 한다.
            st.help = False
        return True

    # h 는 vim 에서 '왼쪽' 이다. l·→ 이 드릴다운 진입이므로 h 는 **나가기** 여야
    # 하는데 예전엔 도움말이었고, 정작 드릴다운에서 h 는 아무 일도 안 했다.
    # 반쪽만 맞는 관례는 안 맞느니만 못하다. 도움말은 ?·F1 로 옮겼다.
    if k in (ord("?"), curses.KEY_F1):
        st.help = True
        st.hrow = 0
        return True
    # ESC 는 종료가 아니다. 느린 SSH 에서 방향키가 ESC 와 나머지로 쪼개져
    # 도착하면(ESCDELAY 기본 1초) ↓ 를 눌렀을 뿐인데 앱이 끝난다. 종료는 q.
    if k == ord("q"):
        return False

    if st.drill:
        m = max(len(st.names()), 1)
        if k in (27, curses.KEY_LEFT, ord("h")):
            st.drill = False
        elif k in (curses.KEY_DOWN, ord("j")):
            st.drow = min(st.drow + 1, m - 1)
        elif k in (curses.KEY_UP, ord("k")):
            st.drow = max(st.drow - 1, 0)
        elif k == curses.KEY_NPAGE:
            st.drow = min(st.drow + page, m - 1)
        elif k == curses.KEY_PPAGE:
            st.drow = max(st.drow - page, 0)
        elif k in (ord("g"), curses.KEY_HOME):
            st.drow = 0
        elif k in (ord("G"), curses.KEY_END):
            st.drow = m - 1
        elif k in (ord("s"), ord("S")):
            st.cycle("ns", 1 if k == ord("s") else -1)
        elif k == ord("r"):
            _keep_selection(st, lambda: _toggle_rev(st))
        # 구간·시장·주체는 드릴다운에서도 듣는다. 예전엔 조용히 무시돼서
        # 나갔다 들어오는 동안 커서를 잃었다.
        elif k in (ord("w"), ord("W")):
            _keep_selection(st, lambda: st.cycle("w", 1 if k == ord("w") else -1))
        elif k in (ord("m"), ord("M")):
            _keep_selection(st, lambda: st.cycle("m", 1 if k == ord("m") else -1))
        elif k in (ord("a"), ord("A")):
            _keep_selection(st, lambda: st.cycle("a", 1 if k == ord("a") else -1))
        elif k == ord("t"):
            # 전 종목 화면은 **여기서도 열린다.** 도움말은 늘 `t` 를 조건 없이
            # 적어 왔는데 이 분기가 그냥 삼켜서, 드릴다운에서만 없는 키였다.
            # 두 층이 겹쳐 보이지는 않으므로(전 종목은 전 섹터를 본다) 드릴다운을
            # 닫고 연다 — 돌아오는 곳이 섹터 표라는 것은 푸터와 도움말이 적는다.
            st.allv, st.drill, st.arow = True, False, 0
        return True

    if st.allv:
        # 전 종목 화면. 구간·시장·주체는 **여기서도 들어야 한다** — 드릴다운이
        # 같은 이유로 이미 그렇게 한다("예전엔 조용히 무시돼서 나갔다 들어오는
        # 동안 커서를 잃었다"). 처음엔 "위 공통 분기가 처리한다" 고 적어 뒀는데
        # 그런 분기는 없었고, 그래서 이 화면에서 w·m·a 가 통째로 죽어 있었다.
        # 정렬만 없다 — 곱 내림차순 고정이고 헤더가 그렇게 적는다.
        m = max(len(st.all_picks()), 1)
        if k in (ord("t"), 27, curses.KEY_LEFT, ord("h")):
            st.allv = False
        elif k in (ord("w"), ord("W")):
            st.cycle("w", 1 if k == ord("w") else -1)
            st.arow = 0
        elif k in (ord("m"), ord("M")):
            st.cycle("m", 1 if k == ord("m") else -1)
            st.arow = 0
        elif k in (ord("a"), ord("A")):
            st.cycle("a", 1 if k == ord("a") else -1)
            st.arow = 0
        elif k in (curses.KEY_DOWN, ord("j")):
            st.arow = min(st.arow + 1, m - 1)
        elif k in (curses.KEY_UP, ord("k")):
            st.arow = max(st.arow - 1, 0)
        elif k == curses.KEY_NPAGE:
            st.arow = min(st.arow + page, m - 1)
        elif k == curses.KEY_PPAGE:
            st.arow = max(st.arow - page, 0)
        elif k in (ord("g"), curses.KEY_HOME):
            st.arow = 0
        elif k in (ord("G"), curses.KEY_END):
            st.arow = m - 1
        return True

    n = max(len(st.rows()), 1)
    if k == ord("t"):
        # 전 종목(곱) 화면 토글. 섹터 표 ↔ 이 화면만 오간다 — 드릴다운에서는
        # 위 분기가 먼저 처리한다.
        st.allv = not st.allv
        st.arow = 0
    elif k in (curses.KEY_ENTER, 10, 13, curses.KEY_RIGHT, ord("l")):
        st.drill = True
        st.drow = 0
    elif k in (curses.KEY_DOWN, ord("j")):
        st.row = min(st.row + 1, n - 1)
    elif k in (curses.KEY_UP, ord("k")):
        st.row = max(st.row - 1, 0)
    elif k == curses.KEY_NPAGE:
        st.row = min(st.row + page, n - 1)
    elif k == curses.KEY_PPAGE:
        st.row = max(st.row - page, 0)
    elif k in (ord("g"), curses.KEY_HOME):
        st.row = 0
    elif k in (ord("G"), curses.KEY_END):
        st.row = n - 1
    elif k == ord("r") and st.sortable:
        # 역순이 없어서 "누가 털렸나"(순매도 상위)로 가는 유일한 길이 G 였다.
        _keep_selection(st, lambda: _toggle_rev(st))
    elif k in (ord("w"), ord("W")):
        st.cycle("w", 1 if k == ord("w") else -1)
    elif k in (ord("m"), ord("M")):
        st.cycle("m", 1 if k == ord("m") else -1)
    elif k in (ord("a"), ord("A")):
        st.cycle("a", 1 if k == ord("a") else -1)
    elif k in (ord("s"), ord("S")) and st.sortable:
        # 종합에서는 **아무 일도 안 한다.** 예전엔 `si` 를 돌려서 헤더 라벨과
        # 힌트바가 따라 움직였는데 표는 그대로였다 — 라벨이 바뀌는 것을 보고
        # 정렬이 됐다고 믿게 만드는 쪽이, 키가 안 듣는 것보다 나쁘다.
        # 화면은 그 사실을 늘 적고 있다(헤더 `정렬없음[G 순]` · 힌트바 한 줄).
        st.cycle("s", 1 if k == ord("s") else -1)
    return True


def _loop(scr, data: dict) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    _init_colors(scr)
    st = State(data)
    while True:
        _draw(scr, st)
        try:
            k = _read_key(scr)
        except KeyboardInterrupt:
            return
        h, _w = scr.getmaxyx()
        rows = layout(h, st.drill).rows
        if not handle_key(st, k, page=rows, help_page=h - 2):
            return


def main() -> None:
    # `get_wch` 가 한글을 한 글자로 주려면 로케일이 **와이드 문자를 아는** 상태여야
    # 한다. 파이썬은 C 로케일로 시작하므로 여기서 사용자 환경을 따라간다 —
    # 안 하면 `ㅈ` 이 다시 바이트로 쪼개진다. 실패해도 앱은 뜬다(영문 키는 듣는다).
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass
    ap = argparse.ArgumentParser(description="섹터 자금흐름 TUI")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="리포트 폴더")
    a = ap.parse_args()
    data = load(a.dir)
    term = os.environ.get("TERM", "")
    if term in ("", "dumb"):
        # dumb 터미널에서는 curses 가 커서를 못 옮긴다. 예전 동작은 실측하면
        # 이랬다 — TERM 미설정은 생 트레이스백으로 죽었고, TERM=dumb 은 헤더와
        # 푸터 두 줄만 나오고 **표가 안 그려졌다**. 표가 전부인 앱이 표를 못
        # 그리면 그렇다고 말하는 편이 낫다.
        raise SystemExit(
            f"TERM={term or '(비어 있음)'} — 이 터미널은 화면 제어(커서 이동·지우기)를"
            f" 못 해 TUI 를 그릴 수 없다.\n"
            f"  TERM=xterm-256color 로 설정하고 다시 실행하라"
            f" (숫자만 필요하면 리포트의 numbers.html 을 보라).")
    curses.wrapper(_loop, data)


if __name__ == "__main__":
    main()
