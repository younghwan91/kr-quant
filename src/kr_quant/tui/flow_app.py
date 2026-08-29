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
import os
import re

from kr_quant.tui.flow_view import (
    HELP, NAME_SORT_COL, NAME_SORTS, SORT_COL, SORTS, State, color_spans,
    detail_lines, footer_line, header_lines, help_desc, help_lines, hint_line,
    name_sort_span, names_lines, sort_span, table_lines, tier_for)

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


# 블룸버그 계열 배색 — 검은 바탕에 앰버가 골격, 값은 국내 관행(상승 빨강/하락 파랑).
C_AMBER, C_UP, C_DOWN, C_HEAD, C_DIM, C_SEL, C_MARK, C_SORT = range(1, 9)
# 정렬 표시는 **헤더만** 한다. 한때 열 전체에 배경을 깔았는데, 배경 위에서
# 부호색(빨강/시안)이 묻혀 읽기 어려웠다. curses 는 색쌍 단위로만 칠해져
# "배경만 바꾸기" 가 없으므로 (전경,배경) 조합을 다 만들어야 했고, 그렇게까지
# 해도 대비가 안 나왔다.
#
# 8색 배색 자체는 건드리지 않는다 — 빨강/시안은 적록색맹에도 갈리고, 부호가
# `+`/`-` 텍스트에도 있어 무색 터미널에서 정보가 남는다. 256색에서는 **같은
# 역할의 더 읽기 좋은 톤**으로만 승격한다(빨강 203 · 청록 80 · 회색 244).
RICH_UP, RICH_DOWN, RICH_DIM = 203, 80, 244

#: 색을 쓸 수 있는 터미널인가. curses.window 에는 속성을 붙일 수 없어서
#: (`scr._colored = ...` 는 AttributeError) 모듈 수준으로 둔다.
_COLORED = False


def _init_colors() -> bool:
    global _COLORED
    _COLORED = False
    # NO_COLOR 표준(no-color.org): **비어 있지 않은** 값으로 설정돼 있으면 색을
    # 쓰지 않는다. 안 보던 시절엔 리다이렉트·로그 캡처에서 색이 그대로 나왔다.
    if os.environ.get("NO_COLOR"):
        return False
    if not curses.has_colors():
        return False
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    rich = getattr(curses, "COLORS", 0) >= 256
    up, down, dim = curses.COLOR_RED, curses.COLOR_CYAN, curses.COLOR_BLUE
    if rich:
        try:
            curses.init_pair(C_UP, RICH_UP, bg)
            curses.init_pair(C_DOWN, RICH_DOWN, bg)
            curses.init_pair(C_DIM, RICH_DIM, bg)
        except curses.error:
            rich = False        # 256색이라 말해놓고 못 받는 터미널이 있다
    if not rich:
        curses.init_pair(C_UP, up, bg)
        curses.init_pair(C_DOWN, down, bg)
        curses.init_pair(C_DIM, dim, bg)
    curses.init_pair(C_AMBER, curses.COLOR_YELLOW, bg)
    curses.init_pair(C_HEAD, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    curses.init_pair(C_SEL, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(C_MARK, curses.COLOR_GREEN, bg)
    curses.init_pair(C_SORT, curses.COLOR_WHITE, curses.COLOR_BLUE)
    _COLORED = True
    return True


def _put(scr, y: int, line: str, base, colored: bool, selected: bool = False) -> None:
    """한 줄 그리기 — 숫자 구간만 부호색으로 덧칠한다.

    선택행은 반전이라 덧칠하지 않는다(반전 위에 색을 얹으면 읽기 어렵다).
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


def _hl_sort(scr, y: int, span, col: bool) -> None:
    """정렬 중인 열 헤더를 덧칠한다 — 어떤 기준으로 줄세웠는지 한눈에 보이게."""
    if not span:
        return
    start, width = span
    attr = (curses.color_pair(C_SORT) if col else curses.A_UNDERLINE) | curses.A_BOLD
    try:
        scr.chgat(y, start, width, attr)
    except curses.error:
        pass


# --- 세로 배치 -------------------------------------------------------------

#: 상세 패널을 접는 높이. 이 아래에서는 패널(3줄)이 표를 통째로 밀어냈다 —
#: 6줄 터미널에서 헤더 위에 패널이 겹쳐 그려져 표가 사라졌고 설명도 없었다.
DETAIL_MIN_H = 10


def layout(h: int, drill: bool = False) -> tuple[int, int, int, int, int]:
    """세로 배치 — (맨 위 헤더 줄 수, 보이는 표 행 수, 상세 패널 줄 수, 힌트바 y, 푸터 y).

    그리는 쪽과 PgUp/PgDn 이 **같은 함수**를 봐야 어긋나지 않는다. 예전엔
    PgDn 이 화면 높이와 무관하게 10줄 고정이라 200x50 에서 반 페이지도 안 갔다.
    힌트바 y 가 음수면 그 줄을 그릴 자리가 없다는 뜻이다.
    """
    foot_y = max(h - 1, 0)
    hint_y = h - 2 if h >= 5 else -1
    limit = hint_y if hint_y >= 0 else foot_y      # 표가 쓸 수 있는 y 의 끝(배타)
    nhead = 2 if drill else 1                      # 표 자체의 머리 줄(제목·열이름)
    detail = 3 if (not drill and h >= DETAIL_MIN_H) else 0
    for head in (2, 1, 0):
        rows = limit - head - nhead - detail
        if rows >= 1:
            return head, rows, detail, hint_y, foot_y
    return min(max(limit, 0), 2), 0, 0, hint_y, foot_y


# --- 힌트바 ---------------------------------------------------------------

#: 표에 열이 없어 `SORT_COL` 이 안 다루는 정렬키 → 도움말의 항목 이름.
_EXTRA_COL = {"tv": "거래대금[억]", "cap_idx": "시총[억]"}


def _help_desc(header: str) -> str:
    """이 화면의 HELP 에서 열 설명 한 줄. 찾기는 ``flow_view.help_desc`` 가 한다
    — 원장 힌트바도 같은 함수를 쓴다(``**`` 떼기가 두 벌이면 갈라진다)."""
    return help_desc(HELP, header)


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
    w = max(w - 1, 1)

    col = _COLORED

    if st.help:
        lines, total = help_lines(w, st.hrow, h - 1)
        for i, line in enumerate(lines):
            base = (curses.color_pair(C_HEAD) | curses.A_BOLD if col and i == 0
                    else curses.A_NORMAL)
            _put(scr, i, line, base, False)
        more = f" {st.hrow + 1}-{min(st.hrow + h - 2, total)} / {total}"
        _put(scr, h - 1, pad_footer(" ↑↓/PgDn:스크롤  g/G:처음·끝  q·Esc:닫기" + more, w),
             curses.color_pair(C_HEAD) if col else curses.A_REVERSE, False)
        scr.refresh()
        return

    head, rows_avail, dh, hint_y, foot_y = layout(h, st.drill)
    hdr = header_lines(st, w)
    for i, line in enumerate(hdr[:head]):
        base = (curses.color_pair(C_HEAD) | curses.A_BOLD if col and i == 0
                else (curses.color_pair(C_AMBER) if col else curses.A_BOLD))
        _put(scr, i, line, base, col and i > 0)
    top = head

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
                base = curses.color_pair(C_SEL) | curses.A_BOLD if col else curses.A_REVERSE
            else:
                base = curses.A_NORMAL
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
            base = curses.color_pair(C_SEL) | curses.A_BOLD if col else curses.A_REVERSE
        elif thin[idx + nhead]:
            base = curses.color_pair(C_DIM) if col else curses.A_DIM
        else:
            base = curses.A_NORMAL
        # 정렬 표시는 **헤더만** — 본문에 배경을 깔면 부호색이 묻혀 읽기 어렵다.
        _put(scr, top + 1 + j, lines[idx + nhead], base,
             col and not thin[idx + nhead], sel)
        shown += 1

    # 상세 패널은 **표 바로 아래**에 붙인다. 바닥 고정이던 시절 200x50 에서는
    # 27개 섹터를 다 그리고도 표와 패널 사이에 빈 줄이 14줄 남았다.
    if dh:
        dtop = min(top + 1 + shown, (hint_y if hint_y >= 0 else foot_y) - dh)
        for i, line in enumerate(detail_lines(st, w)[:dh]):
            base = (curses.color_pair(C_AMBER) | curses.A_BOLD if col and i == 0
                    else curses.A_NORMAL)
            _put(scr, dtop + i, line, base, col and i > 0)
    _draw_hint_and_footer(scr, st, w, hint_y, foot_y, col)
    scr.refresh()


def _draw_hint_and_footer(scr, st: State, w: int, hint_y: int, foot_y: int,
                          col: bool) -> None:
    if hint_y >= 0:
        _put(scr, hint_y, pad_footer(hint_text(st, w), w),
             curses.color_pair(C_AMBER) if col else curses.A_DIM, False)
    _put(scr, foot_y, pad_footer(footer_line(w, st.drill), w),
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
        return True

    n = max(len(st.rows()), 1)
    if k in (curses.KEY_ENTER, 10, 13, curses.KEY_RIGHT, ord("l")):
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
    elif k == ord("r"):
        # 역순이 없어서 "누가 털렸나"(순매도 상위)로 가는 유일한 길이 G 였다.
        _keep_selection(st, lambda: _toggle_rev(st))
    elif k in (ord("w"), ord("W")):
        st.cycle("w", 1 if k == ord("w") else -1)
    elif k in (ord("m"), ord("M")):
        st.cycle("m", 1 if k == ord("m") else -1)
    elif k in (ord("a"), ord("A")):
        st.cycle("a", 1 if k == ord("a") else -1)
    elif k in (ord("s"), ord("S")):
        st.cycle("s", 1 if k == ord("s") else -1)
    return True


def _loop(scr, data: dict) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    _init_colors()
    st = State(data)
    while True:
        _draw(scr, st)
        try:
            k = scr.getch()
        except KeyboardInterrupt:
            return
        h, _w = scr.getmaxyx()
        _head, rows, _dh, _hy, _fy = layout(h, st.drill)
        if not handle_key(st, k, page=rows, help_page=h - 2):
            return


def main() -> None:
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
