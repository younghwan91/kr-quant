"""섹터 자금흐름 TUI — SSH 에서 바로 보는 화면.

렌더 로직은 :mod:`kr_quant.tui.flow_view` 에 있고 여기는 **화면 그리기와 키 입력만**
한다. 표준 라이브러리 curses 만 쓰므로 새 의존성이 없다.

데이터는 일일 리포트가 이미 만들어 둔 ``numbers.html`` 안의 JSON 을 읽는다 —
DB 에 접속하지 않으므로 즉시 뜨고, 화면과 표가 **같은 숫자**를 보게 된다.

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
    FOOTER, FOOTER_DRILL, State, color_spans, detail_lines, header_lines,
    help_lines, name_sort_span, names_lines, sort_span, table_lines)

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
    return json.loads(m.group(1))


# 블룸버그 계열 배색 — 검은 바탕에 앰버가 골격, 값은 국내 관행(상승 빨강/하락 파랑).
C_AMBER, C_UP, C_DOWN, C_HEAD, C_DIM, C_SEL, C_MARK, C_SORT = range(1, 9)

#: 색을 쓸 수 있는 터미널인가. curses.window 에는 속성을 붙일 수 없어서
#: (`scr._colored = ...` 는 AttributeError) 모듈 수준으로 둔다.
_COLORED = False


def _init_colors() -> bool:
    global _COLORED
    _COLORED = False
    if not curses.has_colors():
        return False
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    curses.init_pair(C_AMBER, curses.COLOR_YELLOW, bg)
    curses.init_pair(C_UP, curses.COLOR_RED, bg)
    curses.init_pair(C_DOWN, curses.COLOR_CYAN, bg)
    curses.init_pair(C_HEAD, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    curses.init_pair(C_DIM, curses.COLOR_BLUE, bg)
    curses.init_pair(C_SEL, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(C_MARK, curses.COLOR_GREEN, bg)
    curses.init_pair(C_SORT, curses.COLOR_WHITE, curses.COLOR_BLUE)
    _COLORED = True
    return True


def _put(scr, y: int, line: str, base, colored: bool, selected: bool = False) -> None:
    """한 줄 그리기 — 숫자 구간만 부호색으로 덧칠한다.

    선택행은 반전이라 덧칠하지 않는다(반전 위에 색을 얹으면 읽기 어렵다).
    """
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


def _draw(scr, st: State) -> None:
    scr.erase()
    h, w = scr.getmaxyx()
    w = max(w - 1, 20)

    col = _COLORED

    if st.help:
        lines, total = help_lines(w, st.hrow, h - 1)
        for i, line in enumerate(lines):
            base = (curses.color_pair(C_HEAD) | curses.A_BOLD if col and i == 0
                    else curses.A_NORMAL)
            _put(scr, i, line, base, False)
        more = f" {st.hrow + 1}-{min(st.hrow + h - 2, total)} / {total}"
        _put(scr, h - 1, pad_footer(" ↑↓:스크롤  아무 키나:닫기" + more, w),
             curses.color_pair(C_HEAD) if col else curses.A_REVERSE, False)
        scr.refresh()
        return

    hdr = header_lines(st, w)
    for i, line in enumerate(hdr):
        base = (curses.color_pair(C_HEAD) | curses.A_BOLD if col and i == 0
                else (curses.color_pair(C_AMBER) if col else curses.A_BOLD))
        _put(scr, i, line, base, col and i > 0)
    top = len(hdr)

    if st.drill:
        lines, nhead = names_lines(st, w)
        avail = h - top - 1
        body = lines[nhead:]
        rows_avail = max(avail - nhead, 1)
        first = max(0, min(st.drow - rows_avail // 2, len(body) - rows_avail))
        first = max(first, 0)
        view = lines[:nhead] + body[first:first + rows_avail]
        for i, line in enumerate(view):
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
                _hl_sort(scr, top + i, name_sort_span(st), col)
        _put(scr, h - 1, pad_footer(FOOTER_DRILL, w),
             curses.color_pair(C_HEAD) if col else curses.A_REVERSE, False)
        scr.refresh()
        return

    body_h = h - top - 5           # 헤더 + 상세 3줄 + 푸터
    lines, thin, nhead = table_lines(st, w, body_h)
    rows_avail = max(body_h - nhead, 1)
    total = len(lines) - nhead
    first = max(0, min(st.row - rows_avail // 2, total - rows_avail))
    first = max(first, 0)

    _put(scr, top, lines[0],
         curses.color_pair(C_HEAD) if col else curses.A_REVERSE, False)
    _hl_sort(scr, top, sort_span(st, w), col)
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
        _put(scr, top + 1 + j, lines[idx + nhead], base,
             col and not thin[idx + nhead], sel)

    dtop = h - 4
    for i, line in enumerate(detail_lines(st, w)):
        if dtop + i < h - 1:
            base = (curses.color_pair(C_AMBER) | curses.A_BOLD if col and i == 0
                    else curses.A_NORMAL)
            _put(scr, dtop + i, line, base, col and i > 0)
    _put(scr, h - 1, pad_footer(FOOTER, w),
         curses.color_pair(C_HEAD) if col else curses.A_REVERSE, False)
    scr.refresh()


def pad_footer(text: str, width: int) -> str:
    """푸터를 폭에 맞춘다 — 반전 배경이 줄 끝까지 이어지게."""
    from kr_quant.tui.flow_view import pad
    return pad(text, width)


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
        n = max(len(st.rows()), 1)
        if st.help:
            total = len(help_lines(80, 0, 10**6)[0]) - 1
            if k in (curses.KEY_DOWN, ord("j")):
                st.hrow = min(st.hrow + 1, max(total - 5, 0))
            elif k in (curses.KEY_UP, ord("k")):
                st.hrow = max(st.hrow - 1, 0)
            elif k == curses.KEY_NPAGE:
                st.hrow = min(st.hrow + 10, max(total - 5, 0))
            elif k == curses.KEY_PPAGE:
                st.hrow = max(st.hrow - 10, 0)
            else:
                st.help = False
            continue
        if k in (ord("?"), ord("h")) and not st.drill:
            st.help = True
            st.hrow = 0
            continue
        if st.drill:
            m = max(len(st.names()), 1)
            if k in (ord("q"),):
                return
            elif k == ord("?"):
                st.help = True
                st.hrow = 0
            elif k in (27, curses.KEY_LEFT):
                st.drill = False
            elif k in (curses.KEY_DOWN, ord("j")):
                st.drow = min(st.drow + 1, m - 1)
            elif k in (curses.KEY_UP, ord("k")):
                st.drow = max(st.drow - 1, 0)
            elif k in (ord("s"), ord("S")):
                st.cycle("ns", 1 if k == ord("s") else -1)
            elif k == curses.KEY_NPAGE:
                st.drow = min(st.drow + 10, m - 1)
            elif k == curses.KEY_PPAGE:
                st.drow = max(st.drow - 10, 0)
            continue
        if k in (ord("q"), 27):
            return
        elif k in (curses.KEY_ENTER, 10, 13, curses.KEY_RIGHT, ord("l")):
            st.drill = True
            st.drow = 0
        elif k in (curses.KEY_DOWN, ord("j")):
            st.row = min(st.row + 1, n - 1)
        elif k in (curses.KEY_UP, ord("k")):
            st.row = max(st.row - 1, 0)
        elif k == curses.KEY_NPAGE:
            st.row = min(st.row + 10, n - 1)
        elif k == curses.KEY_PPAGE:
            st.row = max(st.row - 10, 0)
        elif k in (ord("g"),):
            st.row = 0
        elif k in (ord("G"),):
            st.row = n - 1
        elif k in (ord("w"), ord("W")):
            st.cycle("w", 1 if k == ord("w") else -1)
        elif k in (ord("m"), ord("M")):
            st.cycle("m", 1 if k == ord("m") else -1)
        elif k in (ord("a"), ord("A")):
            st.cycle("a", 1 if k == ord("a") else -1)
        elif k in (ord("s"), ord("S")):
            st.cycle("s", 1 if k == ord("s") else -1)


def main() -> None:
    ap = argparse.ArgumentParser(description="섹터 자금흐름 TUI")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="리포트 폴더")
    a = ap.parse_args()
    data = load(a.dir)
    curses.wrapper(_loop, data)


if __name__ == "__main__":
    main()
