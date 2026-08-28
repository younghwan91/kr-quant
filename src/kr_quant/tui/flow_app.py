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
    FOOTER, FOOTER_DRILL, State, detail_lines, header_lines, names_lines,
    table_lines)

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


def _draw(scr, st: State) -> None:
    scr.erase()
    h, w = scr.getmaxyx()
    w = max(w - 1, 20)

    for i, line in enumerate(header_lines(st, w)):
        scr.addstr(i, 0, line, curses.A_BOLD if i == 0 else curses.A_NORMAL)
    top = len(header_lines(st, w))

    if st.drill:
        lines, nhead = names_lines(st, w)
        avail = h - top - 1
        for i, line in enumerate(lines[:avail]):
            attr = curses.A_BOLD if i < nhead else curses.A_NORMAL
            if i == nhead and len(lines) > nhead:
                attr = curses.A_REVERSE
            if i >= nhead + 1 and (i - nhead - 1) == st.drow:
                attr = curses.A_REVERSE | curses.A_BOLD
            scr.addstr(top + i, 0, line, attr)
        scr.addstr(h - 1, 0, FOOTER_DRILL[:w], curses.A_REVERSE)
        scr.refresh()
        return

    body_h = h - top - 5           # 헤더 + 상세 3줄 + 푸터
    lines, thin, nhead = table_lines(st, w, body_h)
    rows_avail = max(body_h - nhead, 1)
    total = len(lines) - nhead
    first = max(0, min(st.row - rows_avail // 2, total - rows_avail))
    first = max(first, 0)

    scr.addstr(top, 0, lines[0], curses.A_REVERSE)
    for j in range(rows_avail):
        idx = first + j
        if idx >= total:
            break
        attr = curses.A_NORMAL
        if thin[idx + nhead]:
            attr = curses.A_DIM
        if idx == st.row:
            attr = curses.A_REVERSE | curses.A_BOLD
        scr.addstr(top + 1 + j, 0, lines[idx + nhead], attr)

    dtop = h - 4
    for i, line in enumerate(detail_lines(st, w)):
        if dtop + i < h - 1:
            scr.addstr(dtop + i, 0, line, curses.A_BOLD if i == 0 else curses.A_NORMAL)
    scr.addstr(h - 1, 0, FOOTER[:w], curses.A_REVERSE)
    scr.refresh()


def _loop(scr, data: dict) -> None:
    curses.curs_set(0)
    st = State(data)
    while True:
        _draw(scr, st)
        try:
            k = scr.getch()
        except KeyboardInterrupt:
            return
        n = max(len(st.rows()), 1)
        if st.drill:
            m = max(len(st.names()), 1)
            if k in (ord("q"),):
                return
            elif k in (27, curses.KEY_LEFT, ord("h")):
                st.drill = False
            elif k in (curses.KEY_DOWN, ord("j")):
                st.drow = min(st.drow + 1, m - 1)
            elif k in (curses.KEY_UP, ord("k")):
                st.drow = max(st.drow - 1, 0)
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
