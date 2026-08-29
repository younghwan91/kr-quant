"""자금 원장 TUI — SSH 에서 바로 보는 화면.

렌더 로직은 :mod:`kr_quant.tui.ledger_view` 에 있고 여기는 **화면 그리기와 키 입력만**
한다. 설계 근거·한계는 ``docs/superpowers/specs/2026-08-29-money-flow-ledger.md``.

``kq-flow`` 와 나란히 산다. 다른 점은 하나다 — ``kq-flow`` 는 섹터의 **동역학**(임펄스·
가속·포텐셜)을 보여주고, 여기는 **회계**를 보여준다. 누가 얼마를 넘겼고 얼마가 미분류로
남았나. 그래서 이 화면에는 파생 지표가 없고 원 금액과 잔여만 있다.

``?`` 는 **키와 열의 뜻**(``ledger_view.LEDGER_HELP``)을 전면 모달로 띄운다.
한계는 없어지지 않았다 — ``v`` 로 도는 네 화면 중 하나로 그대로 있다. 둘은 답하는
질문이 다르다: 도움말은 "이 숫자가 뭐냐", 한계는 "이 숫자가 무엇을 주장하지 않느냐".

기존 TUI 는 8색만 썼다. 여기는 상관 히트맵 때문에 **256색 발산 팔레트**를 쓰되,
색이 없어도 그림이 남도록 부호를 글자에 박아 뒀다(``ledger_view.heat_cell``).

Run:  kq-ledger                     # ~/Documents/kr-quant-reports/latest
      kq-ledger --dir <리포트 폴더>
      kq-ledger --dump              # 색 없는 평문, 파이프·리다이렉트용
"""

from __future__ import annotations

import argparse
import curses

from kr_quant.tui.flow_view import cell_width
from kr_quant.tui.ledger_view import (
    Model, help_total, load, render_text, screen)

DEFAULT_DIR = "~/Documents/kr-quant-reports/latest"

C_HEAD, C_POS, C_NEG, C_DIM, C_SEL, C_BAN = range(1, 7)
#: 히트맵 −5..+5 의 색쌍은 C_HEAT + 5 + level 에 잡는다(0 은 안 칠한다).
C_HEAT = 10

#: 256색 발산 팔레트 (파랑 ← 중립 → 빨강). 국내 관행대로 **양수가 빨강**이다.
#: 8색 터미널에서는 부호만 남기고 세기는 글자(`·░▒▓█`)가 표현한다.
HEAT_256 = {-5: 25, -4: 26, -3: 32, -2: 38, -1: 45,
            1: 217, 2: 210, 3: 203, 4: 196, 5: 160}

_COLORS = 0        # 0=없음, 8=기본, 256=확장


def _init_colors() -> int:
    global _COLORS
    _COLORS = 0
    if not curses.has_colors():
        return 0
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    curses.init_pair(C_HEAD, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    curses.init_pair(C_POS, curses.COLOR_RED, bg)
    curses.init_pair(C_NEG, curses.COLOR_CYAN, bg)
    curses.init_pair(C_DIM, curses.COLOR_BLUE, bg)
    curses.init_pair(C_SEL, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(C_BAN, curses.COLOR_BLACK, curses.COLOR_BLUE)
    _COLORS = 8
    if curses.COLORS >= 256 and curses.COLOR_PAIRS > C_HEAT + 11:
        try:
            for lv, fg in HEAT_256.items():
                curses.init_pair(C_HEAT + 5 + lv, fg, bg)
            _COLORS = 256
        except curses.error:
            pass
    return _COLORS


def _heat_attr(level: int):
    """히트맵 한 칸의 속성. 256색이면 발산 팔레트, 8색이면 부호색만."""
    if level == 0 or not _COLORS:
        return curses.A_NORMAL
    if _COLORS >= 256:
        return curses.color_pair(C_HEAT + 5 + level)
    return curses.color_pair(C_POS if level > 0 else C_NEG) | (
        curses.A_BOLD if abs(level) >= 4 else 0)


def _put(scr, y: int, line: str, attr) -> None:
    try:
        scr.addstr(y, 0, line, attr)
    except curses.error:
        pass          # 오른쪽 아래 끝 칸은 curses 가 항상 실패한다


def _colorize_amounts(scr, y: int, line: str) -> None:
    """숫자 구간을 부호색으로 덧칠한다 — 표시 칸 기준.

    ``flow_view.color_spans`` 와 같은 문제를 푼다: 문자 인덱스로 칠하면 한글이
    섞인 줄에서 한 칸씩 밀린다. 여기서는 폭 계산 없이 갈 수 있다 — 부호 문자
    앞까지의 표시 폭을 누적해서 쓴다.
    """
    def _w(t: str) -> int:
        return sum(cell_width(c) for c in t)

    i = 0
    while i < len(line):
        if line[i] in "+-" and i + 1 < len(line) and line[i + 1].isdigit():
            j = i
            while j < len(line) and (line[j].isdigit() or line[j] in "+-,."):
                j += 1
            attr = curses.color_pair(C_POS if line[i] == "+" else C_NEG)
            try:
                scr.chgat(y, _w(line[:i]), _w(line[i:j]), attr)
            except curses.error:
                pass
            i = j
        else:
            i += 1


def _draw(scr, mo: Model) -> None:
    scr.erase()
    h, w = scr.getmaxyx()
    w = max(w - 1, 1)
    # 화면 전체를 `screen()` 이 만든다 — 앱은 색만 고른다. 예전엔 `h - 1` 을
    # 넘겨 놓고 마지막 두 줄을 앱이 **다시** 만들었다. 같은 줄을 두 곳이
    # 만들면 갈라지고, 실제로 갈라졌다(배너를 문자 슬라이스로 잘랐다).
    s = screen(mo, w, h)
    lines, thin, marks = s["lines"], s["thin"], s["marks"]

    if mo.help:
        for y, line in enumerate(lines[:h]):
            edge = (y == 0 or y == len(lines) - 1)
            attr = (curses.color_pair(C_HEAD) | curses.A_BOLD if _COLORS and edge
                    else (curses.A_REVERSE if edge else curses.A_NORMAL))
            _put(scr, y, line, attr)
        scr.refresh()
        return

    for y, line in enumerate(lines):
        if y >= h - 3:
            break
        selected = (s["cursor"] == y)
        if selected:
            attr = curses.color_pair(C_SEL) if _COLORS else curses.A_REVERSE
        elif y < s["head"]:
            attr = (curses.color_pair(C_HEAD) | curses.A_BOLD if _COLORS
                    else curses.A_BOLD)
        elif thin and y < len(thin) and thin[y]:
            # 종목이 10개 미만인 섹터 — "섹터"로 읽으면 안 되는 칸이다.
            attr = curses.color_pair(C_DIM) if _COLORS else curses.A_DIM
        else:
            attr = curses.A_NORMAL
        _put(scr, y, line, attr)
        if _COLORS and not selected and y >= s["head"] and mo.view != "comove":
            _colorize_amounts(scr, y, line)

    # marks 의 y 는 **이미 최종 화면 좌표**다(`screen()` 의 계약). 여기서
    # head 를 더하면 색이 그만큼 밀린다 — 실제로 그랬다.
    for my, mx, mw, lv in marks:
        if 0 <= my < h - 3:
            try:
                scr.chgat(my, mx, mw, _heat_attr(lv))
            except curses.error:
                pass

    dim = curses.color_pair(C_DIM) if _COLORS else curses.A_DIM
    _put(scr, s["hint_y"], lines[s["hint_y"]], dim)
    _put(scr, s["status_y"], lines[s["status_y"]], dim)
    _put(scr, s["banner_y"], lines[s["banner_y"]],
         curses.color_pair(C_BAN) if _COLORS else curses.A_REVERSE)
    scr.refresh()


def _key(mo: Model, ch: int, page: int) -> bool:
    """키 하나 처리. False 를 돌려주면 종료.

    ESC 는 종료가 **아니다.** 느린 SSH 에서 방향키는 ESC 와 나머지로 쪼개져
    도착하고(ESCDELAY 기본 1초), ESC 가 종료면 ↓ 를 눌렀을 뿐인데 앱이 끝난다.
    ``flow_app`` 이 같은 이유로 뺐다.

    도움말이 열려 있으면 **먼저** 가로챈다. 그리고 거기서 ``q`` 는 **닫기**지
    종료가 아니다 — 키를 배우러 연 화면에서 확인 없이 앱이 끝나면 안 된다.
    ``flow_app.handle_key`` 와 같은 규칙이고, 같은 문구로 화면에 적혀 있다.
    두 앱이 다른 손버릇을 가르치면 안 되기 때문이다.
    """
    if mo.help:
        bottom = max(help_total() - page, 0)
        if ch in (curses.KEY_DOWN, ord("j")):
            mo.help_row = min(mo.help_row + 1, bottom)
        elif ch in (curses.KEY_UP, ord("k")):
            mo.help_row = max(mo.help_row - 1, 0)
        elif ch in (curses.KEY_NPAGE, ord(" ")):
            mo.help_row = min(mo.help_row + page, bottom)
        elif ch == curses.KEY_PPAGE:
            mo.help_row = max(mo.help_row - page, 0)
        elif ch in (ord("g"), curses.KEY_HOME):
            mo.help_row = 0
        elif ch in (ord("G"), curses.KEY_END):
            mo.help_row = bottom
        elif ch in (ord("q"), 27, ord("?"), curses.KEY_ENTER, 10, 13):
            mo.help = False
        return True
    if ch == ord("q"):
        return False
    scroll = "hrow" if mo.view == "limits" else "row"
    if ch in (curses.KEY_DOWN, ord("j")):
        setattr(mo, scroll, getattr(mo, scroll) + 1)
    elif ch in (curses.KEY_UP, ord("k")):
        setattr(mo, scroll, max(0, getattr(mo, scroll) - 1))
    elif ch in (curses.KEY_NPAGE, ord(" ")):
        setattr(mo, scroll, getattr(mo, scroll) + page)
    elif ch == curses.KEY_PPAGE:
        setattr(mo, scroll, max(0, getattr(mo, scroll) - page))
    elif ch == curses.KEY_HOME:
        setattr(mo, scroll, 0)
    elif ch == ord("v"):
        mo.cycle("v")
    elif ch == ord("V"):
        mo.cycle("v", -1)
    elif ch in (ord("w"), ord("W")):
        mo.cycle("w", 1 if ch == ord("w") else -1)
    elif ch in (ord("m"), ord("M")):
        mo.cycle("m", 1 if ch == ord("m") else -1)
    elif ch in (ord("a"), ord("A")):
        mo.cycle("a", 1 if ch == ord("a") else -1)
    elif ch in (ord("s"), ord("S")):
        mo.cycle("s", 1 if ch == ord("s") else -1)
    elif ch == ord("d"):
        mo.detrend = not mo.detrend
    elif ch in (ord("?"), curses.KEY_F1):
        # ``?`` 는 이제 **도움말**이다. 예전엔 한계 화면으로 갔는데, ``?`` 를
        # 누른 사람이 첫 번째로 알고 싶은 건 보통 "이 키가 뭐고 이 열이 뭐냐"
        # 다. 한계는 지우지 않았다 — ``v`` 로 도는 네 화면 중 하나로 그대로
        # 있고, 도움말과 배너와 힌트바가 그리로 가는 길을 말한다.
        mo.help = True
        mo.help_row = 0
    return True


def _loop(scr, mo: Model) -> None:
    curses.curs_set(0)
    _init_colors()
    scr.keypad(True)
    while True:
        _draw(scr, mo)
        try:
            ch = scr.getch()
        except KeyboardInterrupt:
            return
        # 터미널이 사라지면(SSH 끊김·창 닫힘) getch 가 ERR(-1) 을 **즉시** 돌려준다.
        # 아무 분기에도 안 태우면 draw→getch→-1 로 100% CPU 를 태우며 영원히 돈다 —
        # 끊긴 세션마다 서버에 좀비가 쌓인다. ``flow_app`` 과 같은 자리, 같은 버그다.
        if ch == -1:
            return
        # 리사이즈는 "아무 키" 가 아니다. 그냥 다시 그리고 넘어간다.
        if ch == curses.KEY_RESIZE:
            continue
        if not _key(mo, ch, max(1, scr.getmaxyx()[0] - 6)):
            return


def main() -> None:
    ap = argparse.ArgumentParser(description="자금 원장 — 주체×섹터 순매수")
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--dump", action="store_true", help="색 없는 평문으로 찍고 끝낸다")
    ap.add_argument("--width", type=int, default=100, help="--dump 의 폭")
    a = ap.parse_args()
    mo = Model(load(a.dir))
    if a.dump:
        print(render_text(mo, a.width))
        return
    curses.wrapper(_loop, mo)


if __name__ == "__main__":
    main()
