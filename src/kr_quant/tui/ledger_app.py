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
import locale
import os

from kr_quant.tui.flow_app import (
    JAMO_TO_ASCII, RICH_AMBER, RICH_BG, RICH_BODY, RICH_DIM, RICH_DOWN,
    RICH_SEL_BG, RICH_SEL_FG, RICH_UP, normalize_key)
from kr_quant.tui.flow_view import color_spans, is_section
from kr_quant.tui.ledger_view import (
    Model, help_total, load, render_text, screen, status_title_span)

DEFAULT_DIR = "~/Documents/kr-quant-reports/latest"

# --- 색 -------------------------------------------------------------------
#
# **두 앱은 같은 팔레트를 쓴다.** 색 상수는 ``flow_app`` 에서 가져온다 — 검은
# 바탕에 앰버가 골격, 본문 회색, 상승·매수 빨강 / 하락·매도 청록(국내 관행이라
# 방향을 안 뒤집는다), 선택행은 반전 대신 은은한 바탕이다. 여기서 다시 정의하면
# 두 화면이 나란히 뜬 SSH 세션에서 같은 뜻의 값이 다른 색으로 보인다 — 이
# 저장소가 폭 계산·도움말 렌더·힌트바에서 이미 세 번 합친 자리다.
#
# 색으로만 전하는 정보는 **새로 만들지 않는다** — 얇은 섹터의 `~`, 최대일몫의 `!`,
# 히트맵의 부호는 전부 글자라 8색·NO_COLOR·`--dump` 에서도 뜻이 남는다.
C_HEAD, C_POS, C_NEG, C_DIM, C_SEL, C_BAN, C_BODY, C_AMBER = range(1, 9)
#: 히트맵 −5..+5 의 색쌍은 C_HEAT + 5 + level 에 잡는다(0 은 안 칠한다).
C_HEAT = 10

#: 256색 발산 팔레트 (파랑 ← 중립 → 빨강). 국내 관행대로 **양수가 빨강**이다.
#: 8색 터미널에서는 부호만 남기고 세기는 글자(브라유 점 개수)가 표현한다.
HEAT_256 = {-5: 25, -4: 26, -3: 32, -2: 38, -1: 45,
            1: 217, 2: 210, 3: 203, 4: 196, 5: 160}

_COLORS = 0        # 0=없음, 8=기본, 256=확장


def _init_colors(scr=None) -> int:
    """색쌍을 잡고 쓸 수 있는 색 수를 돌려준다. ``flow_app._init_colors`` 와 같은 규율.

    ``NO_COLOR`` 표준(no-color.org)을 원장만 안 보고 있었다 — **비어 있지 않은**
    값으로 설정돼 있으면 색을 안 쓴다. 안 보던 시절엔 리다이렉트·로그 캡처에
    색이 그대로 나갔다(같은 제품인데 한쪽만 지키는 규율은 규율이 아니다).
    """
    global _COLORS
    _COLORS = 0
    if os.environ.get("NO_COLOR"):
        return 0
    if not curses.has_colors():
        return 0
    curses.start_color()
    rich = getattr(curses, "COLORS", 0) >= 256 and curses.COLOR_PAIRS > C_HEAT + 11
    if rich:
        try:
            # 바탕을 터미널 기본값(-1)이 아니라 **검정**으로 박는다. 이 배색의
            # 정체성이 검은 바탕이라, 밝은 테마 터미널에서 앰버만 얹으면 대비가
            # 무너진다(노란 글씨가 흰 종이 위에 뜬다).
            curses.init_pair(C_BODY, RICH_BODY, RICH_BG)
            curses.init_pair(C_AMBER, RICH_AMBER, RICH_BG)
            curses.init_pair(C_POS, RICH_UP, RICH_BG)
            curses.init_pair(C_NEG, RICH_DOWN, RICH_BG)
            curses.init_pair(C_DIM, RICH_DIM, RICH_BG)
            curses.init_pair(C_HEAD, RICH_BG, RICH_AMBER)
            curses.init_pair(C_SEL, RICH_SEL_FG, RICH_SEL_BG)
            curses.init_pair(C_BAN, RICH_BG, RICH_AMBER)
            for lv, fg in HEAT_256.items():
                curses.init_pair(C_HEAT + 5 + lv, fg, RICH_BG)
            _COLORS = 256
        except curses.error:
            rich = False        # 256색이라 말해놓고 못 받는 터미널이 있다
    if not rich:
        # 8색 대체. 앰버 자리는 노랑, 본문은 흰색, 어두운 회색은 `A_DIM` 이다.
        bg = curses.COLOR_BLACK
        curses.init_pair(C_BODY, curses.COLOR_WHITE, bg)
        curses.init_pair(C_AMBER, curses.COLOR_YELLOW, bg)
        curses.init_pair(C_POS, curses.COLOR_RED, bg)
        curses.init_pair(C_NEG, curses.COLOR_CYAN, bg)
        curses.init_pair(C_DIM, curses.COLOR_WHITE, bg)
        curses.init_pair(C_HEAD, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        curses.init_pair(C_SEL, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(C_BAN, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        _COLORS = 8
    # 안 쓴 칸까지 검게 — 배경이 반쪽이면 검은 바탕이 아니라 얼룩이 된다.
    if scr is not None:
        try:
            scr.bkgd(" ", curses.color_pair(C_BODY))
        except curses.error:
            pass
    return _COLORS


def _dim_attr():
    """얇은 섹터의 속성 — 8색에는 어두운 회색이 없어 `A_DIM` 으로 대신한다."""
    if not _COLORS:
        return curses.A_DIM
    return curses.color_pair(C_DIM) | (0 if _COLORS >= 256 else curses.A_DIM)


def _sel_attr():
    """선택행 — 256색이면 **은은한 바탕**, 8색이면 반전.

    반전은 그 줄만 하얗게 타서 표 전체의 리듬을 끊는다. 8색에는 검정과 흰색
    사이의 회색이 없어 흉내낼 수단이 없다 — 그 터미널에서는 선택이 보이는
    쪽이 예쁜 쪽보다 낫다.
    """
    if not _COLORS:
        return curses.A_REVERSE
    return curses.color_pair(C_SEL) | curses.A_BOLD


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


def _hl_span(scr, y: int, span, attr) -> None:
    """뷰가 낸 (시작 표시칸, 폭) 구간에 속성을 덧칠한다.

    좌표를 **여기서 계산하지 않는다** — 어느 칸이 무엇인지는 문자열을 만든 쪽만
    안다(`status_title_span`). 앱이 문구를 다시 뜯으면 문구를 고칠 때 색이
    조용히 어긋난다(``flow_app._hl_span`` 과 같은 관용구).
    """
    if not span:
        return
    start, width = span
    try:
        scr.chgat(y, start, width, attr)
    except curses.error:
        pass


def _colorize_amounts(scr, y: int, line: str) -> None:
    """숫자 구간을 부호색으로 덧칠한다 — 어디를 칠할지는 ``flow_view.color_spans``.

    ⚠️ 원장은 이 판정을 **따로 한 벌** 갖고 있었다. 그래서 ``kq-flow`` 가 고친
    회귀가 여기만 살아 있었다 — 한계 화면 §5 의 ``2026-04-07`` 이 ``-04-07`` 로
    잡혀 하락색으로 칠해졌다(실측). 값이 아닌 것이 값처럼 보이는 건 배색 취향
    문제가 아니다. 같은 질문("이 줄에서 어디가 부호값인가")에 두 곳이 답하면
    반드시 갈라진다 — 그래서 답하는 곳을 하나로 둔다.
    """
    for start, width, role in color_spans(line):
        try:
            scr.chgat(y, start, width,
                      curses.color_pair(C_POS if role == "up" else C_NEG))
        except curses.error:
            pass


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
            if _COLORS and edge:
                attr = curses.color_pair(C_HEAD) | curses.A_BOLD
            elif edge:
                attr = curses.A_REVERSE
            elif _COLORS and is_section(line):
                # 구역 제목(`── 키 ──`)은 앰버 — 긴 모달에서 "여기서부터 다른
                # 이야기" 를 색이 먼저 말한다. 판정은 뷰가 진다(`flow_view`).
                attr = curses.color_pair(C_AMBER) | curses.A_BOLD
            else:
                attr = curses.color_pair(C_BODY) if _COLORS else curses.A_NORMAL
            _put(scr, y, line, attr)
        scr.refresh()
        return

    for y, line in enumerate(lines):
        if y >= h - 3:
            break
        selected = (s["cursor"] == y)
        if selected:
            attr = _sel_attr()
        elif y == 0:
            attr = (curses.color_pair(C_HEAD) | curses.A_BOLD if _COLORS
                    else curses.A_BOLD)
        elif y == s["top_head"] and s["head"] > s["top_head"]:
            # 표 머리의 **첫 줄**(원장·전개의 열 이름, 동시성의 판정줄)만 띠다.
            attr = (curses.color_pair(C_HEAD) | curses.A_BOLD if _COLORS
                    else curses.A_BOLD)
        elif y < s["head"]:
            # 나머지 머리 줄은 앰버 **글자**다. 띠를 다섯 줄 깔면(동시성이 그랬다)
            # 화면 위쪽이 통째로 반전이라 정작 표가 안 읽힌다.
            attr = (curses.color_pair(C_AMBER) | curses.A_BOLD if _COLORS
                    else curses.A_BOLD)
        elif thin and y < len(thin) and thin[y]:
            # 종목이 10개 미만인 섹터 — "섹터"로 읽으면 안 되는 칸이다.
            attr = _dim_attr()
        else:
            attr = curses.color_pair(C_BODY) if _COLORS else curses.A_NORMAL
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

    # 힌트바는 골격(앰버), 상태줄은 본문이다. 둘 다 흐린 색이던 시절엔 상태줄의
    # **값**이 비활성처럼 보였다 — 거기 적힌 것은 표에 없는 값인데 그렇다.
    _put(scr, s["hint_y"], lines[s["hint_y"]],
         curses.color_pair(C_AMBER) if _COLORS else curses.A_DIM)
    _put(scr, s["status_y"], lines[s["status_y"]],
         curses.color_pair(C_BODY) if _COLORS else curses.A_NORMAL)
    # 상태줄에서 **섹터 이름만** 세운다 — 그 줄에서 "지금 무엇을 보고 있는가" 를
    # 말하는 건 이름 하나뿐인데 부속 정보에 묻혀 있었다. 좌표는 뷰가 낸다.
    # 무색 터미널에서는 굵게만 — 글자는 안 바꾸므로 정보가 색에만 실리지 않는다.
    _hl_span(scr, s["status_y"], status_title_span(mo, w),
             (curses.color_pair(C_AMBER) if _COLORS else 0) | curses.A_BOLD)
    _put(scr, s["banner_y"], lines[s["banner_y"]],
         curses.color_pair(C_BAN) | curses.A_BOLD if _COLORS else curses.A_REVERSE)
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
    elif ch in (ord("s"), ord("S")) and mo.sortable:
        # 동시성·한계에서는 **아무 일도 안 한다.** 예전엔 `si` 가 돌아서, 그
        # 화면에서는 아무 것도 안 바뀌는데 돌아오면 정렬이 바뀌어 있었다.
        # 판정은 `Model.sortable` 한 곳이고, 헤더가 그 사실을 적는다
        # (`순서[평균상관]`) — ``kq-flow`` 가 종합 화면에서 밟은 그 자리다.
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


# --- 한글 입력 상태의 키 ---------------------------------------------------
#
# 한영을 켜 둔 채 `v` 를 누르면 터미널에는 `ㅍ` 이 들어온다. 예전엔 그 순간 아무
# 일도 안 일어났다 — 화면 하나 바꾸려면 한영을 껐다 켜야 했다. 두벌식은 자판
# **자리** 대응이라 그 대응만 넣으면 같은 자리가 같은 일을 한다.
#
# 표는 ``flow_app.JAMO_TO_ASCII`` **한 벌**이다. 여기서는 원장에만 있는 키
# (`v`·`d`)를 더할 뿐이다 — 두 앱이 각자 표를 가지면 한쪽만 늘어난다.
#
# ⚠️ `ㅍ`·`ㅇ` 은 Shift 를 눌러도 같은 자모다(두벌식에서 쌍자음이 되는 자음은
# `ㅂㅈㄷㄱㅅ` 뿐이다). 터미널에 도착한 뒤에는 구분할 정보가 **이미 없으므로**
# 소문자 동작으로 떨어뜨린다 — 한글 상태에서 못 하는 것은 `V`(화면 역방향)뿐이고,
# 계속 눌러 한 바퀴 돌면 같은 자리에 온다. 도움말이 그 한계를 한 줄로 적는다.
LEDGER_JAMO = dict(JAMO_TO_ASCII,
                   **{"ㅍ": "v", "ㅇ": "d", "\u1111": "v", "\u110b": "d"})


def LEDGER_JAMO_KEY(ch: str) -> int:
    """한 글자 → :func:`_key` 가 아는 int. 자모면 두벌식에서 같은 자리인 ASCII 키다."""
    return ord(LEDGER_JAMO[ch]) if ch in LEDGER_JAMO else normalize_key(ch)


def _read_key(scr) -> int:
    """키 하나 — 한글 자모까지 온전한 한 글자로 받아 int 로 정규화한다.

    `getch` 는 `ㅍ` 을 세 바이트로 쪼개 주므로 `ord("v")` 와 비교하는 구조로는
    **절대** 안 잡힌다. `get_wch` 는 한 글자로 준다. 그래서 입력 경계에서만
    바꾸고 :func:`_key` 는 여전히 int 를 받는다 — 키 하나가 무엇을 했는지 보는
    검사들이 `ord("v")` 를 넘기고 있고, 그 규약을 흔드는 것보다 변경면적이 작다.

    `get_wch` 는 `getch` 의 ERR(-1) 자리에 **예외**를 쓴다. 터미널이 사라졌을 때
    (SSH 끊김·창 닫힘) 100% CPU 를 태우던 방어가 죽지 않게 같은 뜻의 -1 로 되돌린다.
    """
    try:
        ch = scr.get_wch()
    except curses.error:
        return -1
    if isinstance(ch, str) and len(ch) == 1:
        return LEDGER_JAMO_KEY(ch)
    return normalize_key(ch)


def _loop(scr, mo: Model) -> None:
    curses.curs_set(0)
    _init_colors(scr)
    scr.keypad(True)
    while True:
        _draw(scr, mo)
        try:
            ch = _read_key(scr)
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
    # `get_wch` 가 한글을 한 글자로 주려면 로케일이 **와이드 문자를 아는** 상태여야
    # 한다. 파이썬은 C 로케일로 시작하므로 여기서 사용자 환경을 따라간다 — 안 하면
    # `ㅍ` 이 다시 바이트로 쪼개진다. 실패해도 앱은 뜬다(영문 키는 듣는다).
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass
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
