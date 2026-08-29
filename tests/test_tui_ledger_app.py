"""자금 원장 **앱 층**(curses 배선) 검사 — pty 없이 가짜 스크린으로.

왜 따로 있나: `ledger_app.py` 를 보는 검사가 **하나도 없었다.** 뷰 층 검사는
`render_text`·`screen` 을 직접 부르고, pty 스모크는 키를 넣기만 하고 **눌린 결과를
아무것도 안 본다**(종료코드와 문자열 두 개뿐). 그래서 이런 것들이 조용히 지나갔다.

* 히트맵 색이 `nh` 줄만큼 **밀려** 부호 글자와 색이 서로 다른 쌍을 가리켰다.
* 하단 줄이 문자 슬라이스로 잘려 폭 80(SSH 기본)에서 **키가 통째로 사라졌다.**
* `v`·`a`·`d`·↑↓ 를 죽여도 초록이었다.
* `--dump` 가 `--width` 를 무시해도 초록이었다(CLI 배선 무검사).

`addstr`/`chgat` 를 기록하는 가짜 스크린이면 pty 없이 **좌표**를 검증할 수 있다.
"""

from __future__ import annotations

import curses
import json
import os

import pytest

from kr_quant.tui import ledger_app as app
from kr_quant.tui.flow_view import cell_width
from kr_quant.tui.ledger_view import (
    HEAT_RAMP, VIEWS, Model, heat_cell, screen)

_COMOVE_VI = [v for v, _ in VIEWS].index("comove")


def _payload(nsec: int = 12, ndays: int = 60) -> dict:
    secs = [f"섹터{i:02d}" for i in range(nsec)]
    mk = ["거래소", "코스닥"]

    def cell(seed: str) -> dict:
        return {k: [float(((i * 7 + j * 3 + len(seed) + ord(seed[-1])) % 11) - 5)
                    for i in range(ndays)]
                for j, k in enumerate(("indiv", "forgn", "inst", "etc"))}

    return {"dates": [f"2026-01-{i % 28 + 1:02d}" for i in range(ndays)],
            "sectors": secs, "markets": mk,
            "flows": {m: {s: cell(s + m) for s in secs} for m in mk},
            "cap": {m: {s: 10000.0 + i for i, s in enumerate(secs)} for m in mk},
            "n_by_sector": {m: {s: (200 if i % 3 else 2)
                                for i, s in enumerate(secs)} for m in mk},
            "finalized": True}


class FakeScr:
    """`addstr`/`chgat` 를 좌표째 기록하는 30줄짜리 스크린."""

    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.rows: dict[int, str] = {}
        self.chg: list[tuple[int, int, int]] = []

    def erase(self):
        pass

    def getmaxyx(self):
        return (self.h, self.w)

    def addstr(self, y, x, s, attr=0):
        if not (0 <= y < self.h):
            raise curses.error("bad y")
        self.rows[y] = self.rows.get(y, "") + s

    def chgat(self, y, x, n, attr):
        if not (0 <= y < self.h):
            raise curses.error("bad y")
        self.chg.append((y, x, n))

    def refresh(self):
        pass


def _draw(mo: Model, h: int = 30, w: int = 130) -> FakeScr:
    scr = FakeScr(h, w)
    # 색쌍은 initscr() 뒤에만 잡히므로 무색 경로로 돈다. 히트맵 `chgat` 은
    # 색 여부와 무관하게 불리므로(속성만 A_NORMAL 이 된다) 좌표는 그대로 검증된다.
    app._COLORS = 0
    app._draw(scr, mo)
    return scr


def _cells(text: str) -> list[str]:
    """표시 칸 → 글자. 한글은 두 칸을 차지하므로 두 번째 칸은 ``''`` 로 둔다."""
    out: list[str] = []
    for ch in text:
        out.append(ch)
        out += [""] * (cell_width(ch) - 1)
    return out


def test_heatmap_colour_lands_on_the_cell_it_describes():
    """회귀 — 히트맵 색이 행렬보다 `nh` 줄 아래에 칠해졌다(실측 4줄).

    뷰가 `y - top` 을 내보내고 앱이 다시 `head + my` 를 더했다. 둘 다 자기가
    헤더를 더한다고 믿었다. **이 화면의 존재 이유가 색인데** 부호 글자와 색이
    서로 다른 쌍을 가리켰고, 원장 검사 36개가 고치기 전후로 전부 초록이었다.

    주입: `screen()` 의 marks 를 `(y - top, ...)` 로 되돌리거나 `_draw` 가
    `s["head"] + my` 를 더하게 하면, 칠해진 자리가 공백이라 실패한다.
    """
    mo = Model(_payload())
    mo.vi = _COMOVE_VI
    scr = _draw(mo)
    assert scr.chg, "히트맵에 색이 하나도 안 칠해졌다 — 검사가 헛돈다"
    ramp = set(HEAT_RAMP.strip())
    for y, x, n in scr.chg:
        assert n == 2, (y, x, n)
        cells = _cells(scr.rows.get(y, ""))
        got = "".join(cells[x:x + 2])
        assert got[:1] in "+-", f"y{y} x{x} 에 칠했는데 거기 부호가 없다: {got!r}"
        assert got[1:2] in ramp, f"y{y} x{x} 의 세기 글자가 아니다: {got!r}"
    # 그리고 **빠짐없이** 칠해야 한다 — 뷰가 낸 지시를 앱이 하나도 안 버려야 한다.
    # (범례 줄에도 `+█` 글자가 있으므로 화면 글자만으로 세면 안 된다.)
    want = {(y, x) for y, x, _w, _lv in screen(mo, 129, 30)["marks"]}
    assert want, "뷰가 색칠 지시를 안 냈다 — 검사가 헛돈다"
    painted = {(y, x) for y, x, _n in scr.chg}
    assert want == painted, (
        f"앱이 버리거나 옮겼다: 안 칠함={sorted(want - painted)[:5]} "
        f"헛칠함={sorted(painted - want)[:5]}")


def test_heat_cell_and_marks_agree_on_the_level():
    """색 등급과 글자 등급이 같은 값을 가리키는가.

    주입: `screen()` 의 marks 에서 level 을 0 으로 눕히면(색을 안 칠하는 값)
    그려진 부호 칸과 개수가 어긋나 실패한다.
    """
    mo = Model(_payload())
    mo.vi = _COMOVE_VI
    s = screen(mo, 129, 29)
    assert s["marks"], "색칠 지시가 비었다 — 검사가 헛돈다"
    for y, x, _w, lv in s["marks"]:
        cells = _cells(s["lines"][y])
        assert "".join(cells[x:x + 2]) == heat_cell_at(lv), (y, x, lv)


def heat_cell_at(lv: int) -> str:
    """등급 → 두 글자. 기대값을 `heat_cell` 에서 가져오되 **0 등급은 마크가
    아니다** 는 규칙까지 여기서 못박는다."""
    assert lv != 0, "0 등급은 색칠 지시에 들어오면 안 된다"
    return heat_cell(lv / 10 + (0.001 if lv > 0 else -0.001))


def test_bottom_three_lines_are_drawn_once_each_and_are_not_the_same_line():
    """힌트·상태줄·배너가 **각각 한 줄씩**이어야 한다.

    예전엔 앱이 `screen(mo, w, h - 1)` 로 받아 놓고 마지막 두 줄을 스스로
    다시 그려서, 같은 내용이 인접 두 행에 찍히고 터미널 한 줄을 버렸다.

    주입: `_draw` 가 `h - 1` 을 넘기게 되돌리면 배너가 두 줄에 나와 실패한다.
    """
    mo = Model(_payload())
    scr = _draw(mo, h=24, w=101)
    assert set(scr.rows) == set(range(24)), f"안 그린 줄이 있다: {sorted(set(range(24)) - set(scr.rows))}"
    for y, line in scr.rows.items():
        assert sum(cell_width(c) for c in line) == 100, (y, repr(line))
    assert scr.rows[23] != scr.rows[22], "배너가 두 줄에 찍혔다"
    assert "미관측" in scr.rows[23] and "미관측" not in scr.rows[22]
    assert "?" in scr.rows[23], "폭 100 하단에 ? 가 없다"
    assert "정렬" in scr.rows[21], "힌트바가 없다"


def test_keys_actually_change_what_is_drawn():
    """`v`·`a`·`w`·`m`·`d`·↑↓ 가 **화면을 바꾸는가.**

    pty 스모크는 이 키들을 넣기만 하고 결과를 안 봤다 — 여섯 키를 전부 죽여도
    초록이었다.

    주입: `_key` 에서 아무 분기나 지우면 그 키의 단언이 실패한다.
    """
    mo = Model(_payload())
    base = _draw(mo).rows.copy()

    def after(ch: int) -> dict:
        assert app._key(mo, ch, 10) is True
        return _draw(mo).rows.copy()

    assert after(ord("v")) != base, "v 가 화면을 안 바꾼다"
    app._key(mo, ord("V"), 10)                     # 원장으로 되돌린다
    assert _draw(mo).rows == base, "V 가 v 를 되돌리지 않는다"
    for ch in (ord("a"), ord("w"), ord("m"), ord("s"), curses.KEY_DOWN):
        assert after(ch) != base, f"{chr(ch) if ch < 256 else ch} 가 화면을 안 바꾼다"

    # d(β제거)는 동시성 화면에서만 보인다. 상관 캐시 키에서 detrend 를 빼면
    # 전후 히트맵이 **같아진다** — flow 에서 rows() 캐시가 역순을 안 봐서 r 이
    # 안 먹던 것과 같은 모양의 버그다.
    mo2 = Model(_payload())
    mo2.vi = _COMOVE_VI
    # 판정줄·경고문은 detrend 플래그만 봐도 바뀌므로 **행렬 자체**를 봐야 한다.
    # 화면 전체를 비교하면 캐시 버그가 그 문구 뒤에 숨는다(실측: 안 잡혔다).
    def levels(m: Model):
        return [lv for _y, _x, _w, lv in screen(m, 129, 30)["marks"]]

    before = levels(mo2)
    assert before, "히트맵이 비었다 — 검사가 헛돈다"
    app._key(mo2, ord("d"), 10)
    assert mo2.detrend is True
    assert levels(mo2) != before, "d 를 눌러도 상관행렬이 그대로다(캐시 키에 detrend 가 없나)"


def test_escape_and_help_q_do_not_quit_but_plain_q_does():
    """ESC 는 종료가 아니고, 도움말 안의 `q` 도 종료가 아니다.

    주입: `_key` 첫머리를 `if ch in (ord("q"), 27)` 로 되돌리면 실패한다.
    """
    mo = Model(_payload())
    assert app._key(mo, 27, 10) is True
    assert app._key(mo, ord("?"), 10) is True and mo.help
    assert app._key(mo, ord("q"), 10) is True and not mo.help
    assert app._key(mo, ord("q"), 10) is False


def test_dump_honours_the_width_flag(tmp_path, capsys, monkeypatch):
    """`--dump --width` 가 실제로 그 폭으로 찍는가 — **CLI 배선** 검사.

    뷰 층 검사는 `render_text(mo, width)` 를 직접 부르고, pty 스모크는
    `--width` 를 안 준다. 그래서 배선이 끊겨도 아무도 몰랐다.

    주입: `main()` 에서 `a.width` 를 무시하고 기본값을 쓰게 하면 실패한다.
    """
    with open(os.path.join(tmp_path, "payload.json"), "w", encoding="utf-8") as f:
        json.dump(_payload(), f, ensure_ascii=False)
    seen = []
    for width in (72, 140):
        monkeypatch.setattr("sys.argv",
                            ["kq-ledger", "--dir", str(tmp_path), "--dump",
                             "--width", str(width)])
        app.main()
        out = capsys.readouterr().out
        widths = {sum(cell_width(c) for c in ln) for ln in out.splitlines() if ln}
        assert max(widths) <= width, f"--width {width} 인데 {max(widths)}칸 줄이 있다"
        seen.append(max(widths))
    assert seen[0] < seen[1], f"--width 를 바꿔도 결과가 같다: {seen}"


def test_dump_carries_the_key_and_column_help():
    """평문 덤프에도 도움말이 실린다 — 그 사람에게는 `?` 를 누를 터미널이 없다.

    주입: `render_text` 의 도움말 절을 빼면 실패한다.
    """
    from kr_quant.tui.ledger_view import render_text

    text = render_text(Model(_payload()), 100)
    assert "── 키 ──" in text and "최대일몫[%]" in text
    assert "**" not in text


@pytest.mark.parametrize("h,w", [(4, 2), (5, 8), (10, 41), (24, 81), (60, 300)])
def test_draw_survives_any_terminal_size(h, w):
    """어떤 창 크기에서도 안 죽는다 — 도움말·네 화면 전부.

    주입: `help_screen` 의 폭 단계 고르기를 `next()` 기본값 없이 쓰면 좁은
    폭에서 `StopIteration` 으로 죽는다(flow 가 실제로 그랬다).
    """
    mo = Model(_payload())
    for vi in range(len(VIEWS)):
        mo.vi = vi
        for help_open in (False, True):
            mo.help = help_open
            _draw(mo, h=h, w=w)
