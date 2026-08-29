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


# ------------------------------------------- kq-flow 와 같은 규율(패리티)

def test_hangul_jamo_keys_do_what_their_latin_twins_do():
    """한영을 켜 두면 `v` 가 `ㅍ` 로, `s` 가 `ㄴ` 으로 도착해 아무 일도 안 했다.

    두 앱이 같은 손버릇을 가르쳐야 하므로 대응표도 **한 벌**이다
    (`flow_app.JAMO_TO_ASCII`) — 두 벌이면 한쪽만 늘어난다.

    주입: `_read_key`/`normalize_key` 경로를 걷어내고 `getch` 로 되돌리면 실패한다.
    """
    from kr_quant.tui.flow_app import JAMO_TO_ASCII

    # `v`(화면)·`d`(β제거)는 **원장에만 있는 키**다 — 공용 표에 없으므로 여기서
    # 더한다. 그 둘이 빠지면 한글 상태에서 화면조차 못 바꾼다.
    for jamo, latin in (("ㅍ", "v"), ("ㅇ", "d"), ("ㅈ", "w"), ("ㅁ", "a"),
                        ("ㄴ", "s"), ("ㅗ", "h"), ("ㅓ", "j"), ("ㅏ", "k"),
                        ("ㅂ", "q"), ("ㅉ", "W")):
        a, b = Model(_payload()), Model(_payload())
        a.row = b.row = 1
        ka = app._key(a, app.LEDGER_JAMO_KEY(jamo), 10)
        kb = app._key(b, ord(latin), 10)
        snap = lambda m: (m.vi, m.wi, m.mi, m.ai, m.si, m.row, m.hrow,   # noqa: E731
                          m.detrend, m.help, m.help_row)
        assert (ka, snap(a)) == (kb, snap(b)), f"{jamo!r} 가 {latin!r} 와 다르다"
    # 공용 표를 **다시 구현하지 않는다** — 늘어난 키는 두 앱이 같이 받아야 한다.
    for jamo, latin in JAMO_TO_ASCII.items():
        assert app.LEDGER_JAMO[jamo] == latin, f"공용 표와 갈라졌다: {jamo!r}"

    # 그리고 입력 **경로**가 실제로 그 한 글자를 받는가. `getch` 는 자모를 세
    # 바이트로 쪼개 주므로 이 자리를 안 고치면 표만 있고 아무 일도 안 일어난다.
    class _Scr:
        def __init__(self, out):
            self.out = out

        def get_wch(self):
            if isinstance(self.out, Exception):
                raise self.out
            return self.out

    assert app._read_key(_Scr("ㅍ")) == ord("v")
    assert app._read_key(_Scr("ㅈ")) == ord("w")
    assert app._read_key(_Scr(curses.KEY_DOWN)) == curses.KEY_DOWN
    # 터미널이 사라졌을 때의 -1 방어가 살아 있어야 한다(예외는 ERR 자리다).
    assert app._read_key(_Scr(curses.error("gone"))) == -1
    import inspect
    assert "setlocale" in inspect.getsource(app.main), \
        "로케일을 안 따라가면 자모가 다시 바이트로 쪼개진다"


def test_no_color_env_turns_the_colours_off(monkeypatch):
    """`NO_COLOR` 표준(no-color.org)을 원장만 안 보고 있었다 — 리다이렉트·로그
    캡처에 색이 그대로 나갔다. `kq-flow` 는 이미 본다.

    주입: `_init_colors` 에서 그 검사를 빼면 `curses.has_colors()` 가
    initscr 없이 불려 예외로 실패한다(= 검사가 살아 있다).
    """
    monkeypatch.setenv("NO_COLOR", "1")
    assert app._init_colors(None) == 0
    assert app._COLORS == 0


def test_dates_in_the_body_are_not_painted_as_negative_numbers(monkeypatch):
    """한계 화면 §5 의 `2026-04-07` 이 `-04-07` 로 잡혀 하락색으로 칠해졌다.

    원장이 부호 구간 찾기를 **따로 한 벌** 갖고 있어서, `kq-flow` 가 고친 날짜
    회귀가 여기만 남았다. 이제 두 앱이 `flow_view.color_spans` 하나를 쓴다.

    주입: `_colorize_amounts` 를 되살리거나 `color_spans` 의 앞 제한을 지우면
    첫 단언이 실패한다.
    """
    # 색쌍은 initscr() 뒤에만 잡힌다 — 여기서 보는 것은 **어디를** 칠하느냐다.
    monkeypatch.setattr(curses, "color_pair", lambda n: n * 256)
    mo = Model(_payload())
    mo.vi = [v for v, _ in VIEWS].index("limits")
    scr = FakeScr(30, 121)
    app._COLORS = 8
    app._draw(scr, mo)
    for y, x, n in scr.chg:
        painted = "".join(_cells(scr.rows.get(y, ""))[x:x + n])
        assert not painted.strip().startswith("-0"), \
            f"날짜에 색을 칠했다: {painted!r} · {scr.rows.get(y, '')!r}"
    # 진짜 부호는 여전히 칠한다 — 규칙이 넓어져 다 죽으면 안 된다.
    mo.vi = 0
    scr = FakeScr(30, 121)
    app._draw(scr, mo)
    assert scr.chg, "원장에서 부호색이 통째로 죽었다"
    app._COLORS = 0


def test_the_app_paints_the_sector_name_using_the_coordinates_the_view_gives(
        monkeypatch):
    """상태줄에서 섹터 이름만 세운다. 좌표는 뷰가 낸다 — 앱이 문자열을 다시
    뜯으면 문구를 고칠 때 색이 조용히 어긋난다.

    주입: `_draw` 에서 `status_title_span` 덧칠을 빼면 실패한다.
    """
    from kr_quant.tui.ledger_view import status_title_span

    monkeypatch.setattr(curses, "color_pair", lambda n: n * 256)
    mo = Model(_payload())
    scr = FakeScr(30, 121)
    app._COLORS = 8
    app._draw(scr, mo)
    s = screen(mo, 120, 30)
    want = status_title_span(mo, 120)
    assert want, "뷰가 좌표를 안 냈다 — 검사가 헛돈다"
    assert (s["status_y"], want[0], want[1]) in scr.chg, \
        f"상태줄의 이름을 안 세웠다: {[c for c in scr.chg if c[0] == s['status_y']]}"
    app._COLORS = 0
