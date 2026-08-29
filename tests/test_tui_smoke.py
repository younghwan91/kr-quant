"""TUI 스모크 — **실제 curses 경로**를 의사 터미널에서 돌린다.

렌더 로직만 테스트하면 curses 쪽 오류가 안 잡힌다. 실제로 `scr._curses_window`
에 속성을 붙이려다 AttributeError 로 실행 자체가 죽었는데, 렌더 테스트 12개가
전부 초록이었다. 화면을 띄우는 경로도 한 번은 지나가야 한다.
"""

from __future__ import annotations

import json
import os
import pty
import subprocess
import sys
import tempfile
import threading
import time

import pytest

MINIMAL = {
    "asof": "2026-08-28", "finalized": True, "dates": ["2026-01-01", "2026-08-28"],
    "names": {"005930": {"name": "삼성전자", "sector": "전기/전자", "market": "거래소",
                         "cap": 1000.0,
                         "win": {"20": {"inst": 10.0, "forgn": 1.0, "indiv": -5.0,
                                        "etc": -6.0, "tv": 500.0}}}},
    "n_by_sector": {"거래소": {"전기/전자": 1}},
    "blocks": {f"{w}|{m}": {
        "from": "2026-01-01", "to": "2026-08-28", "k": 1.0, "b": 0.0, "t": 2.0,
        "rows": [{"sector": "전기/전자", "n_all": 1, "thin": False, "G": 0.5,
                  "G_pass": True, "inst": 10.0, "forgn": 1.0, "indiv": -5.0,
                  "etc": -6.0, "accel": 1.0, "ret": -2.0, "exp": 1.0, "x": 3.0,
                  "U": 4.0, "P": 0.1, "xdot": 0.2, "xddot": 0.01,
                  "a_idx": 1.0, "cap_idx": 1000.0,
                  "top": {"buy": [], "sell": [], "n": 1}}]}
        for w in ("5", "20", "60", "120") for m in ("전체", "거래소", "코스닥")},
    "combined": {},
}


def _drain(fd: list[str]):
    """pty 출력을 계속 읽어 버리는 스레드.

    안 읽으면 pty 버퍼(~64KB)가 차는 순간 앱의 refresh() 가 write 에서 막히고,
    앱은 q 를 못 읽는다. 그러면 타임아웃이 나면서 "TUI 가 q 에 종료하지 않았다"
    는 **앱을 가리키는 엉뚱한 메시지**로 실패한다. 키를 몇 개만 더 넣어도
    걸리는 지뢰라, 스모크를 넓히려면 이게 먼저다.
    """
    def run(primary):
        buf = []
        try:
            while True:
                chunk = os.read(primary, 65536)
                if not chunk:
                    break
                buf.append(chunk)
        except OSError:
            pass
        fd.append(b"".join(buf).decode("utf-8", "replace"))
    return run


def _spawn(d: str, app: str, lines: str, cols: str, stderr=subprocess.PIPE,
           env: dict | None = None):
    """pty 를 열고 TUI 앱을 띄운다 — 두 앱(kq-flow·kq-ledger)이 같은 경로를 쓴다."""
    primary, replica = pty.openpty()
    env = {**(env or os.environ), "TERM": "xterm-256color",
           "LINES": lines, "COLUMNS": cols}
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"from kr_quant.tui.{app} import main; main()", "--dir", d],
        stdin=replica, stdout=replica, stderr=stderr, env=env)
    os.close(replica)
    return primary, proc


def _run_tui(d: str, keys: bytes, lines: str = "24", cols: str = "100",
             env: dict | None = None, app: str = "flow_app"):
    """의사 터미널에서 앱을 띄우고 (종료코드, 화면, stderr) 를 돌려준다."""
    primary, proc = _spawn(d, app, lines, cols, env=env)
    screen: list[str] = []
    t = threading.Thread(target=_drain(screen), args=(primary,), daemon=True)
    t.start()
    try:
        os.write(primary, keys)
        _out, err = proc.communicate(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        os.close(primary)
        t.join(timeout=2)
        pytest.fail("TUI 가 q 에 종료하지 않았다")
    os.close(primary)
    t.join(timeout=2)
    return proc.returncode, (screen[0] if screen else ""), err.decode("utf-8", "replace")


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_app_starts_and_quits_in_a_pty():
    """앱이 의사 터미널에서 뜨고 q 로 정상 종료하는가."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "numbers.html"), "w", encoding="utf-8") as f:
            f.write("<script>\nconst D = " + json.dumps(MINIMAL, ensure_ascii=False)
                    + ";\n</script>")
        rc, screen, err = _run_tui(d, b"wmas\x1b[B\n\x1b" b"q")
        assert rc == 0, err[-1500:]
        assert not err.strip(), f"stderr 에 뭔가 나왔다:\n{err[-1500:]}"
        assert "전기/전자" in screen, "표가 안 그려졌다"


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_app_survives_a_long_key_sequence():
    """회귀 — 출력이 pty 버퍼를 넘겨도 살아남는가.

    실측: 키 22개면 23KB 가 나와 64KB 버퍼 아래지만, 읽어주지 않으면
    막힌다. 예전 스모크는 키 8개(8.8KB)라 **우연히** 통과하고 있었다.
    """
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "numbers.html"), "w", encoding="utf-8") as f:
            f.write("<script>\nconst D = " + json.dumps(MINIMAL, ensure_ascii=False)
                    + ";\n</script>")
        keys = b"wwwwmmmaaaassss" * 8 + b"?\x1b" + b"\n\x1b" + b"q"
        rc, _screen, err = _run_tui(d, keys)
        assert rc == 0, err[-1500:]


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
@pytest.mark.parametrize("lines,cols", [("10", "40"), ("24", "80"), ("50", "200")])
def test_app_survives_small_and_large_terminals(lines, cols):
    """좁은 터미널에서도 죽지 않는가. 사용자는 SSH 로 접속한다."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "numbers.html"), "w", encoding="utf-8") as f:
            f.write("<script>\nconst D = " + json.dumps(MINIMAL, ensure_ascii=False)
                    + ";\n</script>")
        rc, _screen, err = _run_tui(d, b"wmas?\x1b\n\x1bq", lines=lines, cols=cols)
        assert rc == 0, err[-1500:]


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_app_exits_when_the_terminal_disappears():
    """회귀 — SSH 가 끊기면(=pty 가 사라지면) 앱이 나가야 한다.

    getch() 가 ERR(-1) 을 즉시 돌려주는데 이걸 아무 분기에도 안 태우면
    draw→getch→-1 로 **100% CPU 를 태우며 영원히 돈다**. 실측 3초에 CPU
    299틱(99.7%), SIGKILL 로만 죽었다. 끊긴 세션마다 서버에 좀비가 쌓인다.
    """
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "numbers.html"), "w", encoding="utf-8") as f:
            f.write("<script>\nconst D = " + json.dumps(MINIMAL, ensure_ascii=False)
                    + ";\n</script>")
        primary, replica = pty.openpty()
        env = {**os.environ, "TERM": "xterm-256color", "LINES": "24", "COLUMNS": "100"}
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "from kr_quant.tui.flow_app import main; main()", "--dir", d],
            stdin=replica, stdout=replica, stderr=subprocess.DEVNULL, env=env)
        os.close(replica)
        # 배수 스레드를 쓰지 않는다 — 읽는 중에 같은 fd 를 닫으면 경합이라
        # 앱이 아니라 테스트 쪽 사정으로 프로세스가 끝날 수 있다.
        time.sleep(0.5)                        # 한 번은 그리게 둔다
        os.close(primary)                      # 터미널이 사라진다
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            ticks = "?"
            try:
                st = open(f"/proc/{proc.pid}/stat").read().split()
                ticks = int(st[13]) + int(st[14])
            except OSError:
                pass
            proc.kill()
            proc.communicate()
            pytest.fail(f"터미널이 사라졌는데 앱이 계속 돈다 — CPU 좀비 ({ticks} 틱)")


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_lone_escape_does_not_quit_the_main_screen():
    """회귀 — 느린 SSH 에서 방향키는 ESC 와 나머지로 쪼개져 도착한다
    (ESCDELAY 기본 1초). ESC 가 종료면 ↓ 를 눌렀을 뿐인데 앱이 끝난다."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "numbers.html"), "w", encoding="utf-8") as f:
            f.write("<script>\nconst D = " + json.dumps(MINIMAL, ensure_ascii=False)
                    + ";\n</script>")
        # ESC 로 끝나버리면 그 뒤의 `?` 가 앱에 닿지 않아 도움말이 안 그려진다.
        # 첫 그림에도 있는 문자열로는 구분이 안 되므로 도움말 전용 문구를 본다.
        rc, screen, err = _run_tui(d, b"\x1b\x1b\x1b" b"?" b"q" b"q")
        assert rc == 0, err[-1500:]
        assert "섹터 표" in screen, (
            "ESC 뒤에 앱이 이미 죽어서 도움말이 안 떴다")


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
@pytest.mark.parametrize("payload,expect", [
    ('{"asof": "2026-08-28"}', "blocks"),          # blocks 없음
    ('{"asof": "x", "blocks": nope}', "JSON"),     # 정규식은 맞고 JSON 이 깨짐
])
def test_broken_report_says_what_is_wrong(payload, expect):
    """회귀 — 예전엔 curses 안에서 생 트레이스백으로 터졌다."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "numbers.html"), "w", encoding="utf-8") as f:
            f.write("<script>\nconst D = " + payload + ";\n</script>")
        proc = subprocess.run(
            [sys.executable, "-c",
             "from kr_quant.tui.flow_app import main; main()", "--dir", d],
            capture_output=True, text=True)
        assert proc.returncode != 0
        msg = proc.stdout + proc.stderr
        assert expect in msg, msg[-800:]
        assert "Traceback" not in msg, f"생 트레이스백이 나왔다:\n{msg[-800:]}"


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_resize_does_not_silently_close_the_help_screen():
    """회귀 — 도움말을 연 채 창 크기를 바꾸면 도움말이 소리 없이 닫혔다.

    ncurses 가 KEY_RESIZE 를 키처럼 주는데, 도움말 분기의 `else: help=False`
    가 그걸 "아무 키" 로 먹었다. 리사이즈는 키가 아니다.
    """
    import fcntl
    import signal
    import struct
    import termios

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "numbers.html"), "w", encoding="utf-8") as f:
            f.write("<script>\nconst D = " + json.dumps(MINIMAL, ensure_ascii=False)
                    + ";\n</script>")
        primary, replica = pty.openpty()
        fcntl.ioctl(replica, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
        env = {**os.environ, "TERM": "xterm-256color"}
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "from kr_quant.tui.flow_app import main; main()", "--dir", d],
            stdin=replica, stdout=replica, stderr=subprocess.PIPE, env=env)
        os.close(replica)
        screen: list[str] = []
        t = threading.Thread(target=_drain(screen), args=(primary,), daemon=True)
        t.start()
        try:
            os.write(primary, b"?")                      # 도움말 열기
            time.sleep(0.4)
            fcntl.ioctl(primary, termios.TIOCSWINSZ,     # 창 크기 변경
                        struct.pack("HHHH", 20, 70, 0, 0))
            # ioctl 만으로는 자식에게 SIGWINCH 가 안 간다(실측: KEY_RESIZE 미도달).
            # 신호를 직접 보내야 ncurses 가 410 을 준다.
            proc.send_signal(signal.SIGWINCH)
            time.sleep(0.6)
            # 도움말이 살아 있어야만 닿는 지점으로 판별한다 — 끝까지 스크롤하면
            # 마지막 줄이 나온다. 닫혔다면 j 는 섹터 행을 움직일 뿐이다.
            # 스크롤 횟수를 상수로 박으면 HELP 가 길어질 때 무관한 이유로 깨진다.
            from kr_quant.tui.flow_view import help_lines
            os.write(primary, b"j" * (help_lines(80, 0, 10**6)[1] + 5))
            time.sleep(0.6)
            os.write(primary, b"q")                      # 도움말 닫기
            time.sleep(0.3)
            os.write(primary, b"q")                      # 앱 종료
            _out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            os.close(primary)
            pytest.fail("종료하지 않았다")
        os.close(primary)
        t.join(timeout=2)
        assert proc.returncode == 0, err.decode("utf-8", "replace")[-1000:]
        assert "섹터 표" in screen[0], "도움말이 아예 안 떴다"
        assert "관망" in screen[0], "리사이즈에 도움말이 닫혀 끝까지 못 내려갔다"


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_no_color_and_dumb_terminal():
    """색: NO_COLOR(표준)를 보고, TERM=dumb 이면 **안내하고 나간다**.

    dumb 에서는 예전엔 빈 화면으로 살아 있으면서 키만 먹었다.
    """
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "numbers.html"), "w", encoding="utf-8") as f:
            f.write("<script>\nconst D = " + json.dumps(MINIMAL, ensure_ascii=False)
                    + ";\n</script>")
        env = {**os.environ, "NO_COLOR": "1"}
        rc, screen, err = _run_tui(d, b"wmas\n\x1bq", env=env)
        assert rc == 0, err[-1500:]
        assert "전기/전자" in screen, "NO_COLOR 에서 표가 안 그려졌다"
        for seq in ("\x1b[31m", "\x1b[33m", "\x1b[36m", "38;5;"):
            assert seq not in screen, f"NO_COLOR 인데 색 시퀀스 {seq!r} 가 나왔다"

        proc = subprocess.run(
            [sys.executable, "-c",
             "from kr_quant.tui.flow_app import main; main()", "--dir", d],
            capture_output=True, text=True, env={**os.environ, "TERM": "dumb"})
        msg = proc.stdout + proc.stderr
        assert proc.returncode != 0, "TERM=dumb 인데 그냥 떴다"
        assert "TERM" in msg and "Traceback" not in msg, msg[-800:]


# --- 키·배치 — curses 없이 상태로 판정한다 ---------------------------------
#
# pty 검사는 "통과했는데 아무것도 확인 못 한" 경우가 이 저장소에서만 세 번
# 나왔다(같은 fd 경합 · 잡을 케이스를 continue 로 건너뜀 · SIGWINCH 미전달).
# 키가 무엇을 했는지는 **상태를 직접 보는 게** 정직하다. 아래 검사들은 전부
# 막으려는 버그를 실제로 주입해 실패하는지 확인했다.

import curses  # noqa: E402

from kr_quant.tui.flow_app import handle_key, layout  # noqa: E402
from kr_quant.tui.flow_view import SORTS, State  # noqa: E402

MANY = json.loads(json.dumps(MINIMAL))
for _i, (_code, _sec) in enumerate([("000010", "전기/전자"), ("000020", "화학"),
                                    ("000030", "은행"), ("000040", "건설")]):
    MANY["names"][_code] = {
        "name": f"종목{_i}", "sector": _sec, "market": "거래소",
        "cap": 100.0 * (_i + 1),
        "win": {w: {"inst": 10.0 * (_i - 1), "forgn": 1.0, "indiv": -5.0,
                    "etc": -6.0, "tv": 100.0 * (_i + 1)}
                for w in ("5", "20", "60", "120")}}
for _key, _b in MANY["blocks"].items():
    _b["rows"] = [dict(_b["rows"][0], sector=_s, inst=float(_v), accel=float(_v),
                       G=_v / 100.0, cap=1000.0)
                  for _s, _v in (("전기/전자", 30), ("화학", 10), ("은행", -20),
                                 ("건설", -40))]
# 종목이 여러 개인 섹터 — 드릴다운 커서 유지를 볼 수 있게. 값은 서로 **달라야**
# 한다: 같으면 정렬이 안정적이라 역순 검사가 뒤집히지 않아도 통과한다.
for _n, (_c, _v) in enumerate((("000021", 7.0), ("000022", -3.0),
                               ("000023", 4.0))):
    MANY["names"][_c] = dict(
        MANY["names"]["000020"], name=f"화학{_n}",
        win={w: dict(MANY["names"]["000020"]["win"][w], inst=_v, tv=_v * 10)
             for w in ("5", "20", "60", "120")})


def _st(data=None) -> State:
    return State(json.loads(json.dumps(data or MANY)))


def _sectors(st: State) -> list[str]:
    return [r["sector"] for r in st.rows()]


def test_h_leaves_the_drill_instead_of_opening_help():
    """h 는 vim 의 '왼쪽' — l 이 들어가기면 h 는 나오기다.

    예전엔 h 가 도움말이었고 드릴다운에서 h 는 **아무 일도 안 했다**.
    반쪽만 맞는 관례는 안 맞느니만 못하다.
    """
    st = _st()
    handle_key(st, ord("l"))
    assert st.drill
    handle_key(st, ord("h"))
    assert not st.drill, "드릴다운에서 h 가 나가지 않았다"
    assert not st.help, "h 가 아직 도움말을 연다"


def test_help_opens_with_question_and_f1():
    for key in (ord("?"), curses.KEY_F1):
        st = _st()
        assert handle_key(st, key)
        assert st.help, f"{key} 로 도움말이 안 열렸다"


def test_q_in_help_closes_and_only_the_next_q_quits():
    """도움말의 q 는 닫기다 — 그리고 그 사실이 화면에 적혀 있어야 한다."""
    from kr_quant.tui.flow_view import HELP, help_lines
    st = _st()
    handle_key(st, ord("?"))
    assert handle_key(st, ord("q")) is True, "도움말의 q 가 앱을 끝냈다"
    assert not st.help
    assert handle_key(st, ord("q")) is False, "그 다음 q 가 종료하지 않았다"
    # 관례(less: q=종료)를 어기는 쪽은 화면에 밝혀야 한다.
    said = help_lines(120, 0, 10**6)[0][0] + " ".join(d for _n, d in HELP)
    assert "한 번 더" in said, "q 가 닫기라는 사실이 어디에도 안 적혀 있다"


def test_help_lists_every_key_the_app_handles():
    """회귀 — 대문자 역방향·g/G·PgUp/PgDn·l/← 이 푸터에도 도움말에도 없었다."""
    from kr_quant.tui.flow_view import HELP, footer_line
    text = " ".join(n + " " + d for n, d in HELP) + footer_line(200)
    for token in ("g", "G", "Home", "End", "PgUp", "PgDn", "W", "M", "A", "S",
                  "r", "h", "l", "Enter", "Esc", "F1", "?", "q"):
        assert token in text, f"도움말에 {token!r} 가 없다"


def test_r_reverses_the_sector_sort():
    """정렬 역순 — 이게 없어서 '누가 털렸나' 로 가는 길이 G 하나뿐이었다."""
    st = _st()
    down = _sectors(st)
    handle_key(st, ord("r"))
    assert st.rev
    assert _sectors(st) == down[::-1], "r 이 순서를 뒤집지 않았다"
    handle_key(st, ord("r"))
    assert _sectors(st) == down, "r 이 토글이 아니다"


def test_r_keeps_the_selected_sector():
    st = _st()
    handle_key(st, curses.KEY_DOWN)
    sec = st.rows()[st.row]["sector"]
    handle_key(st, ord("r"))
    assert st.rows()[st.row]["sector"] == sec, "역순에 커서가 섹터를 놓쳤다"


def test_r_reverses_the_name_list_and_keeps_the_cursor():
    st = _st()
    st.row = _sectors(st).index("화학")
    handle_key(st, ord("l"))
    handle_key(st, curses.KEY_DOWN)
    code = st.names()[st.drow]["code"]
    down = [t["code"] for t in st.names()]
    handle_key(st, ord("r"))
    assert st.nrev and [t["code"] for t in st.names()] != down
    assert st.names()[st.drow]["code"] == code, "역순에 종목 커서를 놓쳤다"


def test_missing_values_stay_last_in_both_directions():
    """역순이라고 결측이 위로 오면 안 된다 — '값 없음'은 작은 값이 아니다."""
    st = _st()
    for _k, b in st.d["blocks"].items():
        b["rows"][1]["G"] = None
    # 인덱스를 박으면 정렬 순서가 바뀔 때 무관한 이유로 깨진다(실제로 깨졌다).
    st.si = [k for k, _ in SORTS].index("G")
    for rev in (False, True):
        st.rev = rev
        vals = [r.get("G") for r in st.rows()]
        assert vals[-1] is None, f"rev={rev} 에서 결측이 맨 뒤가 아니다: {vals}"


def test_window_market_actor_work_inside_the_drill():
    """드릴다운에서 w·m·a 가 조용히 무시되던 회귀.

    Esc → W → Enter 로 나갔다 들어오면 종목 커서를 잃었다.
    """
    st = _st()
    st.row = _sectors(st).index("화학")
    handle_key(st, ord("l"))
    handle_key(st, curses.KEY_DOWN)
    code = st.names()[st.drow]["code"]
    for key in (ord("w"), ord("W"), ord("m"), ord("a")):
        before = st.window, st.market, st.actor
        handle_key(st, key)
        assert st.drill, f"{chr(key)} 가 드릴다운을 닫았다"
        assert (st.window, st.market, st.actor) != before, \
            f"{chr(key)} 가 드릴다운에서 아무 일도 안 했다"
        assert st.rows()[st.row]["sector"] == "화학", \
            f"{chr(key)} 뒤에 섹터 선택을 잃었다"
        if any(t["code"] == code for t in st.names()):
            assert st.names()[st.drow]["code"] == code, \
                f"{chr(key)} 뒤에 종목 커서를 잃었다"


def test_hint_bar_explains_the_column_being_sorted():
    """도움말이 전면 모달이라 '이 숫자가 뭐냐' 를 물은 사람이 그 숫자를 못 봤다.

    푸터 위 한 줄은 늘 떠 있고, s 로 정렬을 바꾸면 같이 바뀐다.
    """
    from kr_quant.tui.flow_app import hint_text
    from kr_quant.tui.flow_view import SORTS, hint_desc

    st = _st()
    seen = set()
    for _ in range(len(SORTS)):
        line = hint_text(st)
        header = line.split("정렬 ", 1)[1].split("▼")[0].split("▲")[0]
        desc = hint_desc(header)
        assert desc, f"'{header}' 의 설명이 어디에도 없다 — 힌트바가 빈다"
        # 힌트바는 **강조** 별표를 뗀다(통과 마커 * 와 헷갈린다).
        assert desc.strip().replace("**", "")[:12] in line, f"열 설명이 없다: {line}"
        assert ("▼" in line) != ("▲" in line), f"정렬 방향 표시가 없다: {line}"
        seen.add(line)
        handle_key(st, ord("s"))
    assert len(seen) > 1, "s 로 정렬을 바꿔도 힌트바가 그대로다"
    handle_key(st, ord("r"))
    assert "▲" in hint_text(st), "r 을 눌러도 힌트바의 방향이 안 바뀐다"
    # 드릴다운에서는 종목 정렬을 설명한다.
    st.rev = False
    handle_key(st, ord("l"))
    assert "순매수" in hint_text(st), hint_text(st)


def _snap(st: State) -> tuple:
    """키 하나가 무엇을 바꿨는지 볼 상태 사진."""
    return (st.wi, st.mi, st.ai, st.si, st.nsi, st.row, st.drow,
            st.drill, st.help, st.hrow, st.rev, st.nrev)


def test_hangul_jamo_keys_do_what_their_latin_twins_do():
    """한영을 켜 두면 `w` 가 `ㅈ` 으로 도착해 아무 일도 안 했다.

    두벌식은 자판 **자리** 대응이라 같은 자리는 같은 일을 해야 한다. 화면
    표기는 영문 그대로다(사용자가 한글 병기를 원하지 않았다).

    주입: `normalize_key` 가 자모를 그대로 `ord` 하면 전부 실패한다.
    """
    from kr_quant.tui.flow_app import normalize_key

    pairs = [("ㅂ", "q"), ("ㅈ", "w"), ("ㅉ", "W"), ("ㄱ", "r"), ("ㄲ", "R"),
             ("ㅁ", "a"), ("ㄴ", "s"), ("ㅎ", "g"), ("ㅡ", "m"),
             ("ㅗ", "h"), ("ㅓ", "j"), ("ㅏ", "k"), ("ㅣ", "l")]
    for jamo, latin in pairs:
        for drill in (False, True):
            a, b = _st(), _st()
            a.drill = b.drill = drill
            # 커서를 가운데로 — 0 에서는 위·아래가 둘 다 제자리라 j/k 가 안 갈린다.
            a.row = b.row = 1
            a.drow = b.drow = 0
            ka = handle_key(a, normalize_key(jamo))
            kb = handle_key(b, ord(latin))
            assert (ka, _snap(a)) == (kb, _snap(b)), (
                f"{jamo!r} 가 {latin!r} 와 다르게 동작한다(드릴{drill})")
    # 첫가끝 자모(U+11xx)로 보내는 IME 도 있다 — 둘 다 받는다.
    assert normalize_key("\u110c") == ord("w")


def test_hangul_cannot_tell_some_capitals_apart_so_they_fall_back(): 
    """두벌식에서 Shift 로 다른 글자가 나오는 자음은 `ㅂㅈㄷㄱㅅ` 뿐이다.

    `A`·`S`·`G`·`M` 은 Shift 를 눌러도 같은 자모라, 터미널에 도착한 뒤에는
    구분할 정보가 **이미 없다.** 앱이 할 수 있는 일은 소문자 동작으로 떨어뜨리는
    것뿐이고, 그 사실은 주석과 도움말에 적혀 있어야 한다.
    """
    from kr_quant.tui.flow_app import normalize_key
    from kr_quant.tui.flow_view import HELP

    for jamo, latin in (("ㅁ", "a"), ("ㄴ", "s"), ("ㅎ", "g"), ("ㅡ", "m")):
        assert normalize_key(jamo) == ord(latin), f"{jamo!r} 가 소문자로 안 떨어진다"
    # 구분되는 쪽은 역방향이 살아 있어야 한다 — 그게 없으면 한글에서 정렬을
    # 되돌릴 길이 아예 없어진다.
    a, b = _st(), _st()
    handle_key(a, normalize_key("ㅈ"))
    handle_key(b, normalize_key("ㅉ"))
    assert a.wi != b.wi, "ㅈ 과 ㅉ 이 같은 방향으로 돈다"
    said = " ".join(n + " " + d for n, d in HELP)
    assert "한영" in said, "한글 상태에서도 듣는다는 사실이 화면 어디에도 없다"


def test_the_screen_never_prints_the_hangul_keys():
    """한글 키를 **적지는** 않는다 — 사용자가 병기를 원하지 않았다."""
    from kr_quant.tui.flow_view import FOOTER_DRILL_TIERS, FOOTER_TIERS, HELP

    text = " ".join(FOOTER_TIERS + FOOTER_DRILL_TIERS
                    + tuple(n + " " + d for n, d in HELP))
    for jamo in "ㅂㅈㅉㄱㄲㅁㄴㅎㅡㅗㅓㅏㅣ":
        assert jamo not in text, f"화면에 자모 {jamo!r} 가 적혀 있다"


def test_footer_grows_and_shrinks_with_the_width():
    """푸터가 한 줄 고정이라 넓은 화면에서 절반이 비고 좁은 화면에서 잘렸다."""
    from kr_quant.tui.flow_view import cell_len as _w, footer_line
    for drill in (False, True):
        wide, narrow = footer_line(200, drill), footer_line(40, drill)
        assert _w(wide) > _w(narrow), "폭이 넓어도 푸터가 안 늘어난다"
        for width in (20, 40, 60, 80, 120, 200):
            line = footer_line(width, drill)
            assert _w(line) <= width or width < _w(footer_line(0, drill)), \
                f"폭 {width} 에서 푸터가 넘친다: {line}"
            # 줄어든 푸터가 "여기가 전부" 로 읽히면 안 된다 — 나머지는 ? 에 있다.
            assert "?" in line, f"폭 {width} 푸터에 ? 안내가 없다: {line}"


def test_page_keys_follow_the_screen_height():
    """PgDn 이 10줄 고정이던 회귀 — 200x50 에서 반 페이지도 안 갔다."""
    assert layout(50)[1] > layout(24)[1] > 10
    st = _st()
    handle_key(st, curses.KEY_NPAGE, page=2)
    assert st.row == 2
    handle_key(st, curses.KEY_PPAGE, page=1)
    assert st.row == 1


def test_home_and_end_move_like_g_and_G():
    st = _st()
    n = len(st.rows())
    handle_key(st, curses.KEY_END)
    assert st.row == n - 1
    handle_key(st, curses.KEY_HOME)
    assert st.row == 0


def test_help_scroll_stops_where_the_last_line_is_at_the_bottom():
    """하한이 total-5 이던 시절엔 끝까지 내리면 화면 아래가 비었다."""
    from kr_quant.tui.flow_view import help_lines
    total = help_lines(80, 0, 10**6)[1]
    st = _st()
    handle_key(st, ord("?"))
    for _ in range(500):
        handle_key(st, curses.KEY_DOWN, help_page=20)
    assert st.hrow == total - 20, f"스크롤 하한이 어긋난다: {st.hrow}"


def test_detail_panel_folds_on_short_screens_so_the_table_survives():
    """6줄 터미널에서 상세 패널이 헤더를 덮어써 표가 통째로 사라졌다."""
    for h in (6, 8, 9):
        lay = layout(h)
        assert lay.detail == 0, f"h={h} 에서 상세 패널이 아직 표를 밀어낸다"
        assert lay.rows >= 1, f"h={h} 에서 표가 한 줄도 안 남았다"
    # 줄 수를 박으면 패널에 줄이 늘 때 무관한 이유로 깨진다 — 있고,
    # 좁아지면 줄어들고, 더 좁아지면 접힌다는 **성질**만 본다.
    assert layout(24).detail >= 3, "정상 높이에서는 상세 패널이 있어야 한다"
    assert layout(24).detail >= layout(11).detail >= 3, "높이가 줄면 패널도 줄어야 한다"


def test_tall_screens_give_the_extra_height_to_the_table():
    """200x50 에서 표와 상세 패널 사이에 빈 줄이 14줄 남던 회귀."""
    lay = layout(50)
    assert lay.head + 1 + lay.rows + lay.gap + lay.detail == lay.hint_y, \
        "세로 배치에 뜻 없는 빈 줄이 남는다"
    assert lay.foot_y == 49 and lay.hint_y == 48


def test_detail_panel_is_set_off_from_the_table_by_a_blank_line():
    """상세 패널이 표 마지막 행에 딱 붙어 패널 첫 줄이 표의 다음 행처럼 읽혔다.

    주입: `layout` 이 `gap` 을 늘 0 으로 내면 첫 단언이 실패한다.
    """
    for h in (24, 30, 50):
        assert layout(h).gap == 1, f"h={h} 에서 표와 패널이 붙어 있다"
    # 드릴다운에는 패널이 없다 — 띄울 것이 없으면 빈 줄도 없다.
    assert layout(50, drill=True).gap == 0, "패널도 없는데 빈 줄만 남았다"


def test_the_blank_line_is_the_first_thing_given_up_when_the_screen_is_short():
    """여백은 **가장 먼저 포기하는** 것이다 — 폭에서 문구를 단계적으로 줄이는
    (`tier_for`) 것과 같은 규율을 높이에도 쓴다.

    주입: `layout` 이 자리를 안 보고 `gap=1` 을 박으면, 낮은 화면에서 패널이
    잘리거나(detail 감소) 표가 비어(rows<1) 실패한다.
    """
    tall = layout(24)
    for h in range(10, 24):
        lay = layout(h)
        assert lay.rows >= 1, f"h={h} 에서 표가 한 줄도 안 남았다"
        assert lay.detail >= 3, f"h={h} 에서 여백이 패널 줄을 잡아먹었다"
        assert lay.head + 1 + lay.rows + lay.gap + lay.detail == lay.hint_y, \
            f"h={h} 세로 배치가 안 맞는다: {lay}"
        # 여백을 지키려고 맨 위 헤더를 버리지는 않는다 — 날짜·구간 줄이 먼저다.
        assert lay.head == tall.head or lay.gap == 0, \
            f"h={h} 에서 빈 줄 하나 때문에 헤더가 사라졌다: {lay}"
    # 낮아질수록 여백은 도로 없어진다(먼저 포기한다는 뜻).
    assert layout(11).gap == 0 and layout(10).gap == 0

# --------------------------------------------------------------- 자금 원장
#
# `kq-ledger` 는 `kq-flow` 와 같은 curses 골격 위에 있으므로, flow 가 밟은 지뢰를
# 그대로 밟는다. 아래 셋은 flow 쪽 회귀(CPU 좀비 · ESC 종료 · 깨진 리포트)를
# **같은 잣대로** 원장에도 태운다 — 한쪽만 고치면 다른 쪽이 조용히 남는다.

LEDGER_MINIMAL = {
    "dates": [f"2026-01-{d:02d}" for d in range(1, 26)],
    "sectors": ["전기/전자", "부동산"],
    "markets": ["거래소", "코스닥"],
    "flows": {m: {s: {k: [float((i + j) % 7 - 3) for i in range(25)]
                      for j, k in enumerate(("indiv", "forgn", "inst", "etc", "tv"))}
                  for s in ("전기/전자", "부동산")}
              for m in ("거래소", "코스닥")},
    "cap": {m: {"전기/전자": 10000.0, "부동산": 300.0} for m in ("거래소", "코스닥")},
    "n_by_sector": {m: {"전기/전자": 200, "부동산": 1} for m in ("거래소", "코스닥")},
    "n_names": 202, "finalized": True,
}


def _ledger_dir(d: str) -> str:
    with open(os.path.join(d, "payload.json"), "w", encoding="utf-8") as f:
        json.dump(LEDGER_MINIMAL, f, ensure_ascii=False)
    return d


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_ledger_app_starts_and_quits_in_a_pty():
    """원장 앱이 네 화면을 다 지나 q 로 정상 종료하는가.

    **256색 초기화**와 히트맵 ``chgat`` 은 렌더 테스트가 못 건드리는 자리다 —
    ``curses.init_pair`` 가 색쌍 한계를 넘으면 여기서만 죽는다.
    """
    with tempfile.TemporaryDirectory() as d:
        rc, screen, err = _run_tui(_ledger_dir(d), b"vvvv" b"wmasd" b"\x1b[B\x1b[B" b"q",
                                   cols="120", app="ledger_app")
        assert rc == 0, err[-1500:]
        assert not err.strip(), f"stderr 에 뭔가 나왔다:\n{err[-1500:]}"
        assert "전기/전자" in screen, "표가 안 그려졌다"
        assert "미관측" in screen, "한계 배너가 화면에 없다"


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
@pytest.mark.parametrize("lines,cols", [("10", "40"), ("24", "80"), ("50", "200")])
def test_ledger_survives_small_and_large_terminals(lines, cols):
    """좁은 터미널에서도 죽지 않는가. 폭 40 은 원장이 표를 안 그리는 구간이다."""
    with tempfile.TemporaryDirectory() as d:
        # `?` 뒤의 첫 q 는 **도움말 닫기**다(종료 아님) — 그래서 q 가 둘이다.
        rc, _screen, err = _run_tui(_ledger_dir(d), b"vvvvwmasd?qq",
                                    lines=lines, cols=cols, app="ledger_app")
        assert rc == 0, err[-1500:]


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_ledger_lone_escape_does_not_quit():
    """회귀 — ESC 는 종료가 아니다.

    느린 SSH 에서 방향키는 ESC 와 나머지로 쪼개져 도착한다(ESCDELAY 기본 1초).
    ESC 가 종료면 ↓ 를 눌렀을 뿐인데 앱이 끝난다. flow_app 이 79e14df 에서
    같은 이유로 뺐고, 원장도 처음엔 `ch in (ord("q"), 27)` 이었다.
    """
    with tempfile.TemporaryDirectory() as d:
        # ESC 로 끝나버리면 그 뒤의 `vvv` 가 안 닿아 한계 화면 문구가 안 나온다.
        rc, screen, err = _run_tui(_ledger_dir(d), b"\x1b\x1b\x1b" b"vvv" b"q",
                                   cols="120", app="ledger_app")
        assert rc == 0, err[-1500:]
        assert "돈에 꼬리표가 없다" in screen, (
            "ESC 뒤에 앱이 이미 죽어서 한계 화면이 안 떴다")


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_ledger_exits_when_the_terminal_disappears():
    """회귀 — SSH 가 끊기면 원장도 나가야 한다.

    getch() 가 ERR(-1) 을 즉시 돌려주는데 아무 분기에도 안 태우면 draw→getch→-1
    로 100% CPU 를 태우며 영원히 돈다. flow_app 과 **같은 골격이라 같은 버그**다.
    """
    with tempfile.TemporaryDirectory() as d:
        primary, proc = _spawn(_ledger_dir(d), "ledger_app", "24", "100",
                               stderr=subprocess.DEVNULL)
        # 배수 스레드를 쓰지 않는다 — 읽는 중에 같은 fd 를 닫으면 경합이라
        # 앱이 아니라 테스트 쪽 사정으로 프로세스가 끝날 수 있다.
        time.sleep(0.5)                        # 한 번은 그리게 둔다
        os.close(primary)                      # 터미널이 사라진다
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            ticks = "?"
            try:
                st = open(f"/proc/{proc.pid}/stat").read().split()
                ticks = int(st[13]) + int(st[14])
            except OSError:
                pass
            proc.kill()
            proc.communicate()
            pytest.fail(f"터미널이 사라졌는데 원장이 계속 돈다 — CPU 좀비 ({ticks} 틱)")


def _payload_without(key: str) -> str:
    """키 하나가 빠진 페이로드. **문 앞 검사가 대괄호 접근과 같은 목록인가**를 본다."""
    return json.dumps({k: v for k, v in LEDGER_MINIMAL.items() if k != key},
                      ensure_ascii=False)


def _payload_with_orphan_market() -> str:
    d = json.loads(json.dumps(LEDGER_MINIMAL, ensure_ascii=False))
    d["markets"] = list(d["markets"]) + ["없는시장"]
    return json.dumps(d, ensure_ascii=False)


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
@pytest.mark.parametrize("payload,expect", [
    ('{"dates": []}', "flows"),        # 스키마가 다름
    ("{nope}", "JSON"),                # JSON 이 깨짐
    # 아래 셋은 예전엔 **생 트레이스백**이었다. 문 앞 검사가 네 키만 봤는데
    # `cap`·`n_by_sector` 와 `flows[market]` 은 뒤에서 대괄호로 집힌다.
    (_payload_without("cap"), "cap"),
    (_payload_without("n_by_sector"), "n_by_sector"),
    (_payload_with_orphan_market(), "없는시장"),
])
def test_ledger_broken_payload_says_what_is_wrong(payload, expect):
    """회귀 — 형식이 바뀐 페이로드가 curses 안에서 생 트레이스백으로 터지면 안 된다.

    주입: `load()` 의 필수 키 목록에서 `cap` 을 빼면 그 파라미터가 트레이스백을
    내며 실패한다(예전 상태다).
    """
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "payload.json"), "w", encoding="utf-8") as f:
            f.write(payload)
        proc = subprocess.run(
            [sys.executable, "-c",
             "from kr_quant.tui.ledger_app import main; main()", "--dir", d],
            capture_output=True, text=True)
        assert proc.returncode != 0
        msg = proc.stdout + proc.stderr
        assert expect in msg, msg[-800:]
        assert "Traceback" not in msg, f"생 트레이스백이 나왔다:\n{msg[-800:]}"


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_ledger_dump_is_pipeable():
    """--dump 는 터미널 없이 돌아야 한다 — SSH 밖으로 내보내는 유일한 경로다."""
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(
            [sys.executable, "-c",
             "from kr_quant.tui.ledger_app import main; main()",
             "--dir", _ledger_dir(d), "--dump"],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr[-1500:]
        assert "### 원장" in r.stdout and "### 한계" in r.stdout
        assert "미관측" in r.stdout
        assert "~" in r.stdout, "얇은 섹터 표시가 평문에서 사라졌다"


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_ledger_shows_the_keys_at_the_default_ssh_width():
    """회귀 — 폭 80(SSH 기본)에서 하단 줄에 **키가 하나도 없었다.**

    `_draw` 가 `(" " + BANNER + "   " + FOOTER)[:w]` 로 붙였는데 그 문자열은
    표시폭 137 이라 배너만 남았다. 도움말이 아무리 좋아도 `?` 가 화면에 없으면
    닿을 수 없다. `--dump` 는 하단 두 줄을 아예 안 찍으므로 이 부류는
    **진짜 터미널을 봐야** 잡힌다.
    """
    with tempfile.TemporaryDirectory() as d:
        rc, screen, err = _run_tui(_ledger_dir(d), b"q", cols="80",
                                   app="ledger_app")
        assert rc == 0, err[-1500:]
        assert "미관측" in screen, "폭 80 에서 경고가 사라졌다"
        assert "도움말" in screen and "종료" in screen, (
            "폭 80 하단에 키가 없다 — ? 를 알 길이 없다")


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_ledger_question_mark_shows_keys_and_columns():
    """`?` 는 **키와 열의 뜻**이다(예전엔 한계 화면이었다). 그리고 거기서 q 는
    닫기지 종료가 아니라, 종료하려면 q 를 한 번 더 눌러야 한다."""
    with tempfile.TemporaryDirectory() as d:
        # 스페이스 두 번은 한 화면씩 아래로 — 열 사전이 그 아래에 있다.
        rc, screen, err = _run_tui(_ledger_dir(d), b"?  qq", cols="100",
                                   app="ledger_app")
        assert rc == 0, err[-1500:]
        assert "키와 열의 뜻" in screen, "? 가 도움말을 안 띄웠다"
        assert "── 키 ──" in screen, "키 목록이 없다"
        assert "오차가 아니라" in screen, "스크롤해도 열 사전이 안 나온다"
        assert "닫기" in screen
