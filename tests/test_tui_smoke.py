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


def _run_tui(d: str, keys: bytes, lines: str = "24", cols: str = "100"):
    """의사 터미널에서 앱을 띄우고 (종료코드, 화면, stderr) 를 돌려준다."""
    primary, replica = pty.openpty()
    env = {**os.environ, "TERM": "xterm-256color", "LINES": lines, "COLUMNS": cols}
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "from kr_quant.tui.flow_app import main; main()", "--dir", d],
        stdin=replica, stdout=replica, stderr=subprocess.PIPE, env=env)
    os.close(replica)
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
