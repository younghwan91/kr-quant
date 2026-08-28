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


@pytest.mark.skipif(sys.platform == "win32", reason="pty 는 POSIX 전용")
def test_app_starts_and_quits_in_a_pty():
    """앱이 의사 터미널에서 뜨고 q 로 정상 종료하는가."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "numbers.html"), "w", encoding="utf-8") as f:
            f.write("<script>\nconst D = " + json.dumps(MINIMAL, ensure_ascii=False)
                    + ";\n</script>")
        primary, replica = pty.openpty()
        env = {**os.environ, "TERM": "xterm-256color", "LINES": "24", "COLUMNS": "100"}
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "from kr_quant.tui.flow_app import main; main()", "--dir", d],
            stdin=replica, stdout=replica, stderr=subprocess.PIPE, env=env)
        os.close(replica)
        try:
            os.write(primary, b"wmas\x1b[B\n\x1b")   # 컨트롤 순회 + Enter + Esc
            os.write(primary, b"q")
            out, err = proc.communicate(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("TUI 가 q 에 종료하지 않았다")
        finally:
            os.close(primary)
        assert proc.returncode == 0, err.decode("utf-8", "replace")[-1500:]
