"""TUI 렌더 — 표시 폭 불변식.

이 저장소는 "표 헤더와 셀 개수가 어긋나 열이 한 칸씩 밀리는" 실수를 두 번 했다.
터미널에서는 한글 2칸·기호 2칸이 섞여 그 실수가 더 쉽다. 그래서 렌더를 curses 에서
떼어 순수 함수로 두고, **모든 행의 표시 폭이 같은지**를 검사한다.
"""

from __future__ import annotations

import pytest

from kr_quant.tui.flow_view import (
    ACTORS, SORTS, WINDOWS, State, detail_lines, fmt_amt, fmt_pct,
    header_lines, pad, table_lines,
)


def _w(text: str) -> int:
    from kr_quant.tui.flow_view import cell_width
    return sum(cell_width(c) for c in text)


@pytest.fixture
def data():
    def row(sec, n, g, thin=False, pass_=False):
        return {"sector": sec, "n_all": n, "thin": thin, "G": g, "G_pass": pass_,
                "inst": 1234.5, "forgn": -20.0, "indiv": -1000.0, "etc": -200.0,
                "accel": 1.23, "ret": -4.56, "x": 7.8, "U": 90.1, "P": 0.12,
                "xdot": 0.3, "xddot": 0.04, "a_idx": 1.23, "cap_idx": 100000.0,
                "top": {"buy": [{"code": "005930", "name": "삼성전자",
                                 "inst": 100.0, "a": 0.5}],
                        "sell": [{"code": "000660", "name": "SK하이닉스",
                                  "inst": -50.0, "a": -0.3}]}}
    rows = [row("전기/전자", 387, 0.68), row("건설", 56, 0.84, pass_=True),
            row("부동산", 3, None, thin=True), row("IT 서비스", 254, 0.62, pass_=True)]
    blocks = {}
    for w in ("5", "20", "60", "120"):
        for m in ("전체", "거래소", "코스닥"):
            blocks[f"{w}|{m}"] = {"from": "2026-01-01", "to": "2026-08-28",
                                  "k": 1.5, "b": 0.1, "t": 2.0, "rows": rows}
    comb = {"windows": [20, 60, 120],
            "rows": [{**r, "per": {"20": 0.5, "60": 0.6, "120": 0.7},
                      "pass_n": 2, "seen": 3} for r in rows]}
    return {"asof": "2026-08-28", "finalized": True, "dates": ["2026-01-01", "2026-08-28"],
            "blocks": blocks, "combined": {m: comb for m in ("전체", "거래소", "코스닥")}}


def test_pad_counts_hangul_as_two_cells():
    assert _w(pad("삼성", 10)) == 10
    assert _w(pad("abc", 10)) == 10
    assert _w(pad("가나다라마바사", 6)) == 6      # 넘치면 자른다


def test_every_table_row_has_identical_display_width(data):
    """회귀 — 열이 한 칸이라도 밀리면 잡는다."""
    st = State(data)
    for width in (80, 100, 132):
        for wi in range(len(WINDOWS)):
            st.wi = wi
            for mi in range(len(st.markets)):
                st.mi = mi
                lines, thin, nhead = table_lines(st, width, 20)
                widths = {_w(x) for x in lines}
                assert widths == {width}, (
                    f"창={WINDOWS[wi]} 시장={st.markets[mi]} 폭={width} "
                    f"→ 행 폭이 섞였다: {sorted(widths)}")
                assert len(thin) == len(lines)


def test_pass_marker_survives_rendering(data):
    """회귀 — 통과 마커가 잘려 사라지면 안 된다.

    처음엔 마커를 G 값에 붙여 렌더했다(`"0.84" + "●"`). ● 는 표시 폭 2 라 열을
    넘겼고 pad 가 **잘라내서** 마커가 사라졌다. 행 전체 폭은 그대로라 폭 검사로는
    안 잡힌다 — 내용을 봐야 잡힌다. (이 테스트 없이 폭 검사만 두면 초록인 채로
    마커가 안 보인다. 실제로 그렇게 한 번 넘어갔다.)
    """
    st = State(data)
    lines, _thin, nhead = table_lines(st, 100, 20)
    body = lines[nhead:]
    by_sector = {}
    for line in body:
        for r in st.rows():
            if line.lstrip().startswith(r["sector"]):
                by_sector[r["sector"]] = line
                break
    passing = [r["sector"] for r in st.rows() if r.get("G_pass")]
    failing = [r["sector"] for r in st.rows()
               if r.get("G") is not None and not r.get("G_pass")]
    assert passing and failing, "픽스처에 통과·미통과가 둘 다 있어야 검사가 성립한다"
    for sec in passing:
        assert "*" in by_sector[sec], f"{sec} 는 통과인데 마커가 없다"
    for sec in failing:
        assert "*" not in by_sector[sec], f"{sec} 는 미통과인데 마커가 있다"


def test_numeric_columns_align_across_rows(data):
    """회귀 — 같은 열의 숫자가 행마다 다른 칸에서 시작하면 안 된다.

    마커를 값에 붙이면 소수점 위치가 행마다 어긋난다. 행 폭은 같으므로 폭 검사로는
    안 잡히고, 화면에서는 명확히 어긋나 보인다.
    """
    def dot_cells(line: str) -> set[int]:
        """소수점의 **표시 칸** 위치. 문자 인덱스로 재면 안 된다 — 한글은 1자가
        2칸이라 섹터명 길이에 따라 인덱스가 달라진다(이 테스트가 처음에 그렇게
        틀렸다). 화면에서 어긋나는지는 표시 칸으로만 판정된다."""
        from kr_quant.tui.flow_view import cell_width
        out, cell = set(), 0
        for c in line:
            if c == ".":
                out.add(cell)
            cell += cell_width(c)
        return out

    st = State(data)
    lines, _thin, nhead = table_lines(st, 100, 20)
    body = [x for x in lines[nhead:] if "." in x]
    cols = [dot_cells(line) for line in body]
    common = set.intersection(*cols) if cols else set()
    assert common, f"모든 행이 공유하는 소수점 칸이 없다 — 열이 어긋났다: {cols[:3]}"


def test_all_control_combinations_render(data):
    st = State(data)
    n = 0
    for wi in range(len(WINDOWS)):
        for mi in range(len(st.markets)):
            for ai in range(len(ACTORS)):
                for si in range(len(SORTS)):
                    st.wi, st.mi, st.ai, st.si = wi, mi, ai, si
                    header_lines(st, 80)
                    table_lines(st, 80, 20)
                    detail_lines(st, 80)
                    n += 1
    assert n == len(WINDOWS) * len(st.markets) * len(ACTORS) * len(SORTS)


def test_header_and_detail_are_exact_width(data):
    st = State(data)
    for width in (80, 120):
        for line in header_lines(st, width) + detail_lines(st, width):
            assert _w(line) == width


def test_empty_data_does_not_crash():
    st = State({"asof": "2026-08-28", "finalized": False, "dates": ["a", "b"],
                "blocks": {}, "combined": {}})
    assert table_lines(st, 80, 10)[0]
    assert detail_lines(st, 80)


def test_formatters():
    assert fmt_amt(1234.5) == "+1,234"
    assert fmt_amt(-1234.5).startswith("-")
    assert fmt_amt(None) == "—"
    assert fmt_pct(1.234) == "+1.23"
    assert fmt_pct(None) == "—"


def test_minus_sign_is_one_cell():
    """회귀 — 음수 부호가 2칸으로 세어지면 음수 행만 한 칸씩 밀린다.

    U+2212(−) 는 ord 가 0x1100 보다 커서 어림 규칙에서 2칸으로 잡혔다.
    TUI 는 ASCII '-' 를 쓰고, 폭 계산은 East Asian Width 표준을 따른다.
    """
    from kr_quant.tui.flow_view import cell_width
    assert cell_width("-") == 1
    assert cell_width("\u2212") == 1      # − 도 1칸이어야 한다
    assert cell_width("가") == 2
    assert _w(fmt_amt(-1234.5)) == _w(fmt_amt(1234.5))
    assert _w(fmt_pct(-1.23)) == _w(fmt_pct(1.23))
