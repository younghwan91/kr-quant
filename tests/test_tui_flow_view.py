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


from kr_quant.tui.flow_view import cell_width


def _w(text: str) -> int:
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


def test_color_spans_use_display_cells_not_char_index():
    """색 구간은 **표시 칸**이어야 한다 — curses addstr 좌표가 표시 칸이기 때문.

    한글이 섞인 줄에서 문자 인덱스를 쓰면 색이 엉뚱한 칸에 칠해진다.
    이 저장소는 같은 함정(문자 인덱스 vs 표시 칸)을 이미 두 번 밟았다.
    """
    from kr_quant.tui.flow_view import cell_width, color_spans

    line = "건설   +1,234  -5.67%p"          # 한글 2자 = 4칸
    spans = color_spans(line)
    roles = [r for _, _, r in spans]
    assert roles == ["up", "down"], roles
    # 첫 구간은 '+' 가 있는 표시 칸에서 시작해야 한다
    start, width, _ = spans[0]
    prefix_cells = sum(cell_width(c) for c in line[:line.index("+")])
    assert start == prefix_cells
    assert width == sum(cell_width(c) for c in "+1,234")


def test_color_spans_marks_pass_flag():
    from kr_quant.tui.flow_view import color_spans
    spans = color_spans("IT 서비스   0.62 *   +100")
    assert any(r == "mark" for _, _, r in spans)


def test_color_spans_on_real_rows(data):
    """실제 렌더 결과에 색 구간이 붙고, 줄 밖으로 안 넘어간다."""
    from kr_quant.tui.flow_view import color_spans
    st = State(data)
    lines, _thin, _nh = table_lines(st, 100, 20)
    for line in lines:
        for start, width, role in color_spans(line):
            assert role in ("up", "down", "mark")
            assert 0 <= start < 100
            assert start + width <= 100


def test_help_covers_every_rendered_column(data):
    """도움말이 실제로 렌더되는 모든 열을 설명하는가.

    열을 추가하면서 도움말을 안 고치면 설명 없는 열이 생긴다 — 그 부류를 막는다.
    """
    from kr_quant.tui.flow_view import HELP, table_lines

    documented = {name for name, _ in HELP if name}
    st = State(data)
    missing = set()
    for wi in range(len(WINDOWS)):
        st.wi = wi
        lines, _thin, _nh = table_lines(st, 132, 20)
        for h in lines[0].split():
            if h and h not in documented and h not in {"섹터", "통과", "일"} \
               and not h.endswith("일"):
                missing.add(h)
    assert not missing, f"도움말에 없는 열: {sorted(missing)}"


def test_help_labels_align_in_display_cells():
    """회귀 — 도움말 라벨이 표시 칸으로 정렬돼야 한다.

    f"{name:>9}" 같은 문자 폭 서식은 한글 라벨(임펄스=6칸)과 ASCII(G=1칸)를
    다른 칸에서 끝내 설명문 시작 위치가 행마다 어긋난다.
    """
    from kr_quant.tui.flow_view import HELP, help_lines

    lines, _ = help_lines(120, 0, 10**6)
    starts = set()
    for (name, desc), line in zip(HELP, lines[1:]):
        if not name or not desc:
            continue
        idx = line.find(desc[:6])
        assert idx > 0, f"설명문을 못 찾았다: {name}"
        starts.add(sum(cell_width(c) for c in line[:idx]))
    assert len(starts) == 1, f"설명문 시작 칸이 섞였다: {sorted(starts)}"


def test_help_lines_fit_width_and_scroll():
    from kr_quant.tui.flow_view import help_lines
    for width in (80, 120):
        lines, total = help_lines(width, 0, 20)
        assert total > 10
        for line in lines:
            assert _w(line) == width
        # 스크롤 끝에서도 안 깨진다
        tail, _ = help_lines(width, total - 2, 20)
        assert all(_w(x) == width for x in tail)


def test_table_and_detail_show_the_same_top_names(data):
    """회귀 — 표의 '순매수 상위' 열과 하단 패널이 같은 목록이어야 한다.

    처음엔 열 이름이 '견인주'였고 하단은 '기관 순매수 상위'라, 같은 값인데 다른
    지표처럼 읽혔다("견인주는 시총 기준이냐"는 질문이 실제로 나왔다).
    """
    st = State(data)
    st.row = 0
    lines, _thin, nhead = table_lines(st, 132, 20)
    row_line = lines[nhead + st.row]
    detail = detail_lines(st, 132)[1]          # '순매수 상위' 줄
    top = (st.rows()[st.row].get("top") or {}).get("buy") or []
    for t in top[:2]:
        assert t["name"] in row_line, f"표에 {t['name']} 가 없다"
        assert t["name"] in detail, f"하단에 {t['name']} 가 없다"


def test_detail_panel_shows_the_value_it_sorts_by(data):
    """회귀 — 금액 순으로 정렬하면서 %p 를 표시하면 순서가 뒤집혀 보인다.

    실제로 '현대건설 +1.34%p' 가 '대우건설 +1.40%p' 위에 오는 화면이 나왔다.
    정렬 기준과 표시값이 같아야 읽는 사람이 순서를 의심하지 않는다.
    """
    st = State(data)
    st.row = 0
    line = detail_lines(st, 132)[1]
    top = (st.rows()[st.row].get("top") or {}).get("buy") or []
    shown = [x for x in top[:3]]
    if len(shown) < 2:
        pytest.skip("픽스처에 상위 종목이 둘 미만")
    vals = []
    for t in shown:
        idx = line.find(t["name"])
        assert idx >= 0
        vals.append((idx, t["inst"]))
    vals.sort()                       # 화면에 나온 순서
    amounts = [v for _, v in vals]
    assert amounts == sorted(amounts, reverse=True), (
        f"화면 순서와 금액 순서가 다르다: {amounts}")


def test_sort_highlight_points_at_the_right_header(data):
    """회귀 — 정렬 하이라이트가 실제 그 열의 헤더 칸을 가리켜야 한다.

    렌더와 하이라이트가 열 정의를 따로 갖고 있으면 조용히 어긋난다.
    두 경로가 같은 table_cols() 를 쓰는지 결과로 검사한다.
    """
    from kr_quant.tui.flow_view import SORTS, SORT_COL, sort_span

    st = State(data)
    for width in (80, 120):
        for si, (key, _label) in enumerate(SORTS):
            st.si = si
            st.wi = 1                       # 종합이 아닌 창
            span = sort_span(st, width)
            header_line = table_lines(st, width, 10)[0][0]
            want = SORT_COL[key]
            if span is None:
                continue
            start, w = span
            got, cell = "", 0
            for ch in header_line:
                if start <= cell < start + w:
                    got += ch
                cell += cell_width(ch)
            assert want in got, f"정렬[{key}] 하이라이트가 '{got.strip()}' 를 가리킨다"


def test_name_sort_highlight_matches_columns():
    from kr_quant.tui.flow_view import NAME_SORTS, NAME_SORT_COL, name_sort_span, names_cols

    class _S:
        pass
    for key, _ in NAME_SORTS:
        st = _S()
        st.name_sort = key
        span = name_sort_span(st)
        assert span, f"{key} 스팬 없음"
        start, w = span
        cell = 0
        for name, cw, _r in names_cols():
            if cell == start:
                assert name == NAME_SORT_COL[key], f"{key} → {name}"
                assert cw == w
                break
            cell += cw + 1
        else:
            pytest.fail(f"{key} 스팬이 어떤 열과도 안 맞는다")
