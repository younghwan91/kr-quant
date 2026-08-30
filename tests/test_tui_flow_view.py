"""TUI 렌더 — 표시 폭 불변식.

이 저장소는 "표 헤더와 셀 개수가 어긋나 열이 한 칸씩 밀리는" 실수를 두 번 했다.
터미널에서는 한글 2칸·기호 2칸이 섞여 그 실수가 더 쉽다. 그래서 렌더를 curses 에서
떼어 순수 함수로 두고, **모든 행의 표시 폭이 같은지**를 검사한다.
"""

from __future__ import annotations

import pytest

from kr_quant.tui.flow_view import (
    NAME_SORTS,
    ACTORS, SORTS, WINDOWS, State, detail_lines, fmt_amt, fmt_pct,
    header_lines, pad, table_lines,
)


from kr_quant.tui.flow_view import (  # noqa: E402  대체안이 쓰는 이름
    SORT_COL, _fit, cell_width, sort_span, table_cols,
)


def _w(text: str) -> int:
    return sum(cell_width(c) for c in text)


@pytest.fixture
def data():
    def row(sec, n, g, thin=False, pass_=False):
        return {"sector": sec, "n_all": n, "thin": thin, "G": g, "G_pass": pass_,
                "inst": 1234.5, "forgn": -20.0, "indiv": -1000.0, "etc": -200.0,
                "cap": 100000.0,
                "accel": 1.23, "ret": -4.56, "x": 7.8, "U": 90.1, "P": 0.12,
                "xdot": 0.3, "xddot": 0.04, "a_idx": 1.23, "cap_idx": 100000.0,
                # 1년 백분위·구간 모양은 **주체마다 다르다** — 값이 같으면
                # "선택된 주체를 따르는가" 검사가 헛돈다.
                "pct1y": {"inst": 96.0, "forgn": 12.0, "indiv": 50.0, "etc": 3.0},
                "spark": {"inst": [1.0, 2.0, -3.0, 0.0, 5.0, -1.0, 2.0, 8.0],
                          "forgn": [-8.0, -2.0, 3.0, 0.0, -5.0, 1.0, -2.0, -1.0],
                          "indiv": [0.0] * 8,
                          "etc": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
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
                                  "k": 1.5, "b": 0.1, "t": 2.0,
                                  # 블록마다 **자기 행**을 갖는다. 12개 블록이 한
                                  # 리스트를 공유하면 "이 창의 값을 읽는가" 를
                                  # 검사할 수 없다(한 창을 바꾸면 전부 바뀐다).
                                  "rows": [dict(r) for r in rows]}
    comb = {"windows": [20, 60, 120],
            "rows": [{**r, "per": {"20": 0.5, "60": 0.6, "120": 0.7},
                      "pass_n": 2, "seen": 3} for r in rows]}
    # 견인주·종목 목록은 `names` 에서 나온다 — 주체를 바꾸면 같이 바뀌어야 하므로
    # 픽스처도 실제 페이로드처럼 4주체를 모두 싣는다.
    #: 창별 배수 — 창마다 값이 **달라야** `최근집중`(5일 ÷ 이 창) 검사가 힘을 갖는다.
    #: 창이 전부 같은 값이던 예전 픽스처에서는 그 비율이 늘 100% 라, 분자와 분모를
    #: 바꿔 넣어도 검사가 통과했다. 20일은 기존 값 그대로 둔다 — 다른 검사들이
    #: 기본 창(20일)을 보고 있어서 여기를 흔들면 무관한 검사가 같이 흔들린다.
    scale = {"5": 0.4, "20": 1.0, "60": 2.5, "120": 3.0}

    def nm(code, name, sec, inst, forgn, ret=0.0):
        return {"name": name, "sector": sec, "market": "거래소", "cap": 50000.0,
                "win": {w: {"inst": inst * s, "forgn": forgn * s, "indiv": -inst * s,
                            "etc": 0.0, "tv": 900.0 * s,
                            # 기관 세부 — 기관 총액이 같아도 이 둘이 채운 것과
                            # 금투가 채운 것은 다른 이야기다.
                            "invtrt": inst * s * 0.6, "penfnd_etc": inst * s * 0.25,
                            "ret": ret * s}
                        for w, s in scale.items()}}
    # 전기/전자는 종목을 넉넉히 둔다 — 파는 쪽이 사는 쪽보다 크고, 누적 80% 를
    # 넘기는 행이 여럿이라야 "절댓값 정렬"·"가로줄 하나" 검사가 힘을 갖는다.
    names = {"005930": nm("005930", "삼성전자", "전기/전자", 100.0, -30.0, 3.1),
             "000660": nm("000660", "SK하이닉스", "전기/전자", -50.0, 80.0, -7.2),
             "066570": nm("066570", "LG전자", "전기/전자", 30.0, -12.0, 0.4),
             "009150": nm("009150", "삼성전기", "전기/전자", -15.0, 25.0, -1.9),
             "000990": nm("000990", "DB하이텍", "전기/전자", 5.0, -3.0, 12.5),
             "000720": nm("000720", "현대건설", "건설", 40.0, -70.0, 2.2),
             "006360": nm("006360", "GS건설", "건설", -20.0, 60.0, -4.4),
             "035420": nm("035420", "NAVER", "IT 서비스", 70.0, -10.0, 8.8),
             "035720": nm("035720", "카카오", "IT 서비스", -60.0, 20.0, -0.6),
             "034020": nm("034020", "두산에너빌리티", "부동산", 10.0, -5.0, 5.0)}
    return {"asof": "2026-08-28", "finalized": True, "dates": ["2026-01-01", "2026-08-28"],
            "names": names,
            "blocks": blocks, "combined": {m: comb for m in ("전체", "거래소", "코스닥")}}


#: **바뀐 열 라벨 대응표** — 새 이름 → 예전 이름(없으면 새로 생긴 열).
#:
#: `HELP` 는 이 작업의 소관이 아니라(다른 에이전트가 쥐고 있다) 열 이름이 바뀐 만큼
#: 도움말이 잠시 뒤처진다. 머지하면서 HELP 를 고치고 **이 표를 비운다**. 표에
#: 남아 있는데 화면에 안 그려지는 열이 있으면 아래 검사가 잡는다.
HELP_PENDING: set[str] = set()   # 새 열은 HELP 에 문구가 들어갈 때까지만 여기 둔다


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

    import re as _re

    # HELP 는 이 작업에서 **손대지 않는다**(다른 에이전트 소관). 열 이름이 바뀐
    # 만큼만 여기 적어 두고, 머지할 때 HELP 를 고치면서 이 표를 비운다.
    # 아래 "안 쓰이는 열" 검사가 있어 표가 조용히 썩지는 않는다.
    documented = {name for name, _ in HELP if name} | set(HELP_PENDING)
    st = State(data)
    missing = set()
    for wi in range(len(WINDOWS)):
        st.wi = wi
        for width in (80, 120, 160):
            lines, _thin, _nh = table_lines(st, width, 20)
            for h in lines[0].split():
                if not h or h in documented:
                    continue
                if _re.fullmatch(r"\d+일\[G\]", h):     # 종합의 창별 G — "N일G" 로 설명
                    continue
                missing.add(h)
    assert not missing, f"도움말에 없는 열: {sorted(missing)}"
    # 대응표가 썩지 않게 — 이미 안 그려지는 열이 남아 있으면 지워야 한다.
    rendered = set()
    for wi in range(len(WINDOWS)):
        st.wi = wi
        for width in (80, 120, 160, 200):
            rendered |= set(table_lines(st, width, 20)[0][0].split())
    stale = set(HELP_PENDING) - rendered
    assert not stale, f"HELP_PENDING 에 안 그려지는 열이 남았다: {sorted(stale)}"


def test_headers_are_not_truncated_by_their_column_width(data):
    """회귀 — 헤더가 열 폭에 안 들어가면 조용히 잘린다.

    실제로 "종목수" 가 폭 5칸에서 "종목" 으로 잘렸고, 도움말 대조 테스트가
    엉뚱하게 실패해서야 드러났다. 잘린 헤더는 뜻이 바뀐다.
    """
    from kr_quant.tui.flow_view import names_cols, table_cols

    st = State(data)
    bad = []
    for wi in range(len(WINDOWS)):
        st.wi = wi
        for width in (80, 120, 160):
            for c in table_cols(st, width):
                if c.header and _w(c.header) > c.width:
                    bad.append(f"{c.header}({_w(c.header)}칸) > 열폭 {c.width}")
    for c in names_cols():
        if c.header and _w(c.header) > c.width:
            bad.append(f"[종목목록] {c.header}({_w(c.header)}칸) > 열폭 {c.width}")
    assert not bad, "헤더가 잘린다: " + "; ".join(sorted(set(bad)))


def test_help_labels_align_in_display_cells():
    """회귀 — 도움말 설명문이 **모두 같은 칸에서** 시작해야 한다.

    f"{name:>9}" 같은 문자 폭 서식은 한글 라벨(임펄스=6칸)과 ASCII(G=1칸)를
    다른 칸에서 끝내 설명문 시작 위치가 행마다 어긋난다.

    긴 라벨(폭 10 초과)은 자기 줄을 쓰고 설명이 다음 줄로 가므로, 여기서는
    **한 줄에 라벨과 설명이 같이 있는** 항목만 본다. 긴 라벨이 온전한지는
    `test_help_labels_are_never_truncated` 가 따로 본다.
    """
    from kr_quant.tui.flow_view import HELP, cell_len, help_lines

    lines = [ln.rstrip() for ln in help_lines(120, 0, 10 ** 6)[0]]
    starts, checked = set(), 0
    for name, desc in HELP:
        if not name or not desc or cell_len(name) > 10:
            continue
        want = desc.replace("**", "")
        hit = [ln for ln in lines if ln.lstrip().startswith(name) and want[:6] in ln]
        assert hit, f"설명문을 못 찾았다: {name}"
        idx = hit[0].find(want[:6])
        starts.add(sum(cell_width(c) for c in hit[0][:idx]))
        checked += 1
    assert checked > 10, f"실제로 확인한 항목이 너무 적다({checked}) — 검사가 헛돈다"
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


def test_help_body_is_not_truncated_at_the_default_ssh_width():
    """회귀 — 도움말 5줄이 폭 80 에서 잘려 있었다.

    도움말은 **읽으러 연 화면**이다. 거기서 설명이 문장 중간에 끊기면
    (`…분포에서 몇 등인` · `…0 으로 준` 처럼) 고칠 곳이 화면에 안 보인다.

    ⚠️ 이 부류를 `help_lines` 의 출력으로는 못 잡는다 — `pad` 가 넘치면 자르고
    모자라면 채워서 **모든 줄이 정확히 width 칸**이 되기 때문이다. 옆에 있는
    `test_help_lines_fit_width_and_scroll` 이 `_w(line) == width` 를 보는데,
    그건 `pad` 가 일을 했다는 뜻일 뿐 글이 온전하다는 뜻이 아니다. 그래서
    여기서는 **자르기 전 원문**을 렌더와 같은 방식으로 조립해 잰다.
    """
    from kr_quant.tui.flow_view import HELP

    over = []
    for name, desc in HELP:
        body = desc.replace("**", "")           # 렌더가 떼는 강조 표기
        raw = (pad(name, 10, right=True) + "  " + body) if name else ("   " + body)
        if _w(raw) > 80:
            over.append((_w(raw), raw))
    assert not over, "폭 80 에서 잘리는 도움말 줄: " + repr(over)


def test_help_section_headers_are_named_by_the_view_not_guessed_by_the_app():
    """구역 제목(``── 키 ──``)을 색칠하는 쪽이 문자열을 다시 뜯으면 안 된다.

    주입: `is_section` 이 늘 False 를 내면 첫 단언이, 아무 줄에나 True 를 내면
    둘째 단언이 실패한다.
    """
    from kr_quant.tui.flow_view import HELP, help_lines, is_section

    heads = [d for n, d in HELP if not n and d.strip().startswith("──")]
    assert heads, "도움말에 구역 제목이 없다 — 검사가 헛돈다"
    for d in heads:
        assert is_section(d), f"구역 제목을 못 알아본다: {d!r}"
    body = help_lines(120, 0, 10 ** 6)[0]
    marked = [ln for ln in body if is_section(ln)]
    assert len(marked) == len(heads), f"구역 제목이 아닌 줄까지 골랐다: {marked}"


def test_footer_shows_one_key_per_action_and_help_keeps_the_reverse_ones():
    """푸터는 **소문자 한 벌**만 적고, 대문자 역방향은 도움말이 진다.

    `w/W m/M a/A s/S` 를 다 적으니 푸터가 길어져 정작 무슨 키가 있는지가 안
    읽혔다. 그렇다고 그냥 지우면 이 저장소가 이미 버그로 친 상태 — "기능이
    화면 어디에도 안 적혀 있다" — 로 돌아간다. 그래서 **옮겼는지**를 본다.

    주입: 푸터 단계에 `w/W` 를 되살리면 앞 절반이, 도움말에서 대문자 설명을
    빼면 뒷 절반이 실패한다.
    """
    from kr_quant.tui.flow_view import (
        FOOTER_DRILL_TIERS, FOOTER_TIERS, HELP, help_desc)

    for ts in (FOOTER_TIERS, FOOTER_DRILL_TIERS):
        for t in ts:
            # `g/G` 는 예외다 — `G`(끝)는 `g`(처음)의 역방향이 아니라 별개
            # 동작이라, 안 적으면 목록 끝으로 가는 길이 화면에서 사라진다.
            for pair in ("w/W", "m/M", "a/A", "s/S", "Enter/l", "h/←"):
                assert pair not in t, f"푸터에 대문자·동의 키 병기가 남았다: {t}"
    assert any("g/G" in t for t in FOOTER_TIERS), "목록 끝으로 가는 G 가 푸터에서 사라졌다"
    # 그 대신 도움말에는 남아 있어야 한다 — 키 자체는 그대로 듣는다.
    for name in ("w W", "m M", "a A", "s S"):
        assert "역방향" in help_desc(HELP, name), f"도움말이 {name} 의 역방향을 안 적는다"
    keys = " ".join(n for n, _d in HELP)
    for token in ("Enter", "l", "h", "Esc", "G", "End", "PgUp"):
        assert token in keys + " " + " ".join(d for _n, d in HELP), \
            f"도움말에 {token!r} 가 없다"


def test_width_tiers_all_go_through_one_helper():
    """회귀 — 폭 단계 고르기가 네 벌 각자 구현이었고 이미 갈라져 있었다.

    푸터는 못 맞으면 마지막 단계로 내려갔는데 도움말 제목은 `next()` 를 기본값
    없이 써서 **폭 6 이하에서 StopIteration 으로 TUI 를 통째로 죽였다.**
    """
    from kr_quant.tui.flow_app import HINT_COMBINED_TIERS
    from kr_quant.tui.flow_view import (
        FOOTER_DRILL_TIERS, FOOTER_TIERS, HELP_TITLE_TIERS, footer_line,
        help_lines, tier_for,
    )
    from kr_quant.tui.ledger_view import BANNER_TIERS, TIMELINE_NOTE_TIERS, banner_for

    tiers = (FOOTER_TIERS, FOOTER_DRILL_TIERS, HELP_TITLE_TIERS,
             HINT_COMBINED_TIERS, BANNER_TIERS, TIMELINE_NOTE_TIERS)
    for ts in tiers:
        # 넓은 것부터여야 `tier_for` 가 "가장 자세한 것" 을 고른다.
        assert [_w(t) for t in ts] == sorted((_w(t) for t in ts), reverse=True), ts
        # 하나도 안 맞으면 마지막(가장 짧은 것). 예외로 죽지 않는다.
        assert tier_for(ts, 0) == ts[-1]
        for width in range(1, 200):
            assert _w(tier_for(ts, width)) <= max(width, _w(ts[-1]))

    # 폭이 아무리 좁아도 죽지 않는다 — 예전엔 폭 1·5·6 이 StopIteration 이었다.
    for width in (1, 5, 6, 7, 40, 200):
        help_lines(width, 0, 5)
        footer_line(width)
        banner_for(width)


def test_table_and_detail_show_the_same_top_names(data):
    """회귀 — 표의 '순매수 상위' 열과 하단 패널이 같은 목록이어야 한다.

    처음엔 열 이름이 '견인주'였고 하단은 '기관 순매수 상위'라, 같은 값인데 다른
    지표처럼 읽혔다("견인주는 시총 기준이냐"는 질문이 실제로 나왔다).
    """
    st = State(data)
    st.row = 0
    # 폭이 모자라면 상위종목 열은 (잘리지 않고) 통째로 빠지므로 넉넉한 폭에서 본다.
    width = 170
    lines, _thin, nhead = table_lines(st, width, 20)
    row_line = lines[nhead + st.row]
    # 줄 번호로 집으면 패널에 줄이 하나 늘 때 무관한 이유로 깨진다(실제로 깨졌다).
    detail = next(ln for ln in detail_lines(st, width) if "순매수 상위" in ln)
    top = (st.rows()[st.row].get("top") or {}).get("buy") or []
    # 표는 1위만, 하단 패널은 상위 3을 보여준다 — 1위는 반드시 일치해야 한다.
    assert top, "상위 종목이 비었다"
    assert top[0]["name"] in row_line, f"표에 {top[0]['name']} 가 없다"
    for t in top[:3]:
        assert t["name"] in detail, f"하단에 {t['name']} 가 없다"


def test_detail_panel_shows_the_value_it_sorts_by(data):
    """회귀 — 금액 순으로 정렬하면서 %p 를 표시하면 순서가 뒤집혀 보인다.

    실제로 '현대건설 +1.34%p' 가 '대우건설 +1.40%p' 위에 오는 화면이 나왔다.
    정렬 기준과 표시값이 같아야 읽는 사람이 순서를 의심하지 않는다.
    """
    st = State(data)
    st.row = 0
    line = next(ln for ln in detail_lines(st, 132) if "순매수 상위" in ln)
    top = (st.rows()[st.row].get("top") or {}).get("buy") or []
    shown = [x for x in top[:3]]
    if len(shown) < 2:
        pytest.skip("픽스처에 상위 종목이 둘 미만")
    vals = []
    for t in shown:
        idx = line.find(t["name"])
        assert idx >= 0
        vals.append((idx, t["flow"]))
    vals.sort()                       # 화면에 나온 순서
    amounts = [v for _, v in vals]
    assert amounts == sorted(amounts, reverse=True), (
        f"화면 순서와 금액 순서가 다르다: {amounts}")


#: 표에 열이 없는 정렬키 — 하이라이트가 없는 게 정상이다.
#: 새 열을 붙이면서 SORT_COL 배선을 잊으면 이 목록이 막는다.
#: 표에 열이 없는 정렬키 — 하이라이트가 없는 게 정상이다. **이 목록이 곧 명세다.**
#: 새 정렬키를 넣고 열을 안 만들면 여기 적어야 하고, 적지 않으면 검사가 잡는다.
NO_COLUMN_SORTS = {"tv", "cap_idx"}


def test_every_sort_key_names_a_column_that_exists(data):
    """회귀 — `SORT_COL` 의 값이 **실재하는 열 이름**인가.

    헤더 이름을 한 글자만 바꿔도 조회가 끊긴다. 그러면 하이라이트가 사라질 뿐
    열은 제자리라 폭 검사도, 헤더-셀 검사도 전부 초록이다. 아래 폭별 검사만으로는
    "그 폭에 열이 없다" 와 구분이 안 되므로, **배선 자체**를 따로 못 박는다.
    """
    st = State(data)
    seen = set()
    for width in (80, 100, 132, 150, 170, 200):
        for wi in range(len(WINDOWS)):
            st.wi = wi
            seen |= {c.header for c in table_cols(st, width)}
    for key, header in SORT_COL.items():
        assert header in seen, (
            f"SORT_COL[{key!r}]={header!r} 라는 열이 어느 폭에도 없다 — "
            f"열 이름을 바꾸면서 배선을 안 고쳤나")
    for key, _label in SORTS:
        assert key in SORT_COL or key in NO_COLUMN_SORTS, (
            f"정렬키 {key!r} 이 SORT_COL 에도 NO_COLUMN_SORTS 에도 없다")


def test_sort_highlight_points_at_the_right_header(data):
    st = State(data)
    checked = 0
    for width in (80, 100, 132, 150, 170):
        for wi, w in enumerate(WINDOWS):
            if w == "종합":
                continue                      # 종합은 정렬 자체가 없다
            st.wi = wi
            for si, (key, label) in enumerate(SORTS):
                st.si = si
                # 폴백을 반영한 **실제** 정렬키를 본다 — 요청한 키가 아니다.
                eff, _fell = st.effective_sort(st.rows())
                cols = _fit(table_cols(st, width), width)
                header = SORT_COL.get(eff)
                span = sort_span(st, width)
                head_line = table_lines(st, width, 10)[0][0]

                if header is None:
                    assert eff in NO_COLUMN_SORTS, (
                        f"정렬키 {eff!r}({label}) 가 SORT_COL 에 없다 — "
                        f"열을 붙이면서 배선을 잊었나")
                    assert span is None, f"열이 없는 {eff!r} 를 하이라이트한다"
                    continue
                on_screen = any(c.header == header for c in cols)
                if not on_screen:
                    assert span is None, (
                        f"폭{width}: 화면에 없는 열 {header!r} 를 하이라이트한다")
                    continue
                # 여기부터는 **반드시** 하이라이트가 있어야 한다.
                assert span is not None, (
                    f"폭{width}·창{w}: {header!r} 이 화면에 있는데 하이라이트가 "
                    f"없다 — SORT_COL 배선이 끊겼나")
                right = next(c.right for c in cols if c.header == header)
                assert _slice(head_line, *span) == pad(header, span[1], right), (
                    f"폭{width}·창{w}: 하이라이트가 {header!r} 칸을 안 가리킨다")
                checked += 1
    assert checked > 50, f"실제로 확인한 조합이 {checked}개뿐 — 검사가 헛돈다"

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
        for c in names_cols():
            if cell == start:
                assert c.header == NAME_SORT_COL[key], f"{key} → {c.header}"
                assert c.width == w
                break
            cell += c.width + 1
        else:
            pytest.fail(f"{key} 스팬이 어떤 열과도 안 맞는다")


def test_lead_name_and_amount_align_in_fixed_cells(data):
    """회귀 — 순매수/순매도 상위의 금액이 줄마다 같은 칸에서 끝나야 한다.

    이름 길이가 제각각이라 '이름 + 금액' 을 그냥 이어붙이면 금액이 들쭉날쭉해진다.
    """
    from kr_quant.tui.flow_view import col_span, color_spans, table_cols

    st = State(data)
    width = 170
    lines, _thin, nhead = table_lines(st, width, 20)
    # 매직넘버 대신 **열 정의에서** 상위종목 열의 구간을 구한다 — 폭이 바뀌면
    # 위치도 바뀌므로 하드코딩하면 무관한 이유로 깨진다(실제로 깨졌다).
    cols = table_cols(st, width)
    spans = [col_span(cols, c.header) for c in cols if c.header.startswith("순매")]
    assert spans and all(spans), "상위종목 열을 못 찾았다"
    ends = set()
    for line in lines[nhead:]:
        if not line.strip():
            continue
        for start, w2, role in color_spans(line):
            if role in ("up", "down") and any(
                    s[0] <= start and start + w2 <= s[0] + s[1] for s in spans):
                ends.add(start + w2)
    assert ends, "상위 종목 금액을 못 찾았다"
    assert len(ends) <= len(spans), f"금액 끝 칸이 흩어졌다: {sorted(ends)}"


def _slice(line: str, start: int, width: int) -> str:
    """표시 칸 기준 슬라이스 — 한글이 2칸이라 문자 인덱스로는 못 자른다."""
    out, cell = "", 0
    for ch in line:
        if start <= cell < start + width:
            out += ch
        cell += cell_width(ch)
    return out


def test_data_cells_sit_under_their_headers(data):
    """회귀 — 헤더 열과 그 아래 데이터 셀이 **같은 칸**에 있는가.

    행 폭 검사는 이 부류를 구조적으로 못 잡는다: 렌더가 마지막에 줄 전체를
    화면 폭으로 패딩하므로, 셀이 통째로 빠지거나 폭이 하나 어긋나도 모든
    행의 표시 폭은 여전히 정확히 width 다. 열이 밀린 채로 초록이 된다.

    `.strip()` 하면 안 된다 — 값이 셀보다 짧을 때 1칸 어긋남이 여백에
    먹혀서 폭 변경(헤더 12 · 셀 13)을 놓친다.
    """
    from kr_quant.tui.flow_view import _fit, col_span, table_cols

    st = State(data)
    for width in (80, 100, 132, 150, 170):
        for wi, w in enumerate(WINDOWS):
            if w == "종합":
                continue
            st.wi = wi
            # 렌더가 보는 것과 **같은** 열 목록이어야 한다 — 폭에 안 들어가
            # 통째로 빠진 열까지 기대하면 검사가 엉뚱한 이유로 실패한다.
            cols = _fit(table_cols(st, width), width)
            lines, _thin, nhead = table_lines(st, width, 30)
            for r, line in zip(st.rows(), lines[nhead:]):
                # 기대값은 렌더 함수를 부르지 않고 **여기서 다시 적는다** —
                # Col.fn 을 그대로 부르면 동어반복이라 아무것도 못 잡는다.
                pct = r.get("pct")
                want = {
                    "섹터": r.get("sector", "—"),
                    "종목[수]": str(r.get("n_all", "—")),
                    "임펄스[억]": fmt_amt(r.get("flow")),
                    "가속[%p]": fmt_pct(r.get("accel")),
                    "수익률[%]": fmt_pct(r.get("ret")),
                    "1년[%ile]": "—" if pct is None else f"{pct:.0f}",
                    "선정": ("—" if r.get("G") is None else f"{r['G']:.2f}"),
                }
                for hdr, val in want.items():
                    span = col_span(cols, hdr)
                    if span is None:          # 이 폭에선 안 보이는 열
                        continue
                    right = next(c.right for c in cols if c.header == hdr)
                    assert _slice(line, *span) == pad(val, span[1], right), (
                        f"폭{width}·창{w} 의 '{hdr}' 열이 헤더와 어긋났다")


def test_stock_list_cells_sit_under_their_headers(data):
    """회귀 — 종목 목록도 같은 검사. 이 화면은 오래 무검증이었다.

    픽스처에 `names` 가 없어 `State.names()` 가 늘 빈 목록을 돌려줬고,
    그래서 셀을 통째로 지워도·정렬을 뒤집어도 전부 초록이었다.
    """
    from kr_quant.tui.flow_view import (
        COL_INVTRT, COL_PENFND, _fit, col_span, fmt_conc, names_cols, names_lines)

    st = State(data)
    cols = names_cols()
    for width in (80, 100, 120, 170):
        # 그 폭에 **온전히 들어간 열만** 본다. 열은 경계에서 통째로 빠지므로
        # (`_fit`), 안 들어간 열의 칸을 읽으면 빈 문자열과 비교하게 된다.
        shown = {c.header for c in _fit(cols, width)}
        for nsi in range(len(NAME_SORTS)):
            st.nsi = nsi
            names = st.names()
            assert names, "픽스처에 종목이 없다 — 이 검사가 헛돈다"
            lines, nhead = names_lines(st, width)
            for t, line in zip(names, lines[nhead:]):
                a = t.get("a")
                want = {
                    "종목": t.get("name", "—"),
                    "코드": t.get("code", ""),
                    "순매수[억]": fmt_amt(t.get("flow")),
                    "시총대비[%p]": fmt_pct(a) if a is not None else "—",
                    "참여율[%]": fmt_pct(t.get("part"), 1),
                    "누적[%]": ("—" if t.get("cum") is None
                                else f"{t['cum']:.0f}"),
                    COL_INVTRT: fmt_amt(t.get("invtrt")),
                    COL_PENFND: fmt_amt(t.get("penfnd")),
                    "상대수익[%p]": fmt_pct(t.get("rrel"), 1),
                    "최근집중[%]": fmt_conc(t.get("conc")),
                }
                for hdr, val in want.items():
                    if hdr not in shown:
                        continue
                    span = col_span(cols, hdr)
                    right = next(c.right for c in cols if c.header == hdr)
                    assert _slice(line, *span) == pad(val, span[1], right), (
                        f"폭{width}·정렬{nsi} 의 '{hdr}' 열이 헤더와 어긋났다")


def test_stock_list_is_actually_sorted_by_the_chosen_key(data):
    """회귀 — 종목 정렬 5종이 실제로 그 키로 줄세우는가.

    정렬 방향을 뒤집어도 어떤 검사도 실패하지 않았다.
    """
    st = State(data)
    for nsi, (key, label) in enumerate(NAME_SORTS):
        st.nsi = nsi
        vals = [t.get(key) for t in st.names()]
        got = [v for v in vals if v is not None]
        if key == "name":
            want = sorted(got)
        elif key == "flow":
            # 섹터를 움직인 건 산 쪽과 판 쪽 **둘 다**라 절댓값 순이다.
            got = [abs(v) for v in got]
            want = sorted(got, reverse=True)
        else:
            want = sorted(got, reverse=True)
        assert got == want, f"정렬[{label}] 이 {key} 순이 아니다: {got}"


def test_numbers_are_never_cut_mid_value_in_narrow_terminals(data):
    """회귀 — 좁은 폭에서 숫자가 자릿수 중간에서 잘리면 **틀린 값**이 된다.

    실측된 증상: 40칸에서 `-1,360` 이 `-1` 로 보였다. 줄을 통째로 자르는 대신
    열 경계에서 떨어뜨려야 한다. 안 보이는 것보다 틀리게 보이는 게 나쁘다.
    """
    from kr_quant.tui.flow_view import col_span, table_cols

    st = State(data)
    for width in range(20, 90, 3):
        cols = table_cols(st, width)
        lines, _thin, nhead = table_lines(st, width, 20)
        for r, line in zip(st.rows(), lines[nhead:]):
            for hdr, val in (("임펄스[억]", fmt_amt(r.get("flow"))),
                             ("가속[%p]", fmt_pct(r.get("accel")))):
                span = col_span(cols, hdr)
                if span is None:
                    continue                      # 이 폭에선 정의되지 않은 열
                # 열이 폭 안에 온전히 들어가면 값이 정확해야 하고, 안 들어가면
                # **아무것도 안 보여야** 한다. 반쪽 숫자는 다른 값이 된다.
                # (여기서 `continue` 로 넘기면 검사가 바로 그 경우를 놓친다.)
                got = _slice(line, span[0], min(span[1], max(width - span[0], 0)))
                if span[0] + span[1] <= width:
                    assert got.strip() == val.strip(), (
                        f"폭{width} 의 '{hdr}' 이 {val!r} 대신 {got!r}")
                else:
                    assert not got.strip(), (
                        f"폭{width} 에서 '{hdr}' 이 {got!r} 로 잘려 보인다 "
                        f"(참값 {val!r}) — 열째로 떨어뜨려야 한다")


def test_thin_sector_warning_survives_without_color(data):
    """회귀 — 얇은 섹터(종목 10개 미만) 경고가 색에만 실려 있으면
    무색 터미널·색맹에서 통째로 사라진다. 글자로도 남아야 한다."""
    st = State(data)
    lines, thin, nhead = table_lines(st, 170, 20)
    marked = [ln for ln, t in zip(lines[nhead:], thin[1:]) if t]
    assert marked, "픽스처에 얇은 섹터가 없다 — 이 검사가 헛돈다"
    for ln in marked:
        assert "~" in ln, f"얇은 섹터인데 글자 표시가 없다: {ln.strip()!r}"


def test_actor_toggle_changes_every_derived_cell_not_just_impulse(data):
    """회귀 — 주체를 바꾸면 그 행의 **파생값이 전부** 따라와야 한다.

    예전엔 임펄스 열만 주체를 따르고 가속·견인주·종목목록은 기관 값이 남아
    한 행 안에 두 주체의 숫자가 섞였다. "외국인이 3,299억 팔았는데 가속 +1.44"
    같은 줄이 나왔다.

    기대값을 `st.rows()` 에서 가져오면 안 된다 — 그건 이미 투영된 행이라
    무엇을 넣어도 자기 자신과 같아 **동어반복**이 된다(실제로 그랬다).
    **원본 블록**에서 직접 계산해 비교한다.
    """
    st = State(data)
    raw = {r["sector"]: r for r in data["blocks"][f"{st.window}|{st.market}"]["rows"]}
    seen = {}
    for ai, (actor, _label) in enumerate(ACTORS):
        st.ai = ai
        for r in st.rows():
            src = raw[r["sector"]]
            assert r["flow"] == src[actor], (
                f"주체[{actor}] 인데 임펄스가 {r['flow']} — 원본 {src[actor]}")
            want = src[actor] / src["cap"] * 100 if src.get("cap") else None
            assert r["accel"] == pytest.approx(want), (
                f"주체[{actor}] 인데 가속이 주체를 안 따른다")
        seen[actor] = [r["flow"] for r in st.rows()]
    assert len({tuple(v) for v in seen.values()}) > 1, (
        "주체를 바꿔도 임펄스가 하나도 안 변한다 — 픽스처가 이 검사를 헛돌게 한다")


def test_sorting_by_impulse_is_monotone_for_every_actor(data):
    """회귀 — 정렬 하이라이트가 가리키는 열은 실제로 정렬돼 있어야 한다.

    정렬키가 문자열 "inst" 로 박혀 있어, 외국인 화면에서 화면은 '임펄스 열로
    줄세웠다' 고 말하면서 그 열의 숫자가 뒤죽박죽이었다.
    """
    st = State(data)
    st.si = next(i for i, (k, _l) in enumerate(SORTS) if k == "flow")
    for ai in range(len(ACTORS)):
        st.ai = ai
        vals = [r["flow"] for r in st.rows() if r.get("flow") is not None]
        assert vals == sorted(vals, reverse=True), (
            f"주체[{ACTORS[ai][0]}]·정렬[임펄스] 인데 내림차순이 아니다: {vals}")


def test_sort_falls_back_when_the_key_is_missing_for_the_whole_block(data):
    """회귀 — 정렬키가 그 블록에서 전멸하면 정렬이 **무음 실패**했다.

    5일 창은 G 가 27/27 결측이라 전 행이 동률이 되어 페이로드 원순서로
    남는데, 헤더는 그 열을 하이라이트하며 '이걸로 줄세웠다' 고 말했다.
    """
    # G 를 통째로 지운 블록을 만든다 — 실데이터의 5일 창과 같은 모양.
    blocks = {k: {**v, "rows": [{**r, "G": None} for r in v["rows"]]}
              for k, v in data["blocks"].items()}
    st = State({**data, "blocks": blocks})
    st.si = next(i for i, (k, _l) in enumerate(SORTS) if k == "G")

    key, fell = st.effective_sort(st.rows())
    assert fell, "전멸한 키인데 폴백하지 않았다"
    assert key != "G"

    vals = [r.get(key) for r in st.rows() if r.get(key) is not None]
    assert vals == sorted(vals, reverse=True), f"폴백 키로도 정렬이 안 됐다: {vals}"
    assert "값이 없다" in header_lines(st, 170)[1], "헤더가 폴백을 밝히지 않는다"

    # 값이 있는 창에서는 폴백하지 않아야 한다(과잉 폴백 방지).
    st2 = State(data)
    st2.si = st.si
    assert not st2.effective_sort(st2.rows())[1], "멀쩡한 키인데 폴백했다"

# ── 열 정의가 하나뿐이라는 것 ────────────────────────────────────────────

def test_render_walks_the_column_definition_and_nothing_else(data):
    """회귀 — 렌더가 열 정의 밖에서 셀을 만들어내면 안 된다.

    예전엔 `table_cols()` 와 `table_lines()` 가 폭 상수와 폭 임계값을 각각
    적었고 분기 모양까지 서로 달랐다. 결과가 같았던 건 우연이다. 이제 셀 수는
    열 수와 **항상** 같아야 한다 — 이 등식이 깨지면 헤더와 값이 어긋난다.
    """
    from kr_quant.tui.flow_view import _fit, table_cols

    st = State(data)
    for wi in range(len(WINDOWS)):
        st.wi = wi
        for width in (60, 80, 100, 132, 150, 200):
            cols = _fit(table_cols(st, width), width)
            need = sum(c.width for c in cols) + len(cols) - 1
            lines, _thin, nhead = table_lines(st, width, 30)
            for line in lines:
                # 열 정의가 쓴 만큼까지가 내용이고 그 뒤는 여백이어야 한다.
                assert not _slice(line, need, max(width - need, 0)).strip(), (
                    f"폭{width}: 열 정의({need}칸) 밖에 내용이 있다: {line!r}")


def test_no_width_threshold_lives_outside_the_column_list():
    """회귀 — 폭 임계값이 렌더 쪽에 다시 나타나면 이중화가 되살아난 것이다."""
    import inspect

    from kr_quant.tui import flow_view

    src = inspect.getsource(flow_view.table_lines) + inspect.getsource(
        flow_view.names_lines) + inspect.getsource(flow_view._render)
    for bad in (">= 100", ">= 132", ">= 150", ">=100", ">=132", ">=150"):
        assert bad not in src, f"렌더에 폭 임계값이 되살아났다: {bad}"


# ── 스파크라인·발산 막대 ─────────────────────────────────────────────────

def test_spark_and_bar_glyphs_are_never_ambiguous_width():
    """⚠️ 폭이 '애매' 한 글자를 표에 넣지 않는다.

    블록 문자 ``▁▂▃▄▅▆▇█`` 는 East Asian Width 가 'A'(Ambiguous) 다. `cell_width`
    는 1칸으로 세지만 한글 로케일 터미널은 2칸으로 그릴 수 있고, 그러면 그 행만
    8칸씩 밀린다. 브라유(U+28xx)는 'N' 이라 그런 여지가 없다.

    이 검사는 그림 글자에만 건다 — 기존 UI 는 이미 '·'·'—'·'²'(전부 'A')를
    쓰고 있어서 전면 금지는 이 작업 범위를 넘는다(보고서에 적었다).

    (미실현 발산 막대는 숫자로 되돌렸다 — 숫자가 10칸을 덜 쓰고 정확하다.
    그래서 여기 검사 대상은 스파크라인 글자만 남았다.)
    """
    import unicodedata

    from kr_quant.tui.flow_view import SPARK, SPARK_EMPTY

    for ch in list(SPARK) + [SPARK_EMPTY]:
        eaw = unicodedata.east_asian_width(ch)
        assert eaw not in ("A", "W", "F"), f"{ch!r} 의 폭이 {eaw} 다"
        assert cell_width(ch) == 1


def test_spark_is_exactly_its_column_width_and_recent_is_rightmost():
    from kr_quant.tui.flow_view import spark

    from kr_quant.tui.flow_view import SPARK, SPARK_EMPTY

    assert _w(spark([1, 2, 3, 4, 5, 6, 7, 8])) == 8
    assert _w(spark([1, 2, 3])) == 8              # 조각이 적어도 폭은 같다
    # 조각이 모자라면 **왼쪽**이 빈다 — 오른쪽 끝이 구간 끝이라는 약속.
    assert spark([1, 2, 3]).startswith(SPARK_EMPTY * 5)
    # 빈 자리와 값 0 은 다른 글자다 — 예전엔 둘 다 빈칸이라 구분이 안 됐다.
    assert SPARK_EMPTY not in spark([0] * 8)

    # 누적이 오르면 오른쪽이 높고, 내리면 오른쪽이 낮다 — 기울기가 곧 답이다.
    up, down = spark([5] * 8), spark([-5] * 8)
    assert SPARK.index(up[-1]) > SPARK.index(up[0]), f"유입인데 안 오른다: {up}"
    assert SPARK.index(down[-1]) < SPARK.index(down[0]), f"유출인데 안 내린다: {down}"

    # 같은 임펄스라도 **언제** 들어왔는지가 모양으로 갈린다.
    early, late = spark([4, 4, 0, 0, 0, 0, 0, 0]), spark([0, 0, 0, 0, 0, 0, 4, 4])
    assert early != late, "앞에서 들어온 것과 지금 들어오는 것이 같아 보인다"
    assert sum(SPARK.index(c) for c in early) > sum(SPARK.index(c) for c in late)

    assert _w(spark([])) == 8                     # 값이 없어도 폭은 같다


def test_year_percentile_and_spark_follow_the_selected_actor(data):
    """회귀 — 오늘 고친 버그가 정확히 '한 행 안에 두 주체의 숫자가 섞이는' 것이었다.

    임펄스만 주체를 따르고 새 열이 기관 값에 머물면, 외국인 화면의 한 행이
    두 주체를 동시에 말한다. 픽스처는 주체마다 다른 값을 싣는다.
    """
    from kr_quant.tui.flow_view import col_span, table_cols

    st = State(data)
    st.wi = WINDOWS.index("20")
    width = 200
    cols = table_cols(st, width)
    raw = data["blocks"]["20|전체"]["rows"][0]
    span = col_span(cols, "1년[%ile]")
    seen = {}
    for ai, (key, _label) in enumerate(ACTORS):
        st.ai = ai
        r = st.rows()[0]
        assert r["pct"] == r["pct1y"][key], f"{key} 의 백분위가 안 따라온다"
        assert r["spark"] == raw["spark"][key], f"{key} 의 스파크라인이 안 따라온다"
        line = table_lines(st, width, 20)[0][1]
        seen[key] = _slice(line, *span).strip()
    assert seen["inst"] == "96" and seen["forgn"] == "12", seen
    assert len(set(seen.values())) == len(ACTORS), f"주체별로 안 갈린다: {seen}"


def test_spark_column_changes_with_the_actor(data):
    from kr_quant.tui.flow_view import col_span, table_cols

    st = State(data)
    st.wi = WINDOWS.index("20")
    width = 200
    span = col_span(table_cols(st, width), "추이[8]")
    got = set()
    for ai in range(len(ACTORS)):
        st.ai = ai
        got.add(_slice(table_lines(st, width, 20)[0][1], *span))
    assert len(got) >= 3, f"주체를 바꿔도 스파크라인이 그대로다: {got}"


# ── 정보 설계 ────────────────────────────────────────────────────────────

def test_default_sort_is_acceleration_not_the_unvalidated_score(data):
    """G 는 세 순위의 평균인 **검증 안 된 탐색 점수**다. 화면 순서를 지배할 근거가
    없다 — 기본은 규모 정규화된 가속이다."""
    assert SORTS[0][0] == "accel", SORTS[0]
    st = State(data)
    assert st.sort_key == "accel"


def test_conclusion_columns_come_after_their_inputs(data):
    """회귀 — 결론(G)이 자기 입력(가속·미실현·풀림)보다 왼쪽에 있으면 안 된다."""
    from kr_quant.tui.flow_view import table_cols

    st = State(data)
    st.wi = WINDOWS.index("20")
    order = [c.header for c in table_cols(st, 200)]
    assert len(order) == len(set(order)), f"헤더가 중복이다: {order}"
    pos = {}
    for i, h in enumerate(order):
        pos.setdefault(h, i)          # 중복이면 col_span 이 가리키는 **앞의 것**
    for inp in ("가속[%p]", "미실현[%p]", "풀림[%p/일²]"):
        assert pos[inp] < pos["선정"], f"{inp} 가 G 보다 오른쪽이다: {order}"
    # 종목수는 판단 변수가 아니라 데이터 품질 주석이다 — 앞자리를 차지하면 안 된다.
    assert pos["종목[수]"] > pos["수익률[%]"], order


def test_the_screen_at_eighty_columns_shows_force_before_conclusion(data):
    """좁은 터미널에서 **무엇이 살아남는가** 가 곧 정보 설계다."""
    st = State(data)
    st.wi = WINDOWS.index("20")
    head = table_lines(st, 80, 20)[0][0]
    for must in ("섹터", "가속[%p]", "임펄스[억]", "1년[%ile]", "추이[8]", "수익률[%]"):
        assert must in head, f"80칸에 {must} 가 없다: {head!r}"


# ── 종목 목록 — 누적 기여율 ─────────────────────────────────────────────

def test_cumulative_contribution_reaches_100_and_marks_the_cut(data):
    st = State(data)
    st.nsi = 0                              # 순매수(절댓값) 정렬
    names = st.names()
    cums = [t["cum"] for t in names]
    assert cums == sorted(cums), f"누적이 단조증가가 아니다: {cums}"
    assert abs(cums[-1] - 100.0) < 1e-6, cums
    cuts = [t for t in names if t["cut"]]
    assert len(cuts) == 1, f"80% 가로줄이 {len(cuts)}개다"
    assert cuts[0]["cum"] >= 80.0
    # 가로줄 **바로 위** 행은 아직 80% 를 못 넘겨야 한다
    i = names.index(cuts[0])
    if i:
        assert names[i - 1]["cum"] < 80.0


def test_cumulative_is_blank_where_it_would_be_meaningless(data):
    """기여도는 |금액| 으로만 정의된다. 종목명 순으로 누적하면 단조증가라서
    뜻이 있어 보이지만 아무 말도 아니다 — 그런 칸은 비운다."""
    st = State(data)
    st.nsi = [k for k, _ in NAME_SORTS].index("name")
    assert all(t["cum"] is None for t in st.names())
    from kr_quant.tui.flow_view import col_span, names_cols, names_lines
    lines, nhead = names_lines(st, 170)
    span = col_span(names_cols(), "누적[%]")
    for line in lines[nhead:]:
        assert _slice(line, *span).strip() == "—"


def test_stock_rows_stay_one_to_one_with_the_name_list(data):
    """회귀 — 80% 가로줄을 별도 행으로 끼워 넣으면 화면 선택(`drow`)이 그 아래
    전 종목에서 한 칸씩 어긋난다. flow_app 은 본문 i 번째 줄 = names()[i] 로 읽는다.
    """
    st = State(data)
    for nsi in range(len(NAME_SORTS)):
        st.nsi = nsi
        names = st.names()
        from kr_quant.tui.flow_view import names_lines
        lines, nhead = names_lines(st, 170)
        assert len(lines) - nhead == len(names), (
            f"정렬{nsi}: 본문 {len(lines) - nhead}줄 vs 종목 {len(names)}개")
        for t, line in zip(names, lines[nhead:]):
            assert t["name"] in line


def test_participation_rate_is_flow_over_turnover(data):
    st = State(data)
    for t in st.names():
        if t.get("tv"):
            assert abs(t["part"] - t["flow"] / t["tv"] * 100) < 1e-9


# ── 1년 백분위·구간 조각을 만드는 쪽(scripts/sector_numbers.py) ──────────
#
# TUI 는 이 두 값을 그리기만 한다. 값이 틀리면 화면은 아무 불평 없이 틀린 숫자를
# 예쁘게 그린다 — 그래서 만드는 쪽을 여기서 같이 잠근다.

def _producer():
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sector_numbers.py"
    spec = importlib.util.spec_from_file_location("_sector_numbers", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rolling_percentile_is_a_rank_not_a_z_score():
    """롤링 창은 겹쳐서 자기상관이 크고 260일에 비겹침 20일 창은 13개뿐이다.
    z 는 그 표본에서 정밀도를 과장한다 — 백분위는 '몇 등' 만 말한다."""
    m = _producer()
    up = list(range(260))                      # 최근이 가장 큼
    assert m.rolling_pct(up, 20) == 100.0
    assert m.rolling_pct(up[::-1], 20) == 0.0


def test_rolling_percentile_does_not_call_a_silent_sector_a_record():
    """회귀 — 동률을 `<=` 로 세면 **0 만 있는 조용한 섹터가 전부 100 백분위**가 된다.

    실제 페이로드에는 그런 섹터가 있다(출판/매체복제: 20일 임펄스 +0억).
    화면에서 '1년 최대치' 라고 외치는데 실은 아무 일도 없었던 것이다.
    """
    m = _producer()
    assert m.rolling_pct([0.0] * 260, 20) == 50.0
    # 한 번만 튀고 다시 잠잠해진 경우도 현재값은 중간이어야 한다
    ser = [0.0] * 260
    ser[100] = 1e9
    assert 0.0 < m.rolling_pct(ser, 20) < 100.0


def test_rolling_percentile_refuses_windows_longer_than_the_history():
    m = _producer()
    assert m.rolling_pct([1.0] * 10, 20) is None


def test_spark_segments_sum_back_to_the_impulse():
    """스파크라인의 맨 오른쪽 칸이 임펄스 열의 그 숫자로 이어진다는 약속은,
    조각이 구간을 **빠짐없이 겹치지 않게** 덮을 때만 성립한다."""
    m = _producer()
    ser = [float(i) for i in range(100)]
    for i0, i1 in ((80, 99), (0, 99), (95, 99), (40, 44)):
        segs = m.spark_segs(ser, i0, i1)
        assert abs(sum(segs) - sum(ser[i0:i1 + 1])) < 1e-9, (i0, i1)
        assert len(segs) == min(8, i1 - i0 + 1)
    # 마지막 조각은 **가장 최근** 날들이다
    segs = m.spark_segs(ser, 80, 99)
    assert abs(segs[-1] - sum(ser[97:100])) < 1e-9 or segs[-1] > segs[0]


def test_help_screen_shows_no_markdown_asterisks():
    """회귀 — 소스의 **강조** 는 읽는 사람 눈에 띄라고 쓴 표기지 화면에
    나갈 글자가 아니다. 힌트바는 떼는데 모달은 안 떼서 `**닫기만**` 이
    그대로 보였다 — 같은 일을 하는 두 경로 중 하나에만 처리가 있었다."""
    from kr_quant.tui.flow_view import HELP, help_lines

    assert any("**" in d for _n, d in HELP), "소스에 강조가 없다 — 검사가 헛돈다"
    lines, _ = help_lines(120, 0, 10**6)
    # 홑별표는 진짜다 — 표의 통과 마커(*)를 설명하는 글자다. 겹별표만 본다.
    bad = [ln.strip() for ln in lines if "**" in ln]
    assert not bad, f"도움말에 마크다운 강조가 그대로 나온다: {bad[:3]}"


def test_help_title_fits_every_width():
    """회귀 — 제목이 85칸이라 80칸(SSH 기본)에서 단어 중간에 잘렸다.
    푸터에 만든 단계 기법을 이 줄에도 쓴다."""
    from kr_quant.tui.flow_view import help_lines

    for width in range(20, 130, 2):
        title = help_lines(width, 0, 5)[0][0].rstrip()
        assert sum(cell_width(c) for c in title) <= width
        assert "q" in title, f"폭{width} 에서 닫는 키가 사라졌다: {title!r}"


def test_hint_bar_fits_the_width_it_is_given(data):
    """회귀 — 힌트바가 head 만 폭을 보고 설명 문장은 안 봐서, 폭 40 에서
    78칸짜리 문장이 나와 `pad` 가 단어 중간에서 잘랐다. 푸터에는 단계를
    만들어 놓고 정작 새로 만든 힌트바에는 안 쓴 셈이었다.

    그 검사는 늘 기본값 width=200 으로만 불러서 좁은 폭을 한 번도 안 봤다.

    그리고 이 검사도 같은 구멍이 있었다 — `st.wi` 를 한 번도 안 바꿔서 **종합
    화면 분기를 한 번도 안 지났고**(그 분기는 폭을 아예 안 보는 고정 문구였다),
    폭도 30 부터라 `room > 4` 가 자르기를 통째로 건너뛰는 구간을 못 봤다.
    실측으로 폭 20~27 에서 88칸짜리 줄이, 폭 20~69 에서 70칸짜리 종합 문구가
    그대로 나갔다. 이제 **모든 구간 × 모든 정렬 × 드릴다운 × 폭 20~200** 을 본다.
    """
    from kr_quant.tui.flow_app import hint_text

    st = State(data)
    for width in range(20, 201):
        for wi in range(len(WINDOWS)):
            st.wi = wi
            for drill in (False, True):
                st.drill = drill
                n = len(NAME_SORTS) if drill else len(SORTS)
                for i in range(n):
                    if drill:
                        st.nsi = i
                    else:
                        st.si = i
                    got = hint_text(st, width)
                    assert sum(cell_width(c) for c in got) <= width, (
                        f"폭{width}·구간{WINDOWS[wi]}·정렬{i}·드릴{drill} 에서 "
                        f"힌트바가 넘친다: {got!r}")


def test_hint_bar_has_its_own_short_lines_and_never_gets_cut_at_80(data):
    """힌트바가 도움말 문장을 통째로 재사용해 폭 80 에서 `…` 로 잘렸다.

    잘리지 않는 자리에서도 한 줄을 비유가 잡아먹었다 — 가속 정렬에서 화면에
    늘 떠 있던 문장이 ``… × 100 [%p]. 물리로 a = F/m.`` 이었다. 값을 읽는 데
    필요한 건 정의와 단위까지고, 비유는 `?` 도움말의 일이다.

    주입: `hint_desc` 가 `HINT_DESC` 를 안 보고 `help_desc` 만 내면(예전 동작)
    폭 80 에서 잘리는 열이 나와 실패한다.
    """
    from kr_quant.tui.flow_app import hint_text
    from kr_quant.tui.flow_view import (
        HINT_DESC, NAME_SORT_COL, SORT_COL, hint_desc)

    st = State(data)
    for width in (80, 90, 100, 132):
        for drill in (False, True):
            st.drill = drill
            n = len(NAME_SORTS) if drill else len(SORTS)
            for i in range(n):
                if drill:
                    st.nsi = i
                else:
                    st.si = i
                line = hint_text(st, width)
                assert "…" not in line, f"폭{width}·드릴{drill} 힌트바가 잘렸다: {line}"

    # 줄세울 수 있는 열은 **전부** 짧은 설명을 가진다. 없으면 도움말 문장으로
    # 떨어지는데, 그건 폴백이지 기본값이 아니다.
    for header in list(SORT_COL.values()) + list(NAME_SORT_COL.values()):
        assert header in HINT_DESC, f"{header!r} 의 짧은 설명이 없다"
    # 그래도 폴백은 살아 있어야 한다 — 열이 하나 늘었을 때 빈 줄이 되면 안 된다.
    assert hint_desc("포텐셜[½kx²]")
    assert "27개" in hint_desc("섹터"), "짧은 설명이 없는 항목의 폴백이 죽었다"


def test_the_long_explanations_stay_in_the_help(data):
    """짧게 만든 건 힌트바뿐이다 — 비유·유래·한계는 도움말에 그대로 남는다.

    주입: `HELP` 에서 물리 비유를 지우면 실패한다. 힌트바를 짧게 만든 대가로
    설명을 **잃지는 않았다**는 사실이 이 검사의 내용이다.
    """
    from kr_quant.tui.flow_view import HELP, HINT_DESC, help_desc

    assert "물리로 a = F/m" in help_desc(HELP, "가속[%p]")
    assert "물리로" not in HINT_DESC["가속[%p]"], "비유가 힌트바에 남았다"
    for header, short in HINT_DESC.items():
        long = help_desc(HELP, header)
        if long:        # 표에 없는 열(종목명 등)은 도움말 항목이 없을 수 있다
            assert len(short) <= len(long), (
                f"{header!r} 의 힌트바 문장이 도움말보다 길다 — 두 벌로 나눈 뜻이 없다")


def test_dates_are_not_painted_as_negative_numbers(data):
    """헤더의 `2026-07-31 ~ 2026-08-28` 이 음수로 잡혀 하락색으로 칠해졌다.

    값이 아닌 것이 값처럼 보이면 그건 배색 취향 문제가 아니다. 표·패널의
    진짜 부호는 그대로 칠해져야 하므로 **양쪽을** 본다.

    주입: `_NUM` 앞에 새로 붙인 "숫자·글자 뒤에서는 안 잡는다" 제한을 지우면
    첫 단언이 실패한다.
    """
    from kr_quant.tui.flow_view import (
        color_spans, detail_lines, header_lines, table_lines)

    st = State(data)
    for line in header_lines(st, 200):
        for start, _w, _role in color_spans(line):
            # 날짜가 있는 줄에서는 아무 것도 안 칠해야 한다(그 줄에 부호값이 없다).
            assert False, f"헤더에 색을 칠했다: {line[start:start + 12]!r} · {line!r}"
    # 그리고 진짜 부호는 여전히 칠한다 — 규칙이 넓어져 다 죽으면 안 된다.
    lines = table_lines(st, 200, 20)[0][1:] + detail_lines(st, 200)
    roles = {role for ln in lines for _s, _w, role in color_spans(ln)}
    assert {"up", "down"} <= roles, f"부호색이 통째로 죽었다: {roles}"


def test_help_labels_are_never_truncated():
    """회귀 — 도움말 라벨이 잘리면 **어느 열 설명인지 알 수 없다.**

    `순매수상위[억]` 이 `순매수상위` 로, `포텐셜[½kx²]` 이 `포텐셜[½kx` 로
    잘려 있었다(7개). 표 헤더와 이름이 안 맞으면 도움말이 자기 일을 못 한다.
    열을 넓히면 설명이 폭 80(SSH 기본)을 넘으므로, 긴 라벨은 자기 줄에 둔다.
    """
    from kr_quant.tui.flow_view import HELP, cell_len, help_lines

    long = [n for n, _d in HELP if n and cell_len(n) > 10]
    assert long, "긴 라벨이 없다 — 이 검사가 헛돈다"
    for width in (80, 100, 132):
        lines = [ln.rstrip() for ln in help_lines(width, 0, 10 ** 6)[0]]
        assert all(cell_len(ln) <= width for ln in lines)
        for name in long:
            assert name in lines, f"폭{width} 에서 {name!r} 라벨이 잘렸다"


# ── ambiguous 폭 · 열 자르기 규칙 ────────────────────────────────────────────

def _wide_w(text: str) -> int:
    """``cell_width`` 를 **안 쓰는** 참조 구현 — 'A' 를 2칸으로 그리는 터미널.

    검사가 `cell_len` 으로 재면 구현과 같은 규칙을 두 번 적어 놓고 대조하는
    꼴이라 늘 초록이다(동어반복). 그래서 여기서 따로 센다.
    """
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F", "A") else 1
               for c in text)


def test_ambiguous_wide_mode_is_off_by_default_and_counts_A_as_two_when_on(monkeypatch):
    """'A' 글자의 폭은 터미널이 정한다 — 그걸 셀 수 있어야 한다."""
    import unicodedata

    from kr_quant.tui import flow_view

    amb = [c for c in "·—↑↓→←×÷²½ΔΣβ▲▼※≠…─"
           if unicodedata.east_asian_width(c) == "A"]
    assert len(amb) >= 15, "이 검사가 겨냥한 글자들이 'A' 가 아니게 됐다"

    # 기본값은 예전 그대로 — 켜지 않으면 한 글자도 다르게 세지 않는다.
    monkeypatch.setattr(flow_view, "AMBIGUOUS_WIDE", False)
    assert all(flow_view.cell_width(c) == 1 for c in amb)
    assert flow_view.cell_width("가") == 2 and flow_view.cell_width("a") == 1

    monkeypatch.setattr(flow_view, "AMBIGUOUS_WIDE", True)
    assert all(flow_view.cell_width(c) == 2 for c in amb), \
        "ambiguous=wide 모드인데 'A' 를 1칸으로 센다"
    # 'A' 아닌 것은 모드와 무관하다 — 브라유 스파크라인이 여기 걸리면 안 된다.
    assert flow_view.cell_width("가") == 2 and flow_view.cell_width("a") == 1
    assert all(flow_view.cell_width(c) == 1 for c in flow_view.SPARK)


def test_no_line_overflows_when_the_terminal_draws_ambiguous_chars_wide(data, monkeypatch):
    """⚠️ 이 화면들은 'A'(Ambiguous) 글자를 31종·440여 곳에서 쓴다.

    `·` `—` `↑` `→` `×` `²` `Σ` … 의 폭은 유니코드가 **정하지 않았고** 터미널이
    정한다. 한국어권에서 흔한 "ambiguous=wide" 설정(PuTTY·iTerm2·mintty)에서는
    2칸으로 그려지는데, 1칸으로 세면 `pad(..., width)` 로 폭에 딱 맞춘 줄이
    넘쳐서 다음 줄로 접히고 **화면 전체가 어긋난다.**

    그래서 모드를 켠 채 모든 표시면을 그려 보고, `cell_width` 를 쓰지 않는
    참조 구현(`_wide_w`)으로 재서 폭을 넘는 줄이 하나도 없어야 한다고 본다.
    """
    from kr_quant.tui import flow_view
    from kr_quant.tui.flow_view import (
        detail_lines, footer_line, header_lines, help_lines, names_lines, table_lines,
    )

    monkeypatch.setattr(flow_view, "AMBIGUOUS_WIDE", True)
    st = State(data)
    bad = []
    for width in (40, 72, 80, 100, 132, 170):
        surfaces = [[footer_line(width)], [footer_line(width, drill=True)],
                    help_lines(width, 0, 10 ** 6)[0]]
        for wi in range(len(WINDOWS)):
            for ai in range(len(ACTORS)):
                st.wi, st.ai = wi, ai
                surfaces += [header_lines(st, width), detail_lines(st, width),
                             table_lines(st, width, 20)[0], names_lines(st, width)[0]]
        for lines in surfaces:
            for line in lines:
                if _wide_w(line) > width:
                    bad.append((width, _wide_w(line), line))
    assert not bad, f"폭을 넘는 줄 {len(bad)}개 — 예: {bad[:3]}"


def test_both_screens_cut_columns_by_the_same_rule():
    """흐름 화면과 원장 화면이 **같은 규칙으로** 열을 떨어뜨린다.

    두 `_fit` 은 열 표현만 다른(``Col`` 대 3-튜플) 같은 로직이 두 벌이었고,
    **둘을 잇는 검사가 하나도 없었다.** 한쪽만 고치면 같은 앱의 두 화면이 열을
    다르게 잘라도 전부 초록이다. 이제 규칙은 `fit_widths` 한 곳에 있고, 이
    검사가 두 호출부를 그 한 곳에 묶어 둔다.
    """
    import random

    from kr_quant.tui.flow_view import Col, _fit as flow_fit, fit_widths
    from kr_quant.tui.ledger_view import _fit as ledger_fit

    rng = random.Random(20260829)
    for _ in range(2000):
        widths = [rng.randint(0, 30) for _ in range(rng.randint(0, 14))]
        total = rng.randint(-5, 200)
        k = fit_widths(widths, total)
        fcols = [Col(f"h{i}", w, False, None) for i, w in enumerate(widths)]
        lcols = [(f"h{i}", w, False) for i, w in enumerate(widths)]
        assert flow_fit(fcols, total) == fcols[:k], (widths, total)
        assert ledger_fit(lcols, total) == lcols[:k], (widths, total)


def test_fit_widths_keeps_the_first_column_and_counts_the_gap():
    """규칙 자체 — 첫 열은 잘려도 남기고, 열 사이 공백 1칸을 센다.

    첫 열을 떨어뜨리면 좁은 터미널에서 표가 통째로 사라지고, 공백을 안 세면
    마지막 열이 한 칸 넘쳐 줄이 접힌다.
    """
    from kr_quant.tui.flow_view import fit_widths, span_at

    assert fit_widths([], 80) == 0
    assert fit_widths([100], 10) == 1          # 첫 열은 잘려도 남는다
    assert fit_widths([5, 5], 10) == 1         # 5+1+5=11 > 10
    assert fit_widths([5, 5], 11) == 2
    assert fit_widths([5, 5, 5], 17) == 3
    assert fit_widths([5, 5, 5], 16) == 2
    assert span_at([5, 5, 5], 0) == (0, 5)
    assert span_at([5, 5, 5], 2) == (12, 5)


def test_the_panel_marks_the_sector_name_and_the_view_gives_the_coordinates(data):
    """패널 첫 줄은 줄 전체가 한 색이라 섹터 이름이 부속 정보에 묻혔다.

    그 줄에서 "지금 무엇을 보고 있는가" 를 말하는 건 이름 하나뿐이다. 좌표는
    **뷰가 낸다** — 앱이 문자열을 다시 뜯어 길이를 추측하면 문구를 고칠 때 색이
    조용히 어긋난다(`col_span`·`name_sort_span`·`is_section` 과 같은 관용구).

    주입: `detail_title_span` 이 문자 수(`len`)로 폭을 내면 한글 섹터에서
    칠하는 칸이 절반으로 어긋나 실패한다.
    """
    from kr_quant.tui.flow_view import cell_len, detail_lines, detail_title_span

    st = State(data)
    for row in range(min(5, len(st.rows()))):
        st.row = row
        sector = st.rows()[row].get("sector")
        for width in (40, 80, 120, 200):
            line = detail_lines(st, width)[0]
            span = detail_title_span(st, width)
            assert span, f"폭{width} 에서 강조할 자리를 못 냈다"
            start, w = span
            # 칠하는 칸이 **섹터 이름 위**여야 한다 — 한글이 두 칸이라 문자
            # 인덱스로 세면 절반씩 어긋난다. 표시 칸으로 되짚어 확인한다.
            cells = []
            for ch in line:
                cells.append(ch)
                cells += [""] * (cell_width(ch) - 1)
            painted = "".join(cells[start:start + w])
            assert sector.startswith(painted.rstrip()) and painted.strip(), (
                f"폭{width}: 칠하는 자리가 이름이 아니다 {painted!r} vs {sector!r}")
            assert start + w <= width, f"폭{width} 를 넘겨 칠한다: {span}"
            if width >= cell_len(sector) + 2:
                assert w == cell_len(sector), f"이름 전체를 안 덮는다: {span}"
    # 색이 없어도 줄은 그대로다 — 강조는 속성일 뿐 글자를 바꾸지 않는다.
    assert detail_lines(st, 200)[0].strip().startswith(st.rows()[st.row]["sector"])


def test_the_combined_screen_neither_sorts_nor_claims_to(data):
    """종합에서 `s`·`r` 은 눌리는데 표는 안 바뀌었다 — 그런데 헤더 라벨은 바뀌어서
    정렬이 된 것처럼 보였다.

    이 저장소가 리포트 쪽에서 CI 로 막아 온 부류다(README §7 D 계층: "파라미터를
    바꿨는데 계산은 동일"). 화살표를 안 그리기로 한 판단(`header_lines` 주석)을
    열 이름까지 밀고 간다 — 새 동작을 만드는 게 아니라 이미 내린 판단을 끝까지
    적용하는 일이다.

    주입: `handle_key` 의 `and st.sortable` 을 지우면 상태가 바뀌어 실패하고,
    헤더가 `정렬[…]` 을 다시 적으면 마지막 단언이 실패한다.
    """
    from kr_quant.tui.flow_app import handle_key, hint_text
    from kr_quant.tui.flow_view import State, header_lines

    st = State(data)
    st.wi = WINDOWS.index("종합")
    if not st.rows():
        pytest.skip("이 픽스처에 종합 블록이 없다")
    before = (st.si, st.rev, st.nsi, st.nrev, st.row, st.wi, st.mi, st.ai)
    order = [r.get("sector") for r in st.rows()]
    for key in (ord("s"), ord("S"), ord("r")):
        assert handle_key(st, key) is True
        assert (st.si, st.rev, st.nsi, st.nrev, st.row, st.wi, st.mi,
                st.ai) == before, f"{chr(key)!r} 가 종합에서 상태를 바꿨다"
        assert [r.get("sector") for r in st.rows()] == order
    # 그리고 화면이 그 사실을 말한다 — 키가 조용히 안 듣기만 하면 그것도 거짓말이다.
    head = header_lines(st, 200)[1]
    assert "정렬[" not in head, f"없는 정렬을 헤더가 말한다: {head.strip()!r}"
    assert "정렬" in head and "없" in head, f"정렬이 없다는 사실이 없다: {head.strip()!r}"
    assert "정렬" in hint_text(st, 200) and "없다" in hint_text(st, 200)

    # 드릴다운은 **실제로** 정렬한다 — 종합에서 들어가도 그렇다.
    handle_key(st, ord("l"))
    assert st.drill and st.sortable
    codes = [t.get("code") for t in st.names()]
    handle_key(st, ord("s"))
    assert st.nsi != 0, "드릴다운에서 s 까지 죽였다"
    handle_key(st, ord("r"))
    assert st.nrev, "드릴다운에서 r 까지 죽였다"
    assert codes, "픽스처에 종목이 없다 — 이 검사가 헛돈다"


def test_combined_screen_shows_the_four_actors_and_says_which_window(data):
    """종합 구간에서만 4주체가 `— · — · — · —` 였다.

    원인은 렌더가 아니라 데이터다 — 종합 블록은 5·20·60·120 을 **G 순위**로
    섞은 축이라 `inst`·`forgn`·`indiv`·`etc` 가 아예 없다. 이 화면의 견인주는
    같은 처지에서 이미 **종목에서 대표 창(20일)으로 더해** 답을 만들고 있었다
    (`_leads`). 4주체도 같은 규칙으로 만든다 — 그리고 **어느 창인지 적는다.**

    주입: `_rows_uncached` 의 종합 갈래에서 `_actor_sums` 를 빼면 대시가 돌아와
    실패하고, `_actors_line` 의 꼬리표를 지우면 마지막 단언이 실패한다.
    """
    from kr_quant.tui.flow_view import ACTORS, State, detail_lines, fmt_amt

    st = State(data)
    st.wi = WINDOWS.index("종합")
    if not st.rows():
        pytest.skip("이 픽스처에 종합 블록이 없다")
    line = detail_lines(st, 200)[1]
    assert "—" not in line, f"종합에서 4주체가 아직 대시다: {line.strip()!r}"
    for _key, ko in ACTORS:
        assert ko in line, f"{ko} 가 없다: {line.strip()!r}"
    # 값은 **그 섹터 종목들의 20일 순매수 합**이어야 한다 — 화면이 스스로 지어낸
    # 숫자면 안 된다. 기대값을 여기서 따로 더해 맞춰 본다.
    sec = st.rows()[st.row]["sector"]
    want = {}
    for _code, nm in data["names"].items():
        if nm.get("sector") != sec:
            continue
        w = (nm.get("win") or {}).get("20") or {}
        for key, _ko in ACTORS:
            if w.get(key) is not None:
                want[key] = want.get(key, 0.0) + w[key]
    assert want, "픽스처에 그 섹터 종목이 없다 — 이 검사가 헛돈다"
    for key, ko in ACTORS:
        assert f"{ko} {fmt_amt(want.get(key))}" in line, (
            f"{ko} 가 종목 합({want.get(key)})과 다르다: {line.strip()!r}")
    # 어느 창의 값인지가 화면에 있어야 한다 — 없으면 위 표(종합)와 이 줄(20일)이
    # 다른 것을 말하는데 화면은 같은 것처럼 보인다.
    assert "20일" in line, f"대표 창을 안 적었다: {line.strip()!r}"


def test_the_drill_title_says_20_days_when_the_window_is_combined(data):
    """종합에서 Enter 를 누르면 나오는 목록은 **20일** 인데 제목이 `종합일 기준`
    이라고 적었다 — 화면이 스스로 없는 구간을 말했다.

    주입: 제목을 `f"{st.window}일 기준"` 으로 되돌리면 실패한다.
    """
    from kr_quant.tui.flow_view import State, names_lines

    st = State(data)
    st.wi = WINDOWS.index("종합")
    if not st.rows():
        pytest.skip("이 픽스처에 종합 블록이 없다")
    st.drill = True
    title = names_lines(st, 200)[0][0]
    assert "종합일" not in title, f"없는 구간을 말한다: {title.strip()!r}"
    assert "20일" in title, f"어느 구간인지 안 적었다: {title.strip()!r}"


def test_detail_panel_does_not_repeat_what_the_table_already_shows(data):
    """상세 패널 첫 줄이 표에 있는 값을 한 번 더 적고 있었다.

    ``· 미실현 +2.9%p(포텐셜 60)`` 은 미실현이 **발산 막대**였던 시절 "도형은
    순서를, 숫자는 크기를" 이라며 붙인 것이다. 막대를 숫자로 되돌리면서
    (`Col("미실현[%p]", …, _num("x", 1))`) 이쪽을 안 지워 같은 값이 두 자리에
    남았다. 포텐셜은 x 를 제곱해 **부호를 지운 값**이라 방향이 핵심인 이
    화면에서 무슨 질문에 답하는지도 불분명하다.

    주입: 첫 줄에 `f" · 미실현 {fmt_pct(x, 1)}%p"` 를 되살리면 실패한다.
    """
    from kr_quant.tui.flow_view import detail_lines, table_cols

    st = State(data)
    # `섹터`·`종목`(수) 는 이 줄이 지고 있는 일 자체다 — 무슨 섹터의 몇 종목인가.
    headers = {c.header.split("[")[0] for c in table_cols(st, 200)
               if c.header} - {"섹터", "종목"}
    assert "미실현" in headers, "표에 미실현 열이 없다 — 이 검사가 헛돈다"
    for row in range(min(5, len(st.rows()))):
        st.row = row
        first = detail_lines(st, 200)[0]
        for h in headers:
            assert h not in first, (
                f"패널 첫 줄이 표의 {h!r} 열을 또 적는다: {first.strip()!r}")
        assert "포텐셜" not in first, f"부호를 지운 값이 남았다: {first.strip()!r}"
        # 대신 이 줄이 지고 있는 일 — 무슨 섹터인지·몇 종목인지·어떻게 여는지.
        assert "Enter" in first and "종목" in first, first


def test_detail_panel_answers_who_took_the_other_side(data):
    """회귀 — "기관이 팔았다" 다음 질문은 **"그럼 누가 받았지"** 다.

    화면이 한 번에 한 주체만 보여주므로 그 답은 앱을 바꿔야(`kq-ledger`)
    알 수 있었다. 주식은 누가 사면 누가 판 것이고 4주체 합은 0 에 닫히므로,
    답은 같은 페이로드 안에 이미 있다 — 화면을 바꿀 이유가 없다.
    """
    from kr_quant.tui.flow_view import ACTORS, cell_len, detail_lines

    st = State(data)
    raw = {r["sector"]: r for r in data["blocks"][f"{st.window}|{st.market}"]["rows"]}
    for row in range(min(3, len(st.rows()))):
        st.row = row
        line = next(ln for ln in detail_lines(st, 170) if "기타법인" in ln)
        # 라벨은 없다 — 넷을 다 적는 줄에 "반대편"(선택 주체를 뺀 나머지)이라는
        # 이름이 붙어 있었다. 들여쓰기는 다른 패널 줄과 같은 한 칸이다.
        assert "반대편" not in line, f"틀린 라벨이 아직 붙어 있다: {line!r}"
        assert line.startswith(" ") and not line.startswith("  "), repr(line)
        src = raw[st.rows()[row]["sector"]]
        for key, ko in ACTORS:
            assert ko in line, f"{ko} 가 4주체 줄에 없다"
            assert fmt_amt(src[key]) in line, (
                f"{ko} 금액이 원본과 다르다 — 기대 {fmt_amt(src[key])}, 줄: {line!r}")
        # 네 주체가 서로 다른 값이어야 이 검사가 헛돌지 않는다.
        assert len({src[k] for k, _ko in ACTORS}) > 1, "픽스처가 4주체를 같게 뒀다"

    # 좁아지면 단계를 내려 줄인다. **자르기 전 원문**을 재야 한다 —
    # `detail_lines` 가 `pad` 로 감싸므로 출력으로는 넘침이 원리상 안 보인다
    # (이 저장소가 오늘만 세 번 밟은 함정이다).
    from kr_quant.tui.flow_view import _actors_line

    st.row = 0
    r = st.rows()[0]
    for width in range(24, 180, 3):
        raw_line = _actors_line(r, width)
        assert cell_len(raw_line) + 1 <= width or raw_line == " —", (
            f"폭{width} 에서 {cell_len(raw_line)}칸: {raw_line!r}")
        # 폭 80(SSH 기본)에서는 네 주체가 다 들어가야 한다 — 라벨(`반대편: `,
        # 6칸)이 붙어 있던 시절엔 `기타법인` 이 먼저 잘렸는데, 예시처럼 그
        # 종목의 주역이 기타법인인 경우가 있어 그게 실질 손실이었다.
        if width >= 80:
            assert "기타법인" in raw_line, f"폭{width} 인데 주체가 빠졌다"


# ── 종목 목록의 세 축 (자금 성격 · 지속성 · 가격) ──────────────────────────
#
# 이 화면의 여섯 열은 전부 **같은 분자**(선택 주체의 구간 순매수)를 분모만 바꿔
# 여섯 번 보여줬다: 금액 그대로 · 섹터 합 대비 · 거래대금 대비 · 시총 대비, 그리고
# 그 두 분모 자체. 축이 하나뿐이면 "이 섹터에서 어느 종목을 살까" 에 답할 수 없다.
# 여기서 잠그는 것은 새로 들어온 세 축이 **다른 질문에 답하는가** 다.

def test_stock_list_has_three_axes_not_one_number_rescaled(data):
    from kr_quant.tui.flow_view import COL_INVTRT, COL_PENFND, names_cols

    heads = [c.header for c in names_cols()]
    for h in (COL_INVTRT, COL_PENFND, "최근집중[%]", "상대수익[%p]"):
        assert h in heads, f"'{h}' 열이 없다: {heads}"


def test_the_new_axes_survive_at_eighty_columns():
    """폭 80 에서 **결론과 자금**이 남는다 — 선정 · 순매수 · 누적 · 투신.

    예전에는 여기서 `상대수익` 까지 지켰는데, `선정` 이 왼쪽으로 오면서 네 칸을
    가져갔다(87 부터 보인다). 그 교환을 의식적으로 한다 — 80칸 화면에서 한 열만
    남길 수 있다면 네 축을 겹쳐 놓은 점수가 그중 하나보다 낫다. 이 검사가
    바뀌는 것 자체가 그 판단을 기록으로 남기는 일이다.
    """
    from kr_quant.tui.flow_view import COL_INVTRT, fit_names, view_width

    heads = [c.header for c in fit_names(view_width(80))]
    for h in ("선정", "순매수[억]", "누적[%]", COL_INVTRT):
        assert h in heads, f"폭 80 에서 '{h}' 가 잘려 나갔다: {heads}"
    # 네 축은 더 넓은 화면에서 전부 살아난다 — 사라진 것이 아니라 밀린 것이다.
    wide = [c.header for c in fit_names(view_width(130))]
    for h in (COL_INVTRT, "연기금[억]", "상대수익[%p]", "최근집중[%]", "참여율[%]"):
        assert h in wide, f"폭 130 에서도 '{h}' 가 없다: {wide}"

def test_recent_concentration_is_the_short_window_over_this_one(data):
    """최근집중 = 5일 순매수 ÷ 이 창의 순매수 × 100.

    창이 5 ⊂ 20 ⊂ 60 ⊂ 120 으로 겹치는 것이 여기선 자산이다 — 이미 실린 값의
    비율일 뿐이라 페이로드가 한 바이트도 안 늘어난다.
    """
    st = State(data)                                  # 기본 창 20일
    seen = 0
    for t in st.names():
        w5 = data["names"][t["code"]]["win"]["5"]["inst"]
        assert t["conc"] is not None, t
        assert abs(t["conc"] - w5 / t["flow"] * 100) < 1e-9
        seen += 1
    assert seen, "픽스처에 종목이 없다 — 이 검사가 헛돈다"


def test_recent_concentration_follows_the_window_the_list_shows(data):
    """60일 화면의 분모는 60일이다. 창을 바꿨는데 값이 그대로면 라벨만
    바뀐 것이다 — 이 저장소가 리포트 쪽에서 CI 로 막아 온 부류다."""
    st = State(data)
    by_win = {}
    for w in ("20", "60", "120"):
        st.wi = WINDOWS.index(w)
        by_win[w] = [t["conc"] for t in st.names()]
    assert by_win["20"] != by_win["60"] != by_win["120"], by_win
    st.wi = WINDOWS.index("60")
    for t in st.names():
        w5 = data["names"][t["code"]]["win"]["5"]["inst"]
        assert abs(t["conc"] - w5 / t["flow"] * 100) < 1e-9


def test_recent_concentration_is_blank_where_the_ratio_stops_meaning_anything(data):
    """분모가 0 근처면 비율이 폭발한다 — 5일 +40억 ÷ 20일 +0.2억 = 20,000%.
    그건 '집중' 이 아니라 나눗셈 사고다. 못 읽는 칸은 비운다.
    가장 짧은 창(5일)에서는 분자와 분모가 같아 늘 100% 라 역시 비운다."""
    import copy

    d = copy.deepcopy(data)
    d["names"]["005930"]["win"]["20"]["inst"] = 0.2
    st = State(d)
    t = next(t for t in st.names() if t["code"] == "005930")
    assert t["conc"] is None, t["conc"]

    st = State(data)
    st.wi = WINDOWS.index("5")
    assert all(t["conc"] is None for t in st.names())


def test_relative_return_is_the_stock_minus_its_sector(data):
    """상대수익 = 종목 구간수익률 − 섹터 구간수익률.

    섹터의 `미실현`(x)을 종목에 내리지 않는 이유: 그 k·b 는 27개 섹터
    횡단면에서 적합된 것이라 2,645종목 위에서는 적합된 적이 없다. 뺄셈은
    회귀 없이 "섹터는 갔는데 이건 안 갔다" 를 직접 답한다.
    """
    st = State(data)
    sec_ret = st.selected()["ret"]
    seen = 0
    for t in st.names():
        assert abs(t["rrel"] - (t["ret"] - sec_ret)) < 1e-9, t
        seen += 1
    assert seen


def test_relative_return_reads_the_window_the_list_reads(data):
    """종합 축은 창을 섞은 것이라 그 행에 수익률이 없다. 목록은 20일을 쓰므로
    기준선도 **20일 블록**에서 꺼내야 한다 — 다른 창끼리 빼면 뜻이 없다."""
    import copy

    d = copy.deepcopy(data)
    for m in ("전체", "거래소", "코스닥"):
        for r in d["blocks"][f"20|{m}"]["rows"]:
            r["ret"] = 10.0
        for r in d["blocks"][f"60|{m}"]["rows"]:
            r["ret"] = -99.0
    st = State(d)
    st.wi = WINDOWS.index("종합")
    seen = 0
    for t in st.names():
        assert abs(t["rrel"] - (t["ret"] - 10.0)) < 1e-9, t
        seen += 1
    assert seen


def test_fund_detail_columns_are_not_the_institution_total_rescaled(data):
    """투신·연기금은 `기관` 한 덩어리에서 뽑아낼 수 없는 값이다.

    실측(20거래일): SK하이닉스는 기관 −13,157억인데 투신 +1,549 · 연기금 +4,614 ·
    금투 −22,611 이다. 기관 총액만 보면 "판다" 지만 이 화면이 믿는 두 주체는
    사고 있었다. 그래서 페이로드에서 **따로** 와야 한다.
    """
    import copy

    d = copy.deepcopy(data)
    w = d["names"]["000660"]["win"]["20"]
    w["invtrt"], w["penfnd_etc"] = 1548.6, 4614.2      # 기관은 그대로 −50.0
    st = State(d)
    t = next(t for t in st.names() if t["code"] == "000660")
    assert t["invtrt"] == 1548.6
    assert t["penfnd"] == 4614.2
    assert t["flow"] == -50.0


def test_a_report_without_the_new_fields_still_renders(data):
    """회귀 — 옛 페이로드에는 투신·연기금·종목수익률이 없다.

    화면이 깨지면 안 된다. 값이 없는 칸은 `—` 로 남고 나머지 열은 그대로여야
    하며, 폭 불변식도 지켜져야 한다. (이 저장소의 서명 같은 실패 모드는 그
    반대다 — producer 가 키를 빠뜨려도 화면이 예쁘게 `—` 를 그리는 것. 그쪽은
    `scripts/verify_report.py` 가 잡는다.)
    """
    import copy

    from kr_quant.tui.flow_view import names_lines

    d = copy.deepcopy(data)
    for nmv in d["names"].values():
        for w in nmv["win"].values():
            for k in ("invtrt", "penfnd_etc", "ret"):
                w.pop(k, None)
    st = State(d)
    for nsi in range(len(NAME_SORTS)):
        st.nsi = nsi
        names = st.names()
        assert names
        assert all(t["invtrt"] is None and t["penfnd"] is None
                   and t["rrel"] is None for t in names)
        for width in (80, 120, 170):
            lines, nhead = names_lines(st, width)
            assert len(lines) - nhead == len(names)
            for ln in lines:
                assert _w(ln) == width, (width, repr(ln))


def test_the_curses_last_cell_rule_lives_in_one_place():
    """회귀 — curses 는 오른쪽 아래 칸에 쓰면 터지므로 앱은 늘 마지막 칸을 비운다.

    그 −1 이 앱 안에만 있으면 "폭 80 에서 이 열이 보이는가" 를 묻는 검사는 80 으로
    묻고 화면은 79 로 그린다. 열 합계가 정확히 80 인 배치가 통과하는데 실제 80칸
    터미널에서는 마지막 열이 안 보인다 — `상대수익[%p]` 이 실측 캡처에서 그렇게
    사라졌다. 규칙은 :func:`view_width` 하나이고, 앱은 그것을 부른다.
    """
    import inspect
    import re

    from kr_quant.tui import flow_app
    from kr_quant.tui.flow_view import view_width

    assert view_width(80) == 79
    assert view_width(1) == 1                      # 폭 0 으로 내려가지 않는다
    src = inspect.getsource(flow_app._draw)
    assert "view_width(" in src, "앱이 폭 규칙을 안 부른다"
    assert not re.search(r"\bw\s*-\s*1\b", src), "앱에 −1 이 다시 나타났다"


def test_recent_concentration_is_hidden_on_rows_that_do_not_move_the_sector(data):
    """값이 틀린 게 아니라, 볼 이유가 없는 줄이 화면에서 **가장 큰 숫자**를 달고 있었다.

    실측(건설 20일): 금호건설은 순매수 +3억(섹터 총유량의 0.049%)인데 최근집중이
    −287 로 떠서, 눈이 그 줄로 끌려갔다. 화면이 중요하지 않은 것을 크게 말한 것이다.
    목록 안에서의 몫이 :data:`CONC_MIN_SHARE` 미만이면 비운다.
    """
    import copy

    from kr_quant.tui.flow_view import CONC_MIN_SHARE

    d = copy.deepcopy(data)
    # 섹터를 압도하는 종목 하나를 넣어 나머지의 몫을 임계 아래로 민다.
    d["names"]["005930"]["win"]["20"]["inst"] = 500000.0
    d["names"]["005930"]["win"]["5"]["inst"] = 200000.0
    st = State(d)
    rows = {t["code"]: t for t in st.names()}
    tot = sum(abs(t["flow"]) for t in rows.values())
    big, small = rows["005930"], rows["000990"]         # DB하이텍 +5억
    assert abs(big["flow"]) / tot * 100 >= CONC_MIN_SHARE
    assert abs(small["flow"]) / tot * 100 < CONC_MIN_SHARE
    assert big["conc"] is not None, "섹터를 움직인 행의 최근집중까지 지우면 안 된다"
    assert small["conc"] is None, (
        f"몫 {abs(small['flow']) / tot * 100:.3f}% 인 행에 최근집중 "
        f"{small['conc']} 이 떴다")


def test_the_recent_concentration_gate_is_a_share_not_an_amount(data):
    """회귀 — 고정 금액 임계를 쓰면 얇은 섹터가 통째로 빈다.

    섹터 규모는 세 자릿수 배로 갈린다(실측: 누적 80% 를 처음 넘긴 행의 |순매수| 가
    거래소/제조 0.8억 · 전기/전자 2,435억). 고정 10억을 걸면 자기 섹터를 실제로
    움직인 행의 7.7% 가 함께 지워진다. 작은 섹터에서도 **1등 종목은 보여야** 한다.
    """
    import copy

    d = copy.deepcopy(data)
    for code in ("005930", "000660", "066570", "009150", "000990"):
        for w, v in d["names"][code]["win"].items():
            v["inst"] = v["inst"] / 1000.0             # 섹터 전체를 1/1000 로
    st = State(d)
    rows = st.names()
    assert all(abs(t["flow"]) < 1.0 for t in rows), "축소가 안 됐다 — 검사가 헛돈다"
    # 금액으로는 전부 1억 미만이지만 몫은 그대로다. 반올림 하한(CONC_MIN_DEN)이
    # 걸려 비는 것은 맞고, **몫 때문에** 비어서는 안 된다.
    st2 = State(data)
    shares = {t["code"]: abs(t["flow"]) for t in st2.names()}
    top = max(shares, key=shares.get)
    assert next(t for t in st2.names() if t["code"] == top)["conc"] is not None


def test_blanked_concentration_sorts_to_the_back_not_to_zero(data):
    """'값이 없는 것은 작은 값이 아니다' — 비운 칸은 정렬에서도 결측이어야 한다.

    가두기(−100~+200 클리핑)를 안 고른 이유가 여기 있다. 경계값을 넣으면 그것이
    진짜 값처럼 정렬에 끼어들어 새 거짓말이 된다.
    """
    import copy

    d = copy.deepcopy(data)
    d["names"]["005930"]["win"]["20"]["inst"] = 500000.0
    d["names"]["005930"]["win"]["5"]["inst"] = 200000.0
    st = State(d)
    st.nsi = [k for k, _ in NAME_SORTS].index("conc")
    vals = [t["conc"] for t in st.names()]
    assert None in vals, "픽스처에 비워진 행이 없다 — 이 검사가 헛돈다"
    first_none = vals.index(None)
    assert all(v is None for v in vals[first_none:]), f"결측이 중간에 섞였다: {vals}"


def test_an_old_report_says_so_instead_of_showing_four_blank_columns(data):
    """옛 페이로드로 띄우면 새 열 넷이 통째로 `—` 다.

    하위호환 경로가 제대로 작동한 결과라 화면은 안 깨지는데, 보는 사람에게는
    **빈 열이 넷 늘어난 것**으로만 보인다 — "고장났나" 로 읽힌다(실제로 사용자가
    그렇게 봤다). 화면이 왜 비는지 스스로 말해야 한다.
    """
    import copy

    from kr_quant.tui.flow_view import names_lines

    d = copy.deepcopy(data)
    for nmv in d["names"].values():
        for w in nmv["win"].values():
            for k in ("invtrt", "penfnd_etc", "ret"):
                w.pop(k, None)
    old, new = State(d), State(data)
    assert old.legacy_names and not new.legacy_names
    for width in (80, 120, 170):
        assert "옛 리포트" in names_lines(old, width)[0][0], width
        assert "옛 리포트" not in names_lines(new, width)[0][0], width
        for ln in names_lines(old, width)[0]:
            assert _w(ln) == width, (width, repr(ln))


def test_a_quiet_day_is_not_called_an_old_report(data):
    """키가 **없는 것**과 있는데 0 인 것은 다르다. 값이 0 인지로 판정하면 정말로
    전부 0 인 조용한 날에 화면이 "옛 리포트" 라고 거짓말을 한다."""
    import copy

    d = copy.deepcopy(data)
    for nmv in d["names"].values():
        for w in nmv["win"].values():
            w["invtrt"] = w["penfnd_etc"] = w["ret"] = 0.0
    assert not State(d).legacy_names


def test_participation_is_never_shown_without_its_liquidity_partner():
    """`참여율[%]` 이 보이면 `거래대금[억]` 도 보여야 한다 — 어느 폭에서도.

    참여율은 순매수 ÷ 거래대금이라 **비율이다.** 혼자서는 "살 수 있는가" 를 못
    답한다. 실측(2026-08-28, 20일): 참여율 11~14% 구간에 DB금융스팩12호(거래대금
    3억)와 삼성SDI(43,543억)가 같이 있다 — 유동성이 14,000배 다른데 같은 칸에
    비슷한 숫자가 뜬다. 짝이 없으면 살 수 없는 종목이 상단에 섞인다.

    예전 순서는 참여율(102) → 시총대비(115) → 시총(129) → 거래대금(143) 이라
    폭 120 터미널에서 참여율만 보이고 거래대금이 안 보였다.
    """
    from kr_quant.tui.flow_view import fit_names, view_width

    for w in range(40, 220):
        heads = [c.header for c in fit_names(view_width(w))]
        if "참여율[%]" in heads:
            assert "거래대금[억]" in heads, (
                f"폭 {w}: 참여율은 보이는데 거래대금이 없다 — {heads}")


def test_liquidity_pair_is_adjacent():
    """참여율과 거래대금은 **붙어 있어야** 한 쌍으로 읽힌다.

    사이에 다른 열이 끼면 눈이 둘을 짝으로 안 본다. 이 검사가 없으면 나중에
    누가 열을 하나 끼워 넣어도 위 검사는 통과한다(둘 다 보이기는 하므로).
    """
    from kr_quant.tui.flow_view import names_cols

    heads = [c.header for c in names_cols()]
    assert heads.index("거래대금[억]") == heads.index("참여율[%]") + 1, heads

# ── 합성 페이로드로 만드는 쪽을 잠근다 ─────────────────────────────────
#
# 아래 두 검사는 **실데이터에 기대지 않는다.** k ≤ 0 인 블록도, 표의 종목수와
# 목록 길이가 갈리는 상황도 어느 날 리포트에는 있고 어느 날에는 없다 — 그런
# 검사는 CI 에서 조용히 무력해진다. 원하는 상황을 직접 만들어 걸어 둔다.


def _synth_payload(k_sign: float = 1.0, live_by_sector=None, n_days: int = 130):
    """가속도와 수익률의 횡단면 기울기가 정확히 ``k_sign`` 인 페이로드.

    자금흐름도 수익률도 **마지막 날 하루에만** 싣는다. 그러면 어느 창을 잡아도
    같은 (가속도, 수익률) 이 나와 블록마다 k 가 같고, 검사가 창 선택에 안 흔들린다.
    """
    mkt = "거래소"
    secs = [f"S{i:02d}" for i in range(12)]
    dates = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_days)]
    n = len(dates)
    live_by_sector = live_by_sector or {}
    flows, ret, retw, cap, names, n_by = {mkt: {}}, {mkt: {}}, {mkt: {}}, {mkt: {}}, {}, {mkt: {}}
    for i, s in enumerate(secs):
        z = [0.0] * n
        inst = list(z)
        inst[-1] = i * 100.0                 # 가속도 = inst/cap*100 = i
        flows[mkt][s] = {"inst": inst, "forgn": list(z), "indiv": list(z),
                         "etc": list(z), "tv": [1000.0] * n}
        r = list(z)
        r[-1] = k_sign * i                   # 수익률 = k_sign · 가속도
        w = list(z)
        w[-1] = 1.0
        ret[mkt][s] = r
        retw[mkt][s] = w
        cap[mkt][s] = 10000.0
        live = live_by_sector.get(s, 12)
        n_by[mkt][s] = 12                    # 마스터에는 12개가 남아 있다
        for j in range(12):
            code = f"{i:02d}{j:04d}"
            win = {}
            if j < live:                     # 그중 `live` 개만 이 구간에 거래됐다
                for wkey in ("5", "20", "60", "120"):
                    win[wkey] = {"inst": 0.0, "forgn": 0.0, "indiv": 0.0,
                                 "etc": 0.0, "tv": 1.0, "invtrt": 0.0,
                                 "penfnd_etc": 0.0, "ret": 0.0}
            names[code] = {"name": f"N{code}", "sector": s, "market": mkt,
                           "cap": 100.0, "win": win}
    return {"dates": dates, "sectors": secs, "markets": [mkt], "flows": flows,
            "detail": {mkt: {s: {} for s in secs}}, "ret": ret, "retw": retw,
            "iret": {mkt: {s: None for s in secs}}, "cap": cap, "cap_by_month": {},
            "names": names, "n_by_sector": n_by, "n_names": len(names),
            "finalized": True, "detail_labels": {}}


def test_potential_is_withheld_where_the_stiffness_is_not_positive():
    """회귀 — `U = ½kx²` 는 **k>0 에서만** |미실현| 의 순증가 변환이다.

    k<0 인 블록에서는 전 섹터의 U 가 음수가 되고, 화면이 "포텐셜 큰 순" 이라
    말하면서 |미실현| 이 **작은 순**으로 줄세운다. 실측(2026-08-28 리포트):
    5일|코스닥 k=−0.587(t=−0.24) 에서 22개 섹터 전부 U<0, ρ(|x|,U) = −1.0000.
    도움말은 "4개 구간 전부 Spearman 1.0000" 이라 적었는데 그건 시장=전체에서만
    잰 값이었다 — 실측 주장이 실측 범위를 넘어 일반화돼 있었다.
    """
    m = _producer()
    pos = m.build(_synth_payload(k_sign=+1.0))
    neg = m.build(_synth_payload(k_sign=-1.0))
    for key, B in pos["blocks"].items():
        assert B["k"] > 0, (key, B["k"])
        us = [r["U"] for r in B["rows"] if r["x"] is not None]
        assert us and all(u is not None and u >= 0 for u in us), key
        # k>0 이면 |미실현| 이 클수록 U 가 크다 — 순위가 정확히 같아야 한다
        pairs = sorted((abs(r["x"]), r["U"]) for r in B["rows"] if r["x"] is not None)
        assert [p[1] for p in pairs] == sorted(p[1] for p in pairs), key
    for key, B in neg["blocks"].items():
        assert B["k"] < 0, (key, B["k"])
        assert all(r["U"] is None for r in B["rows"]), key
        # 미실현은 남는다 — 뜻을 잃은 것은 U 뿐이다
        assert any(r["x"] is not None for r in B["rows"]), key


def test_missing_potential_sorts_last_not_first():
    """결측은 '작은 값' 이 아니다 — 포텐셜이 없는 블록에서 U 로 정렬하면
    화면은 정렬 대체(`effective_sort`)로 내려가고, 그 사실을 헤더에 적는다."""
    m = _producer()
    st = State(m.build(_synth_payload(k_sign=-1.0)))
    st.si = [k for k, _ in SORTS].index("U")
    key, fell = st.effective_sort(st.rows())
    assert fell and key != "U"
    assert "값이 없다" in header_lines(st, 200)[1]


def test_the_table_counts_the_same_stocks_the_drilldown_lists():
    """회귀 — 표가 "387종목" 이라 적고 그 줄에서 Enter 를 누르면 386개가 나왔다.

    원인은 `n_by_sector`(260일 창에 한 번이라도 거래된 종목)로 세면서 드릴다운은
    그 창의 집계가 있는 종목만 그렸기 때문이다. 벤더 마스터(`stocks`)에는 수급
    보고가 두 달 전에 끊긴 이름이 남아 있다(실측 17종목). 더 나쁜 건 `MIN_NAMES`
    판정이 이 수로 이뤄져 **죽은 이름 하나가 얇은 섹터 경고를 지웠다**는 것이다.
    """
    m = _producer()
    d = m.build(_synth_payload(live_by_sector={"S03": 7, "S07": 9}))
    for bk, B in d["blocks"].items():
        st = State(d)
        st.wi = WINDOWS.index(bk.split("|")[0])
        st.mi = st.markets.index(bk.split("|")[1])
        for ri, r in enumerate(st.rows()):
            st.row = ri
            assert r["n_all"] == len(st.names()), (bk, r["sector"])
        # 마스터에는 12개가 남아 있지만 거래된 것은 7·9 개다
        got = {r["sector"]: (r["n_all"], r["thin"]) for r in B["rows"]}
        assert got["S03"] == (7, True), got["S03"]
        assert got["S07"] == (9, True), got["S07"]
        assert got["S00"] == (12, False), got["S00"]


def _pick_payload(rows, gsec=0.6):
    """선정점수 전용 합성 페이로드 — 한 섹터에 후보를 넉넉히 둔다.

    기본 픽스처의 건설에는 순매수 양수인 종목이 하나뿐이라 ``len(cand) < 2`` 로
    순위가 아예 안 매겨진다. 그러면 관문을 지워도 점수가 계속 ``None`` 이라
    **검사가 변이에 안 죽는다.** 원하는 상황을 직접 만든다.

    ``rows`` 는 ``(code, name, inst, invtrt, penfnd, ret5, ret20, tv)``.
    최근집중은 5일÷20일이므로 창별 ``inst`` 로 그 비율을 만든다.
    """
    names = {}
    for code, name, inst, iv, pf, i5, tv in rows:
        names[code] = {
            "name": name, "sector": "S", "market": "거래소", "cap": 100000.0,
            "win": {"5": {"inst": i5, "forgn": 0.0, "indiv": 0.0, "etc": 0.0,
                          "tv": tv * 0.25, "invtrt": iv * 0.25,
                          "penfnd_etc": pf * 0.25, "ret": 0.0},
                    "20": {"inst": inst, "forgn": 0.0, "indiv": 0.0, "etc": 0.0,
                           "tv": tv, "invtrt": iv, "penfnd_etc": pf,
                           "ret": 0.0}}}
    block = {"from": "2026-08-01", "to": "2026-08-28", "k": 1.0, "b": 0.0, "t": 2.0,
             "rows": [{"sector": "S", "n_all": len(rows), "thin": False,
                       "inst": sum(r[2] for r in rows), "forgn": 0.0,
                       "indiv": 0.0, "etc": 0.0, "cap": 100000.0,
                       "accel": 1.0, "ret": 0.0, "a_idx": 1.0, "cap_idx": 100000.0,
                       "G": gsec, "G_pass": gsec is not None,
                       "pct1y": {}, "spark": {}, "top": {}}]}
    return {"asof": "2026-08-28", "finalized": True,
            "dates": ["2026-08-01", "2026-08-28"], "names": names,
            "blocks": {f"{w}|{m}": block for w in (5, 20, 60, 120)
                       for m in ("전체", "거래소", "코스닥")},
            "combined": {}}


#: (코드, 이름, 20일 기관, 투신, 연기금, 5일 기관, 거래대금)
#: 네 축이 서로 다른 순서를 갖도록 짰다 — 한 축만 보면 1위가 달라진다.
_PICK_ROWS = [
    ("A", "가", 1000.0, 600.0, 300.0, 700.0, 5000.0),   # 믿는돈 크고 최근집중 높음
    ("B", "나", 900.0, 100.0, 50.0, 90.0, 4000.0),      # 집중 낮음
    ("C", "다", 800.0, 400.0, 300.0, 200.0, 3000.0),
    ("D", "라", 700.0, 300.0, 200.0, 500.0, 2000.0),
]


def _pick_rows(data, sector="건설"):
    from kr_quant.tui.flow_view import State

    st = State(data)
    st.row = next(i for i, r in enumerate(st.rows()) if r["sector"] == sector)
    return st.names()


def _pick_rows(data, sector="건설"):
    st = State(data)
    st.row = next(i for i, r in enumerate(st.rows()) if r["sector"] == sector)
    return st.names()


def _scores(rows=None):
    from kr_quant.tui.flow_view import State

    st = State(_pick_payload(rows or _PICK_ROWS))
    st.row = 0
    return {t["code"]: t.get("pick") for t in st.names()}


def test_pick_score_is_withheld_when_a_gate_is_broken():
    """관문 셋 중 하나라도 어기면 점수가 **없다** — 순위가 아니라 탈락이다.

    관문은 탈락을, 점수는 순서를 정한다. 둘을 한 수로 섞으면 "왜 빠졌는지" 와
    "왜 앞인지" 가 같은 숫자에 뭉개진다.
    """
    base = _scores()
    assert all(v is not None for v in base.values()), base

    def broken(idx, **kw):
        rows = [list(r) for r in _PICK_ROWS]
        for k, v in kw.items():
            rows[idx][{"inst": 2, "invtrt": 3, "penfnd": 4, "tv": 6}[k]] = v
        return _scores([tuple(r) for r in rows])

    # |순매수| 는 크게 두어 목록 상단에 남긴다 — 작게 만들면 누적 모집단에서
    # 먼저 빠져 나가, 부호 관문을 지워도 검사가 통과한다(실측으로 밟았다).
    assert broken(0, inst=-1200.0)["A"] is None, "순매수 ≤ 0 인데 점수가 붙었다"
    assert broken(0, invtrt=-1.0)["A"] is None, "투신 < 0 인데 점수가 붙었다"
    assert broken(0, penfnd=-1.0)["A"] is None, "연기금 < 0 인데 점수가 붙었다"
    assert broken(0, tv=50.0)["A"] is None, "거래대금 < 100억인데 점수가 붙었다"
    # 남은 셋은 그대로 매겨진다 — 한 줄이 빠져도 화면이 죽지 않는다.
    assert all(broken(0, invtrt=-1.0)[c] is not None for c in ("B", "C", "D"))


def test_pick_score_stops_at_the_eighty_percent_marker():
    """`-` 마커 **아래** 행에는 점수가 없다 — 모집단이 거기서 끝난다.

    그 아래는 순매수가 사실상 0 이라 순위에 넣으면 노이즈가 분위를 밀어낸다.
    마커가 붙은 행은 **포함**한다 — 도움말이 그 줄까지를 "이 섹터를 움직인
    종목" 이라 부르므로, 화면의 가로줄과 점수의 경계가 같아야 한다.
    """
    from kr_quant.tui.flow_view import State

    # 앞 하나가 압도적이라 두 번째 줄에서 80% 를 넘긴다 → 3·4번째는 마커 아래.
    rows = [("A", "가", 5000.0, 500.0, 300.0, 2000.0, 9000.0),
            ("B", "나", 3000.0, 400.0, 200.0, 1000.0, 8000.0),
            ("C", "다", 200.0, 100.0, 50.0, 80.0, 3000.0),
            ("D", "라", 100.0, 60.0, 30.0, 40.0, 2000.0)]
    st = State(_pick_payload(rows))
    st.row = 0
    got = {t["code"]: (t.get("cut"), t.get("pick")) for t in st.names()}
    marked = [c for c, (cut, _) in got.items() if cut]
    assert len(marked) == 1, f"마커는 한 줄에만: {got}"
    seen_marker = False
    for code, (cut, pick) in got.items():
        if seen_marker:
            assert pick is None, f"{code} 는 마커 아래인데 점수가 있다: {got}"
        seen_marker = seen_marker or cut
    assert got[marked[0]][1] is not None, f"마커가 붙은 줄에 점수가 없다: {got}"


def test_pick_score_gives_ties_the_same_rank():
    """동률은 **같은 순위**를 받는다 — `1년[%ile]` 과 같은 규약.

    안 그러면 순서가 입력 순서에 따라 임의로 갈리고, 세 축이 같은 두 종목이
    그 축들에서 0 과 1 을 나눠 가져 네 번째 축의 차이가 묻힌다.
    """
    from kr_quant.tui.flow_view import State

    rows = [("A", "가", 1000.0, 500.0, 300.0, 400.0, 5000.0),
            ("B", "나", 1000.0, 500.0, 300.0, 400.0, 5000.0)]
    st = State(_pick_payload(rows))   # 네 축이 완전히 같다
    st.row = 0
    got = {t["code"]: t["pick"] for t in st.names()}
    assert got["A"] == got["B"], f"같은 값인데 순위가 갈렸다: {got}"


def test_pick_score_puts_the_name_that_has_not_moved_above_the_one_that_has():
    """상대수익만 **부호를 뒤집는다** — 섹터만큼 안 간 쪽이 위여야 한다.

    다른 세 축을 똑같이 두고 상대수익만 갈라 놓으면, 뒤집기를 빼먹었을 때
    순서가 정확히 반대가 된다.
    """
    from kr_quant.tui.flow_view import State

    rows = [("A", "가", 1000.0, 500.0, 300.0, 400.0, 5000.0),
            ("B", "나", 1000.0, 500.0, 300.0, 400.0, 5000.0)]
    d = _pick_payload(rows)
    # 섹터 수익률 0 이므로 종목 수익률이 그대로 상대수익이 된다.
    d["names"]["A"]["win"]["20"]["ret"] = -5.0    # 덜 갔다 → 위여야 한다
    d["names"]["B"]["win"]["20"]["ret"] = +30.0   # 이미 갔다
    st = State(d)
    st.row = 0
    got = {t["code"]: t["pick"] for t in st.names()}
    assert got["A"] > got["B"], f"이미 간 종목이 위다: {got}"


def test_pick_score_leads_the_row_like_the_sector_markers_do():
    """결론을 **맨 앞**에 둔다 — 섹터 표의 `*`·`~` 가 섹터 이름 옆에 서는 자리다.

    이 화면은 "고르는" 화면이라 눈이 먼저 후보를 좁히고 그 다음 오른쪽에서
    근거를 확인한다. 오른쪽 끝에 두었더니 폭 123 부터 보여서 대부분의 터미널에서
    없는 열이었다(실측).
    """
    from kr_quant.tui.flow_view import names_cols

    heads = [c.header for c in names_cols()]
    assert heads.index("선정") == 2, heads       # 종목 · 코드 · 선정
    assert heads.index("선정") < heads.index("순매수[억]"), heads

def test_pick_score_does_not_change_when_you_sort_by_it():
    """선정점수는 **화면 정렬에 안 흔들린다** — 자기 정렬로 자기를 지우면 안 된다.

    모집단을 `cum`/`cut` 에서 읽으면, 그 둘이 순매수 정렬에서만 채워지므로
    `선정점수` 로 정렬하는 순간 전 행이 `—` 가 된다. 점수는 그 종목이 섹터 안에서
    갖는 성질이지 지금 무엇으로 줄세웠나가 아니다.
    """
    from kr_quant.tui.flow_view import State

    st = State(_pick_payload(_PICK_ROWS))
    st.row = 0
    by_flow = {t["code"]: t.get("pick") for t in st.names()}
    assert any(v is not None for v in by_flow.values()), by_flow
    from kr_quant.tui.flow_view import NAME_SORTS

    for _ in range(len(NAME_SORTS)):
        st.cycle("ns", 1)
        got = {t["code"]: t.get("pick") for t in st.names()}
        assert got == by_flow, (
            f"'{st.name_sort}' 정렬에서 점수가 달라졌다: {got} vs {by_flow}")


def test_the_cross_sector_list_multiplies_instead_of_averaging():
    """층 **사이**는 곱이다 — 평균이면 죽은 섹터의 1등이 좋은 섹터의 3등을 이긴다.

    각 점수 **안에서**는 축 하나가 낮아도 나머지가 메우는 것이 맞다(그래서
    평균이다). 층 사이에서는 아니다. 사용자의 질문이 "섹터도 높고 종목도 높은
    것" 이므로 AND 를 뜻하는 연산이어야 한다.
    """
    from kr_quant.tui.flow_view import State

    st = State(_pick_payload(_PICK_ROWS))
    rows = st.all_picks()
    assert rows, "곱 목록이 비었다 — 픽스처가 관문을 못 넘는다"
    for t in rows:
        assert abs(t["both"] - t["gsec"] * t["pick"]) < 1e-12, t
    # 내림차순 고정
    assert [t["both"] for t in rows] == sorted((t["both"] for t in rows), reverse=True)
    # 양쪽 다 있어야 한다 — 한쪽이라도 없으면 목록에 안 나온다
    assert all(t.get("gsec") is not None and t.get("pick") is not None for t in rows)


def test_the_cross_sector_list_drops_a_stock_whose_sector_failed_its_gate():
    """섹터가 관문에 걸리면 그 안의 종목은 아무리 좋아도 안 나온다."""
    import copy

    from kr_quant.tui.flow_view import State

    d = _pick_payload(_PICK_ROWS)
    base = {t["code"] for t in State(d).all_picks()}
    assert base, base
    killed = copy.deepcopy(d)
    for key, B in killed["blocks"].items():
        for r in B["rows"]:
            r["G"] = None          # 섹터 점수 없음 = 관문 탈락
    assert State(killed).all_picks() == [], "섹터가 탈락했는데 종목이 남았다"


def test_every_footer_tier_that_names_enter_also_names_the_all_stocks_screen():
    """`Enter:종목` 을 적는 단계는 `t:전종목` 도 적는다.

    회귀 — 새 화면을 만들고 푸터 **한 단계에만** 넣었다. 넓은 터미널은 다른
    단계를 쓰므로, 정작 화면이 넓은 사람에게 그 키가 안 보였다. 이 코드베이스는
    "듣는데 화면 어디에도 안 적혀 있다" 를 버그로 취급해 왔다.
    """
    from kr_quant.tui.flow_view import FOOTER_TIERS

    for tier in FOOTER_TIERS:
        if "종목" in tier and "Enter" in tier:
            assert "t:전종목" in tier, f"이 단계에 t 가 없다: {tier}"


def test_the_cross_sector_title_says_which_windows_it_multiplied(data):
    """종합에서는 **두 층이 다른 구간을 본다** — 제목이 그것을 적어야 한다.

    섹터 점수는 종합축(세 창의 등가중 평균)인데 종목 점수는 그 섹터의 20일
    목록에서 나온다. 예전엔 `20일 기준` 이라고만 적어서 섹터 쪽도 20일인 것처럼
    읽혔다. 곱이 뜻을 가지려면 무엇과 무엇을 곱했는지가 화면에 있어야 한다.
    """
    from kr_quant.tui.flow_view import State, all_lines

    st = State(data)
    while st.window != "종합":
        st.cycle("w", 1)
    title = all_lines(st, 200)[0][0]
    assert "섹터 종합" in title and "종목 20일" in title, title
    # 일반 구간에서는 한 구간뿐이라 그 말을 안 붙인다 — 없는 구분을 만들지 않는다.
    st.cycle("w", 1)
    assert st.window != "종합"
    assert "섹터 종합" not in all_lines(st, 200)[0][0]


def test_the_cross_sector_title_never_loses_what_it_multiplied(data):
    """제목은 어느 폭에서도 **곱한 대상**을 잃지 않는다.

    한 줄 고정이면 좁은 화면에서 `pad` 가 뒤를 자른다 — "섹터 종합(20·60·120일)
    × 종목 20일" 이 "…섹터 종합(20·60" 으로 끝나면 무엇을 곱했는지가 오히려
    잘못 읽힌다. 잘린 설명은 설명이 아니다.
    """
    from kr_quant.tui.flow_view import State, all_lines, cell_len, view_width

    st = State(data)
    while st.window != "종합":
        st.cycle("w", 1)
    for w in range(40, 160):
        title = all_lines(st, view_width(w))[0][0]
        assert cell_len(title) <= view_width(w), (w, title)
        assert "…" not in title, (w, title)
        # 두 층을 곱했다는 사실은 어느 단계에서도 남는다.
        assert "×" in title, f"폭 {w} 에서 곱한 대상이 사라졌다: {title!r}"


def test_the_html_report_calls_the_score_the_same_name_as_the_tui():
    """HTML 리포트와 TUI 가 **같은 값을 같은 이름**으로 불러야 한다.

    회귀 — 섹터 점수를 `G[0~1]` → `선정` 으로 바꾸면서 TUI 만 고치고
    `scripts/templates/sector_numbers.html` 을 안 고쳤다. 같은 리포트를 두 도구가
    다른 이름으로 부르면, 화면에서 배운 이름으로 HTML 을 찾을 수 없다.
    """
    from pathlib import Path

    from kr_quant.tui.flow_view import SORT_COL

    tpl = Path(__file__).resolve().parents[1] / "scripts" / "templates" / "sector_numbers.html"
    html = tpl.read_text(encoding="utf-8")
    assert f'"{SORT_COL["G"]}"' in html, (
        f"HTML 표의 열 이름이 TUI 의 '{SORT_COL['G']}' 와 다르다")
    assert "성장 G" not in html, "옛 이름이 남아 있다"
