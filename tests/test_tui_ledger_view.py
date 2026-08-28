"""자금 원장 렌더 — 표시 폭 불변식 + **설계 규칙의 회귀**.

이 저장소의 교훈(GUARDRAILS §4-5): "구현했다"와 "작동한다"는 다르고, 통과만 보는
테스트는 규칙이 죽어도 초록이다. 그래서 여기 있는 검사는 전부 **위반을 주입하면
실제로 실패하도록** 짜여 있다. 각 검사의 독스트링에 "무엇을 주입하면 깨지나"를 적는다.

지키려는 규칙은 셋이다.

1. 표시 폭 — 한글 2칸·U+2212 1칸. 열이 한 칸이라도 밀리면 잡는다.
2. **잔여를 숨기지 않는다** — 화면의 4주체 합의 부호를 뒤집은 값이 잔여 열과 같아야
   한다. 잔여를 4주체에 안분해 0 으로 만들면(설계 §2.3 이 금지한 것) 깨진다.
3. **부호는 색이 아니라 글자에 있다** — 히트맵에서 색을 빼도 부호가 남아야 한다.
"""

from __future__ import annotations

import pytest

from kr_quant.tui.flow_view import cell_width
from kr_quant.tui.ledger_view import (
    ACTORS, ACTOR_KEYS, BANNER, LIMITS, SORTS, VIEWS, WINDOWS, Model,
    col_span, heat_cell, heat_level, ledger_cols, ledger_lines, comove_lines,
    residual,
    LEDGER_MIN_W, SPARK, THIN_N, TIMELINE_MIN_W, _fit, downsample, load, render_text,
    screen, signed_bar, spark, status_line, timeline_lines,
)


def _w(text: str) -> int:
    return sum(cell_width(c) for c in text)


@pytest.fixture
def data():
    """합성 페이로드 — 실제 리포트 폴더에 의존하지 않는다.

    4주체 합이 정확히 0 이 **아니게** 만든다(잔여가 0 이면 규칙 2 를 검사할 수 없다).
    """
    n = 40
    dates = [f"2026-0{1 + i // 28}-{1 + i % 28:02d}" for i in range(n)]
    secs = ["전기/전자", "운송장비/부품", "부동산", "출판/매체복제"]
    mkts = ["거래소", "코스닥"]

    def cell(seed: int):
        out = {}
        for j, k in enumerate(ACTOR_KEYS):
            out[k] = [round((-1) ** (i + j) * (seed + i * (j + 1)) * 1.5, 2)
                      for i in range(n)]
        out["tv"] = [1000.0] * n
        # 4주체 합을 일부러 0 에서 띄운다 — 미분류 주체가 있는 실제 데이터처럼.
        out["indiv"] = [v + 3.25 for v in out["indiv"]]
        return out

    return {
        "dates": dates, "sectors": secs, "markets": mkts,
        "flows": {m: {s: cell(1 + i * 7 + len(m)) for i, s in enumerate(secs)}
                  for m in mkts},
        "cap": {m: {s: 10000.0 * (i + 1) for i, s in enumerate(secs)} for m in mkts},
        # 부동산·출판은 일부러 **얇게** 둔다 — 얇은 섹터가 하나도 없으면
        # ~ 표시 회귀가 아무것도 안 지킨다(거래소/부동산은 실제로 3종목이다).
        "n_by_sector": {m: {"전기/전자": 200, "운송장비/부품": 70,
                            "부동산": 1, "출판/매체복제": 2} for m in mkts},
        "n_names": 100, "finalized": True,
    }


# ------------------------------------------------------------------ 표시 폭

def test_every_line_has_identical_display_width(data):
    """회귀 — 어떤 화면·어떤 조작 조합에서도 모든 행의 표시 폭이 같다.

    주입해 보라: ``pad(v, c[1], c[2])`` 를 ``str(v)`` 로 바꾸면 실패한다.
    """
    mo = Model(data)
    for width in (80, 100, 132):
        for vi in range(len(VIEWS)):
            mo.vi = vi
            for wi in range(len(WINDOWS)):
                mo.wi = wi
                for ai in range(len(ACTORS)):
                    mo.ai = ai
                    s = screen(mo, width, 40)
                    for line in s["lines"]:
                        assert _w(line) == width, (VIEWS[vi][0], width, repr(line))


def _cells(line: str) -> list[str]:
    """표시 칸 → 그 칸을 차지한 문자. 한글은 두 칸을 차지하므로 두 번 들어간다."""
    out = []
    for c in line:
        out += [c] * cell_width(c)
    return out


def test_numeric_columns_start_at_the_same_display_cell(data):
    """열이 한 칸이라도 밀리면 잡는다 — **총 폭이 같은 것으로는 안 잡힌다**.

    이 저장소가 두 번 겪은 실수가 정확히 이것이다(헤더와 셀 개수 불일치). 줄 전체를
    ``pad(line, width)`` 로 감싸면 총 폭은 늘 맞으므로, 총 폭 검사는 통과하면서
    열은 밀린다. 그래서 **열 경계 칸이 공백인지**를 본다.

    주입해 보라: ``ledger_lines`` 의 ``pad(v, c[1], c[2])`` 를 ``str(v)`` 로 바꾸면
    실패한다.
    """
    mo = Model(data)
    for width in (72, 80, 100, 132):
        # 렌더가 _fit 으로 열을 떨어뜨리므로 검사도 **같은 목록**을 봐야 한다.
        # ledger_cols() 만 보면 렌더가 안 그리는 열의 경계를 검사하게 된다.
        cols = _fit(ledger_cols(), width)
        lines, _t, nh = ledger_lines(mo, width)
        bounds = [col_span(cols, c[0])[0] for c in cols[1:]]
        for line in lines:
            cells = _cells(line)
            for b in bounds:
                if b - 1 < len(cells):
                    assert cells[b - 1] == " ", (width, b, repr(line))


def test_headers_are_not_truncated_by_their_column_width(data):
    """헤더가 자기 열 폭에 들어가야 한다.

    ``기타법인[억]`` 은 표시 폭 12 다. 주입해 보라: ``_col`` 의
    ``max(want, _w(name))`` 를 ``want`` 로 바꾸면 ``기타법인[`` 로 잘려 실패한다.
    """
    for name, w, _r in ledger_cols():
        assert _w(name) <= w, (name, _w(name), w)
    # 정의만이 아니라 **그려진 헤더**에도 온전히 남아야 한다. 스파크라인 열처럼
    # 폭이 데이터에 따라 정해지는 열은 정의 검사만으로는 안 잡힌다.
    mo = Model(data)
    import re
    for width in range(TIMELINE_MIN_W, 140):
        head = timeline_lines(mo, width)[0][0].rstrip()
        tail = head.split("진폭[억]")[-1].strip()
        assert re.fullmatch(r"(누적추이\[\d+점\]|추이\[\d+\]|추이|)", tail), (
            width, repr(tail))


def test_minus_sign_and_blocks_are_one_cell():
    """막대 글자와 부호가 폭 계산에서 어긋나지 않는다."""
    for ch in "█▌▐▁▂▃▄▅▆▇·░▒▓":
        assert cell_width(ch) == 1, ch
    assert cell_width("−") == 1        # U+2212 는 1칸이다


@pytest.mark.parametrize("half", [0, 1, 6, 12])
def test_signed_bar_width_is_constant(half):
    """막대는 값이 무엇이든 폭이 ``2*half`` 다 — 아니면 뒤 열이 밀린다.

    주입해 보라: 양수 가지의 ``pad(bar, half)`` 를 ``bar`` 로 바꾸면 실패한다.
    """
    for v in (0.0, 1.0, -1.0, 5.0, -5.0, 1e9, -1e9, 0.4, -0.4):
        assert _w(signed_bar(v, 5.0, half)) == 2 * half, (v, half)


def test_signed_bar_points_the_right_way():
    """양수는 오른쪽, 음수는 왼쪽. 반칸 꼬리도 바깥쪽을 향한다."""
    pos = signed_bar(5.0, 5.0, 4)
    neg = signed_bar(-5.0, 5.0, 4)
    assert pos[:4] == "    " and pos[4:] == "████"
    assert neg[:4] == "████" and neg[4:] == "    "
    # 반칸: 양수 꼬리는 `▌`(칸의 왼쪽 절반), 음수 꼬리는 `▐`(칸의 오른쪽 절반).
    assert signed_bar(5.0 * 3 / 8, 5.0, 4).strip() == "█▌"
    assert signed_bar(-5.0 * 3 / 8, 5.0, 4).strip() == "▐█"


# ------------------------------------------------------- 규칙 2: 잔여를 숨기지 않는다

def test_residual_is_minus_sum_of_four_actors(data):
    """``잔여 = −Σ(4주체)``. 주입해 보라: ``residual`` 의 부호를 떼면 실패한다."""
    cell = data["flows"]["거래소"]["전기/전자"]
    for i in (0, 7, 39):
        assert residual(cell, i) == pytest.approx(-sum(cell[k][i] for k in ACTOR_KEYS))


def test_ledger_row_residual_matches_the_four_numbers_on_screen(data):
    """**화면에 찍힌** 네 숫자의 합과 잔여 열이 맞아야 한다.

    이게 설계 §2.3 의 회귀다. 주입해 보라: ``Model.rows`` 에서 잔여를 4주체에
    안분해(예: ``r[k] += r["resid"]/4; r["resid"] = 0``) 0 으로 만들면 실패한다.
    미분류 주체가 측정된 주체의 옷을 입는 것을 여기서 잡는다.
    """
    mo = Model(data)
    for r in mo.rows():
        assert r["resid"] == pytest.approx(-sum(r[k] for k in ACTOR_KEYS), abs=1e-6)
        # 그리고 실제로 0 이 아니다 — 0 이면 이 검사는 아무것도 안 지킨다.
    assert any(abs(r["resid"]) > 1e-6 for r in mo.rows())


def test_residual_and_four_actors_are_all_on_screen_or_none_are(data):
    """**그려진 줄**에 4주체와 잔여가 함께 있거나, 아예 표를 안 그리거나 둘 중 하나다.

    한때 이 검사가 ``ledger_cols()`` 만 봤다. 그런데 렌더는 그 목록을 ``_fit`` 으로
    한 번 더 줄이므로, 정의에 잔여가 있어도 **화면에는 없을 수** 있었다 — 실제로
    폭 60 에서 잔여가 조용히 사라졌는데 검사는 초록이었다. 검사는 정의가 아니라
    사용자가 보는 것을 봐야 한다.

    주입해 보라: ``ledger_lines`` 의 ``width < LEDGER_MIN_W`` 분기를 지우면
    폭 60 에서 4주체 일부만 남은 표가 그려져 실패한다.
    """
    mo = Model(data)
    need = [f"{ko}[억]" for _, ko in ACTORS] + ["잔여[억]"]
    for width in (40, 60, 70, LEDGER_MIN_W, 80, 96, 104, 132):
        head = ledger_lines(mo, width)[0][0]
        if width < LEDGER_MIN_W:
            assert "안 그린다" in head, (width, head)
            assert not any(n in head for n in need), (width, head)
        else:
            for n in need:
                assert n in head, (width, n, head)


def test_narrow_ledger_says_why_instead_of_showing_part_of_it(data):
    """부분 표 대신 이유를 말한다 — 검산할 수 없는 숫자는 안 보이는 게 낫다."""
    lines, thin, nh = ledger_lines(Model(data), 60)
    assert nh == len(lines) == len(thin) == 2
    assert "−Σ4주체" in lines[1]


def test_thin_sector_warning_survives_without_color(data):
    """얇은 섹터 경고(~)가 글자로 남는다.

    색(A_DIM/파랑)에만 실으면 무색 터미널·색맹·``--dump`` 에서 통째로 사라진다.
    ``flow_view`` 가 같은 이유로 ~ 글리프를 넣었다.

    주입해 보라: 표시 문자열을 ``""`` 로 되돌리면(색에만 의존) 실패한다.
    """
    mo = Model(data)
    lines, thin, nh = ledger_lines(mo, 132)
    marked = [ln for ln in lines[nh:] if "~" in ln]
    assert marked, "얇은 섹터가 글자로 표시되지 않았다"
    # 표시된 줄과 thin 플래그가 같은 행을 가리켜야 한다 — 어긋나면 엉뚱한 섹터에
    # 경고가 붙는다.
    for i, ln in enumerate(lines[nh:], start=nh):
        assert ("~" in ln) == thin[i], (i, ln, thin[i])
    assert any(r["n"] < THIN_N for r in mo.rows()), "얇은 섹터가 없으면 검사가 무의미하다"


def test_columns_are_dropped_at_boundaries_not_mid_number(data):
    """폭이 모자라면 열 경계에서 떨어뜨린다 — -1,360 이 -1 로 보이면 안 된다.

    ``flow_view._fit`` 를 재사용한다. 주입해 보라: ``_fit(...)`` 을 벗겨
    ``ledger_cols()`` 를 그대로 쓰면, 잘린 자리에서 숫자가 자릿수 중간에서
    끊겨 이 검사가 실패한다.
    """
    mo = Model(data)
    # 폭을 **훑는다**. 처음엔 다섯 개만 골라 봤는데 전부 열 경계에 딱 떨어지는
    # 폭이라, _fit 을 통째로 지워도 검사가 초록이었다. 잘림은 경계에 걸치는 폭
    # 에서만 보인다 — 표본이 아니라 전수를 봐야 한다.
    for width in range(LEDGER_MIN_W, 140):
        cols = _fit(ledger_cols(), width)
        end = sum(c[1] + 1 for c in cols) - 1
        assert end <= width, (width, end)
        lines, _t, nh = ledger_lines(mo, width)
        for ln in lines[nh:]:
            tail = "".join(_cells(ln)[end:])
            assert not any(ch.isdigit() for ch in tail), (width, end, repr(ln))


def test_sparkline_is_never_truncated(data):
    """스파크라인 칸 수가 헤더가 광고한 점 수와 **정확히** 같아야 한다.

    잘리면 구간의 일부만 그려 놓고 전부인 척한다 — 자릿수가 잘린 숫자보다 더
    나쁘다. 숫자는 이상해 보이기라도 하지만 짧은 스파크라인은 멀쩡해 보인다.

    주입해 보라: ``sw = min(mo.window, width - used)`` 를 예전처럼
    ``max(8, min(mo.window, width - used - 1))`` 로 되돌리면 폭 48~56 에서
    줄이 폭을 넘어 pad 가 뒤를 잘라내고 실패한다.
    """
    import re
    mo = Model(data)
    # 가드(TIMELINE_MIN_W) 아래까지 훑는다 — 가드를 지우면 거기서 잘린다.
    for width in range(30, 200):
        lines, _t, nh = timeline_lines(mo, width)
        if width < TIMELINE_MIN_W:
            assert nh == len(lines) == 2, (width, lines)
            assert not any(c in SPARK for ln in lines for c in ln), (width, lines)
            continue
        # 헤더는 점 수를 **온전한 토큰으로** 광고해야 한다. 폭에 안 맞는 이름을
        # 그대로 쓰면 "누적추이" 로 잘려 광고가 사라진다.
        m = re.search(r"추이\[(\d+)", lines[0])
        assert m, (width, repr(lines[0]))
        want = int(m.group(1))
        for ln in lines[nh:]:
            got = sum(1 for c in ln if c in SPARK)
            assert got == want, (width, want, got, repr(ln))


def test_narrow_timeline_says_why_instead_of_a_clipped_sparkline(data):
    lines, thin, nh = timeline_lines(Model(data), TIMELINE_MIN_W - 1)
    assert nh == len(lines) == len(thin) == 2
    assert "전부인 척" in lines[1]


def test_market_order_is_fixed_not_hash_order(data):
    """"m 두 번 = 코스닥" 손버릇이 조용히 깨지면 안 된다."""
    d = dict(data)
    d["markets"] = ["코스닥", "거래소"]        # 파일 순서가 뒤집혀 와도
    assert Model(d).markets == ["전체", "거래소", "코스닥"]


def test_load_says_what_is_missing_instead_of_a_traceback(tmp_path):
    """회귀 — 형식이 바뀐 페이로드는 curses 안에서 생 트레이스백으로 터졌다."""
    (tmp_path / "payload.json").write_text('{"dates": []}', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        load(str(tmp_path))
    for k in ("dates", "sectors", "markets", "flows"):
        assert k in str(e.value)
    (tmp_path / "payload.json").write_text("{nope}", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        load(str(tmp_path))
    assert "JSON" in str(e.value)


# --------------------------------------------------- 규칙 3: 부호는 글자에 있다

def test_heatmap_sign_survives_without_color():
    """색을 못 쓰는 터미널·평문 덤프에서도 부호가 남는다.

    주입해 보라: ``heat_cell`` 이 ``"██"`` 를 돌려주게 하면(색에만 의존) 실패한다.
    """
    assert heat_cell(0.9).startswith("+")
    assert heat_cell(-0.9).startswith("-")
    assert heat_cell(0.0).strip() == ""
    for c in (-1.0, -0.3, 0.0, 0.3, 1.0):
        assert _w(heat_cell(c)) == 2, c


def test_heat_level_is_monotone_and_clamped():
    assert heat_level(1.0) == 5 and heat_level(-1.0) == -5
    assert heat_level(0.0) == 0
    prev = -6
    for c in [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]:
        lv = heat_level(c)
        assert lv >= prev
        prev = lv


def test_comove_reports_the_null_next_to_the_observation(data):
    """동시성 화면은 관측과 **널**을 나란히 찍는다 — 무늬만 보여주지 않는다.

    주입해 보라: ``comove_lines`` 의 첫 줄에서 널 항을 지우면 실패한다.
    """
    mo = Model(data)
    mo.vi = [v for v, _ in VIEWS].index("comove")
    lines, marks, head = comove_lines(mo, 132)
    assert "널" in lines[0] and "관측" in lines[0]
    assert "옮겨갔다" in lines[1]        # 해석과 사실을 구분하는 문장이 본문에 있다
    assert marks, "히트맵 색칠 지시가 비어 있다"


def test_short_window_refuses_to_show_correlations(data):
    """5일짜리 구간에서는 상관을 내지 않는다 — 노이즈를 무늬로 보여주지 않는다."""
    mo = Model(data)
    mo.wi = WINDOWS.index(5)
    lines, marks, _h = comove_lines(mo, 100)
    assert marks == []
    assert "상관을 내지 않는다" in lines[0]


# ---------------------------------------------------------------- 그 외 회귀

def test_downsampling_keeps_the_head_of_the_series():
    """구간을 줄일 때 **앞을 잘라내면** 안 된다 — 누적이 시작점을 잃는다.

    주입해 보라: ``downsample`` 을 ``values[-n:]`` 로 바꾸면 첫 값이 3 이 아니라
    30 이 되어 실패한다. 그 버그는 화면에서 "최근에야 움직였다"로 보인다.
    """
    got = downsample(list(range(40)), 10)
    assert len(got) == 10
    assert got[0] == 3 and got[-1] == 39      # 각 묶음의 끝점
    assert got[0] != 30, "뒤 n개만 남기고 있다 — 앞이 잘렸다"
    assert downsample([1.0, 2.0], 10) == [1.0, 2.0]
    assert downsample([], 5) == []


def test_timeline_uses_the_full_window_not_just_the_tail(data):
    mo = Model(data)
    mo.wi = WINDOWS.index(60)          # 구간(40일) > 스파크라인 칸(좁은 폭)
    lines, thin, nh = timeline_lines(mo, 60)
    assert nh == 2 and len(lines) == nh + len(mo.sectors)
    assert len(thin) == len(lines)


def test_timeline_shows_amplitude_so_rows_are_comparable(data):
    """행마다 세로 눈금이 다르다는 걸 숫자가 알려줘야 한다(진폭 열)."""
    mo = Model(data)
    lines, _t, nh = timeline_lines(mo, 120)
    assert "진폭[억]" in lines[0]
    # 경고는 스파크라인 열 폭에 밀어 넣지 않고 **별도 헤더 줄**에 둔다 — 짧은 구간에서
    # 잘려 사라지면 안 된다. 주입해 보라: 다시 첫 줄 뒤에 붙이면 20일 구간에서 잘린다.
    assert nh == 2
    for width in (80, 100, 200):
        assert "세로 눈금은 행마다 다르다" in timeline_lines(mo, width)[0][1]


def test_spark_is_flat_when_the_series_is_flat():
    assert len(set(spark([3.0] * 5))) == 1
    assert spark([]) == ""
    assert len(spark([1.0, 2.0, 3.0])) == 3


def test_banner_is_on_screen_not_in_a_footnote(data):
    """한계는 각주가 아니라 사용자가 읽는 자리에 있어야 한다."""
    mo = Model(data)
    assert "미관측" in BANNER and "섹터→섹터" in BANNER
    # 배너는 **모든 화면**의 마지막 줄이다. 주입해 보라: screen() 에서 배너를 빼고
    # 앱이 붙이게 되돌리면, 순수 함수로는 못 잡아 이 검사가 실패한다.
    for vi in range(len(VIEWS)):
        mo.vi = vi
        s = screen(mo, 120, 30)
        assert BANNER in s["lines"][s["banner_y"]], VIEWS[vi][0]
        assert len(s["lines"]) == 30
    mo.vi = [v for v, _ in VIEWS].index("limits")
    body = screen(mo, 100, 40)["lines"]
    assert any("돈에 꼬리표가 없다" in line for line in body)
    assert any("주변합" in line for line in LIMITS)
    # 원장 화면의 상태줄은 커서가 놓인 칸의 전체 수치를 보여준다(툴팁의 등가물).
    mo.vi = 0
    assert "잔여" in status_line(mo, 200)


def test_limits_screen_names_both_refusals(data):
    """거부는 둘이다 — 섹터→섹터와 주체→주체. 하나만 적으면 나머지를 잊는다."""
    text = "\n".join(LIMITS)
    assert "섹터 A" in text and "섹터 B" in text
    assert "주체 → 주체" in text and "C(4,2)=6" in text


def test_all_control_combinations_render(data):
    mo = Model(data)
    for vi in range(len(VIEWS)):
        mo.vi = vi
        for si in range(len(SORTS)):
            mo.si = si
            for mi in range(len(mo.markets)):
                mo.mi = mi
                s = screen(mo, 100, 30)
                assert s["lines"]


def test_empty_flows_do_not_crash(data):
    d = dict(data)
    d["flows"] = {m: {} for m in data["markets"]}
    mo = Model(d)
    for vi in range(len(VIEWS)):
        mo.vi = vi
        assert screen(mo, 100, 30)["lines"]


def test_render_text_covers_every_view_and_restores_state(data):
    mo = Model(data)
    mo.vi = 1
    out = render_text(mo, 100)
    for _v, ko in VIEWS:
        assert f"### {ko}" in out
    assert mo.vi == 1, "render_text 가 화면 상태를 되돌리지 않았다"


def _cw(text: str) -> int:
    """표시 칸 수. (이 파일의 `_cells` 는 이미 다른 뜻으로 쓰인다.)"""
    from kr_quant.tui.flow_view import cell_width
    return sum(cell_width(c) for c in text)


def test_dump_respects_the_requested_width(data):
    """회귀 — `--dump` 만 폭을 안 지켰다. curses 경로는 14,800 조합이 완벽한데
    평문 경로는 `LIMITS` 를 그대로 이어붙여 폭 80(터미널 기본)에서도 넘쳤다.

    `--dump` 는 SSH 밖으로 내보내는 유일한 경로다. 그리고 그 검사는 `--dump`
    만 주고 **`--width` 를 한 번도 안 줬다** — 폭을 무시해도 초록이었다.
    """
    from kr_quant.tui.ledger_view import render_text

    mo = Model(data)
    for width in (40, 60, 80, 100, 132, 200):
        for line in render_text(mo, width).splitlines():
            assert _cw(line.rstrip()) <= width, (
                f"폭{width} 요청인데 {_cw(line.rstrip())}칸: {line.rstrip()!r}")


def test_limits_prose_is_wrapped_not_dropped():
    """한계 문단은 **표를 대신하는 문장**이라 잘리면 뜻이 바뀐다.

    curses 경로는 `pad` 로 잘라 문장을 버렸고 평문 경로는 넘쳤다 — 같은 글을
    두 경로가 다르게 다뤘다. 이제 둘 다 `limits_body` 를 쓴다.
    """
    from kr_quant.tui.ledger_view import LIMITS, limits_body

    want = "".join(LIMITS).replace(" ", "").replace("**", "")
    for width in (40, 80, 120):
        body = limits_body(width)
        assert all(_cw(ln) <= width for ln in body), f"폭{width} 에서 넘친다"
        assert "".join(body).replace(" ", "") == want, (
            f"폭{width} 에서 문장이 사라지거나 늘었다")


def test_banner_never_loses_the_unobserved_half():
    """회귀 — 배너를 잘라내면 '미관측' 이 사라져 **관측된 것만 남는다.**
    이 화면이 존재하는 이유가 그 경고라, 좁다고 없앨 수 없다."""
    from kr_quant.tui.ledger_view import banner_for

    for width in range(20, 130, 2):
        got = banner_for(width)
        assert "미관측" in got, f"폭{width} 배너에서 미관측이 사라졌다: {got!r}"
        assert _cw(got) + 1 <= width, f"폭{width} 에서 배너가 넘친다: {got!r}"


def test_no_screen_shows_markdown_asterisks(data):
    """소스의 **강조** 는 읽는 사람 눈에 띄라고 쓴 표기지 화면에 나갈 글자가
    아니다. `--dump` 뿐 아니라 **모든 화면**을 본다 — 한 곳만 검사하면 다음
    문구가 다른 화면으로 새어 나온다(실제로 동시성 헤더에 남아 있었다)."""
    from kr_quant.tui.ledger_view import LIMITS, render_text, screen

    assert any("**" in t for t in LIMITS), "소스에 강조가 없다 — 검사가 헛돈다"
    mo = Model(data)
    assert "**" not in render_text(mo, 100)
    for width in (60, 100, 160):
        for vi in range(len(VIEWS)):
            mo.vi = vi
            for detrend in (False, True):
                mo.detrend = detrend
                # screen() 은 **dict** 를 돌려준다. 리스트로 가정했더니
                # 빈 문자열을 검사하고 있었다 — 통과했지만 아무것도 안 봤다.
                lines = screen(mo, width, 30)["lines"]
                assert lines, "화면이 비었다 — 검사가 헛돈다"
                assert "**" not in "".join(lines), (
                    f"화면{vi}·폭{width} 에 마크다운 강조가 나온다")
