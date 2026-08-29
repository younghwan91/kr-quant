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

from kr_quant.tui.flow_view import cell_width, span_at
from kr_quant.tui.ledger_view import (
    ACTORS, ACTOR_KEYS, BANNER, LIMITS, SORTS, VIEWS, WINDOWS, Model,
    heat_cell, heat_level, ledger_cols, ledger_lines, comove_lines,
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
        widths = [c[1] for c in cols]
        bounds = [span_at(widths, i)[0] for i in range(1, len(cols))]
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
        tail = head.split("눈금[억]")[-1].strip()
        assert re.fullmatch(r"(누적추이\[\d+점\]|추이\[\d+\]|추이|)", tail), (
            width, repr(tail))


def test_picture_glyphs_are_one_cell_even_when_ambiguous_is_wide(monkeypatch):
    """그림 글자는 **어떤 터미널에서도** 1칸이어야 한다.

    `signed_bar`·`heat_cell` 은 글자가 1칸이라고 **가정하고 칸을 센다**
    (`" " * half`, 칸당 2글자). 그 가정이 깨지면 막대·히트맵이 자기 폭의 두 배를
    먹고 뒤 열을 통째로 민다. 예전 글자표(`█▌▁▂▃·▒▓`)는 EAW 'A' 라 터미널이
    정하는 값이었고, `KQ_AMBIGUOUS_WIDE=1` 을 켜면 실제로 어긋났다.

    ⚠️ `cell_width` 만으로 재면 **모드가 꺼진 기본값에서는 'A' 도 1** 이라
    아무것도 안 잡는다. 모드를 켜고 잰다.

    주입: `SPARK_SIGNED`·`HEAT_RAMP`·`BLOCK_FULL` 중 아무거나 블록 문자로
    되돌리면 실패한다.
    """
    from kr_quant.tui import flow_view
    from kr_quant.tui.ledger_view import BLOCK_FULL, BLOCK_L, BLOCK_R, HEAT_RAMP, SPARK

    monkeypatch.setattr(flow_view, "AMBIGUOUS_WIDE", True)
    assert cell_width("█") == 2, "모드가 안 켜졌다 — 이 검사가 헛돈다"
    for ch in SPARK + HEAT_RAMP.strip() + BLOCK_FULL + BLOCK_L + BLOCK_R:
        assert cell_width(ch) == 1, f"{ch!r} 이 모드에서 2칸이 된다"
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
    from kr_quant.tui.ledger_view import BLOCK_FULL, BLOCK_L, BLOCK_R

    pos = signed_bar(5.0, 5.0, 4)
    neg = signed_bar(-5.0, 5.0, 4)
    assert pos[:4] == "    " and pos[4:] == BLOCK_FULL * 4
    assert neg[:4] == BLOCK_FULL * 4 and neg[4:] == "    "
    # 반칸: 양수 꼬리는 칸의 **왼쪽** 절반, 음수 꼬리는 칸의 **오른쪽** 절반이다.
    assert signed_bar(5.0 * 3 / 8, 5.0, 4).strip() == BLOCK_FULL + BLOCK_L
    assert signed_bar(-5.0 * 3 / 8, 5.0, 4).strip() == BLOCK_R + BLOCK_FULL
    # 그리고 그 두 글자는 **서로 다른 쪽**을 채운다 — 같으면 음수 막대의 꼬리가
    # 한 칸 떨어져 보인다(예전 `▏▎▍…` 여덟 단계가 그랬다).
    assert BLOCK_L != BLOCK_R


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
    assert nh == len(lines) == len(thin) == 3
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
            assert nh == len(lines) == 3, (width, lines)
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
    assert nh == len(lines) == len(thin) == 3
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


def test_timeline_shows_its_scale_so_rows_are_comparable(data):
    """크기 눈금이 행마다 다르다는 걸 숫자가 알려줘야 한다(눈금 열)."""
    mo = Model(data)
    lines, _t, nh = timeline_lines(mo, 120)
    assert "눈금[억]" in lines[0]
    # 경고는 스파크라인 열 폭에 밀어 넣지 않고 **별도 헤더 줄**에 둔다 — 짧은 구간에서
    # 잘려 사라지면 안 된다. 주입해 보라: 다시 첫 줄 뒤에 붙이면 20일 구간에서 잘린다.
    assert nh == 2
    for width in (80, 100, 200):
        assert "가운데가 0" in timeline_lines(mo, width)[0][1]


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


# ---------------------------------------------------------------- 도움말 · 힌트

def _help_raw(entry) -> str:
    """도움말 한 줄을 **자르기 전** 원문으로 조립한다.

    ⚠️ `help_body` 의 출력으로는 잘림을 못 잡는다 — `pad` 가 넘치면 자르고
    모자라면 채워서 **모든 줄이 정확히 width 칸**이 되기 때문이다. flow 쪽
    `test_help_body_is_not_truncated_at_the_default_ssh_width` 와 같은 수법이다.
    """
    from kr_quant.tui.flow_view import pad
    from kr_quant.tui.ledger_view import HELP_LABEL_W

    name, desc = entry
    body = desc.replace("**", "")               # 렌더가 떼는 강조 표기
    return ((pad(name, HELP_LABEL_W, right=True) + "  " + body) if name
            else ("   " + body))


def test_ledger_help_body_is_not_truncated_at_the_default_ssh_width():
    """도움말은 **읽으러 연 화면**이다. 폭 80(SSH 기본)에서 문장이 끊기면
    고칠 곳이 화면에 안 보인다.

    주입: LEDGER_HELP 의 아무 설명에나 몇 글자를 붙이거나 HELP_LABEL_W 를 키우면
    원문이 80칸을 넘어 실패한다.
    """
    from kr_quant.tui.ledger_view import HELP_LABEL_W, LEDGER_HELP

    over = [(_cw(_help_raw(e)), _help_raw(e)) for e in LEDGER_HELP
            if _cw(_help_raw(e)) > 80]
    assert not over, "폭 80 에서 잘리는 도움말 줄: " + repr(over)
    # 라벨도 잘리면 안 된다 — flow 는 지금 `순매수상위[억]` 이 `순매수상위` 로
    # 잘려 있다(라벨 폭 10). 원장은 그걸 반복하지 않는다.
    long_labels = [n for n, _ in LEDGER_HELP if _cw(n) > HELP_LABEL_W]
    assert not long_labels, f"라벨이 {HELP_LABEL_W}칸을 넘어 잘린다: {long_labels}"


def test_help_explains_every_column_the_screens_draw():
    """화면에 보이는 열은 **전부** 도움말에 있어야 한다.

    기대값을 도움말에서 가져오면 동어반복이다. 그래서 열 이름을 **화면을 그리는
    코드**(`ledger_cols`)와 전개 화면의 열에서 뽑는다.

    주입: `LEDGER_HELP` 에서 `최대1일[%]` 항목을 지우면 실패한다. 열을 새로
    추가하고 설명을 안 써도 실패한다 — 그게 이 검사의 목적이다.
    """
    from kr_quant.tui.ledger_view import _SORT_HELP, LEDGER_HELP

    named = {n for n, _ in LEDGER_HELP if n}
    cols = [c[0] for c in ledger_cols() if c[0]]
    cols += ["누적[억]", "눈금[억]", "추이"]     # 전개 화면의 열
    missing = [c for c in cols if c not in named]
    assert not missing, f"열이 화면에 있는데 도움말에 없다: {missing}"
    # 얇은 섹터 마커는 이름이 없는 1칸 열이라 위 목록에 안 잡힌다. flow 도움말에는
    # 있는데 원장에는 없었다 — 같은 글자, 같은 제품, 한쪽만 설명하면 안 된다.
    assert "~" in named, "~ 마커 설명이 없다"
    # 정렬 키도 전부 — `절대크기` 는 열이 없어서 특히 빠지기 쉽다.
    for key, _ko in SORTS:
        if key in _SORT_HELP:
            assert _SORT_HELP[key] in named, f"정렬 {key} 의 설명이 없다"


def test_help_documents_every_key_the_app_handles():
    """앱이 처리하는 글자 키는 전부 도움말에 적혀 있어야 한다.

    기대 목록을 손으로 적으면 키를 늘렸을 때 검사가 조용히 뒤처진다. 그래서
    `ledger_app._key` 의 **소스에서** `ord("x")` 를 긁어 온다.

    주입: `_key` 에 새 키 분기를 넣고 도움말에 안 적으면 실패한다.
    """
    import inspect
    import re

    from kr_quant.tui import ledger_app
    from kr_quant.tui.ledger_view import LEDGER_HELP

    src = inspect.getsource(ledger_app._key)
    keys = {m for m in re.findall(r'ord\("(.)"\)', src)} - {" "}
    assert len(keys) >= 10, f"키를 못 긁었다 — 검사가 헛돈다: {keys}"
    text = "\n".join(f"{n} {d}" for n, d in LEDGER_HELP)
    missing = [k for k in sorted(keys) if k not in text]
    assert not missing, f"앱이 처리하는데 도움말에 없는 키: {missing}"


def test_question_mark_opens_help_and_q_closes_it_without_quitting(data):
    """`?` 는 도움말이고, 거기서 `q` 는 **닫기**지 종료가 아니다.

    키를 배우러 연 화면에서 확인 없이 앱이 끝나면 안 된다 — `kq-flow` 와 같은
    규칙이라야 두 앱에서 다른 손버릇을 안 배운다.

    주입: `_key` 의 `?` 를 예전처럼 `mo.vi = _LIMITS_VI` 로 되돌리면 첫 단언에서,
    도움말 안의 `q` 를 종료로 두면 둘째 단언에서 실패한다.
    """
    from kr_quant.tui.ledger_app import _key

    mo = Model(data)
    assert _key(mo, ord("?"), 10) is True
    assert mo.help, "? 가 도움말을 안 열었다"
    assert mo.view != "limits", "? 가 아직 한계 화면으로 간다"
    # 닫는 키 넷 — flow 와 같은 집합이다(q·Esc·?·Enter).
    for ch in (ord("q"), 27, ord("?"), 10):
        mo.help, mo.help_row = True, 3
        assert _key(mo, ch, 10) is True, f"{ch!r} 이 앱을 끝냈다"
        assert not mo.help, f"{ch!r} 로 도움말이 안 닫혔다"
    # 닫힌 뒤의 q 는 종료다.
    assert _key(mo, ord("q"), 10) is False
    # 도움말 안에서는 화면 키가 **먹지 않는다** — 뒤에서 화면이 몰래 바뀌면
    # 닫았을 때 다른 곳에 서 있다.
    mo.help, before = True, (mo.vi, mo.wi)
    _key(mo, ord("v"), 10)
    _key(mo, ord("w"), 10)
    assert (mo.vi, mo.wi) == before, "도움말 뒤에서 화면이 바뀌었다"
    # 한계는 지워지지 않았다 — v 로 여전히 도달한다.
    mo.help = False
    for _ in range(len(VIEWS)):
        mo.cycle("v")
        if mo.view == "limits":
            break
    assert mo.view == "limits", "한계 화면이 없어졌다"


def test_help_scrolls_and_never_runs_past_the_end(data):
    """도움말은 스크롤된다. 하한을 넘겨 화면 아래가 비면 안 된다.

    주입: 하한을 `help_total()` 로 두면(마지막 한 줄만 남는 자리) 실패한다.
    """
    import curses

    from kr_quant.tui.ledger_app import _key
    from kr_quant.tui.ledger_view import help_total

    mo = Model(data)
    mo.help = True
    for _ in range(500):
        _key(mo, curses.KEY_DOWN, 10)
    assert mo.help_row == help_total() - 10, f"스크롤 하한이 어긋난다: {mo.help_row}"
    bottom = screen(mo, 100, 24)["lines"]
    assert len(bottom) == 24
    _key(mo, curses.KEY_HOME, 10)
    assert mo.help_row == 0
    first = screen(mo, 100, 24)["lines"]
    assert first != bottom, "스크롤해도 화면이 같다 — 검사가 헛돈다"
    assert any("── 키 ──" in ln for ln in first), "맨 위가 키 절이 아니다"


def test_help_screen_fits_any_width_and_shows_the_closing_keys(data):
    """폭 1 에서도 안 죽고, 넓으면 **닫는 법**이 적혀 있다.

    flow 에서 폭 6 이하가 `StopIteration` 으로 TUI 를 통째로 죽인 적이 있다.

    주입: `help_screen` 의 푸터를 `tier_for` 없이 한 줄 고정으로 두면 좁은 폭에서
    폭 단언이 깨진다.
    """
    mo = Model(data)
    mo.help = True
    for width in list(range(1, 20)) + [40, 60, 80, 100, 200]:
        s = screen(mo, width, 24)
        assert len(s["lines"]) == 24, width
        for ln in s["lines"]:
            assert _cw(ln) == width, f"폭{width}: {ln!r}"
    wide = screen(mo, 120, 24)["lines"]
    assert "닫기" in wide[-1] and "종료" in wide[-1], wide[-1]
    assert "q·Esc·?·Enter" in wide[0] or "q·Esc·?·Enter" in wide[-1], wide[0]
    # 푸터도 **잘리면 안 된다.** 폭에 맞는 단계를 고르지 않고 한 줄로 고정하면
    # pad 가 잘라내는데, 모든 줄이 정확히 width 칸이라 폭 검사로는 안 잡힌다.
    from kr_quant.tui.ledger_view import HELP_FOOT_TIERS

    for width in (30, 40, 60, 80, 120, 200):
        foot = screen(mo, width, 24)["lines"][-1].rstrip()
        assert any(foot.startswith(t) for t in HELP_FOOT_TIERS), (
            f"폭{width} 도움말 푸터가 잘렸다: {foot!r}")


def test_hint_bar_names_the_column_you_sorted_by(data):
    """푸터 위 한 줄은 **지금 줄세운 열**의 뜻을 말한다.

    `?` 가 전면 모달이라, 숫자를 보면서 답을 읽으려면 이 줄이 필요하다.

    주입: `hint_text` 가 정렬과 무관하게 한 문장을 돌려주게 만들면 실패한다.
    """
    from kr_quant.tui.ledger_view import hint_text

    mo = Model(data)
    seen = set()
    for _ in range(len(SORTS)):
        seen.add(hint_text(mo, 200))
        mo.cycle("s")
    assert len(seen) == len(SORTS), f"정렬을 바꿔도 힌트가 그대로다: {seen}"
    # 설명이 있어야 힌트다 — 열 이름만 되풀이하면 아무 도움이 안 된다.
    mo.si = [k for k, _ in SORTS].index("spike")
    line = hint_text(mo, 200)
    assert "최대일몫[%]" in line and "하루가 차지한 몫" in line, line
    # 주체를 바꾸면 그 주체의 열을 가리킨다.
    mo.si = [k for k, _ in SORTS].index("actor")
    mo.ai = 1
    assert ACTORS[1][1] in hint_text(mo, 200)
    # 정렬이 없는 화면은 **없는 정렬을 있는 척하지 않는다.**
    mo.vi = [v for v, _ in VIEWS].index("comove")
    assert "정렬" not in hint_text(mo, 200)
    assert "이동이 아니다" in hint_text(mo, 200)


def test_hint_bar_is_on_every_screen_and_fits_the_width(data):
    """힌트바는 모든 화면에 있고 폭을 정확히 채운다. 폭 1 에서도 안 죽는다.

    주입: `screen()` 에서 힌트 줄을 빼면 `hint_y` 자리에 상태줄이 와 실패한다.
    """
    from kr_quant.tui.flow_view import pad
    from kr_quant.tui.ledger_view import hint_text

    mo = Model(data)
    for vi in range(len(VIEWS)):
        mo.vi = vi
        for width in (1, 5, 12, 40, 80, 120, 200):
            s = screen(mo, width, 24)
            assert s["hint_y"] == s["status_y"] - 1 == s["banner_y"] - 2
            line = s["lines"][s["hint_y"]]
            assert _cw(line) == width
            assert line == pad(hint_text(mo, width), width), (vi, width)


def test_bottom_line_keeps_both_the_banner_and_the_keys():
    """회귀 — 폭 80(SSH 기본)에서 **키가 통째로 사라져 있었다.**

    앱이 `(" " + BANNER + "   " + FOOTER)[:w]` 로 문자 슬라이스해 붙였는데 그
    문자열은 표시폭 137 이라, 폭 80 에서는 배너만 남고 `?` 조차 화면에 없었다.
    도움말이 아무리 좋아도 닿을 수 없으면 없는 것과 같다.

    `--dump` 는 상태줄·푸터를 안 찍으므로 이 부류는 **`screen()` 을 봐야** 잡힌다.

    주입: `banner_footer` 를 `" " + BANNER + "   " + FOOTER` 로 되돌리면 폭 80
    에서 `?` 가 없어 실패한다.
    """
    from kr_quant.tui.ledger_view import banner_footer

    # 폭 34 미만에서는 가장 짧은 배너와 가장 짧은 푸터가 같이 못 들어간다.
    # 그때는 배너가 이긴다(이 화면이 존재하는 이유가 그 경고다).
    for width in range(34, 202, 2):
        line = banner_footer(width)
        assert _cw(line) == width, f"폭{width}: {_cw(line)}칸"
        assert "미관측" in line, f"폭{width} 에서 경고가 사라졌다: {line!r}"
        assert "?" in line and "q" in line, f"폭{width} 에서 키가 사라졌다: {line!r}"


def test_the_app_does_not_rebuild_the_bottom_lines(data):
    """`screen()` 이 만든 마지막 세 줄을 앱이 **다시 만들지 않는다.**

    예전엔 앱이 `h - 1` 을 넘겨 놓고 하단을 스스로 그렸다 — 그래서 순수 함수
    검사도, 렌더 스모크 14,800 조합도 그 줄을 한 번도 본 적이 없다.

    주입: `_draw` 가 하단을 다시 조립하게 되돌리면 여기 단언이 깨진다.
    """
    import inspect

    from kr_quant.tui import ledger_app

    src = inspect.getsource(ledger_app._draw)
    assert "screen(mo, w, h)" in src, "앱이 화면 높이를 줄여 넘긴다"
    assert "BANNER" not in src and "FOOTER" not in src, (
        "앱이 하단 줄을 스스로 조립한다 — screen() 과 갈라진다")
    for name in ("hint_y", "status_y", "banner_y"):
        assert f's["{name}"]' in src, f"{name} 을 안 쓴다"


def test_help_and_hint_never_show_markdown_asterisks(data):
    """도움말·힌트는 `screen()` 을 거치는 **새 경로**다. 강조 표기를 여기서도 뗀다.

    주입: `help_lines` 의 `desc.replace("**", "")` 를 지우면 실패한다.
    """
    from kr_quant.tui.ledger_view import LEDGER_HELP

    assert any("**" in d for _n, d in LEDGER_HELP), "소스에 강조가 없다 — 검사가 헛돈다"
    mo = Model(data)
    mo.help = True
    for width in (40, 80, 120):
        for row in (0, 20, 60):
            mo.help_row = row
            lines = screen(mo, width, 24)["lines"]
            assert lines and "**" not in "".join(lines), (width, row)
    mo.help = False
    assert "**" not in render_text(mo, 100)
    assert "── 키 ──" in render_text(mo, 100), "평문 덤프에 도움말이 없다"


def test_narrow_notices_are_never_truncated(data):
    """좁아서 표를 못 그릴 때의 "왜 안 그리나" 안내는 **어떤 폭에서도 온전해야**
    한다.

    예전엔 한 줄 고정이라 `pad` 가 잘랐고, 하필 전개 안내는 **어떤 폭에서도**
    온전히 못 나왔다 — 54칸이 필요한 문장인데 폭 48 미만에서만 뜬다. 검사가
    아슬아슬 들어가는 폭만 표본으로 골라서 몰랐다.

    주입: `_too_narrow`·`tier_for` 를 예전 f-string 한 줄로 되돌리면, 잘린 줄이
    문장 끝 글자로 안 끝나 실패한다.
    """
    from kr_quant.tui.ledger_view import (
        LEDGER_NARROW_TIERS, TIMELINE_NARROW_TIERS, timeline_lines)

    mo = Model(data)
    ends = ("칸.", "칸)", "칸", "좁다", "넓혀라")
    for width in range(5, LEDGER_MIN_W):
        lines, _thin, nh = ledger_lines(mo, width)
        assert nh == 3 and len(lines) == 3, width
        assert lines[0].rstrip().endswith(ends), (width, lines[0].rstrip())
        assert lines[1].rstrip() in LEDGER_NARROW_TIERS, (width, lines[1].rstrip())
        for ln in lines:
            assert _cw(ln) == width
    for width in range(5, TIMELINE_MIN_W):
        lines, _thin, nh = timeline_lines(mo, width)
        assert lines[0].rstrip().endswith(ends), (width, lines[0].rstrip())
        assert lines[1].rstrip() in TIMELINE_NARROW_TIERS, (width, lines[1].rstrip())


def test_no_line_overflows_when_the_terminal_draws_ambiguous_chars_wide(data, monkeypatch):
    """⚠️ 원장 화면의 'A'(Ambiguous) 글자 — 블록 문자 ``▁▂▃█▒▓▌`` 도 'A' 다.

    폭 계산이 `flow_view.cell_width` 한 곳을 지나므로, 거기서 'A' 를 2칸으로
    세면 이 화면도 같이 맞는다. 그 배선이 살아 있는지 본다 — `cell_width` 를
    안 쓰는 참조 구현으로 재서 폭을 넘는 줄이 없어야 한다.
    """
    import unicodedata

    from kr_quant.tui import flow_view

    def wide_w(text: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in ("W", "F", "A") else 1
                   for c in text)

    # 전제 — 화면에 'A' 글자가 실제로 있어야 이 검사가 뭔가를 지킨다.
    # (그림 글자는 브라유로 옮겨 'A' 가 아니다. 남은 것은 `·`·`—` 같은 산문 글자다.)
    mo0 = Model(data)
    seen = set()
    for vi in range(len(VIEWS)):
        mo0.vi = vi
        seen |= {c for ln in screen(mo0, 100, 40)["lines"] for c in ln}
    assert any(unicodedata.east_asian_width(c) == "A" for c in seen), \
        "화면에 'A' 글자가 하나도 없다 — 이 검사가 겨냥한 위험이 사라졌나?"

    monkeypatch.setattr(flow_view, "AMBIGUOUS_WIDE", True)
    mo = Model(data)
    bad = []
    for width in (72, 80, 100, 132):
        for vi in range(len(VIEWS)):
            mo.vi = vi
            for wi in range(len(WINDOWS)):
                mo.wi = wi
                for line in screen(mo, width, 40)["lines"]:
                    if wide_w(line) > width:
                        bad.append((VIEWS[vi][0], width, wide_w(line), line))
    assert not bad, f"폭을 넘는 줄 {len(bad)}개 — 예: {bad[:3]}"
# ═══════════════════════════════════════════════════════════════════════════
# 정보 설계 회귀 — 독립 리뷰가 실측으로 잡아낸 것들.
# 각 검사의 독스트링에 "무엇을 주입하면 깨지나"를 적는다(tests/mutations.toml).
# ═══════════════════════════════════════════════════════════════════════════

def _corr_payload() -> dict:
    """상관 구조를 **손으로 정한** 페이로드.

    ``갑``·``을`` 은 같이 가고 ``병`` 은 둘과 반대로 간다. 그러면 평균상관
    내림차순의 답이 데이터가 아니라 **정의**로 정해진다 — 구현에서 기대값을
    끌어오지 않고도 순서를 검사할 수 있다.
    """
    n = 40
    wave = [((-1) ** i) * (10.0 + i % 5) for i in range(n)]
    series = {"갑": wave, "을": [v * 1.1 + 0.3 for v in wave],
              "정": [v * 0.7 - 0.2 for v in wave], "병": [-v * 0.9 for v in wave]}
    # 셋이 같이 가고 하나가 반대로 간다. 셋 이하면 같이 가는 쪽의 평균이 대칭
    # 때문에 0 이 되어 "따로 노는 쪽" 이 안 갈린다.
    secs = ["갑", "을", "병", "정"]

    def cell(vals):
        return {"indiv": list(vals), "forgn": [v * 0.5 for v in vals],
                "inst": [v * 0.25 for v in vals], "etc": [1.0] * n}

    return {"dates": [f"2026-01-{i % 28 + 1:02d}" for i in range(n)],
            "sectors": secs, "markets": ["거래소"],
            "flows": {"거래소": {s: cell(series[s]) for s in secs}},
            "cap": {"거래소": {s: 1000.0 for s in secs}},
            "n_by_sector": {"거래소": {s: 50 for s in secs}},
            "finalized": True}


# ------------------------------------------------ 1. 동시성 — 순서와 결론

def test_comove_rows_are_ordered_by_mean_correlation_not_by_name():
    """행·열이 **평균상관 내림차순**이어야 한다. 가나다순은 무늬가 없다.

    27×27 을 이름순으로 늘어놓으면 첫 글자와 상관 사이에 아무 관계가 없으니
    당연히 아무것도 안 보인다. 평균상관으로 줄세워야 "누가 시장 공통요인과
    따로 노는가"가 그림에 나온다.

    기대 순서를 구현에서 안 가져온다 — 페이로드가 갑·을을 같이 가게, 병을
    반대로 가게 **만들어 놓았다**.

    주입: ``_corr_ordered`` 의 ``keys.sort(...)`` 를 지우면(= 페이로드 순서)
    ``병`` 이 가운데로 오지 않고 맨 뒤로도 안 가서 실패한다.
    """
    from kr_quant.tui.ledger_view import Model

    mo = Model(_corr_payload())
    mo.wi = WINDOWS.index(20)
    keys, _m = mo.corr_matrix()
    mean = mo.corr_means()
    assert keys[-1] == "병", f"반대로 가는 섹터가 맨 아래가 아니다: {keys}"
    assert set(keys[:3]) == {"갑", "을", "정"}, keys
    assert keys != sorted(keys), "가나다순과 구분이 안 된다 — 검사가 헛돈다"
    vals = [mean[k] for k in keys]
    assert vals == sorted(vals, reverse=True), f"내림차순이 아니다: {vals}"
    assert mean["병"] < 0 < mean["갑"], mean


def test_comove_prints_the_mean_column_and_names_the_outliers():
    """그림을 못 봐도 결론이 남아야 한다 — ``평균`` 열과 셋째 줄.

    좁은 창·평문 덤프·스크롤 아래에서는 히트맵을 통째로 못 본다. 그때도
    "어느 섹터가 따로 노는가"는 글자로 남아야 한다.

    주입: ``평균`` 열을 빼거나(라벨 뒤 ``pad(f"{mv*100:+.0f}")`` 삭제)
    ``comove_outliers`` 줄을 안 넣으면 실패한다.
    """
    from kr_quant.tui.ledger_view import Model

    mo = Model(_corr_payload())
    mo.wi = WINDOWS.index(20)
    lines, _marks, head = comove_lines(mo, 100)
    assert "평균" in lines[head - 1], lines[head - 1]
    assert "병" in lines[2] and "따로" in lines[2], lines[2]
    body = lines[head:head + 4]
    # 각 행 라벨 뒤에 부호 있는 정수가 붙는다 — 그게 그 섹터의 평균상관 ×100.
    got = [ln.split()[2] for ln in body]
    assert all(g[0] in "+-" for g in got), got
    assert int(got[0]) > 0 > int(got[-1]), got


def test_comove_diagonal_is_blank_so_the_eye_has_a_baseline():
    """자기상관 1.0 은 27칸을 먹으면서 아는 사실만 되풀이한다. 비운다.

    주입: ``heat_diag()`` 대신 ``heat_cell(m[i][j])`` 를 쓰면 대각선이
    ``+█`` 로 차서 실패한다.
    """
    from kr_quant.tui.ledger_view import Model

    mo = Model(_corr_payload())
    mo.wi = WINDOWS.index(20)
    lines, marks, head = comove_lines(mo, 100)
    keys, _m = mo.corr_matrix()
    x0 = 13 + 1 + 4 + 1
    for i in range(len(keys)):
        cells = _cells(lines[head + i])
        at = "".join(cells[x0 + 2 * i:x0 + 2 * i + 2])
        assert at == "  ", f"{i}번 대각선이 안 비었다: {at!r}"
    assert not any(y == head + i and x == x0 + 2 * i
                   for i in range(len(keys)) for y, x, _w, _lv in marks), (
        "대각선에 색칠 지시가 남아 있다")


def test_comove_verdict_is_never_truncated_into_its_opposite():
    """회귀 — 폭 80 에서 첫 줄이 ``→ 로테이션`` 에서 끊겨 **뜻이 뒤집혔다.**

    원문은 "로테이션은 우연보다 **드물다**" 인데, 잘린 자리에서 읽으면 정확히
    반대로 읽힌다. 이 저장소는 배너·푸터·도움말 제목에 전부 단계를 깔아 뒀는데
    가장 결론적인 이 줄에만 없었다.

    주입: ``tier_for(comove_verdict_tiers(obs, nul), width)`` 를
    ``comove_verdict_tiers(obs, nul)[0]`` 로 되돌리면 폭 80 에서 실패한다.
    """
    from kr_quant.tui.ledger_view import Model, comove_verdict_tiers

    mo = Model(_corr_payload())
    mo.wi = WINDOWS.index(20)
    obs, nul = mo.obs_neg_frac(), mo.null_neg_frac()
    tiers = comove_verdict_tiers(obs, nul)
    assert "드물다" in tiers[0] or "읽지 마라" in tiers[0], tiers[0]
    for width in range(30, 200):
        got = comove_lines(mo, width)[0][0].rstrip()
        assert got in tiers, f"폭{width} 판정줄이 어느 단계도 아니다: {got!r}"
        assert _cw(got) <= width


def test_comove_header_does_not_advertise_a_sort_that_does_nothing(data):
    """``s`` 는 동시성 화면에서 **아무 일도 하지 않는다.** 칩이 그걸 광고하면 안 된다.

    정렬 3종을 다 돌려도 출력이 같다는 사실을 여기서 못박는다 — 그래서 칩은
    정렬이 아니라 그 화면이 실제로 쓰는 순서(평균상관)를 말해야 한다.

    주입: ``header_lines`` 의 ``order`` 를 늘 ``정렬[...]`` 로 되돌리면 실패한다.
    """
    from kr_quant.tui.ledger_view import header_lines

    mo = Model(data)
    mo.vi = [v for v, _ in VIEWS].index("comove")
    seen = set()
    for _ in range(len(SORTS)):
        head = header_lines(mo, 120)[1]
        assert "정렬" not in head, head
        assert "순서[평균상관]" in head, head
        seen.add(tuple(comove_lines(mo, 120)[0]))
        mo.cycle("s")
    assert len(seen) == 1, "정렬이 동시성 행렬을 바꾼다 — 이 검사의 전제가 틀렸다"
    # 그리고 표 화면에서는 여전히 정렬을 말해야 한다.
    mo.vi = 0
    assert "정렬" in header_lines(mo, 120)[1]


def test_undefined_correlation_is_not_the_same_glyph_as_no_correlation():
    """분산 0 은 "무상관" 이 아니라 **잴 수 없음**이다. 빈칸을 나눠 쓰면 안 된다.

    주입: ``_corr`` 이 분산 0 에서 ``0.0`` 을 돌려주게 되돌리면 두 칸이 같아져
    실패한다.
    """
    from kr_quant.tui.ledger_view import HEAT_UNDEF, _corr

    flat, moving = [1.0] * 10, [float(i) for i in range(10)]
    c = _corr(flat, moving)
    assert c != c, "분산 0 인데 nan 이 아니다"
    assert heat_cell(c) == HEAT_UNDEF
    assert heat_cell(0.0) != HEAT_UNDEF
    assert _w(heat_cell(c)) == 2


# --------------------------------------------- 2. 전개 — 기준선과 글자 폭

def test_spark_glyphs_are_one_cell_in_every_locale():
    """회귀 — ``▁▂▃▄▅▆▇█`` 는 EAW 'A' 라 한글 로케일에서 **2칸이 될 수 있다.**

    20점이 40칸이 되면 ``pad`` 가 뒤를 잘라내고 구간의 절반만 그린 채 전부인
    척한다. ``cell_width`` 로는 못 잡는다 — 그 함수는 'A' 를 1 로 센다.
    ``kq-flow`` 가 같은 이유로 브라유(EAW 'N')로 옮겼고, 원장만 안 옮겨져 있었다.

    주입: ``SPARK_SIGNED`` 를 ``▁▂▃▄▅▆▇█`` 계열로 되돌리면 실패한다.
    """
    import unicodedata

    from kr_quant.tui.ledger_view import SPARK, SPARK_SIGNED

    assert len(SPARK_SIGNED) == 5 and set(SPARK) == set(SPARK_SIGNED.values())
    for ch in SPARK:
        eaw = unicodedata.east_asian_width(ch)
        assert eaw not in ("A", "W", "F"), f"{ch!r} 의 폭이 {eaw} 다"
        assert cell_width(ch) == 1


def test_spark_puts_zero_at_the_same_height_in_every_row():
    """회귀 — **20일 내내 판 섹터가 "높다가 없어졌다" 로 읽혔다.**

    예전 ``lo, hi = min(values), max(values)`` 는 행마다 바닥이 달라서, 계속
    내려가기만 한 누적 경로가 ``████████…▁▁`` 로 나왔다. 실측(20일 기관)에서
    운송장비/부품이 정확히 그랬다.

    ⚠️ ``kq-flow`` 처방(0 을 범위에 접어 넣기)만으로는 **안 고쳐진다.** hi=0 이
    되어 경로가 맨 위에서 시작할 뿐이다. 0 을 세로 한가운데 **고정**해야 한다.

    주입: ``spark`` 을 ``lo,hi = min(v),max(v)`` 든 ``min(0,*v),max(0,*v)`` 든
    범위 정규화로 되돌리면 아래 단언이 깨진다.
    """
    from kr_quant.tui.ledger_view import SPARK_SIGNED, spark

    up = {SPARK_SIGNED[1], SPARK_SIGNED[2]}
    down = {SPARK_SIGNED[-1], SPARK_SIGNED[-2]}
    sell = spark([-100.0 * i for i in range(1, 21)])       # 20일 내내 팔았다
    assert set(sell) <= down, f"계속 팔았는데 위쪽 글자가 있다: {sell}"
    buy = spark([100.0 * i for i in range(1, 21)])
    assert set(buy) <= up, f"계속 샀는데 아래쪽 글자가 있다: {buy}"
    # 그리고 0 은 **크기와 무관하게** 같은 자리다 — 큰 행과 작은 행이 같은 글자.
    assert spark([-1.0, -2.0])[0] == spark([-1e9, -2e9])[0]
    assert spark([0.0, 5.0])[0] == SPARK_SIGNED[0], "정확히 0 인 점이 0 글자가 아니다"


def test_timeline_scale_column_matches_the_points_actually_drawn(data):
    """``눈금[억]`` 은 **그려지는 점들**로 재야 한다.

    예전 ``진폭[억]`` 은 원본 260점의 최고−최저였는데 그림은 다운샘플된 점으로
    정규화된다. 폭 80·260일에서 최대 21.5% 어긋났다(실측: 일반서비스 29,112억
    vs 그림 22,839억). "숫자가 그림의 눈금을 알려준다"는 계약이 거짓이었다.

    기대값을 ``spark_scale`` 에서 안 가져온다 — ``downsample`` 로 직접 만든다.

    주입: ``spark_scale(show)`` 를 ``max(cum) - min(cum)`` 이나
    ``spark_scale(cum)`` 으로 되돌리면 구간 > 칸 인 폭에서 실패한다.
    """
    from kr_quant.tui.ledger_view import downsample

    mo = Model(data)
    mo.wi = WINDOWS.index(60)          # 구간(40일) > 스파크라인 칸
    width = 60
    lines, _t, nh = timeline_lines(mo, width)
    from kr_quant.tui.ledger_view import _col

    cols = _fit([_col("섹터", 13, False), _col("", 1, False),
                 _col("누적[억]", 11), _col("눈금[억]", 11)], width)
    used = sum(c[1] + 1 for c in cols)
    sw = min(mo.window, width - used)
    assert sw < mo.window, "구간이 칸보다 안 길면 이 검사는 아무것도 안 잡는다"
    start, w = span_at([c[1] for c in cols], 3)          # 눈금[억] 열
    ok = False
    for ln, r in zip(lines[nh:], mo.rows()):
        v = mo.daily(r["sector"], mo.actor)[-mo.window:]
        cum, acc = [], 0.0
        for x in v:
            acc += x
            cum.append(acc)
        want = max(abs(c) for c in downsample(cum, sw))
        got = float("".join(_cells(ln)[start:start + w]).strip().replace(",", ""))
        assert abs(got - round(want)) <= 1, (r["sector"], got, want)
        if abs(max(abs(c) for c in cum) - want) > 1:
            ok = True          # 원본으로 재면 달라지는 행이 실제로 있다
    assert ok, "다운샘플 전후가 같은 데이터다 — 이 검사가 헛돈다"


# ------------------------------------------- 3·4. 잔여몫 · 최대일몫

def _cancelling_payload() -> dict:
    """일별 잔여가 **부호를 바꿔 상쇄되는** 페이로드.

    실측이 그랬다(20일·전시장): 전기/전자는 구간 합계 −175억인데 일별 |잔여|
    합이 1,934억으로 11배, IT 서비스는 48배다. 표가 합계만 보여주면 그 사실이
    지워진다 — 한계 §5 가 정확히 그걸 경고하는데 화면이 그 짓을 하고 있었다.
    """
    n = 20
    dates = [f"2026-02-{i + 1:02d}" for i in range(n)]
    # indiv 가 하루 걸러 ±100 을 오간다 → 잔여도 ∓100 을 오가고 합계는 0 이다.
    swing = [100.0 if i % 2 == 0 else -100.0 for i in range(n)]
    cell = {"indiv": swing, "forgn": [0.0] * n, "inst": [0.0] * n,
            "etc": [0.0] * n}
    return {"dates": dates, "sectors": ["상쇄"], "markets": ["거래소"],
            "flows": {"거래소": {"상쇄": cell}},
            "cap": {"거래소": {"상쇄": 1000.0}},
            "n_by_sector": {"거래소": {"상쇄": 50}}, "finalized": True}


def test_residual_share_is_measured_daily_not_on_the_cancelled_total():
    """``잔여몫`` 을 구간 합계로 재면 상쇄가 사실을 지운다.

    이 페이로드는 잔여가 하루 걸러 ±100 이라 **합계는 정확히 0** 이다. 합계
    기준이면 몫도 0% 라 "잔여가 없다"고 말하게 되는데, 실제로는 매일 그 섹터
    순매매의 100% 가 미분류 주체였다.

    주입: ``r["resid_pct"]`` 를 ``abs(r["resid"]) / Σ|4주체 합계|`` 로 바꾸면
    0.0 이 나와 실패한다.
    """
    mo = Model(_cancelling_payload())
    r = mo.rows()[0]
    assert abs(r["resid"]) < 1e-6, "합계가 0 이 아니면 이 검사의 전제가 틀렸다"
    assert r["resid_pct"] == pytest.approx(100.0), r["resid_pct"]
    # 그리고 그 값이 **화면에** 있어야 한다.
    lines, _t, nh = ledger_lines(mo, 132)
    cols = _fit(ledger_cols(), 132)
    names = [c[0] for c in cols]
    start, w = span_at([c[1] for c in cols], names.index("잔여몫[%]"))
    assert "".join(_cells(lines[nh])[start:start + w]).strip() == "100.0"


def _spiky_payload() -> dict:
    """한 섹터는 **하루짜리 블록딜**, 다른 섹터는 매일 고르게.

    `data` 픽스처는 매끈해서 어느 행도 문턱을 안 넘는다 — 그러면 마커 검사가
    "아무 행에도 안 붙는다"만 확인하고 통과한다(아무것도 안 지키는 초록).
    """
    n = 20
    even = [10.0] * n
    block = [0.0] * n
    block[7] = 500.0
    zero = [0.0] * n

    def cell(vals):
        return {"indiv": list(vals), "forgn": list(zero), "inst": list(zero),
                "etc": [1.0] * n}

    secs = ["블록딜", "고르게"]
    return {"dates": [f"2026-03-{i + 1:02d}" for i in range(n)],
            "sectors": secs, "markets": ["거래소"],
            "flows": {"거래소": {"블록딜": cell(block), "고르게": cell(even)}},
            "cap": {"거래소": {s: 1000.0 for s in secs}},
            "n_by_sector": {"거래소": {s: 50 for s in secs}}, "finalized": True}


def test_spike_column_marks_what_beats_the_even_anchor(data):
    """``최대일몫`` 은 이름만으로 안 잡힌다 — 균등(=100÷구간일수) 대비 표시가 필요하다.

    17.9% 가 큰지 작은지는 20일 균등이 5.0% 라는 걸 알아야 판단된다. 문턱을
    넘은 행에는 **글자** 마커를 붙인다(색은 무색 터미널·`--dump` 에서 사라진다).

    주입: ``mark`` 을 늘 ``" "`` 로 두면 실패한다. ``SPIKE_MARK_MULT`` 를 크게
    키워도(예: 100) 아무 행에도 안 붙어 실패한다.
    """
    from kr_quant.tui.ledger_view import SPIKE_MARK_MULT

    mo = Model(_spiky_payload())
    assert mo.uniform_spike() == pytest.approx(100.0 / mo.window)
    lines, _t, nh = ledger_lines(mo, 132)
    cols = _fit(ledger_cols(), 132)
    names = [c[0] for c in cols]
    start, w = span_at([c[1] for c in cols], names.index("최대일몫[%]"))
    hot = cold = 0
    for ln, r in zip(lines[nh:], mo.rows()):
        got = "".join(_cells(ln)[start:start + w]).rstrip()
        want = r["spike"] >= SPIKE_MARK_MULT * mo.uniform_spike()
        assert got.endswith("!") == want, (r["sector"], r["spike"], got)
        hot += want
        cold += not want
    assert hot and cold, f"한쪽만 있어서 아무것도 안 잡는다: {hot}/{cold}"


def test_status_line_anchors_the_spike_and_names_the_actor(data):
    """``a`` 를 누르면 ``최대일몫`` 만 조용히 바뀌는데 헤더에 주체가 없다.

    상태줄이 주체와 균등 앵커를 같이 댄다. ``--dump`` 는 상태줄을 안 찍으므로
    이 부류는 ``status_line`` 을 직접 봐야 잡힌다.

    주입: 상태줄에서 ``· 균등 {…}%`` 를 지우거나 ``({mo.actor_ko})`` 를 떼면 실패한다.
    """
    mo = Model(data)
    seen = set()
    for ai in range(len(ACTORS)):
        mo.ai = ai
        line = status_line(mo, 200)
        assert f"최대일몫({ACTORS[ai][1]})" in line, line
        assert f"균등 {mo.uniform_spike():.1f}%" in line, line
        seen.add(line)
    assert len(seen) == len(ACTORS), "주체를 바꿔도 상태줄이 그대로다"
    # 폭이 좁아도 **앞쪽 우선순위**는 살아남는다 — 예전엔 뒤가 통째로 잘렸다.
    for width in (60, 80, 100, 132, 200):
        line = status_line(mo, width)
        assert _cw(line) == width
        assert "최대일몫" in line, (width, line)


# ------------------------------------------------- 5. 좁은 폭 안내

def test_narrow_notice_names_a_screen_that_works_at_this_width(data):
    """"창을 넓혀라" 는 SSH 로 붙은 창에 자주 막다른 길이다.

    폭 70 에서 원장만 안 그려질 뿐 전개(48칸)·동시성·한계는 멀쩡히 돈다. 그때
    화면의 75% 가 빈 줄인데 안내는 그걸 한 글자도 안 말했다.

    ⚠️ 없는 것을 있다고 하면 안내가 아니라 오도다 — 폭 48 미만에서는 전개도
    안 되므로 그 폭의 안내는 전개를 말하면 안 된다.

    주입: 셋째 줄을 지우면 첫 단언에서, ``narrow_also`` 가 폭과 무관하게
    ``NARROW_ALSO_TIERS`` 만 쓰게 하면 둘째 단언에서 실패한다.
    """
    from kr_quant.tui.ledger_view import (
        NARROW_ALSO_NO_TIMELINE_TIERS, NARROW_ALSO_TIERS, narrow_also)

    mo = Model(data)
    for width in range(5, LEDGER_MIN_W):
        lines, thin, nh = ledger_lines(mo, width)
        assert nh == len(lines) == len(thin) == 3, (width, lines)
        also = lines[2].rstrip()
        tiers = (NARROW_ALSO_TIERS if width >= TIMELINE_MIN_W
                 else NARROW_ALSO_NO_TIMELINE_TIERS)
        if width >= _cw(tiers[-1]):
            assert also in tiers, f"폭{width} 안내가 잘렸다: {also!r}"
            assert "?" in also, f"폭{width} 에 도움말 가는 길이 없다: {also!r}"
        if width < TIMELINE_MIN_W:
            assert "전개" not in also, (
                f"폭{width} 에서는 전개도 안 되는데 전개를 권한다: {also!r}")
        assert _cw(lines[2]) == width
    # 그리고 안내가 가리키는 화면은 **실제로 그 폭에서 그려져야** 한다.
    assert "전개" in narrow_also(TIMELINE_MIN_W)
    mo.vi = [v for v, _ in VIEWS].index("timeline")
    assert timeline_lines(mo, TIMELINE_MIN_W)[2] == 2, "전개가 그 폭에서 안 그려진다"


# ------------------------------------------------------------- 6. 성능

def test_rows_are_cached_on_everything_that_changes_them(data):
    """``rows()`` 는 한 번 그릴 때 세 곳에서 불린다(1회 23ms). 캐시한다.

    ⚠️ 캐시 키에 **주체**가 있어야 한다 — 정렬(선택주체)도 ``최대일몫`` 도
    고른 주체의 값이다. ``kq-flow`` 가 바로 이 자리에서 물렸다(역순이 키에
    빠져 ``r`` 이 안 먹었다).

    주입: ``self._rows_cache`` 를 안 쓰면 첫 단언(동일 객체)에서, 키에서
    ``self.actor`` 를 빼면 둘째 단언에서 실패한다.
    """
    mo = Model(data)
    assert mo.rows() is mo.rows(), "같은 상태인데 표를 다시 만든다"
    mo.si = [k for k, _ in SORTS].index("actor")
    first = mo.rows()
    mo.ai = (mo.ai + 1) % len(ACTORS)
    assert mo.rows() is not first, (
        "주체를 바꿨는데 같은 표를 돌려준다 — 캐시 키에 주체가 없다")
    # 그리고 그 표는 **값이 실제로 다르다** — 최대일몫은 고른 주체의 값이다.
    mo.si = [k for k, _ in SORTS].index("spike")
    mo.ai = 0
    a0 = [r["spike"] for r in mo.rows()]
    mo.ai = 1
    assert [r["spike"] for r in mo.rows()] != a0, "최대일몫이 주체를 안 본다"
    # 구간·시장·정렬도 키에 있어야 한다.
    base = mo.rows()
    for what in ("w", "m", "s"):
        mo.cycle(what)
        assert mo.rows() is not base, f"{what} 를 바꿨는데 같은 표를 돌려준다"
        mo.cycle(what, -1)


def test_screens_do_not_compute_what_they_throw_away(data):
    """안 쓸 것을 먼저 만들고 나서 되돌아오지 않는다.

    ``status_line`` 은 한계 화면에서도 ``selected()``→``rows()`` 를 먼저 불렀고,
    ``comove_lines`` 는 구간 20일 미만에서도 27×27 을 만든 뒤 버렸다(5·10일).

    주입: 두 함수에서 가드를 다시 아래로 내리면 각각 실패한다.
    """
    mo = Model(data)
    mo.vi = [v for v, _ in VIEWS].index("limits")
    mo.rows = lambda: (_ for _ in ()).throw(AssertionError("한계 화면이 표를 만들었다"))
    assert "미관측" in status_line(mo, 120)

    mo2 = Model(data)
    mo2.vi = [v for v, _ in VIEWS].index("comove")
    mo2.wi = WINDOWS.index(5)
    mo2.corr_matrix = lambda: (_ for _ in ()).throw(
        AssertionError("짧은 구간에서 27×27 을 만들었다"))
    lines, marks, _h = comove_lines(mo2, 120)
    assert marks == [] and "상관을 내지 않는다" in lines[0]
