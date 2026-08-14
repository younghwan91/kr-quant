"""``earnings_yoy_panel(knowledge_col=...)`` — 정정공시 룩어헤드 방지.

``earnings`` 는 정정공시를 새 행으로 쌓는다. 어느 분기가 공시됐나(``avail_date``)와
그 분기의 **어느 버전을 그때 알고 있었나**(``knowledge_date``)는 서로 독립인 축이라,
스칼라 as-of 하나로는 표현되지 않는다. 이 파일이 그 두 축을 못박는다.

터지지 않고 수치만 좋아지는 종류의 누출이라 — 2024년에 정정된 숫자를 2020년 셀에
넣으면 백테스트가 답을 보고 문제를 푼다 — 합성 데이터로 명시 검증한다.
"""

from __future__ import annotations

import pandas as pd

from kr_quant.features.fundamentals import earnings_yoy_panel

KN = "knowledge_date"


def _rows(*recs) -> pd.DataFrame:
    return pd.DataFrame(
        list(recs), columns=["code", "period", "avail_date", KN, "yoy"])


def _at(panel, date, code="A"):
    row = panel[(panel["date"] == date) & (panel["code"] == code)]
    return float(row["yoy"].iloc[0])


DATES = ["2020-06-01", "2022-06-01", "2025-06-01", "2026-06-01"]


def test_restatement_is_invisible_before_it_was_filed():
    """2024년 정정본을 2022년 셀이 보면 룩어헤드다."""
    ea = _rows(
        ("A", "2020Q1", "2020-05-15", "2020-05-15", 0.10),   # 최초 보고
        ("A", "2020Q1", "2020-05-15", "2024-03-01", 0.99),   # 4년 뒤 정정
    )
    p = earnings_yoy_panel(ea, DATES, knowledge_col=KN)
    assert _at(p, "2022-06-01") == 0.10, "정정 이전 시점은 최초 보고치여야 한다"
    assert _at(p, "2025-06-01") == 0.99, "정정 이후 시점은 정정본이어야 한다"


def test_without_knowledge_col_behavior_is_unchanged():
    """기본 경로는 건드리지 않는다 — 발표 수치·패리티 테스트가 여기 고정돼 있다."""
    ea = _rows(("A", "2020Q1", "2020-05-15", "2020-05-15", 0.10))
    a = earnings_yoy_panel(ea[["code", "avail_date", "yoy"]], DATES)
    b = earnings_yoy_panel(ea, DATES, knowledge_col=KN)
    pd.testing.assert_frame_equal(a, b)


def test_newer_period_wins_over_a_later_restatement_of_an_old_one():
    """핵심 반례 — max(avail, knowledge) 로 정렬하면 여기서 틀린다.

    2024Q4 를 이미 아는 상태에서 2020Q1 이 나중에 정정되면, 정정본의 valid_from 이
    더 커서 '가장 최근'으로 뽑힌다. 신호는 **최신 분기**여야 한다.
    """
    ea = _rows(
        ("A", "2020Q1", "2020-05-15", "2020-05-15", 0.10),
        ("A", "2024Q4", "2025-03-15", "2025-03-15", 0.50),   # 최신 분기
        ("A", "2020Q1", "2020-05-15", "2025-06-01", 0.99),   # 그 뒤 옛 분기 정정
    )
    p = earnings_yoy_panel(ea, DATES, knowledge_col=KN)
    assert _at(p, "2026-06-01") == 0.50, "옛 분기의 정정본이 최신 분기를 밀어내면 안 된다"


def test_late_collected_filing_is_not_usable_before_collection():
    """공시일과 수집일이 다르면 '알게 된 날' 이후부터만 쓸 수 있다."""
    ea = _rows(("A", "2020Q1", "2020-05-15", "2022-09-01", 0.10))
    p = earnings_yoy_panel(ea, ["2020-06-01", "2021-06-01", "2023-06-01"],
                           knowledge_col=KN)
    assert pd.isna(p[(p["date"] == "2021-06-01")]["yoy"].iloc[0])
    assert _at(p, "2023-06-01") == 0.10


def test_age_days_measures_filing_age_not_restatement_age():
    """age_days 는 PEAD 의 '신선도'다 — 정정 때문에 리셋되면 드리프트 창이 왜곡된다."""
    ea = _rows(
        ("A", "2020Q1", "2020-05-15", "2020-05-15", 0.10),
        ("A", "2020Q1", "2020-05-15", "2024-03-01", 0.99),
    )
    p = earnings_yoy_panel(ea, ["2024-06-01"], knowledge_col=KN)
    row = p[p["date"] == "2024-06-01"].iloc[0]
    assert row["yoy"] == 0.99
    expected = (pd.Timestamp("2024-06-01") - pd.Timestamp("2020-05-15")).days
    assert int(row["age_days"]) == expected


def test_multiple_codes_stay_independent():
    ea = _rows(
        ("A", "2020Q1", "2020-05-15", "2020-05-15", 0.10),
        ("A", "2020Q1", "2020-05-15", "2024-03-01", 0.99),
        ("B", "2020Q1", "2020-05-15", "2020-05-15", 0.20),
    )
    p = earnings_yoy_panel(ea, DATES, knowledge_col=KN)
    assert _at(p, "2022-06-01", "A") == 0.10
    assert _at(p, "2025-06-01", "A") == 0.99
    assert _at(p, "2022-06-01", "B") == 0.20
    assert _at(p, "2025-06-01", "B") == 0.20


def test_three_versions_pick_the_one_current_at_each_date():
    ea = _rows(
        ("A", "2020Q1", "2020-05-15", "2020-05-15", 0.10),
        ("A", "2020Q1", "2020-05-15", "2022-01-01", 0.50),
        ("A", "2020Q1", "2020-05-15", "2025-01-01", 0.99),
    )
    p = earnings_yoy_panel(ea, ["2020-06-01", "2023-06-01", "2026-06-01"],
                           knowledge_col=KN)
    assert _at(p, "2020-06-01") == 0.10
    assert _at(p, "2023-06-01") == 0.50
    assert _at(p, "2026-06-01") == 0.99


def test_restatement_does_not_disturb_other_quarters():
    """정정 하나가 무관한 날짜의 값까지 바꾸면 안 된다.

    구현 중 실제로 두 번 틀렸던 자리다. 구간 스냅샷을 ``max(avail, knowledge)`` 로
    거르면(1차), 또는 버전 축으로 걸러도 평범한 행까지 함께 거르면(2차), 구간 *안에서*
    공시되는 최신 분기가 통째로 잘려 무관한 과거 날짜의 신호가 바뀐다.
    """
    base = _rows(
        ("A", "2020Q1", "2020-05-15", "2020-05-15", 0.10),
        ("A", "2021Q1", "2021-05-15", "2021-05-15", 0.20),
        ("A", "2023Q1", "2023-05-15", "2023-05-15", 0.30),
    )
    restated = pd.concat(
        [base, _rows(("A", "2020Q1", "2020-05-15", "2025-01-01", 9.99))],
        ignore_index=True)
    dates = ["2020-06-01", "2021-06-01", "2024-06-01", "2026-06-01"]

    a = earnings_yoy_panel(base, dates, knowledge_col=KN)
    b = earnings_yoy_panel(restated, dates, knowledge_col=KN)
    pd.testing.assert_frame_equal(
        a.sort_values(["code", "date"]).reset_index(drop=True),
        b.sort_values(["code", "date"]).reset_index(drop=True),
    )


def test_restatement_of_the_newest_quarter_flips_on_its_knowledge_date():
    """그 분기가 최신이면 정정은 실제로 신호를 바꿔야 한다 — 경계일 기준으로."""
    ea = _rows(
        ("A", "2026Q1", "2026-05-15", "2026-05-15", 1.00),
        ("A", "2026Q1", "2026-05-15", "2026-07-01", 9.99),
    )
    p = earnings_yoy_panel(ea, ["2026-06-30", "2026-07-02"], knowledge_col=KN)
    assert _at(p, "2026-06-30") == 1.00
    assert _at(p, "2026-07-02") == 9.99
