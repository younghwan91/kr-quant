"""생존편향 스멜테스트 유닛테스트 (합성, DB 불요).

핵심 스멜 판별을 검증:
  (1) 전 종목 생존 패널 → frac_disappear≈0, frac_full_span=1 (생존필터 스멜 높음).
  (2) 종목이 중간에 사라지는 패널 → frac_disappear 높고 full_span 낮음.
  (3) 종가 패널(NaN=부재)도 존재행렬과 동일하게 동작.
  (4) assert_point_in_time: 다년 생존필터 → RAISE, 사라짐 있으면 통과, 짧은 구간 스킵.
  (5) 리포터 원칙 — 반환 dict 에 PASS/FAIL bool 필드가 없다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kr_quant.validation.universe_hygiene import (
    assert_point_in_time,
    survivorship_report,
)


def _dates(n: int) -> list[str]:
    return [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]


def test_all_survive_flags_survivorship_smell():
    # 5 종목 × 10 날짜, 전부 항상 존재 → 아무도 안 사라짐 = 생존필터 스멜.
    codes = ["A", "B", "C", "D", "E"]
    dates = _dates(10)
    pres = np.ones((5, 10))
    rep = survivorship_report(codes, dates, pres)
    assert rep["n_disappear_before_end"] == 0
    assert rep["frac_disappear_before_end"] == 0.0
    assert rep["frac_full_span"] == 1.0          # 전원 첫날+마지막날 존재
    assert rep["survival_rate"] == 1.0


def test_disappearing_codes_lower_full_span():
    # C,D 는 중간에 사라진다(대리 상폐), A,B,E 는 끝까지 생존.
    codes = ["A", "B", "C", "D", "E"]
    dates = _dates(10)
    pres = np.ones((5, 10))
    pres[2, 5:] = 0.0    # C: index5 부터 부재
    pres[3, 7:] = 0.0    # D: index7 부터 부재
    rep = survivorship_report(codes, dates, pres)
    assert rep["n_disappear_before_end"] == 2
    assert rep["frac_disappear_before_end"] == pytest.approx(2 / 5)
    assert rep["frac_full_span"] == pytest.approx(3 / 5)   # A,B,E
    # C 의 last-seen 은 index4, D 는 index6 → last_seen max 는 마지막날(생존종목).
    assert rep["last_seen_idx"]["min"] == 4
    assert rep["last_seen_idx"]["max"] == 9


def test_close_panel_treats_nan_and_zero_as_absent():
    # 종가 패널: NaN·0 = 부재. DataFrame 경로도 함께 검증.
    codes = ["A", "B"]
    dates = _dates(6)
    close = pd.DataFrame(
        [[100, 101, 102, 103, 104, 105],       # A: 항상 거래
         [50, 51, np.nan, np.nan, np.nan, np.nan]],  # B: index2 부터 부재
        index=codes, columns=dates,
    )
    rep = survivorship_report(codes, dates, close)
    assert rep["n_disappear_before_end"] == 1          # B
    assert rep["frac_full_span"] == pytest.approx(1 / 2)
    # first-seen 은 둘 다 index0.
    assert rep["frac_first_seen_at_start"] == 1.0


def test_ipo_midpanel_not_counted_as_disappeared():
    # 중간 상장(IPO) 종목: first-seen 이 늦지만 끝까지 생존 → 사라진 게 아님.
    codes = ["OLD", "IPO"]
    dates = _dates(8)
    pres = np.ones((2, 8))
    pres[1, :4] = 0.0        # IPO: index4 부터 존재
    rep = survivorship_report(codes, dates, pres)
    assert rep["n_disappear_before_end"] == 0          # 둘 다 마지막날 존재
    assert rep["frac_last_seen_at_end"] == 1.0
    assert rep["n_present_at_start"] == 1              # OLD 만 첫날 존재
    assert rep["frac_first_seen_at_start"] == pytest.approx(1 / 2)


def test_assert_raises_on_survivorship_filtered_multiyear():
    # 다년(≥250 날짜) 전원생존 → 명백한 생존필터 → RAISE.
    n = 300
    codes = ["A", "B", "C"]
    dates = _dates(n)
    pres = np.ones((3, n))
    with pytest.raises(AssertionError, match="생존편향 스멜"):
        assert_point_in_time(codes, dates, pres)


def test_assert_passes_when_codes_disappear():
    # 다년 구간이지만 일부 종목이 사라지면(정상) 통과, 리포트 반환.
    n = 300
    codes = ["A", "B", "C", "D"]
    dates = _dates(n)
    pres = np.ones((4, n))
    pres[3, 150:] = 0.0     # D 상폐 → 1/4 사라짐 > 문턱 1%
    rep = assert_point_in_time(codes, dates, pres)
    assert rep["n_disappear_before_end"] == 1


def test_assert_skips_short_span():
    # 짧은 구간(<250 날짜)에서는 전원생존이어도 스멜을 주장하지 않는다(스킵).
    codes = ["A", "B"]
    dates = _dates(20)
    pres = np.ones((2, 20))
    rep = assert_point_in_time(codes, dates, pres)   # RAISE 안 함
    assert rep["frac_disappear_before_end"] == 0.0


def test_report_has_no_verdict_field():
    # 리포터-not-판정기: PASS/FAIL bool 필드가 없어야 한다.
    codes = ["A", "B"]
    dates = _dates(5)
    rep = survivorship_report(codes, dates, np.ones((2, 5)))
    keys = {k.lower() for k in rep}
    assert not (keys & {"pass", "fail", "verdict", "ok", "go"})
    assert not any(isinstance(v, bool) for v in rep.values())
