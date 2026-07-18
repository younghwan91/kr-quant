"""validation/ 유닛테스트 (합성, DB 불요) — no-lookahead fold·부트스트랩 하단·민감도 표.

핵심 과최적 방지 장치를 검증:
  (1) rolling_folds가 frozen 6-fold 튜플을 정확히 재현(fold-shopping 방지의 기준).
  (2) fold 슬라이스가 no-lookahead — train 진입 < 경계, test 진입 >= 경계, 겹침 0.
  (3) _boot_lower < 평균(보수적) + 소표본 가드가 센티넬(-1.0) 반환.
  (4) sensitivity_table 표의 shape가 grid 크기와 일치.
"""

from __future__ import annotations

import numpy as np

from kr_quant.validation import (
    FOLDS,
    _boot_lower,
    entry_mask,
    fold_slices,
    oos_fixed,
    purge_embargo,
    rolling_folds,
    sensitivity_table,
    slice_by_entry,
)

# 기존 연구 스크립트에 박혀있던 frozen 6-fold (train 3년 → test 1년, 마지막은 상한 연장).
FROZEN = [
    ("2017-01-01", "2020-01-01", "2020-01-01", "2021-01-01"),
    ("2018-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
    ("2019-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
    ("2020-01-01", "2023-01-01", "2023-01-01", "2024-01-01"),
    ("2021-01-01", "2024-01-01", "2024-01-01", "2025-01-01"),
    ("2022-01-01", "2025-01-01", "2025-01-01", "2027-01-01"),
]


def test_rolling_folds_reproduces_frozen_set():
    """생성기 기본 인자 == 기존 frozen 6-fold 튜플 (NamedTuple은 평범 튜플과 == 비교)."""
    folds = rolling_folds()
    assert len(folds) == 6
    assert list(folds) == FROZEN
    assert list(FOLDS) == FROZEN  # 모듈 상수도 동일


def test_rolling_folds_generalizes():
    """인자를 바꾸면 다른 window/기간의 fold를 만든다(생성기 일반성)."""
    folds = rolling_folds(first_test_year=2021, last_test_year=2022,
                          train_years=2, test_years=1, final_test_hi=None)
    assert list(folds) == [
        ("2019-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
        ("2020-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
    ]


def test_fold_split_is_no_lookahead():
    """각 fold: train_hi <= test_lo, train 진입 < 경계, test 진입 >= 경계, 겹침 0."""
    # 각 fold의 train_hi 직전/직후에 진입일을 촘촘히 배치한 합성 진입 배열.
    edates = np.array([
        "2019-06-01", "2021-12-31", "2022-01-01", "2022-06-01", "2024-01-01",
    ], dtype=object)
    for f in FOLDS:
        assert f.train_hi <= f.test_lo  # 학습 구간이 평가 구간보다 앞선다
        tr = entry_mask(edates, f.train_lo, f.train_hi)
        te = entry_mask(edates, f.test_lo, f.test_hi)
        # train 진입은 모두 train_hi 미만, test 진입은 모두 test_lo 이상
        assert all(d < f.train_hi for d in edates[tr])
        assert all(d >= f.test_lo for d in edates[te])
        # train_hi == test_lo이므로 두 구간은 절대 겹치지 않는다
        assert not (tr & te).any()


def test_slice_by_entry_matches_mask():
    """slice_by_entry == entry_mask 적용 (no-lookahead 슬라이스 기본 연산)."""
    rets = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    edates = np.array(["2019-05-01", "2021-08-01", "2023-05-01",
                       "2023-08-01", "2025-02-01"], dtype=object)
    got = slice_by_entry(rets, edates, "2022-01-01", None)
    assert set(np.round(got, 3)) == {0.3, -0.4, 0.5}
    # train 쪽
    got_tr = slice_by_entry(rets, edates, None, "2022-01-01")
    assert set(np.round(got_tr, 3)) == {0.1, -0.2}


def test_boot_lower_below_mean_and_small_guard():
    """부트스트랩 2.5% 하단 < 평균(보수적), 표본<5는 센티넬(-1.0)."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.02, 0.1, 500)
    lo = _boot_lower(r, seed=0)
    assert lo < r.mean()
    assert _boot_lower(np.array([0.1, 0.2, 0.3, 0.4])) == -1.0  # n=4 < 5 가드


def test_fold_slices_learns_theta_on_train_only():
    """θ는 TRAIN feature 분위에서만 학습 → TEST에 적용. look-ahead 없음."""
    fold = FOLDS[2]  # train 2019~2022, test 2022~2023
    entry = np.array(["2020-01-01", "2020-06-01", "2020-09-01", "2021-01-01",
                      "2021-06-01", "2022-03-01", "2022-06-01", "2022-09-01"],
                     dtype=object)
    feature = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.05, 0.6])
    value = np.array([1.0, -1.0, 2.0, -1.0, 3.0, 4.0, -1.0, 2.0])
    out = fold_slices(entry, feature, value, fold, frac=0.5, min_train=3)
    assert out is not None
    theta, base, sel = out
    # θ = TRAIN(5개 진입, feature 0.1~0.5)의 상위50% 분위 = 0.3
    assert abs(theta - 0.3) < 1e-9
    # base = TEST 전부(3개), sel = TEST 중 feature>=θ (0.9, 0.6 → 2개)
    assert len(base) == 3
    assert len(sel) == 2
    # TRAIN 부족이면 None
    assert fold_slices(entry, feature, value, fold, frac=0.5, min_train=99) is None


def test_oos_fixed_evaluates_each_fold_test_window():
    """oos_fixed는 fold마다 TEST 슬라이스의 stat을 낸다(고정 params 재최적화 없음)."""
    # simulate: params 무시, 연도별로 다른 진입/수익 고정 반환.
    edates = np.array([f"{y}-06-01" for y in range(2020, 2026)], dtype=object)
    rets = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])

    def simulate(_params):
        return rets, edates

    arr = oos_fixed(FOLDS, simulate, {"any": 1})
    assert len(arr) == len(FOLDS)
    # 각 fold TEST에 진입 1건 → 그 수익이 기대값(마지막 fold는 2025~2027, 2025건만)
    assert np.allclose(arr, [0.1, 0.2, -0.3, 0.4, -0.5, 0.6])


def test_purge_removes_boundary_straddling_train_trade():
    """경계를 걸친 트레이드(진입 TRAIN·실현 TEST)는 TRAIN에서 purge, TEST는 불변."""
    fold = FOLDS[2]  # train 2019~2022, test 2022~2023 (train_hi == test_lo == 2022-01-01)
    entry = np.array([
        "2021-06-01",   # 0 TRAIN, 잘 앞선 실현 → 유지
        "2021-12-20",   # 1 TRAIN 진입인데 실현이 TEST로 넘어감 → purge
        "2022-06-01",   # 2 TEST
    ], dtype=object)
    # (a) 명시적 exit_dates
    exit_dates = np.array(["2021-06-10", "2022-01-10", "2022-06-20"], dtype=object)
    m = purge_embargo(entry, fold, exit_dates=exit_dates)
    assert list(m.train) == [True, False, False]  # 0 유지·1 purge·2 는 TEST(TRAIN 아님)
    assert list(m.test) == [False, False, True]  # TEST 창은 그대로
    # (b) max_hold 로도 동일: 진입 2021-12-20 + 30달력일 = 2022-01-19 >= test_lo
    m2 = purge_embargo(entry, fold, max_hold=30)
    assert list(m2.train) == [True, False, False]
    assert list(m2.test) == [False, False, True]
    # 실현이 경계 못 넘으면(짧은 보유) purge 안 함
    m3 = purge_embargo(entry, fold, max_hold=1)
    assert list(m3.train) == [True, True, False]


def test_embargo_drops_train_trades_in_pre_test_buffer():
    """embargo_days>0 이면 test_lo 직전 완충구간 진입 TRAIN 제거(실현이 경계 전이어도)."""
    fold = FOLDS[2]  # test_lo == 2022-01-01
    entry = np.array([
        "2021-06-01",   # 0 완충 밖 → 유지
        "2021-12-20",   # 1 [2021-12-02, 2022-01-01) 완충 안 → 제거
        "2022-06-01",   # 2 TEST(embargo는 TRAIN만 건드림)
    ], dtype=object)
    # 실현이 전부 경계 이전이라 purge 단독으론 아무것도 안 빠진다.
    exit_dates = np.array(["2021-06-05", "2021-12-25", "2022-06-10"], dtype=object)
    base = purge_embargo(entry, fold, exit_dates=exit_dates, embargo_days=0)
    assert list(base.train) == [True, True, False]  # embargo 0 → 완충 없음
    m = purge_embargo(entry, fold, exit_dates=exit_dates, embargo_days=30)
    # embargo_lo = 2022-01-01 - 30d = 2021-12-02; 진입 2021-12-20 >= 그것 → 제거
    assert list(m.train) == [True, False, False]
    assert list(m.test) == [False, False, True]  # TEST 평가창 불변


def test_purge_embargo_default_reproduces_entry_mask():
    """기본(exit/max_hold 없음·embargo 0)은 entry_mask 소속을 정확히 재현(회귀 고정)."""
    edates = np.array([
        "2019-06-01", "2021-12-31", "2022-01-01", "2022-06-01", "2024-01-01",
        "2025-06-01",
    ], dtype=object)
    for f in FOLDS:
        m = purge_embargo(edates, f)
        assert list(m.train) == list(entry_mask(edates, f.train_lo, f.train_hi))
        assert list(m.test) == list(entry_mask(edates, f.test_lo, f.test_hi))


def test_sensitivity_table_shape():
    """표 행수 == 모든 grid 값 개수 합. 각 행에 train/oos stat dict."""
    edates = np.array([f"{y}-06-01" for y in range(2018, 2025)], dtype=object)
    rets = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15])

    def simulate(_params):
        return rets, edates

    grids = {"window": [3, 5, 8], "stop": [0.05, 0.10]}
    center = {"window": 5, "stop": 0.10}
    rows = sensitivity_table(simulate, center, grids, split="2022-01-01")
    assert len(rows) == 3 + 2  # grid 값 총합
    assert {r["param"] for r in rows} == {"window", "stop"}
    # center 값 행이 정확히 표시된다
    centers = [r for r in rows if r["is_center"]]
    assert {(r["param"], r["value"]) for r in centers} == {("window", 5), ("stop", 0.10)}
    # 각 행에 train/oos stat dict
    for r in rows:
        assert "expectancy" in r["train"] and "n" in r["oos"]
