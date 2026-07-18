"""diagnostics 모듈 — DB-free 합성 단위 테스트 (R-멀티플 분포·취약성·게이트 리포터).

전부 손으로 만든 배열로 검증한다: (1) r_multiples 는 손절폭으로 스케일된다,
(2) dist_shape 오른꼬리 점유가 알려진 skewed 배열에서 정확히 합산된다,
(3) fragility 괴물 의존도·최장 연패가 손으로 만든 배열에서 정확하다,
(4) gate_report 는 bool PASS/FAIL 키 없는 dict 를 돌려준다.
"""

from __future__ import annotations

import numpy as np

from kr_quant.diagnostics import (
    conviction_analysis,
    dist_shape,
    fragility_report,
    gate_report,
    hold_curve,
    max_loss_streak,
    monster_share,
    r_multiples,
    selection_curve,
    tail_removal,
    win_conditional,
)


# --- r_multiples ----------------------------------------------------------------

def test_r_multiples_scales_by_stop():
    ret = np.array([0.30, -0.10, 0.15])
    # 손절 10% → R = ret/0.10; 손절 5% → 두 배로 스케일
    np.testing.assert_allclose(r_multiples(ret, 0.10), [3.0, -1.0, 1.5])
    np.testing.assert_allclose(r_multiples(ret, 0.05), [6.0, -2.0, 3.0])


def test_r_multiples_drops_nonfinite():
    ret = np.array([0.10, np.nan, np.inf, -0.10])
    np.testing.assert_allclose(r_multiples(ret, 0.10), [1.0, -1.0])


# --- dist_shape -----------------------------------------------------------------

def test_dist_shape_right_tail_share_sums_correctly():
    # 알려진 skewed 배열: wins=[0.5,0.5,4,6] → 양수익질량=11; ≥3R=[4,6] 합=10
    R = np.array([-1.0, -1.0, -1.0, 0.5, 0.5, 4.0, 6.0])
    s = dist_shape(R)
    assert s["n"] == 7
    assert abs(s["pos_mass"] - 11.0) < 1e-12
    rt3 = s["right_tail"][3.0]
    assert abs(rt3["freq"] - 2 / 7) < 1e-12
    assert abs(rt3["share"] - 10 / 11) < 1e-12
    # ≥5R = [6] 만; share = 6/11
    assert abs(s["right_tail"][5.0]["share"] - 6 / 11) < 1e-12
    # 왼꼬리(≤−0.9): 손절 3건 = 3/7
    assert abs(s["left_tail_share"] - 3 / 7) < 1e-12
    assert abs(s["expectancy_R"] - R.mean()) < 1e-12


def test_dist_shape_empty_is_nan_safe():
    s = dist_shape(np.array([]))
    assert s["n"] == 0
    assert np.isnan(s["expectancy_R"])


# --- selection_curve ------------------------------------------------------------

def test_selection_curve_bends_with_feature():
    # feature 강할수록 R 크게 설계 → 상위%로 갈수록 기대값R 상승
    R = np.arange(1.0, 101.0)          # 1..100
    feature = np.arange(1.0, 101.0)    # 완전 상관
    rows = selection_curve(R, feature, fracs=(1.0, 0.10))
    full = next(r for r in rows if r["frac"] == 1.0)
    top = next(r for r in rows if r["frac"] == 0.10)
    assert full["n"] == 100
    assert top["n"] == 10
    assert top["expectancy_R"] > full["expectancy_R"]
    # 상위 10% = 91..100 평균 95.5
    assert abs(top["expectancy_R"] - 95.5) < 1e-9


# --- conviction_analysis --------------------------------------------------------

def test_conviction_analysis_train_oos_and_verdict():
    rng = np.random.default_rng(0)
    n = 400
    feature = rng.normal(0, 1, n)
    # R 이 feature 에 단조 증가 → 예측력 있음
    R = feature * 1.0 + rng.normal(0, 0.1, n)
    # 절반 TRAIN(2021), 절반 OOS(2023)
    entry = np.array(["2021-06-01"] * (n // 2) + ["2023-06-01"] * (n // 2))
    out = conviction_analysis(feature, R, entry, train_hi="2022-01-01", nq=5)
    assert out["nq"] == 5
    assert len(out["train"]) == 5 and len(out["oos"]) == 5
    assert out["train_monotonic"] is True
    assert out["oos_monotonic"] is True
    assert out["verdict"] == "predictive"


def test_conviction_analysis_too_few_is_none():
    feature = np.arange(10.0)
    R = np.arange(10.0)
    entry = np.array(["2021-01-01"] * 10)
    out = conviction_analysis(feature, R, entry, train_hi="2022-01-01", nq=5)
    assert out["train"] is None and out["oos"] is None
    assert out["verdict"] is None


# --- hold_curve -----------------------------------------------------------------

def test_hold_curve_generic_arrays():
    results = {
        60: {"R": np.array([-1.0, -1.0, 2.0, 3.0]), "reason": np.array([1, 1, 2, 0])},
        90: {"R": np.array([-1.0, 4.0, 5.0, 6.0])},
    }
    rows = hold_curve(results, right_tails=(3.0,), top_frac=0.25)
    assert [r["hold"] for r in rows] == [60, 90]
    r60 = rows[0]
    assert abs(r60["expectancy_R"] - 0.75) < 1e-12
    # time_exit_share: reason==0 → 1/4
    assert abs(r60["time_exit_share"] - 0.25) < 1e-12
    # 90 은 reason 없음 → time_exit_share 키 없음
    assert "time_exit_share" not in rows[1]


# --- fragility: monster_share ---------------------------------------------------

def test_monster_share_hand_built():
    # 총합=10, 상위 2건=[5,3]=8 → 0.8
    R = np.array([5.0, 3.0, 1.0, 1.0, -1.0, 1.0])
    assert abs(monster_share(R, k=2) - 0.8) < 1e-12


def test_monster_share_zero_total_is_nan():
    R = np.array([1.0, -1.0])  # 합=0
    assert np.isnan(monster_share(R, k=1))


# --- fragility: max_loss_streak -------------------------------------------------

def test_max_loss_streak_hand_built():
    # 손실(≤0) 연속: [win, loss, loss, loss, win, loss] → 최장 3
    R = np.array([1.0, -1.0, -0.5, -1.0, 2.0, -1.0])
    assert max_loss_streak(R) == 3


def test_max_loss_streak_orders_by_entry():
    # 배열 순서로는 연패가 안 보이지만 시간순으로 정렬하면 2연패
    R = np.array([-1.0, 2.0, -1.0])
    entry = np.array(["2023-01-01", "2023-03-01", "2023-02-01"])
    # 시간순 R = [-1(1월), -1(2월), 2(3월)] → 최장 2연패
    assert max_loss_streak(R, entry=entry) == 2


# --- fragility: tail_removal / win_conditional ----------------------------------

def test_tail_removal_flips_expectancy():
    # 엣지가 상위 1건에 의존: [-1,-1,-1,10] mean=1.75; 상위1 제거 → mean=-1
    R = np.array([-1.0, -1.0, -1.0, 10.0])
    tr = tail_removal(R, k=1)
    assert abs(tr["expectancy_full"] - 1.75) < 1e-12
    assert abs(tr["expectancy_ex"] - (-1.0)) < 1e-12
    assert abs(tr["cum_ex"] - (-3.0)) < 1e-12


def test_win_conditional_fields():
    R = np.array([-1.0, -1.0, 2.0, 4.0])
    wc = win_conditional(R)
    assert abs(wc["win_rate"] - 0.5) < 1e-12
    assert abs(wc["median_win"] - 3.0) < 1e-12
    assert abs(wc["max_win"] - 4.0) < 1e-12
    assert abs(wc["median_loss"] - (-1.0)) < 1e-12


# --- gate_report ----------------------------------------------------------------

_VERDICT_KEYS = {"pass", "fail", "passed", "failed", "ok", "verdict", "go", "nogo", "deploy"}


def _assert_no_boolean_verdict(d: dict) -> None:
    """gate_report 반환 dict(중첩 포함)에 bool 값·판정 키가 없어야 한다."""
    for k, v in d.items():
        assert k.lower() not in _VERDICT_KEYS, f"판정 키 발견: {k}"
        assert not isinstance(v, bool), f"bool 판정 값 발견: {k}={v}"
        if isinstance(v, dict):
            _assert_no_boolean_verdict(v)


def test_gate_report_returns_dict_without_pass_fail():
    rng = np.random.default_rng(1)
    R = rng.normal(0.2, 1.0, 300)
    rep = gate_report(
        R,
        fold_expectancies=[0.1, -0.05, 0.2, 0.15, -0.1, 0.05],
        cost_curve={0.0046: 0.20, 0.008: 0.10, 0.010: -0.02, 0.015: -0.10},
        n_boot=500,
    )
    assert isinstance(rep, dict)
    _assert_no_boolean_verdict(rep)
    # 신호들이 숫자/분포로 보고되는지
    assert rep["n"] == 300
    assert isinstance(rep["expectancy_ci"], tuple) and len(rep["expectancy_ci"]) == 2
    assert rep["fold_consistency"]["n_folds"] == 6
    assert rep["fold_consistency"]["n_positive"] == 4
    # 엣지가 죽는 비용: 기대값이 처음 ≤0 인 0.010
    assert abs(rep["cost_edge_dies"] - 0.010) < 1e-12


def test_gate_report_optional_blocks_absent():
    R = np.array([1.0, -1.0, 2.0, -1.0, 3.0])
    rep = gate_report(R, n_boot=200)
    assert "fold_consistency" not in rep
    assert "cost_edge_dies" not in rep
    _assert_no_boolean_verdict(rep)


def test_gate_report_fragility_report_shape():
    R = np.array([-1.0, -1.0, 2.0, 4.0, -1.0, 6.0])
    fr = fragility_report(R)
    assert set(fr) == {"n", "monster_share", "max_loss_streak", "tail_removal",
                       "median_trade", "win_conditional"}
    assert fr["n"] == 6
