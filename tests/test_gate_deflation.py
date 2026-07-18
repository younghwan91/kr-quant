"""gate_report 의 다중검정 보정(Deflated Sharpe + t-haircut) — DB-free 합성 단위 테스트.

검증(§4·5·11 공백 폐쇄):
  (1) DSR: deflated Sharpe < raw Sharpe 이고, 시도 config 수 N 이 커지면 더 깎인다
      (E[maxSharpe|H0]↑ → deflated↓, P(Sharpe>SR0)↓, t-haircut 배수↑).
  (2) n_trials=None 이면 deflation 블록이 없다(하위호환 — 기본 동작 불변).
  (3) deflation 이 있어도 bool 판정 키가 **재귀적으로** 없다(REPORTER 불변).
"""

from __future__ import annotations

import numpy as np

from kr_quant.diagnostics.gate_report import deflated_sharpe, gate_report

# --- 판정 키 부재(재귀) — test_prop_gate 와 동일 잣대 --------------------------------

_VERDICT_KEYS = {"pass", "fail", "passed", "failed", "ok", "verdict",
                 "go", "nogo", "deploy", "accept", "reject"}


def _assert_no_boolean_verdict(obj) -> None:
    """반환 구조(중첩 dict/list)에 bool 판정 값·판정 키가 없어야 한다."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                assert k.lower() not in _VERDICT_KEYS, f"판정 키 발견: {k}"
            _assert_no_boolean_verdict(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_no_boolean_verdict(v)


def _edge_R(seed: int = 0, n: int = 400) -> np.ndarray:
    """양(+)의 엣지가 있는 R-멀티플 표본: 손절 −1R 절단 + 오른꼬리."""
    rng = np.random.default_rng(seed)
    ret = np.where(rng.random(n) < 0.40, -0.10, rng.exponential(0.08, n) + 0.04)
    return ret / 0.10


# --- (1) DSR < raw Sharpe, N 증가에 단조 깎임 --------------------------------------

def test_deflated_sharpe_below_raw_and_decreasing_in_trials():
    R = _edge_R(seed=1)
    mu, sd = R.mean(), R.std(ddof=1)
    raw_sharpe = mu / sd
    assert raw_sharpe > 0  # 합성 엣지가 양(+) 이어야 의미 있는 검정

    reps = [gate_report(R, n_trials=n) for n in (1, 10, 100, 1000)]
    defl = [r["deflation"] for r in reps]

    # observed_sharpe 는 N 과 무관하게 동일
    for d in defl:
        assert abs(d["observed_sharpe"] - raw_sharpe) < 1e-9

    # deflated Sharpe < raw Sharpe (N≥2 에서 벤치마크 SR0>0 이므로 깎인다)
    for d in defl[1:]:
        assert d["deflated_sharpe"] < d["observed_sharpe"]

    # N 이 커질수록: E[maxSharpe|H0] 상승 → deflated 하락 → P(Sharpe>SR0) 하락
    sr0 = [d["expected_max_sharpe_h0"] for d in defl]
    dfl = [d["deflated_sharpe"] for d in defl]
    pgt = [d["prob_deflated_sharpe"] for d in defl]
    hair = [d["t_haircut"]["haircut_multiple"] for d in defl]
    assert sr0 == sorted(sr0)              # 비감소
    assert dfl == sorted(dfl, reverse=True)  # 비증가
    assert pgt == sorted(pgt, reverse=True)  # 비증가
    assert hair == sorted(hair)            # t-haircut 배수는 N 과 함께 상승
    assert sr0[0] == 0.0                   # N=1 은 뽑을 게 하나 → 벤치마크 0


def test_prob_sharpe_gt0_is_probability():
    R = _edge_R(seed=2)
    d = gate_report(R, n_trials=50)["deflation"]
    assert 0.0 <= d["prob_sharpe_gt0"] <= 1.0
    assert 0.0 <= d["prob_deflated_sharpe"] <= 1.0
    # 관측 Sharpe>0 → SR0 벤치마크가 더 높으니 P(>SR0) ≤ P(>0)
    assert d["prob_deflated_sharpe"] <= d["prob_sharpe_gt0"] + 1e-12


def test_helper_matches_report_block():
    R = _edge_R(seed=3)
    mu, sd = R.mean(), R.std(ddof=1)
    sr = mu / sd
    sdp = R.std()
    skew = ((R - mu) ** 3).mean() / sdp ** 3
    kurt = ((R - mu) ** 4).mean() / sdp ** 4
    direct = deflated_sharpe(sr, 25, len(R), skew, kurt)
    block = gate_report(R, n_trials=25)["deflation"]
    assert direct == block


# --- (2) n_trials=None → deflation 블록 없음 (하위호환) ------------------------------

def test_no_trials_omits_deflation_block():
    R = _edge_R(seed=4)
    rep = gate_report(R)  # n_trials 기본 None
    assert "deflation" not in rep
    # 기존 키는 그대로
    assert {"n", "expectancy_R", "expectancy_ci", "monster_share",
            "max_loss_streak"} <= set(rep)


# --- (3) deflation 있어도 REPORTER 불변 (bool 판정 키 없음) --------------------------

def test_deflation_present_still_no_verdict_key():
    R = _edge_R(seed=5)
    rep = gate_report(
        R,
        n_trials=100,
        fold_expectancies=[0.1, -0.05, 0.2, 0.0],
        cost_curve={0.0046: 0.3, 0.02: -0.1},
    )
    assert "deflation" in rep
    _assert_no_boolean_verdict(rep)


# --- 퇴화 방어: 표본<2 여도 예외 없이 NaN --------------------------------------------

def test_degenerate_sample_is_nan_safe():
    d = gate_report(np.array([0.5]), n_trials=10)["deflation"]
    assert np.isnan(d["observed_sharpe"])
    assert np.isnan(d["prob_sharpe_gt0"])
    # t-haircut 은 표본과 무관하게 계산된다
    assert d["t_haircut"]["haircut_multiple"] > 1.0
