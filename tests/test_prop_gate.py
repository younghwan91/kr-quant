"""prop_gate 하버스 — DB-free 합성 단위 테스트 (Step 0 게이트 배터리).

전부 손으로 만든 배열/합성으로 검증한다:
  (1) 모든 폴드에 알려진 양(+)의 엣지가 있으면 clean-OOS 폴드 카운트가 높다.
  (2) 순수 노이즈(평균 0)는 폴드 일관성이 chance 수준, 음성대조도 chance.
  (3) 반환 dict 에 기대 키가 있고 bool PASS/FAIL·verdict 키가 **재귀적으로** 없다.
  (4) R-멀티플 스케일링(÷stop)·미접촉창 슬라이싱이 손으로 만든 날짜에서 정확.
"""

from __future__ import annotations

import numpy as np

from research.experiments.prop_gate import (
    prop_gate,
    random_entry_control,
)

# --- 판정 키 부재(재귀) ----------------------------------------------------------

_VERDICT_KEYS = {"pass", "fail", "passed", "failed", "ok", "verdict",
                 "go", "nogo", "deploy", "accept", "reject"}


def _assert_no_boolean_verdict(obj) -> None:
    """반환 구조(중첩 dict/list 포함)에 bool 판정 값·판정 키가 없어야 한다.

    단, positive/inside_train 등 서술적 bool 플래그는 폴드 행 안의 사실 라벨이지
    배포 판정이 아니므로 허용한다 — 금지 대상은 _VERDICT_KEYS 키뿐.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                assert k.lower() not in _VERDICT_KEYS, f"판정 키 발견: {k}"
            _assert_no_boolean_verdict(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_no_boolean_verdict(v)


def _positive_edge(seed: int = 0, n: int = 800):
    """모든 연도에 동일한 양(+)의 엣지: 45% 손절(-1R), 55% 양(+) 지수꼬리 +0.05 쉬프트."""
    rng = np.random.default_rng(seed)
    years = rng.integers(2018, 2027, size=n)
    entry = np.array([f"{y:04d}-06-15" for y in years])
    ret = np.where(rng.random(n) < 0.40, -0.10, rng.exponential(0.06, n) + 0.06)
    return entry, ret


def _pure_noise(seed: int = 0, n: int = 2500):
    """평균 0 노이즈: 대칭 정규 수익(엣지 없음). 손절폭 stop=0.10 로 정규화."""
    rng = np.random.default_rng(seed)
    years = rng.integers(2018, 2027, size=n)
    entry = np.array([f"{y:04d}-06-15" for y in years])
    ret = rng.normal(0.0, 0.10, n)
    return entry, ret


# --- (1) 알려진 양(+)의 엣지 → 높은 clean-OOS 폴드 카운트 -------------------------

def test_positive_edge_high_clean_fold_count():
    entry, ret = _positive_edge(seed=1)
    rep = prop_gate(entry, ret, stop=0.10, label="edge", verbose=False)
    fb = rep["folds"]
    # 모든 폴드에 엣지가 있으니 clean-OOS 폴드 대부분이 양수여야 한다
    assert fb["clean_oos_valid"] >= 3
    assert fb["clean_oos_positive"] >= fb["clean_oos_valid"] - 1
    # OOS 기준 비용 기대값 R > 0
    assert rep["gate_report"]["expectancy_R"] > 0
    # 미접촉창도 표본이 있고 양(+) 방향
    assert rep["untouched"]["n"] > 0


# --- (2) 순수 노이즈 → chance 수준 폴드 일관성, 음성대조도 chance ------------------

def test_pure_noise_is_chance_level():
    entry, ret = _pure_noise(seed=2)
    rep = prop_gate(entry, ret, stop=0.10, label="noise", verbose=False)
    fb = rep["folds"]
    # 노이즈: 전 폴드 양수(clean 전부)는 아니어야 한다(chance ~50%)
    assert fb["clean_oos_positive"] < fb["clean_oos_valid"] or fb["clean_oos_valid"] <= 1
    # OOS 기대값 R 은 0 근처
    assert abs(rep["gate_report"]["expectancy_R"]) < 0.15


def test_negative_control_reports_false_positive_rate():
    entry, ret = _pure_noise(seed=3, n=600)
    ctrl = random_entry_control(entry, ret, stop=0.10, n_per_draw=len(ret),
                                n_draws=40, seed=5, verbose=False)
    assert ctrl["n_draws"] == 40
    assert len(ctrl["raw_fold_positive"]) == 40
    assert len(ctrl["clean_fold_positive"]) == 40
    # 위양성률은 [0,1] 비율로 보고된다(게이트 느슨함 보정 숫자)
    assert 0.0 <= ctrl["raw_ge5_frac"] <= 1.0
    assert 0.0 <= ctrl["clean_all_positive_frac"] <= 1.0
    # 노이즈 풀이므로 OOS 기대값 평균은 0 근처
    assert abs(ctrl["oos_expectancy_R_mean"]) < 0.15


# --- (3) 반환 dict 구조 + 판정 키 부재 -------------------------------------------

def test_return_dict_keys_and_no_verdict():
    entry, ret = _positive_edge(seed=4)
    rep = prop_gate(entry, ret, stop=0.10, verbose=False)
    expected = {"label", "stop", "n_total", "entry_range", "primary_cost",
                "cost_sweep", "cost_edge_dies", "folds", "distribution",
                "fragility", "untouched", "gate_report"}
    assert expected <= set(rep)
    # 슬리피지 스윕은 비용 하나당 한 행
    assert len(rep["cost_sweep"]) == 4
    # 폴드 블록: raw 는 6폴드, clean 은 그 부분집합
    assert rep["folds"]["raw_valid"] <= 6
    assert rep["folds"]["clean_oos_valid"] <= rep["folds"]["raw_valid"]
    # 재귀적으로 bool 판정 키 없음(REPORTER)
    _assert_no_boolean_verdict(rep)


# --- (4) R-멀티플 스케일링 + 미접촉창 슬라이싱 정확성 ------------------------------

def test_r_multiple_scaling_by_stop():
    # 손으로 만든 수익: 진입 비용 0 가정 위해 costs=(0.0,) → R = ret/stop
    entry = np.array(["2023-06-15", "2023-06-16", "2023-06-17", "2023-06-18"])
    ret = np.array([0.30, -0.10, 0.15, 0.05])
    rep = prop_gate(entry, ret, stop=0.10, costs=(0.0,), verbose=False)
    # OOS(≥2022) = 전부 4건, 기대값 R = mean([3,-1,1.5,0.5]) = 1.0
    assert rep["gate_report"]["n"] == 4
    assert abs(rep["gate_report"]["expectancy_R"] - 1.0) < 1e-9
    # stop 절반이면 R 두 배 → 기대값 2.0
    rep2 = prop_gate(entry, ret, stop=0.05, costs=(0.0,), verbose=False)
    assert abs(rep2["gate_report"]["expectancy_R"] - 2.0) < 1e-9


def test_untouched_window_slicing():
    # 진입일을 미접촉창 [2025-07-01, 2026-07-01) 안팎으로 배치
    entry = np.array([
        "2025-06-30",  # 창 직전 (제외)
        "2025-07-01",  # 창 시작 (포함)
        "2025-12-01",  # 창 안 (포함)
        "2026-06-30",  # 창 끝 직전 (포함)
        "2026-07-01",  # 창 끝 (제외)
        "2024-01-01",  # 창 밖 (제외)
    ])
    ret = np.array([0.10, 0.20, 0.30, 0.40, 0.50, -0.10])
    rep = prop_gate(entry, ret, stop=0.10, costs=(0.0,), verbose=False)
    u = rep["untouched"]
    # 포함되는 3건: 0.20, 0.30, 0.40 → R = [2,3,4], 기대값 3.0
    assert u["n"] == 3
    assert abs(u["expectancy_R"] - 3.0) < 1e-9


def test_cost_sweep_edge_dies_monotone():
    # 얇은 양(+) 엣지가 비용 상승에 따라 죽는지: OOS 기대값 R 이 비용 증가에 단조 감소
    entry, ret = _positive_edge(seed=6)
    rep = prop_gate(entry, ret, stop=0.10, verbose=False)
    exps = [cs["oos_expectancy_R"] for cs in rep["cost_sweep"]]
    # 비용 오름차순이므로 기대값은 비증가(같은 R 에서 상수 차감)
    assert all(exps[i] >= exps[i + 1] - 1e-12 for i in range(len(exps) - 1))


# --- 다중검정 원장 배선 -------------------------------------------------------

def test_prop_gate_records_config_and_derives_n_trials(tmp_path, monkeypatch):
    """config 를 주면 원장에 적히고 N 이 거기서 나온다 — 손으로 세지 않는다."""
    import numpy as np

    from kr_quant.diagnostics import trials
    from research.experiments.prop_gate import prop_gate

    monkeypatch.setattr(trials, "_repo_root", lambda: tmp_path)
    rng = np.random.default_rng(0)
    n = 400
    entry = np.array([f"2023-{1 + i % 12:02d}-15" for i in range(n)])
    ret = rng.normal(0.01, 0.05, n)

    rep = prop_gate(entry, ret, 0.10, label="tl", config={"a": 1}, verbose=False)
    assert rep["gate_report"]["deflation"]["n_trials"] == 1

    # 같은 config 재실행 → 시행 아님
    rep = prop_gate(entry, ret, 0.10, label="tl", config={"a": 1}, verbose=False)
    assert rep["gate_report"]["deflation"]["n_trials"] == 1

    # 다른 config → 시행
    rep = prop_gate(entry, ret, 0.10, label="tl", config={"a": 2}, verbose=False)
    assert rep["gate_report"]["deflation"]["n_trials"] == 2


def test_prop_gate_without_config_is_unchanged(tmp_path, monkeypatch):
    """config 미지정이면 원장을 만들지도, deflation 을 보고하지도 않는다."""
    import numpy as np

    from kr_quant.diagnostics import trials
    from research.experiments.prop_gate import prop_gate

    monkeypatch.setattr(trials, "_repo_root", lambda: tmp_path)
    rng = np.random.default_rng(1)
    n = 300
    entry = np.array([f"2023-{1 + i % 12:02d}-15" for i in range(n)])
    rep = prop_gate(entry, rng.normal(0.01, 0.05, n), 0.10, label="none", verbose=False)
    assert rep["gate_report"].get("deflation") is None
    assert not (tmp_path / "research").exists()
