"""가드레일 린트 자체의 회귀 테스트 — 린트를 검사하는 린트.

이 파일이 없어서 실제로 일이 났다. 규칙 (e)가 **파일 단위**로 구현돼 있어서, 파일
어딘가에 ``config=`` 가 한 번만 있으면 같은 파일의 나머지 게이트 호출이 전부 면제됐다.
그 결과 민감도 스윕 18~27칸이 다중검정 원장을 통째로 우회했고 Deflated Sharpe 의
haircut 이 0 이 됐는데도 CI 는 초록이었다. **막으려던 실패 모드를 린트가 못 보고 있었고,
린트를 검사하는 것이 아무것도 없어서 아무도 몰랐다.**

그래서 여기서는 "위반 없는 트리에서 통과한다"만 보지 않는다 — 각 규칙에 대해 **위반을
주입하면 실제로 실패하는지**를 확인한다. 통과만 보는 테스트는 규칙이 죽어도 초록이다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_guardrails", REPO / "scripts" / "check_guardrails.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def lint():
    return _load()


def test_repo_is_currently_clean(lint):
    """현재 트리는 모든 규칙을 통과한다."""
    assert lint.main() == 0


# --- (e) 호출 지점 단위 원장 규칙 ------------------------------------------------
#
# 아래 세 케이스가 이 규칙의 핵심이다. 특히 `two_calls_one_config` 는 예전 파일 단위
# 구현이 놓쳤던 바로 그 모양이다.

_TWO_CALLS_ONE_CONFIG = '''
from prop_gate import prop_gate

def preregistered(a, b):
    return prop_gate(a, b, 0.04, label="x", config={"stop": 0.04}, log_dir="d")

def sweep(a, b):
    rows = []
    for s in (0.03, 0.04, 0.05):
        rows.append(prop_gate(a, b, s, label=f"sens_{s}", verbose=False))
    return rows
'''

_MULTILINE_CALL_NO_CONFIG = '''
from prop_swing_common import gate_sim

def sweep(a, b):
    return gate_sim(
        a, b, 0.04,
        "sens",
        log_dir="d",
    )
'''

_MULTILINE_CALL_WITH_CONFIG = '''
from prop_swing_common import gate_sim

def sweep(a, b):
    return gate_sim(
        a, b, 0.04,
        "sens",
        config={"stop": 0.04},
        log_dir="d",
    )
'''


def _run_rule_e(lint, tmp_path, monkeypatch, source: str) -> list[str]:
    gate = tmp_path / "fake_gate.py"
    gate.write_text(source, encoding="utf-8")
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "EXPERIMENTS", tmp_path)
    return lint.check_gates_record_trials()


def test_rule_e_catches_second_call_without_config(lint, tmp_path, monkeypatch):
    """파일에 config= 가 이미 한 번 있어도, config 없는 다른 호출은 잡아야 한다.

    예전 파일 단위 구현이 통과시키던 정확한 모양이다 — 이 테스트가 그 회귀를 막는다.
    """
    out = _run_rule_e(lint, tmp_path, monkeypatch, _TWO_CALLS_ONE_CONFIG)
    assert len(out) == 1, out
    assert "prop_gate()" in out[0]
    assert ":10" in out[0]      # 사전등록 호출(5행)이 아니라 스윕 호출을 지목해야 한다


def test_rule_e_catches_multiline_call(lint, tmp_path, monkeypatch):
    """여러 줄로 쪼갠 호출도 잡아야 한다(정규식 구현이 못 보던 모양)."""
    out = _run_rule_e(lint, tmp_path, monkeypatch, _MULTILINE_CALL_NO_CONFIG)
    assert len(out) == 1, out
    assert "gate_sim()" in out[0]


def test_rule_e_accepts_multiline_call_with_config(lint, tmp_path, monkeypatch):
    """config= 를 제대로 넘긴 여러 줄 호출은 통과 — 거짓 양성이 없어야 규칙이 살아남는다."""
    assert _run_rule_e(lint, tmp_path, monkeypatch, _MULTILINE_CALL_WITH_CONFIG) == []


def test_rule_e_ignores_the_harness_definition(lint, tmp_path, monkeypatch):
    """하버스 자신의 def 안에서의 호출은 대상이 아니다(자기 자신을 고발하면 규칙이 죽는다)."""
    src = '''
def gate_sim(a, b, stop, label, *, config, log_dir):
    return prop_gate(a, b, stop, label=label, config=config, log_dir=log_dir)
'''
    assert _run_rule_e(lint, tmp_path, monkeypatch, src) == []


# --- (a) src → research 경계 ----------------------------------------------------

def test_rule_a_catches_research_import(lint, tmp_path, monkeypatch):
    (tmp_path / "leaky.py").write_text(
        "from research.signals import contrarian_retail\n", encoding="utf-8"
    )
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "SRC", tmp_path)
    out = lint.check_src_no_research_import()
    assert len(out) == 1 and "boundary" in out[0]


# --- (b) 게이트 → VERDICT -------------------------------------------------------

def test_rule_b_catches_gate_without_verdict(lint, tmp_path, monkeypatch):
    (tmp_path / "orphan_gate.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "EXPERIMENTS", tmp_path)
    monkeypatch.setattr(lint, "LOGS", tmp_path / "logs")
    out = lint.check_gates_have_verdict()
    assert len(out) == 1 and "verdict" in out[0]


# --- (c) 하드코딩 판정 ----------------------------------------------------------

def test_rule_c_catches_literal_verdict(lint, tmp_path, monkeypatch):
    (tmp_path / "judgey.py").write_text('verdict = "PASS"\n', encoding="utf-8")
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "EXPERIMENTS", tmp_path)
    out = lint.check_no_literal_verdict()
    assert len(out) == 1 and "reporter" in out[0]


def test_rule_c_ignores_prose_in_comments(lint, tmp_path, monkeypatch):
    (tmp_path / "prose.py").write_text(
        '# PASS/FAIL bool 도 두지 않는다\nx = 1\n', encoding="utf-8"
    )
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "EXPERIMENTS", tmp_path)
    assert lint.check_no_literal_verdict() == []


# --- (g2) 생존자 전용 조인 -------------------------------------------------------

_DELETED_LOADER_SHAPE = '''
QUERY = """
    SELECT sd.code, sd.date, sd.close, s.name, s.market
    FROM supply_demand sd
    JOIN stocks s ON s.code = sd.code
    WHERE sd.date >= ?
"""
'''


def test_rule_g2_catches_multiline_survivor_join(lint, tmp_path, monkeypatch):
    """2026-08-16 에 삭제한 로더 3개의 실제 모양 — 여러 줄 SQL 이라 한 줄 정규식은 못 봤다."""
    (tmp_path / "loader.py").write_text(_DELETED_LOADER_SHAPE, encoding="utf-8")
    monkeypatch.setattr(lint, "REPO", tmp_path)
    out = lint.check_no_survivor_only_join()
    assert len(out) == 1, out
    assert "universe" in out[0] and "delisted_stocks" in out[0]


def test_rule_g2_allows_supply_demand_without_the_join(lint, tmp_path, monkeypatch):
    """폐지분을 배제하지 않는 supply_demand 읽기는 통과해야 한다(거짓 양성 방지)."""
    (tmp_path / "ok.py").write_text(
        'Q = "SELECT code, date, individual FROM supply_demand WHERE date >= ?"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(lint, "REPO", tmp_path)
    assert lint.check_no_survivor_only_join() == []
