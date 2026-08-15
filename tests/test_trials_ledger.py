"""다중검정 원장 — 시행 수를 사람 기억이 아니라 파일에서 읽는다.

Deflated Sharpe 의 N 을 손으로 적으면 규율이 아니라 부탁이 된다. 이 원장이 그 N 을
게이트 호출에서 자동으로 만들어낸다. 여기서 지키는 성질은 두 가지 —
**재실행은 시행이 아니다**(중복 제거)와 **다른 config 는 시행이다**(누락 없음).
"""

from __future__ import annotations

import json

from kr_quant.diagnostics.trials import (
    config_fingerprint,
    count_trials,
    ledger_path,
    read_trials,
    record_trial,
)

CFG_A = {"window": 8, "stop": 0.10, "hold": 60}
CFG_B = {"window": 8, "stop": 0.10, "hold": 20}


def test_absent_ledger_counts_zero(tmp_path):
    assert count_trials("nope", logs_dir=tmp_path) == 0
    assert read_trials("nope", logs_dir=tmp_path) == []


def test_distinct_configs_each_count_once(tmp_path):
    record_trial("x", CFG_A, logs_dir=tmp_path)
    record_trial("x", CFG_B, logs_dir=tmp_path)
    assert count_trials("x", logs_dir=tmp_path) == 2


def test_rerunning_the_same_config_is_not_a_new_trial(tmp_path):
    """시행은 '다르게 시도한 횟수'지 '실행 횟수'가 아니다.

    같은 게이트를 두 번 돌렸다고 N 이 2가 되면 DSR 이 거짓으로 깎인다.
    """
    for _ in range(5):
        record_trial("x", CFG_A, logs_dir=tmp_path)
    assert count_trials("x", logs_dir=tmp_path) == 1
    assert len(read_trials("x", logs_dir=tmp_path)) == 1


def test_key_order_does_not_change_the_fingerprint():
    a = config_fingerprint({"a": 1, "b": 2})
    b = config_fingerprint({"b": 2, "a": 1})
    assert a == b


def test_labels_have_independent_ledgers(tmp_path):
    """음성대조(label='rand')가 실제 셋업의 N 을 오염시키면 안 된다."""
    record_trial("real", CFG_A, logs_dir=tmp_path)
    for i in range(10):
        record_trial("rand", {**CFG_A, "seed": i}, logs_dir=tmp_path)
    assert count_trials("real", logs_dir=tmp_path) == 1
    assert count_trials("rand", logs_dir=tmp_path) == 10


def test_ledger_is_append_only_jsonl_next_to_the_verdict(tmp_path):
    record_trial("x", CFG_A, note="사전등록", logs_dir=tmp_path)
    record_trial("x", CFG_B, logs_dir=tmp_path)
    p = ledger_path("x", logs_dir=tmp_path)
    assert p.name == "TRIALS.jsonl" and p.parent.name == "x"

    lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [r["config"] for r in lines] == [CFG_A, CFG_B], "기록 순서가 보존돼야 한다"
    assert lines[0]["note"] == "사전등록"


def test_corrupt_lines_do_not_break_counting(tmp_path):
    record_trial("x", CFG_A, logs_dir=tmp_path)
    p = ledger_path("x", logs_dir=tmp_path)
    with p.open("a", encoding="utf-8") as fh:
        fh.write("{ this is not json\n\n")
    record_trial("x", CFG_B, logs_dir=tmp_path)
    assert count_trials("x", logs_dir=tmp_path) == 2
