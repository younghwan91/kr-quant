"""Naver consensus parsing. Pure JSON in -> numbers out (no network)."""

from __future__ import annotations

from kr_quant.collectors.naver_consensus import parse_consensus


def test_parse_consensus_extracts_target_and_recommendation():
    payload = {"consensusInfo": {
        "itemCode": "005930", "createDate": "2026-07-09",
        "recommMean": "4.04", "priceTargetMean": "513,958",
    }}
    tm, rm, bd = parse_consensus(payload)
    assert tm == 513958.0
    assert rm == 4.04
    assert bd == "2026-07-09"


def test_parse_consensus_no_coverage_returns_none():
    assert parse_consensus({}) == (None, None, None)
    assert parse_consensus({"consensusInfo": {}}) == (None, None, None)


def test_parse_consensus_handles_partial_fields():
    tm, rm, bd = parse_consensus({"consensusInfo": {"priceTargetMean": "1,000"}})
    assert tm == 1000.0
    assert rm is None
    assert bd is None
