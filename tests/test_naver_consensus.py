"""Naver consensus parsing. Pure JSON in -> numbers out (no network)."""

from __future__ import annotations

from kr_quant.collectors.naver_consensus import parse_consensus, parse_estimate


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


def _finance(titles, eps_cols):
    return {"financeInfo": {
        "trTitleList": titles,
        "rowList": [{"title": {"name": "EPS"}, "columns": eps_cols}],
    }}


def test_parse_estimate_extracts_forward_and_prior_eps():
    payload = _finance(
        [{"key": "202412", "isConsensus": "N"}, {"key": "202512", "isConsensus": "N"},
         {"key": "202612", "isConsensus": "Y"}],
        {"202412": {"value": "4,950"}, "202512": {"value": "6,564"},
         "202612": {"value": "46,664"}},
    )
    fwd, prev, ey = parse_estimate(payload)
    assert fwd == 46664.0        # 2026 consensus estimate
    assert prev == 6564.0        # latest actual (2025)
    assert ey == "202612"


def test_parse_estimate_no_consensus_year_returns_none():
    payload = _finance([{"key": "202512", "isConsensus": "N"}], {"202512": {"value": "1"}})
    assert parse_estimate(payload) == (None, None, None)
    assert parse_estimate({}) == (None, None, None)
