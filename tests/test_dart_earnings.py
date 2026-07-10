"""DART earnings parsing. Pure JSON in -> numbers out (no network)."""

from __future__ import annotations

import math

from kr_quant.collectors.dart_earnings import (
    parse_net_income,
    yoy_growth,
)


def _payload(rows, status="000"):
    return {"status": status, "list": rows}


def test_parse_net_income_picks_current_and_prior():
    payload = _payload([
        {"account_nm": "매출액", "thstrm_amount": "1,000", "frmtrm_amount": "900"},
        {"account_nm": "당기순이익", "thstrm_amount": "2,206,125", "frmtrm_amount": "974,571"},
    ])
    ni, nip = parse_net_income(payload)
    assert ni == 2206125.0
    assert nip == 974571.0


def test_parse_net_income_handles_loss_label_and_blanks():
    payload = _payload([
        {"account_nm": "당기순이익(손실)", "thstrm_amount": "-500", "frmtrm_amount": ""},
    ])
    ni, nip = parse_net_income(payload)
    assert ni == -500.0
    assert nip is None


def test_parse_net_income_missing_or_error_returns_none():
    assert parse_net_income({"status": "013"}) == (None, None)  # no data
    assert parse_net_income(_payload([{"account_nm": "자산총계", "thstrm_amount": "5"}])) == (None, None)


def test_yoy_growth_math_and_guards():
    assert yoy_growth(200.0, 100.0) == 1.0            # +100%
    assert yoy_growth(50.0, 100.0) == -0.5            # -50%
    assert yoy_growth(10.0, -20.0) == 1.5             # divides by |prior|
    assert yoy_growth(10.0, 0) is None                # no divide-by-zero
    assert yoy_growth(None, 100.0) is None
