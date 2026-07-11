"""DART earnings parsing. Pure JSON in -> numbers out (no network)."""

from __future__ import annotations


from kr_quant.collectors.dart_earnings import (
    parse_financials,
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


def test_parse_financials_extracts_all_three_lines():
    payload = _payload([
        {"account_nm": "매출액", "thstrm_amount": "1,000", "frmtrm_amount": "900"},
        {"account_nm": "영업이익", "thstrm_amount": "300", "frmtrm_amount": "250"},
        {"account_nm": "당기순이익", "thstrm_amount": "200", "frmtrm_amount": "100"},
    ])
    ni, nip, rev, revp, oi, oip = parse_financials(payload)
    assert (ni, nip) == (200.0, 100.0)
    assert (rev, revp) == (1000.0, 900.0)
    assert (oi, oip) == (300.0, 250.0)


def test_parse_financials_revenue_variant_and_missing_legs():
    # 수익(매출액) 변형은 revenue로 잡히고, 영업이익 없으면 op_income None, 순이익은 정상.
    payload = _payload([
        {"account_nm": "수익(매출액)", "thstrm_amount": "5,000", "frmtrm_amount": "4,000"},
        {"account_nm": "당기순이익(손실)", "thstrm_amount": "-50", "frmtrm_amount": ""},
    ])
    ni, nip, rev, revp, oi, oip = parse_financials(payload)
    assert (ni, nip) == (-50.0, None)
    assert (rev, revp) == (5000.0, 4000.0)
    assert (oi, oip) == (None, None)          # 영업이익 라인 없음 → 크래시 없이 None


def test_parse_financials_error_status_all_none():
    assert parse_financials({"status": "013"}) == (None,) * 6


def test_parse_net_income_still_backward_compatible():
    # 기존 시그니처·동작 불변 (하위호환).
    payload = _payload([
        {"account_nm": "당기순이익", "thstrm_amount": "2,206,125", "frmtrm_amount": "974,571"},
    ])
    assert parse_net_income(payload) == (2206125.0, 974571.0)
