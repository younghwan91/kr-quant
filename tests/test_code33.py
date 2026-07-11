"""Code 33 acceleration flag — synthetic financials panel in → is_code33 out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.features.fundamentals import code33_panel

_AVAILS = ["2019-05-15", "2019-08-15", "2019-11-15", "2020-03-30", "2020-05-15"]


def _rows(code, netinc, revenue, op_income, prior_netinc=100.0, prior_rev=1000.0, prior_margin=0.10):
    """Build 5 quarterly financial rows for one code."""
    out = []
    for k, av in enumerate(_AVAILS):
        rev = revenue[k]
        out.append({
            "code": code, "period": f"Q{k}", "avail_date": av,
            "netinc": netinc[k], "netinc_prior": prior_netinc,
            "revenue": rev, "revenue_prior": prior_rev,
            "op_income": (op_income[k] if op_income[k] is not None else None),
            "op_income_prior": prior_rev * prior_margin,
        })
    return out


def _accelerating():
    # yoy_eps/rev = .05,.10,.20,.35,.55 ; margin = .11,.12,.14,.17,.21 (all rising).
    netinc = [105, 110, 120, 135, 155]
    revenue = [1050, 1100, 1200, 1350, 1550]
    margins = [0.11, 0.12, 0.14, 0.17, 0.21]
    op_income = [revenue[k] * margins[k] for k in range(5)]
    return _rows("ACC", netinc, revenue, op_income)


def _decelerating():
    netinc = [155, 135, 120, 110, 105]          # yoy shrinking
    revenue = [1550, 1350, 1200, 1100, 1050]
    op_income = [revenue[k] * 0.10 for k in range(5)]
    return _rows("DEC", netinc, revenue, op_income)


def _missing_revenue():
    netinc = [105, 110, 120, 135, 155]
    revenue = [np.nan] * 5                       # no revenue → margin/rev YoY NaN
    op_income = [None] * 5
    return _rows("MIS", netinc, revenue, op_income)


_TRADING = ["2019-01-01", "2019-06-01", "2019-12-01", "2020-04-01", "2020-06-01"]


def _panel():
    fin = pd.DataFrame(_accelerating() + _decelerating() + _missing_revenue())
    return code33_panel(fin, _TRADING).set_index(["code", "date"])["is_code33"]


def test_accelerating_code_flags_code33_after_three_quarters():
    p = _panel()
    assert p[("ACC", "2020-06-01")]           # 3+ consecutive accelerations
    assert p[("ACC", "2020-04-01")]           # already true by the 4th quarter


def test_no_lookahead_before_first_filing():
    p = _panel()
    assert not p[("ACC", "2019-01-01")]       # before any avail_date → False


def test_decelerating_and_missing_revenue_are_false():
    p = _panel()
    assert not p[("DEC", "2020-06-01")]       # YoY shrinking → not accelerating
    assert not p[("MIS", "2020-06-01")]       # revenue missing → margin/rev NaN → False


def test_components_present_for_accelerator():
    fin = pd.DataFrame(_accelerating())
    out = code33_panel(fin, _TRADING)
    last = out[(out["code"] == "ACC") & (out["date"] == "2020-06-01")].iloc[0]
    assert np.isfinite(last["yoy_eps"]) and last["yoy_eps"] > 0
    assert np.isfinite(last["yoy_rev"]) and np.isfinite(last["yoy_margin"])
