"""``read_earnings`` — 정정공시 버전 중 as-of 한 건만 고르는지.

``earnings``는 PK에 ``knowledge_date``가 들어가 정정본이 새 행으로 쌓인다. 읽는 쪽이
그냥 ``SELECT * FROM earnings``를 하면 (code, period)당 행이 여럿이 되어 하류 패널이
조용히 중복된다 — 터지지 않고 수치만 틀리는 종류라 테스트로 못박는다.
"""

from __future__ import annotations

import sqlite3

import pytest

from kr_quant.storage import SCHEMA, read_earnings

FIRST = ("000020", "2026Q1", "2026-05-15", "2026-05-15", 100.0, 90.0, 1.0, 1.0, 1.0, 1.0)
RESTATED = ("000020", "2026Q1", "2026-05-15", "2026-08-13", 111.0, 90.0, 1.0, 1.0, 1.0, 1.0)
OTHER = ("000030", "2026Q1", "2026-05-16", "2026-05-16", 200.0, 180.0, 1.0, 1.0, 1.0, 1.0)


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    c.executemany("INSERT INTO earnings VALUES (?,?,?,?,?,?,?,?,?,?)", [FIRST, RESTATED, OTHER])
    yield c
    c.close()


def test_restatement_is_a_new_row_not_an_overwrite(con):
    """스키마 전제: 같은 (code, period)에 버전 2개가 공존한다."""
    n = con.execute(
        "SELECT count(*) FROM earnings WHERE code='000020' AND period='2026Q1'"
    ).fetchone()[0]
    assert n == 2


def test_read_earnings_returns_one_row_per_key(con):
    df = read_earnings(con)
    assert len(df) == 2
    assert not df.duplicated(["code", "period"]).any()


def test_asof_before_restatement_returns_original_figure(con):
    """백테스트가 2026-06-01을 돌 때 8월 정정본을 보면 룩어헤드다."""
    df = read_earnings(con, asof="2026-06-01")
    assert float(df.loc[df["code"] == "000020", "netinc"].iloc[0]) == 100.0
    assert not df.duplicated(["code", "period"]).any()


def test_asof_after_restatement_returns_restated_figure(con):
    df = read_earnings(con, asof="2026-08-13")
    assert float(df.loc[df["code"] == "000020", "netinc"].iloc[0]) == 111.0


def test_asof_none_is_latest_known(con):
    df = read_earnings(con)
    assert float(df.loc[df["code"] == "000020", "netinc"].iloc[0]) == 111.0


def test_asof_before_any_filing_is_empty(con):
    assert read_earnings(con, asof="2019-01-01").empty


def test_unversioned_key_is_unaffected_by_asof(con):
    """정정 이력이 없는 종목은 어느 as-of에서도 같은 값이어야 한다."""
    for asof in (None, "2026-06-01", "2026-08-13"):
        df = read_earnings(con, asof=asof)
        assert float(df.loc[df["code"] == "000030", "netinc"].iloc[0]) == 200.0
