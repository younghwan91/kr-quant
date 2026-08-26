"""Storage layer (read side): connect() dispatch, market_cap_asof. No network.

Write-side tests (upsert idempotency, to_int/to_float, ON CONFLICT SQL) moved
to quant-airflow/tests/test_storage.py along with the upsert_* functions
themselves — see quant-airflow/collectors/storage.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kr_quant.storage import connect, market_cap_asof


def test_connect_dispatches_postgres_dsn_to_psycopg2():
    """A postgresql:// path opens Postgres instead of sqlite — no real connection made."""
    fake_module = MagicMock()
    with patch.dict("sys.modules", {"psycopg2": fake_module}):
        connect("postgresql://user:pw@localhost:5432/kr_quant")
    fake_module.connect.assert_called_once_with("postgresql://user:pw@localhost:5432/kr_quant")


def _insert_bar(con, code, date, close):
    con.execute(
        "INSERT INTO daily_bars(code, date, open, high, low, close, volume, trade_value) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
        (code, date, close, close, close, close),
    )
    con.commit()


def _insert_shares(con, code, date, shares):
    con.execute(
        "INSERT INTO shares_outstanding_history(code, date, shares_outstanding) VALUES (?, ?, ?)",
        (code, date, shares),
    )
    con.commit()


def test_market_cap_asof_normal_case(tmp_path):
    con = connect(tmp_path / "t.db")
    _insert_bar(con, "005930", "2026-02-01", 70000)
    _insert_shares(con, "005930", "2026-02-01", 1000000)

    assert market_cap_asof(con, "005930", "2026-02-01") == 70000 * 1000000
    con.close()


def test_market_cap_asof_avoids_lookahead_bias(tmp_path):
    con = connect(tmp_path / "t.db")
    _insert_shares(con, "005930", "2026-01-01", 1000000)
    _insert_shares(con, "005930", "2026-03-01", 2000000)  # simulates a later stock split
    _insert_bar(con, "005930", "2026-02-01", 70000)

    # 2026-02-01 is between the two share counts — must use the earlier
    # (on-or-before) 1,000,000 figure, never the later 2,000,000 one.
    assert market_cap_asof(con, "005930", "2026-02-01") == 70000 * 1000000
    con.close()


def test_market_cap_asof_returns_none_without_shares_data(tmp_path):
    con = connect(tmp_path / "t.db")
    _insert_bar(con, "005930", "2026-02-01", 70000)

    assert market_cap_asof(con, "005930", "2026-02-01") is None
    con.close()


def test_market_cap_asof_treats_zero_shares_as_missing(tmp_path):
    """0 주식수는 결측이다 — close*0 = 0 이 정상 시총으로 나가면 안 된다.

    조회가 `date <= 조회일 ORDER BY date DESC LIMIT 1` 이라, 0 행이 한 번 들어오면
    그 종목의 최신 점이 되어 이후 모든 조회를 이긴다. None 만 보는 가드로는 못 막는다.
    """
    con = connect(tmp_path / "t.db")
    _insert_shares(con, "005930", "2026-01-01", 1000000)
    _insert_shares(con, "005930", "2026-01-15", 0)      # 파싱 실패가 0 으로 적재된 행
    _insert_bar(con, "005930", "2026-02-01", 70000)

    assert market_cap_asof(con, "005930", "2026-02-01") is None
    con.close()


def test_market_cap_asof_bulk_matches_per_row_on_zero_shares(tmp_path):
    """벌크도 같은 semantics — 더 오래된 유효 행으로 조용히 대체하지 않는다."""
    import pandas as pd

    from kr_quant.storage import market_cap_asof_bulk

    con = connect(tmp_path / "t.db")
    _insert_shares(con, "005930", "2026-01-01", 1000000)
    _insert_shares(con, "005930", "2026-01-15", 0)
    _insert_shares(con, "000660", "2026-01-01", 500000)
    _insert_bar(con, "005930", "2026-02-01", 70000)
    _insert_bar(con, "000660", "2026-02-01", 20000)

    df = pd.DataFrame({"code": ["005930", "000660"], "date": ["2026-02-01", "2026-02-01"]})
    out = market_cap_asof_bulk(con, df)

    assert pd.isna(out.iloc[0])                       # 0 행이 이긴 종목 → 결측
    assert out.iloc[1] == 20000 * 500000              # 정상 종목은 그대로
    assert pd.isna(out.iloc[0]) == (market_cap_asof(con, "005930", "2026-02-01") is None)
    con.close()
