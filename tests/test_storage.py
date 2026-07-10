"""Storage layer: schema, numeric coercion, idempotent upserts. No network."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kr_quant.storage import (
    DAILY_BAR_COLUMNS,
    SUPPLY_DEMAND_COLUMNS,
    _upsert,
    connect,
    market_cap_asof,
    to_float,
    to_int,
    upsert_daily_bars,
    upsert_shares_outstanding,
    upsert_stocks,
    upsert_supply_demand,
)


def test_to_int_handles_kiwoom_strings():
    assert to_int("+322500") == 322500
    assert to_int("-1979879") == -1979879
    assert to_int("") == 0
    assert to_int(None) == 0
    assert to_int("abc") == 0


def test_to_float_handles_signs():
    assert to_float("+7.86") == 7.86
    assert to_float("") == 0.0


def test_upsert_is_idempotent(tmp_path):
    con = connect(tmp_path / "t.db")
    upsert_stocks(con, [{"code": "005930", "name": "삼성전자",
                         "market": "거래소", "sector": "전기/전자", "kind": "A"}])

    record = tuple(
        [{"code": "005930", "date": "20260612", "close": 322500, "flu_rt": 7.86,
          "acc_trde_qty": 31006148, "individual": -1979879, "foreign_": 971587,
          "institution": 1097529, "fnnc_invt": 0, "insrnc": 0, "invtrt": 0,
          "bank": 0, "penfnd_etc": 0, "samo_fund": 0, "natn": 0, "etc_corp": 0}[c]
         for c in SUPPLY_DEMAND_COLUMNS]
    )
    upsert_supply_demand(con, [record])
    upsert_supply_demand(con, [record])  # same PK again

    n = con.execute("SELECT COUNT(*) FROM supply_demand").fetchone()[0]
    assert n == 1  # INSERT OR REPLACE → no duplicate
    row = con.execute("SELECT foreign_ FROM supply_demand").fetchone()
    assert row["foreign_"] == 971587
    con.close()


def test_connect_dispatches_postgres_dsn_to_psycopg2():
    """A postgresql:// path opens Postgres instead of sqlite — no real connection made."""
    fake_module = MagicMock()
    with patch.dict("sys.modules", {"psycopg2": fake_module}):
        connect("postgresql://user:pw@localhost:5432/kr_quant")
    fake_module.connect.assert_called_once_with("postgresql://user:pw@localhost:5432/kr_quant")


def test_upsert_uses_on_conflict_for_postgres_connection():
    """Non-sqlite connections get ON CONFLICT DO UPDATE, not INSERT OR REPLACE."""
    fake_con = MagicMock()
    fake_cursor = MagicMock()
    fake_con.cursor.return_value.__enter__.return_value = fake_cursor

    n = _upsert(fake_con, "daily_bars", ["code", "date", "close"], [("005930", "20260706", 100)])

    assert n == 1
    sql = fake_cursor.executemany.call_args[0][0]
    assert "ON CONFLICT (code,date) DO UPDATE SET close=EXCLUDED.close" in sql
    fake_con.commit.assert_called_once()


def _bar(code, date, close):
    values = {"code": code, "date": date, "open": close, "high": close,
              "low": close, "close": close, "volume": 0, "trade_value": 0}
    return tuple(values[c] for c in DAILY_BAR_COLUMNS)


def test_market_cap_asof_normal_case(tmp_path):
    con = connect(tmp_path / "t.db")
    upsert_daily_bars(con, [_bar("005930", "2026-02-01", 70000)])
    upsert_shares_outstanding(con, [("005930", "2026-02-01", 1000000)])

    assert market_cap_asof(con, "005930", "2026-02-01") == 70000 * 1000000
    con.close()


def test_market_cap_asof_avoids_lookahead_bias(tmp_path):
    con = connect(tmp_path / "t.db")
    upsert_shares_outstanding(con, [
        ("005930", "2026-01-01", 1000000),
        ("005930", "2026-03-01", 2000000),  # simulates a later stock split
    ])
    upsert_daily_bars(con, [_bar("005930", "2026-02-01", 70000)])

    # 2026-02-01 is between the two share counts — must use the earlier
    # (on-or-before) 1,000,000 figure, never the later 2,000,000 one.
    assert market_cap_asof(con, "005930", "2026-02-01") == 70000 * 1000000
    con.close()


def test_market_cap_asof_returns_none_without_shares_data(tmp_path):
    con = connect(tmp_path / "t.db")
    upsert_daily_bars(con, [_bar("005930", "2026-02-01", 70000)])

    assert market_cap_asof(con, "005930", "2026-02-01") is None
    con.close()
