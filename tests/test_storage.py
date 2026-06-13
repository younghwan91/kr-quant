"""Storage layer: schema, numeric coercion, idempotent upserts. No network."""

from __future__ import annotations

from kr_quant.storage import (
    SUPPLY_DEMAND_COLUMNS,
    connect,
    to_float,
    to_int,
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
