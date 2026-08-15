"""Per-stock supply/demand chart (viz/supply_demand_chart.py)."""

from __future__ import annotations

import pytest

from kr_quant.storage import connect
from kr_quant.viz.supply_demand_chart import build_chart, fetch_series

# Write helpers (upsert_stocks etc.) moved to quant-airflow/collectors/
# storage.py; this repo's storage.py is read-only, so tests insert rows with
# raw SQL instead. Column names are INVESTOR_COLUMNS' *keys* (individual,
# foreign_, institution, ...) — the actual DB/table columns; the dict's
# values (ind_invsr, frgnr_invsr, ...) are just Kiwoom-API-field comments.
_SD_COLS = [
    "code", "date", "close", "flu_rt", "acc_trde_qty",
    "individual", "foreign_", "institution", "fnnc_invt", "insrnc",
    "invtrt", "bank", "penfnd_etc", "samo_fund", "natn", "etc_corp",
]


def _insert_supply_demand(con, rows: list[dict]) -> None:
    cols = ",".join(_SD_COLS)
    placeholders = ",".join(["?"] * len(_SD_COLS))
    con.executemany(
        f"INSERT INTO supply_demand({cols}) VALUES({placeholders})",
        [tuple(r.get(c, 0) for c in _SD_COLS) for r in rows],
    )
    con.commit()


def _insert_stock(con, code: str, name: str) -> None:
    con.execute(
        "INSERT INTO stocks(code, name, market, sector, kind) VALUES(?,?,?,?,?)",
        (code, name, "거래소", "", ""),
    )
    con.commit()


def _seed(con, code="005930", name="삼성전자"):
    _insert_stock(con, code, name)
    _insert_supply_demand(con, [
        {"code": code, "date": "20260101", "close": 100, "individual": -10, "foreign_": 5, "institution": 5},
        {"code": code, "date": "20260102", "close": 102, "individual": -8, "foreign_": 4, "institution": 4},
        {"code": code, "date": "20260103", "close": 105, "individual": -20, "foreign_": 12, "institution": 8},
    ])


def test_fetch_series_returns_stock_name_and_ordered_rows(tmp_path):
    con = connect(str(tmp_path / "t.db"))
    _seed(con)
    name, rows = fetch_series(con, "005930")
    assert name == "삼성전자"
    assert [r["date"] for r in rows] == ["20260101", "20260102", "20260103"]


def test_fetch_series_falls_back_to_code_when_stock_unknown(tmp_path):
    con = connect(str(tmp_path / "t.db"))
    _insert_supply_demand(con, [{"code": "999999", "date": "20260101", "close": 1}])
    name, rows = fetch_series(con, "999999")
    assert name == "999999"  # no stocks row -> name defaults to the code itself
    assert len(rows) == 1


def test_build_chart_writes_a_real_png(tmp_path):
    con = connect(str(tmp_path / "t.db"))
    _seed(con)
    out = build_chart(con, "005930", tmp_path / "chart.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_build_chart_raises_on_no_data(tmp_path):
    con = connect(str(tmp_path / "t.db"))
    with pytest.raises(ValueError, match="005930"):
        build_chart(con, "005930", tmp_path / "chart.png")
