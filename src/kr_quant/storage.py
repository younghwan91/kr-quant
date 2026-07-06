"""Storage layer for collected Kiwoom datasets — sqlite or Postgres/TimescaleDB.

Defines the schema and small helpers used by collectors and strategies.
Collectors produce plain records; this module persists them idempotently on
natural keys. ``connect()`` dispatches on the connection string: a
``postgresql://``/``postgres://`` DSN opens Postgres (psycopg2, imported
lazily so sqlite-only use never needs it installed); anything else opens a
local sqlite file exactly as before.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_PG_PREFIXES = ("postgresql://", "postgres://")

# ka10059 (투자자기관별종목별) net-buy fields → DB columns.
# Order matters: it defines the column order for ``supply_demand`` inserts.
INVESTOR_COLUMNS: dict[str, str] = {
    "individual": "ind_invsr",   # 개인
    "foreign_": "frgnr_invsr",   # 외국인
    "institution": "orgn",       # 기관계
    "fnnc_invt": "fnnc_invt",    # 금융투자
    "insrnc": "insrnc",          # 보험
    "invtrt": "invtrt",          # 투신
    "bank": "bank",              # 은행
    "penfnd_etc": "penfnd_etc",  # 연기금 등
    "samo_fund": "samo_fund",    # 사모펀드
    "natn": "natn",              # 국가
    "etc_corp": "etc_corp",      # 기타법인
}

SUPPLY_DEMAND_COLUMNS: list[str] = [
    "code",
    "date",
    "close",
    "flu_rt",
    "acc_trde_qty",
    *INVESTOR_COLUMNS.keys(),
]

# ka10081 (주식일봉차트) candle fields → DB columns. Order defines insert order.
DAILY_BAR_COLUMNS: list[str] = [
    "code",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_value",
]

_INVESTOR_COL_DDL = ",\n            ".join(f"{c} INTEGER" for c in INVESTOR_COLUMNS)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS stocks (
    code   TEXT PRIMARY KEY,
    name   TEXT,
    market TEXT,
    sector TEXT,
    kind   TEXT
);
CREATE TABLE IF NOT EXISTS supply_demand (
    code         TEXT NOT NULL,
    date         TEXT NOT NULL,
    close        INTEGER,
    flu_rt       REAL,
    acc_trde_qty INTEGER,
    {_INVESTOR_COL_DDL},
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_sd_date ON supply_demand(date);
CREATE TABLE IF NOT EXISTS daily_bars (
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        INTEGER,
    high        INTEGER,
    low         INTEGER,
    close       INTEGER,
    volume      INTEGER,
    trade_value INTEGER,
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_db_date ON daily_bars(date);
CREATE TABLE IF NOT EXISTS short_selling (
    code            TEXT NOT NULL,
    date            TEXT NOT NULL,
    close           INTEGER,
    volume          INTEGER,
    short_qty       INTEGER,   -- 당일 공매도 수량 (shrts_qty)
    short_balance   INTEGER,   -- 공매도 잔고 수량 (ovr_shrts_qty)
    short_ratio     REAL,      -- 공매도 비중 % (trde_wght)
    short_avg_price INTEGER,   -- 공매도 평균가 (shrts_avg_pric)
    short_value     INTEGER,   -- 공매도 거래대금 (shrts_trde_prica)
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_ss_date ON short_selling(date);
CREATE TABLE IF NOT EXISTS credit_balance (
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,
    close       INTEGER,
    new_qty     INTEGER,   -- 신규 신용매수 (new)
    repay_qty   INTEGER,   -- 상환 (rpya)
    balance_qty INTEGER,   -- 신용잔고 수량 (remn)
    balance_amt INTEGER,   -- 신용잔고 금액 (amt)
    balance_rt  REAL,      -- 신용잔고율 % (remn_rt)
    credit_rt   REAL,      -- 신용비율 % (shr_rt)
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_cb_date ON credit_balance(date);
CREATE TABLE IF NOT EXISTS sector_index (
    code        TEXT NOT NULL,  -- 업종코드 (001=KOSPI 종합, 101=KOSDAQ 종합 등)
    name        TEXT,
    date        TEXT NOT NULL,
    open        INTEGER,
    high        INTEGER,
    low         INTEGER,
    close       INTEGER,
    volume      INTEGER,
    trade_value INTEGER,
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_si_date ON sector_index(date);
"""


def default_db_path() -> Path:
    """Default DB location: ``<repo>/data/kr_quant.db`` (gitignored)."""
    return Path(__file__).resolve().parents[2] / "data" / "kr_quant.db"


def connect(db_path: str | Path | None = None) -> Any:
    """Open a connection with row access.

    ``db_path`` starting with ``postgresql://``/``postgres://`` opens Postgres
    (e.g. TimescaleDB) via psycopg2. Anything else is treated as a sqlite file
    path (default: ``<repo>/data/kr_quant.db``, dirs created as needed).
    """
    if isinstance(db_path, str) and db_path.startswith(_PG_PREFIXES):
        import psycopg2  # noqa: PLC0415 — optional dep, only needed for this path

        con = psycopg2.connect(db_path)
        # Schema (tables, hypertables, compression policy) is provisioned by
        # kr-quant-airflow/sql/init_timescale.sql, not here — init_db() only
        # applies to the sqlite path.
        return con

    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    init_db(con)
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    con.commit()


def _is_pg(con: Any) -> bool:
    return not isinstance(con, sqlite3.Connection)


def _upsert(
    con: Any,
    table: str,
    cols: list[str],
    records: list[tuple],
    *,
    pk_cols: tuple[str, ...] = ("code", "date"),
) -> int:
    """Insert/replace ``records`` (tuples ordered by ``cols``) into ``table``.

    sqlite: ``INSERT OR REPLACE``. Postgres: ``INSERT ... ON CONFLICT DO
    UPDATE`` on ``pk_cols`` — same natural-key upsert semantics either way.
    """
    if not records:
        return 0
    if _is_pg(con):
        placeholders = ",".join(["%s"] * len(cols))
        update_cols = [c for c in cols if c not in pk_cols]
        set_clause = ",".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        sql = (
            f"INSERT INTO {table}({','.join(cols)}) VALUES({placeholders}) "
            f"ON CONFLICT ({','.join(pk_cols)}) DO UPDATE SET {set_clause}"
        )
        with con.cursor() as cur:
            cur.executemany(sql, records)
    else:
        placeholders = ",".join(["?"] * len(cols))
        sql = f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES({placeholders})"
        con.executemany(sql, records)
    con.commit()
    return len(records)


def to_int(s: object) -> int:
    """Kiwoom numeric strings (``'+322500'``, ``'-1979879'``, ``''``) → int."""
    text = str(s or "").replace("+", "").strip()
    try:
        return int(text)
    except ValueError:
        return 0


def to_float(s: object) -> float:
    text = str(s or "").replace("+", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


_STOCKS_COLS = ["code", "name", "market", "sector", "kind"]


def upsert_stocks(con: Any, stocks: list[dict]) -> int:
    """Insert/replace stock master rows. Returns the number written."""
    records = [tuple(s.get(c) for c in _STOCKS_COLS) for s in stocks]
    return _upsert(con, "stocks", _STOCKS_COLS, records, pk_cols=("code",))


def upsert_supply_demand(con: Any, records: list[tuple]) -> int:
    """Insert/replace supply_demand rows (tuples ordered by SUPPLY_DEMAND_COLUMNS)."""
    return _upsert(con, "supply_demand", SUPPLY_DEMAND_COLUMNS, records)


def upsert_daily_bars(con: Any, records: list[tuple]) -> int:
    """Insert/replace daily_bars rows (tuples ordered by DAILY_BAR_COLUMNS)."""
    return _upsert(con, "daily_bars", DAILY_BAR_COLUMNS, records)


_SHORT_SELLING_COLS = [
    "code", "date", "close", "volume",
    "short_qty", "short_balance", "short_ratio", "short_avg_price", "short_value",
]

_CREDIT_BALANCE_COLS = [
    "code", "date", "close",
    "new_qty", "repay_qty", "balance_qty", "balance_amt", "balance_rt", "credit_rt",
]


def upsert_short_selling(con: Any, records: list[tuple]) -> int:
    """Insert/replace short_selling rows."""
    return _upsert(con, "short_selling", _SHORT_SELLING_COLS, records)


def upsert_credit_balance(con: Any, records: list[tuple]) -> int:
    """Insert/replace credit_balance rows."""
    return _upsert(con, "credit_balance", _CREDIT_BALANCE_COLS, records)


_SECTOR_INDEX_COLS = [
    "code", "name", "date", "open", "high", "low", "close", "volume", "trade_value",
]


def upsert_sector_index(con: Any, records: list[tuple]) -> int:
    """Insert/replace sector_index rows."""
    return _upsert(con, "sector_index", _SECTOR_INDEX_COLS, records)
