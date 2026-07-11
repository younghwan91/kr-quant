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
CREATE TABLE IF NOT EXISTS shares_outstanding_history (
    code               TEXT NOT NULL,
    date               TEXT NOT NULL,
    shares_outstanding INTEGER,  -- sqlite INTEGER is dynamically 64-bit already;
    PRIMARY KEY (code, date)     -- Postgres side (init_timescale.sql) must use BIGINT, not INTEGER(32bit) — 삼성전자(58억주) overflows it
);
CREATE INDEX IF NOT EXISTS idx_sh_date ON shares_outstanding_history(date);
CREATE TABLE IF NOT EXISTS earnings (
    code            TEXT NOT NULL,
    period          TEXT NOT NULL,   -- e.g. '2020Q1'
    avail_date      TEXT,            -- lookahead-safe availability date (period-end + filing lag)
    netinc          REAL,
    netinc_prior    REAL,
    revenue         REAL,
    revenue_prior   REAL,
    op_income       REAL,
    op_income_prior REAL,
    PRIMARY KEY (code, period)
);
CREATE INDEX IF NOT EXISTS idx_earnings_avail_date ON earnings(avail_date);
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
        try:
            with con.cursor() as cur:
                cur.executemany(sql, records)
        except Exception:
            # A failed statement leaves the whole Postgres transaction aborted
            # until rolled back — without this, every later upsert on this
            # connection fails with InFailedSqlTransaction even for unrelated,
            # valid records (cascading one bad row into the entire run).
            con.rollback()
            raise
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


_SHARES_OUTSTANDING_COLS = ["code", "date", "shares_outstanding"]


def upsert_shares_outstanding(con: Any, records: list[tuple]) -> int:
    """Insert/replace shares_outstanding_history rows."""
    return _upsert(con, "shares_outstanding_history", _SHARES_OUTSTANDING_COLS, records)


_EARNINGS_COLS = [
    "code", "period", "avail_date",
    "netinc", "netinc_prior", "revenue", "revenue_prior", "op_income", "op_income_prior",
]


def upsert_earnings(con: Any, records: list[tuple]) -> int:
    """Insert/replace earnings rows (tuples ordered by _EARNINGS_COLS)."""
    return _upsert(con, "earnings", _EARNINGS_COLS, records, pk_cols=("code", "period"))


def market_cap_asof(con: Any, code: str, date: str) -> int | float | None:
    """Market cap for ``code`` on exactly ``date``: close * shares outstanding.

    ``close`` must exist for that exact date (no guessing). Shares outstanding
    is looked up as the most recent row with ``date <= date`` — never a row
    dated after the given date — to avoid lookahead bias (e.g. a stock split
    recorded later must not be applied retroactively to an earlier market cap).
    Returns ``None`` if either lookup is empty.
    """
    ph = "%s" if _is_pg(con) else "?"
    cur = con.cursor()
    cur.execute(
        f"SELECT close FROM daily_bars WHERE code={ph} AND date={ph}",
        (code, date),
    )
    bar_row = cur.fetchone()
    if bar_row is None:
        return None
    close = bar_row[0]

    cur.execute(
        f"SELECT shares_outstanding FROM shares_outstanding_history "
        f"WHERE code={ph} AND date<={ph} ORDER BY date DESC LIMIT 1",
        (code, date),
    )
    shares_row = cur.fetchone()
    if shares_row is None:
        return None
    shares = shares_row[0]

    if close is None or shares is None:
        return None
    return close * shares


def market_cap_asof_bulk(con: Any, df: Any) -> Any:
    """Vectorized :func:`market_cap_asof` for many ``(code, date)`` rows at once.

    Byte-identical semantics to the per-row function — close comes from
    ``daily_bars`` on the exact date (NOT any close column the caller's
    frame may already carry, e.g. ``supply_demand.close``: that is a
    different, not-necessarily-equal price series in this DB — verified by
    spot-checking real rows, e.g. code 118000 on 2026-03-25 has
    ``daily_bars.close=2410`` vs ``supply_demand.close=241``). Shares
    outstanding is the most recent row with ``date <= date`` (lookahead-safe,
    same rule as the per-row SQL ``date<=? ORDER BY date DESC LIMIT 1``).

    This issues 2 bulk queries total instead of 2 queries per row —
    ``market_cap_asof`` called in a per-row Python loop over a multi-year
    dataset means millions of round trips to Postgres.

    Args:
        df: Must have ``code`` and ``date`` columns.

    Returns:
        A ``pandas.Series`` aligned with ``df.index``: ``market_cap`` per
        row, ``NaN`` where close or shares-outstanding is unavailable
        (mirrors ``market_cap_asof`` returning ``None``).
    """
    import pandas as pd  # noqa: PLC0415 — pandas is a strategy-layer dep, not storage's

    if df.empty:
        return pd.Series([], dtype=float, index=df.index)

    codes = df["code"].unique().tolist()
    ph = "%s" if _is_pg(con) else "?"
    placeholders = ",".join([ph] * len(codes))
    bars = pd.read_sql_query(
        f"SELECT code, date, close FROM daily_bars WHERE code IN ({placeholders})",
        con,
        params=codes,
    )
    bars["date"] = bars["date"].astype(str)

    shares = pd.read_sql_query(
        f"SELECT code, date, shares_outstanding FROM shares_outstanding_history "
        f"WHERE code IN ({placeholders})",
        con,
        params=codes,
    )
    if bars.empty or shares.empty:
        return pd.Series([float("nan")] * len(df), index=df.index)

    shares["date"] = pd.to_datetime(shares["date"].astype(str))
    # merge_asof with `by=` still requires the `on` column sorted globally
    # (not just within each `by` group) — sort by date first, code second.
    shares = shares.sort_values(["date", "code"])

    d = df[["code"]].reset_index(drop=True).copy()
    d["date_str"] = df["date"].astype(str).to_numpy()
    d["_row"] = d.index
    d = d.merge(bars, left_on=["code", "date_str"], right_on=["code", "date"], how="left")
    d["date"] = pd.to_datetime(d["date_str"])
    d_sorted = d.sort_values(["date", "code"])

    merged = pd.merge_asof(
        d_sorted,
        shares[["code", "date", "shares_outstanding"]],
        on="date",
        by="code",
        direction="backward",
    )
    merged = merged.sort_values("_row")
    market_cap = (merged["close"] * merged["shares_outstanding"]).to_numpy()
    return pd.Series(market_cap, index=df.index)
