"""SQLite storage layer for collected Kiwoom datasets.

Defines the schema and small, dependency-free helpers used by collectors and
strategies. Kept deliberately thin: collectors produce plain records, this
module persists them idempotently (``INSERT OR REPLACE`` on natural keys).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

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
"""


def default_db_path() -> Path:
    """Default DB location: ``<repo>/data/kr_quant.db`` (gitignored)."""
    return Path(__file__).resolve().parents[2] / "data" / "kr_quant.db"


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (creating dirs/schema as needed) a connection with row access."""
    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    init_db(con)
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    con.commit()


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


def upsert_stocks(con: sqlite3.Connection, stocks: list[dict]) -> int:
    """Insert/replace stock master rows. Returns the number written."""
    con.executemany(
        "INSERT OR REPLACE INTO stocks(code, name, market, sector, kind) "
        "VALUES(:code, :name, :market, :sector, :kind)",
        stocks,
    )
    con.commit()
    return len(stocks)


def upsert_supply_demand(con: sqlite3.Connection, records: list[tuple]) -> int:
    """Insert/replace supply_demand rows (tuples ordered by SUPPLY_DEMAND_COLUMNS)."""
    if not records:
        return 0
    placeholders = ",".join("?" * len(SUPPLY_DEMAND_COLUMNS))
    con.executemany(
        f"INSERT OR REPLACE INTO supply_demand({','.join(SUPPLY_DEMAND_COLUMNS)}) "
        f"VALUES({placeholders})",
        records,
    )
    con.commit()
    return len(records)
