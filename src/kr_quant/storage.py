"""Storage layer (read side) — sqlite or Postgres/TimescaleDB.

Collectors (and their write-side upsert helpers) now live in
quant-airflow/collectors/storage.py — an intentionally independent copy,
not a shared package. This module keeps only what strategies/features read:
``connect()``/``default_db_path()`` (also used for the local sqlite dev
fallback) and ``market_cap_asof()``/``market_cap_asof_bulk()``.
``connect()`` dispatches on the connection string: a
``postgresql://``/``postgres://`` DSN opens Postgres (psycopg2, imported
lazily so sqlite-only use never needs it installed); anything else opens a
local sqlite file exactly as before.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

_PG_PREFIXES = ("postgresql://", "postgres://")


def load_env_db() -> str:
    """**명시적 opt-in** — CWD 의 ``.env`` 에서 KR_QUANT_DB 를 읽어 환경에 실어준다.

    러너마다 복붙돼 있던 .env 파서의 유일한 본체다. 부르는 쪽이 명시적으로 부를 때만
    동작한다 — :func:`db_default` 는 이걸 부르지 않는다. 그래야 KR_QUANT_DB 를
    export 하지 않은 셸이 여전히 로컬 sqlite 폴백을 **보장**받는다(공유 운영 DB 로
    조용히 붙어버리면 안 된다).

    값을 ``os.environ`` 에 되실어 이후 ``db_default()`` 와 하위 프로세스가 본다.
    설정된 게 없으면 ``""``.
    """
    v = os.environ.get("KR_QUANT_DB")
    if v:
        return v
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("KR_QUANT_DB"):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    os.environ["KR_QUANT_DB"] = v
                    return v
    return ""


def db_default() -> str:
    """CLI ``--db`` flag default: ``KR_QUANT_DB`` env var if set, else local sqlite.

    Lets an analysis-only checkout point at the shared TimescaleDB once (via
    ``.env``/``export KR_QUANT_DB=postgresql://...``) instead of passing
    ``--db`` on every command. ``.env`` 을 스스로 읽지는 않는다 — :func:`load_env_db`
    를 명시적으로 부른 호출자만 그 경로를 탄다.
    """
    return os.environ.get("KR_QUANT_DB") or str(default_db_path())

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
    -- kiwoom = 11개 분류 전체. naver = 폐지 종목 부분 백필(기관·외국인 순매매만,
    -- 나머지는 NULL — 0 은 '순매매 없음'이고 NULL 은 '모름'이라 구분해야 한다).
    -- 외국인 정의도 다르다: 네이버 값은 4개 분류 합이 0이 되도록 맞춘 수치다.
    source       TEXT NOT NULL DEFAULT 'kiwoom',
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_sd_date ON supply_demand(date);
-- source: 'kiwoom' = 상장 종목(ka10081, trade_value 는 보고된 거래대금),
--         'naver'  = 상장폐지 종목 백필(siseJson, trade_value 는 close*volume/1e6
--                    근사 — 실측 오차 중앙값 0.7%, p95 3.6%).
-- 폐지 종목이 이 테이블에 들어오는 이유는 생존편향 때문이다: 수집 소스가 현재 상장
-- 종목만 돌려주므로, 그냥 두면 백테스트가 살아남은 회사만 보고 성적을 잰다.
CREATE TABLE IF NOT EXISTS daily_bars (
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        INTEGER,
    high        INTEGER,
    low         INTEGER,
    close       INTEGER,
    volume      INTEGER,
    trade_value INTEGER,
    source      TEXT NOT NULL DEFAULT 'kiwoom',
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
-- 정정공시는 기존 행을 덮어쓰지 않고 새 버전으로 쌓인다(PK에 knowledge_date 포함).
-- 그래서 (code, period)당 행이 여럿일 수 있다 — 읽는 쪽은 반드시 as-of로 한 버전만
-- 골라야 하며, 그냥 SELECT 하면 패널이 조용히 중복된다. kr_quant.storage.read_earnings()
-- 를 쓸 것.
CREATE TABLE IF NOT EXISTS earnings (
    code            TEXT NOT NULL,
    period          TEXT NOT NULL,   -- e.g. '2020Q1'
    avail_date      TEXT,            -- lookahead-safe availability date (period-end + filing lag)
    knowledge_date  TEXT NOT NULL,   -- 이 값을 알게 된 날(수집일) — 정정공시는 새 행
    netinc          REAL,
    netinc_prior    REAL,
    revenue         REAL,
    revenue_prior   REAL,
    op_income       REAL,
    op_income_prior REAL,
    PRIMARY KEY (code, period, knowledge_date)
);
CREATE INDEX IF NOT EXISTS idx_earnings_avail_date ON earnings(avail_date);
CREATE INDEX IF NOT EXISTS idx_earnings_asof ON earnings(code, period, knowledge_date DESC);
CREATE TABLE IF NOT EXISTS consensus (
    code         TEXT NOT NULL,
    date         TEXT NOT NULL,   -- 스냅샷 수집일 (오늘)
    target_mean  REAL,            -- 목표주가 평균
    recomm_mean  REAL,            -- 투자의견 평균 (1~5, 5=강력매수)
    base_date    TEXT,            -- 컨센서스 기준일(네이버 createDate)
    fwd_eps      REAL,            -- 향후 컨센서스 EPS
    prev_eps     REAL,            -- 직전 확정 EPS
    est_year     TEXT,            -- fwd_eps가 가리키는 연도(예: '202612')
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_consensus_date ON consensus(date);
CREATE TABLE IF NOT EXISTS daily_bars_adjusted (
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,   -- price_adjust.adjust_prices()의 back-adjust 배수 적용 후이므로
    high        REAL,   -- daily_bars(원자료, INTEGER)와 달리 REAL — 분할비율이 실수라 정수로
    low         REAL,   -- 안 떨어짐(예: 1주→4주 분할이면 종가가 1/4배가 됨)
    close       REAL,
    volume      INTEGER,      -- 기본은 미조정 원본 거래량 그대로(adjust_volume=False)
    trade_value INTEGER,      -- 거래대금은 가격조정과 무관(가격×수량이 아니라 원 보고값)
    source      TEXT NOT NULL DEFAULT 'kiwoom',  -- daily_bars.source 전파(근사 거래대금 식별)
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_dba_date ON daily_bars_adjusted(date);
CREATE TABLE IF NOT EXISTS delisted_stocks (
    code            TEXT NOT NULL,
    name            TEXT,
    market          TEXT,
    last_trade_date TEXT,   -- daily_bars 기준 마지막 거래일(상장폐지일 근사), 이력 없으면 NULL
    PRIMARY KEY (code)
);
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
        # quant-airflow/sql/init_timescale.sql, not here — init_db() only
        # applies to the sqlite path.
        return con

    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    init_db(con)
    return con


# 기존 sqlite 파일에 나중에 추가된 컬럼 — {테이블: [(컬럼, DDL 조각)]}.
# ``CREATE TABLE IF NOT EXISTS`` 는 이미 있는 테이블을 고치지 않으므로, 마이그레이션
# 이전에 만들어진 로컬 DB 는 컬럼이 없는 채로 남는다. 그 상태에서 새 인덱스
# (idx_earnings_asof 는 knowledge_date 를 참조)를 만들면 init_db 가 통째로 죽는다 —
# Postgres 없이 쓰는 경로가 막힌다(실측). Postgres 쪽은 sql/migrations/ 가 담당한다.
_SQLITE_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "earnings": [("knowledge_date", "TEXT")],
    "daily_bars": [("source", "TEXT NOT NULL DEFAULT 'kiwoom'")],
    "daily_bars_adjusted": [("source", "TEXT NOT NULL DEFAULT 'kiwoom'")],
}


def _add_missing_columns(con: sqlite3.Connection) -> None:
    """기존 파일에 빠진 컬럼을 채운다. 새 파일에는 no-op."""
    for table, cols in _SQLITE_ADDED_COLUMNS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}  # noqa: S608 — 모듈 상수
        if not have:
            continue                      # 테이블 자체가 없다 = 새 DB, SCHEMA 가 만든다
        for name, ddl in cols:
            if name not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")  # noqa: S608 — 모듈 상수
    con.commit()


def init_db(con: sqlite3.Connection) -> None:
    _add_missing_columns(con)
    con.executescript(SCHEMA)
    con.commit()


def _is_pg(con: Any) -> bool:
    return not isinstance(con, sqlite3.Connection)


# kr_quant.price_adjust.rebuild_adjusted_table() writes daily_bars_adjusted —
# the one write path kept here (not moved to quant-airflow) because
# price_adjust.py's split-detection logic is imported in-process by the backtest
# strategies, so the module as a whole stays in kr-quant;
# weekly_price_adjust.py's DAG task still invokes it via
# `python -m kr_quant.price_adjust --rebuild-db` (PYTHONPATH-based, no pip
# install needed) rather than through quant-airflow/collectors.
DAILY_BAR_COLUMNS: list[str] = [
    "code", "date", "open", "high", "low", "close", "volume", "trade_value",
]


def _upsert(
    con: Any,
    table: str,
    cols: list[str],
    records: list[tuple],
    *,
    pk_cols: tuple[str, ...] = ("code", "date"),
) -> int:
    if not records:
        return 0
    if _is_pg(con):
        # psycopg2 의 executemany 는 행마다 INSERT 문을 한 번씩 왕복시킨다 —
        # rebuild_adjusted_table() 처럼 570만 행을 upsert 하면 몇 시간이 걸린다
        # (실측: 40분 넘게 돌고도 진행 중). execute_values 는 한 문장에 여러 VALUES
        # 튜플을 실어 보내 같은 일을 수십 배 빠르게 끝낸다. 수집기 쪽
        # (quant-airflow/collectors/storage.py)은 이미 이 방식이다.
        import psycopg2.extras

        update_cols = [c for c in cols if c not in pk_cols]
        set_clause = ",".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        sql = (
            f"INSERT INTO {table}({','.join(cols)}) VALUES %s "
            f"ON CONFLICT ({','.join(pk_cols)}) DO UPDATE SET {set_clause}"
        )
        template = "(" + ",".join(["%s"] * len(cols)) + ")"
        try:
            with con.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, sql, records, template=template, page_size=1000)
        except Exception:
            con.rollback()
            raise
    else:
        placeholders = ",".join(["?"] * len(cols))
        sql = f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES({placeholders})"
        con.executemany(sql, records)
    con.commit()
    return len(records)


# 조정가 테이블은 원본의 source 를 그대로 실어 나른다 — 백테스트가 읽는 건 이쪽이라,
# "이 행의 trade_value 는 근사치"라는 사실이 여기 없으면 소비되는 자리에서 알 수 없다.
ADJUSTED_BAR_COLUMNS: list[str] = [*DAILY_BAR_COLUMNS, "source"]


def upsert_daily_bars_adjusted(con: Any, records: list[tuple]) -> int:
    """Insert/replace daily_bars_adjusted rows (tuples ordered by ADJUSTED_BAR_COLUMNS)."""
    return _upsert(con, "daily_bars_adjusted", ADJUSTED_BAR_COLUMNS, records)


_EARNINGS_READ_COLS = (
    "code", "period", "avail_date", "netinc", "netinc_prior",
    "revenue", "revenue_prior", "op_income", "op_income_prior",
)


def read_earnings(con: Any, *, asof: str | None = None, cols: "tuple[str, ...] | None" = None,
                  all_versions: bool = False):
    """Earnings as a long frame with **one row per (code, period)** — as-of ``asof``.

    ``earnings`` is versioned: a DART restatement lands as a new row keyed by
    ``knowledge_date`` rather than overwriting the original. A plain
    ``SELECT * FROM earnings`` therefore returns several rows per (code, period)
    once restatements exist, which silently duplicates every downstream panel.
    This picks the newest version whose ``knowledge_date <= asof``.

    Args:
        con: Open connection (sqlite3 or psycopg2).
        asof: Ceiling on ``knowledge_date`` — a single instant. ``None`` means
            latest-known.
        cols: Columns to return (default :data:`_EARNINGS_READ_COLS`).
        all_versions: 접지 말고 모든 버전을 그대로 준다(``knowledge_date`` 포함).

            **백테스트는 이쪽을 써야 한다.** 스칼라 ``asof`` 는 "한 시점 기준"만
            표현할 수 있는데, 백테스트가 필요한 건 "각 바 t 시점 기준"이다. 버전
            선택은 날짜별로 달라져야 하므로 읽는 단계가 아니라
            :func:`kr_quant.features.fundamentals.earnings_yoy_panel` 의
            ``knowledge_col`` 로 넘겨 as-of 조인에 함께 태운다.

            ``asof`` 없이 접어서 쓰면(기본값) 최신 정정본이 과거 날짜 셀에 들어가
            조용한 룩어헤드가 된다 — 터지지 않고 성과만 좋아지는 종류다.

    Returns a pandas DataFrame.
    """
    import pandas as pd

    sel = ", ".join(cols or _EARNINGS_READ_COLS)
    if all_versions:
        if asof is not None:
            raise ValueError("all_versions 와 asof 는 함께 쓸 수 없다 — "
                             "버전 선택을 날짜별로 하려고 전부 넘기는 것이다")
        return pd.read_sql_query(f"SELECT {sel} FROM earnings", con)  # noqa: S608 — 모듈 상수
    if _is_pg(con):
        where = "WHERE knowledge_date <= %s " if asof else ""
        sql = (  # noqa: S608 — column names come from a module constant, not user input
            f"SELECT DISTINCT ON (code, period) {sel} FROM earnings {where}"
            "ORDER BY code, period, knowledge_date DESC"
        )
        params: tuple = (asof,) if asof else ()
    else:
        # sqlite has no DISTINCT ON — pick the max knowledge_date per key instead.
        cap = "AND x.knowledge_date <= ? " if asof else ""
        sql = (  # noqa: S608 — same
            f"SELECT {sel} FROM earnings e WHERE e.knowledge_date = ("
            "SELECT MAX(x.knowledge_date) FROM earnings x "
            f"WHERE x.code = e.code AND x.period = e.period {cap})"
        )
        params = (asof,) if asof else ()
    return pd.read_sql_query(sql, con, params=params or None)


PRICE_TABLE = "daily_bars_adjusted"   # 분할조정. 백테스트는 원자료(daily_bars) 금지.
_PRICE_READ_COLS = ("code", "date", "open", "high", "low", "close", "volume", "trade_value")


def read_prices(con: Any, *, cols: "tuple[str, ...] | None" = None,
                table: str = PRICE_TABLE, require_delisted: bool = True):
    """백테스트용 가격 패널 — **상장폐지 종목 포함**을 로딩 시점에 검사한다.

    생존편향은 "가장 큰 숨은 인플레이터"인데(GUARDRAILS §3), 검증 스택은 넘겨받은
    트레이드만 믿으므로 유니버스가 이미 생존자만 담고 있으면 **어떤 게이트도 그걸
    못 잡는다**(§4 공백 2). 잡을 수 있는 유일한 지점이 데이터를 읽는 이 자리다.

    Args:
        con: 열린 연결.
        cols: 반환 컬럼(기본 :data:`_PRICE_READ_COLS`).
        table: 가격 테이블. 기본 분할조정 테이블 — 백테스트가 원자료를 읽으면
            액면분할이 수익률로 둔갑한다.
        require_delisted: DB 에 폐지 종목 시세가 있는데 이 조회 결과엔 없으면
            ``AssertionError``. 유니버스를 좁히는 WHERE 를 붙였거나 폐지 백필이
            안 돌았다는 뜻이다. 폐지 시세가 DB 에 아예 없으면(신규 DB 등) 통과시킨다 —
            없는 걸 요구할 수는 없고, 그 경우는 백필 자체가 선행 과제다.

    Returns: pandas DataFrame.
    """
    import pandas as pd

    sel = ", ".join(cols or _PRICE_READ_COLS)
    df = pd.read_sql_query(f"SELECT {sel} FROM {table}", con)  # noqa: S608 — 컬럼·테이블은 모듈 상수
    if require_delisted:
        _assert_universe_has_delisted(con, df, table=table)
    return df


def _assert_universe_has_delisted(con: Any, df, *, table: str) -> None:
    """조회 결과가 폐지 종목을 담고 있는지 확인(DB 에 있는 경우에 한해)."""
    ph = "%s" if _is_pg(con) else "?"
    sql = f"SELECT count(DISTINCT code) FROM {table} WHERE source = {ph}"  # noqa: S608 — 모듈 상수
    if _is_pg(con):
        with con.cursor() as cur:
            cur.execute(sql, ("naver",))
            in_db = cur.fetchone()[0]
    else:
        in_db = con.execute(sql, ("naver",)).fetchone()[0]
    if not in_db:
        return                      # DB 에 폐지 시세 자체가 없음 — 백필이 선행 과제
    if "code" not in df.columns:
        return                      # code 를 안 뽑은 조회는 유니버스 판정 대상이 아님
    loaded = set(df["code"].astype(str).unique())
    ph2 = "%s" if _is_pg(con) else "?"
    sql2 = f"SELECT DISTINCT code FROM {table} WHERE source = {ph2}"  # noqa: S608 — 모듈 상수
    if _is_pg(con):
        with con.cursor() as cur:
            cur.execute(sql2, ("naver",))
            delisted = {r[0] for r in cur.fetchall()}
    else:
        delisted = {r[0] for r in con.execute(sql2, ("naver",)).fetchall()}
    if not (delisted & loaded):
        raise AssertionError(
            f"{table} 에 상장폐지 종목 {in_db}개가 있는데 로딩된 유니버스에 하나도 없다 — "
            f"생존편향이 들어간다. 유니버스를 좁히는 WHERE 를 걷어내거나, "
            f"의도한 것이면 read_prices(require_delisted=False) 로 명시하라."
        )


def market_cap_asof(con: Any, code: str, date: str) -> int | float | None:
    """Market cap for ``code`` on exactly ``date``: close * shares outstanding.

    ``close`` must exist for that exact date (no guessing). Shares outstanding
    is looked up as the most recent row with ``date <= date`` — never a row
    dated after the given date — to avoid lookahead bias (e.g. a stock split
    recorded later must not be applied retroactively to an earlier market cap).
    Returns ``None`` if either lookup is empty, or if shares outstanding is
    non-positive (treated as missing — a listed name cannot have 0 shares).
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

    # 0 주식수는 "진짜 0"이 아니라 결측이다 — 상장돼 있으면 0 일 수 없다. 가드가 None 만
    # 보면 close*0 = 0 이 정상 시총으로 나가고, 이걸 분모로 쓰는 소비자에서 inf 가 난다.
    # 게다가 조회가 `date <= 조회일 ORDER BY date DESC LIMIT 1` 이라 0 행이 한 번 들어오면
    # 그 종목의 최신 점이 되어 이후 모든 조회를 이긴다. 상류(quant-airflow collectors)가
    # 2026-08-27 에 막았지만 이 테이블에는 kiwoom·dart·krx + 수동 백필이 함께 쓴다.
    if close is None or shares is None or shares <= 0:
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
    # per-row 와 동일 semantics: 주식수가 0 이하면 결측으로 떨어뜨린다(더 오래된 유효
    # 행으로 대체하지 않는다 — 그러면 두 함수의 결과가 갈린다).
    merged.loc[merged["shares_outstanding"] <= 0, "shares_outstanding"] = float("nan")
    market_cap = (merged["close"] * merged["shares_outstanding"]).to_numpy()
    return pd.Series(market_cap, index=df.index)
