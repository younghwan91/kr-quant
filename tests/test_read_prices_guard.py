"""``read_prices`` — 유니버스에 상장폐지 종목이 들어있는지 로딩 시점에 검사.

생존편향은 검증 스택이 못 잡는다. walk-forward·음성대조·fragility 는 전부 **넘겨받은
트레이드**만 보므로, 유니버스가 이미 생존자만 담고 있으면 모든 게이트가 초록불이면서
성적만 부풀려진다(GUARDRAILS §4 공백 2). 잡을 수 있는 유일한 지점이 데이터를 읽는
자리라, 그 자리에서 assert 한다.
"""

from __future__ import annotations

import pytest

from kr_quant.storage import DAILY_BAR_COLUMNS, connect, read_prices

COLS = [*DAILY_BAR_COLUMNS, "source"]


def _row(code, date, source):
    return (code, date, 100, 110, 90, 100, 10, 1, source)


def _seed(con, rows):
    ph = ",".join(["?"] * len(COLS))
    con.executemany(
        f"INSERT INTO daily_bars_adjusted({','.join(COLS)}) VALUES({ph})", rows)  # noqa: S608 — 모듈 상수
    con.commit()


def test_universe_with_delisted_names_loads(tmp_path):
    con = connect(tmp_path / "t.db")
    _seed(con, [_row("000020", "2020-01-02", "kiwoom"),
                _row("000030", "2020-01-02", "naver")])
    df = read_prices(con)
    assert set(df["code"]) == {"000020", "000030"}
    con.close()


def test_missing_delisted_names_raises(tmp_path):
    """DB 엔 폐지 종목이 있는데 로딩 결과에 없으면 = 유니버스가 좁혀졌다."""
    con = connect(tmp_path / "t.db")
    _seed(con, [_row("000020", "2020-01-02", "kiwoom"),
                _row("000030", "2020-01-02", "naver")])

    # 생존자만 남기는 조회를 흉내낸다 — read_prices 를 우회한 자리와 같은 상황.
    import pandas as pd
    survivors = pd.read_sql_query(
        "SELECT code, date, close FROM daily_bars_adjusted WHERE source = 'kiwoom'", con)
    from kr_quant.storage import _assert_universe_has_delisted
    with pytest.raises(AssertionError, match="생존편향"):
        _assert_universe_has_delisted(con, survivors, table="daily_bars_adjusted")
    con.close()


def test_db_without_any_delisted_rows_passes(tmp_path):
    """폐지 시세가 DB 에 아예 없으면 통과 — 없는 걸 요구할 수는 없다(백필이 선행)."""
    con = connect(tmp_path / "t.db")
    _seed(con, [_row("000020", "2020-01-02", "kiwoom")])
    assert len(read_prices(con)) == 1
    con.close()


def test_opt_out_is_explicit(tmp_path):
    """의도적으로 좁힌 유니버스는 인자로 명시해야 한다 — 조용히 되면 안 된다."""
    con = connect(tmp_path / "t.db")
    _seed(con, [_row("000020", "2020-01-02", "kiwoom"),
                _row("000030", "2020-01-02", "naver")])
    df = read_prices(con, require_delisted=False)
    assert len(df) == 2
    con.close()


def test_default_table_is_split_adjusted():
    """백테스트가 원자료를 읽으면 액면분할이 수익률로 둔갑한다."""
    from kr_quant.storage import PRICE_TABLE
    assert PRICE_TABLE == "daily_bars_adjusted"
