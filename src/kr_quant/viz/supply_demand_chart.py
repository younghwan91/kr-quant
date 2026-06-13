"""Render a stock's investor supply/demand chart from the collected DB.

Top panel: close price. Bottom panel: cumulative net buying of individual /
foreign / institutional investors. Saves a PNG (works headless via the Agg
backend). Korean labels render if NanumGothic is installed.

CLI:
    kq-chart --code 005930
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ..storage import connect, default_db_path  # noqa: E402

for _fp in (Path.home() / ".local/share/fonts").glob("NanumGothic*.ttf"):
    fm.fontManager.addfont(str(_fp))
if any("NanumGothic" in f.name for f in fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


def fetch_series(con: sqlite3.Connection, code: str) -> tuple[str, list[sqlite3.Row]]:
    """Return (stock_name, rows ordered by date ascending) for ``code``."""
    name_row = con.execute("SELECT name FROM stocks WHERE code=?", (code,)).fetchone()
    name = name_row["name"] if name_row else code
    rows = con.execute(
        "SELECT date, close, individual, foreign_, institution "
        "FROM supply_demand WHERE code=? ORDER BY date",
        (code,),
    ).fetchall()
    return name, rows


def build_chart(con: sqlite3.Connection, code: str, out_path: str | Path) -> Path:
    """Build and save the supply/demand chart. Returns the output path."""
    name, rows = fetch_series(con, code)
    if not rows:
        raise ValueError(f"{code}: DB에 데이터가 없습니다. 먼저 kq-collect 로 수집하세요.")

    dates = [r["date"] for r in rows]
    price = [abs(r["close"]) for r in rows]

    def cumsum(key: str) -> list[int]:
        acc, out = 0, []
        for r in rows:
            acc += r[key] or 0
            out.append(acc)
        return out

    cum_i, cum_f, cum_o = cumsum("individual"), cumsum("foreign_"), cumsum("institution")
    x = range(len(dates))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}
    )
    fig.suptitle(f"{name} ({code}) 투자자별 수급", fontsize=14, fontweight="bold")

    ax1.plot(x, price, color="black", lw=1.4)
    ax1.set_ylabel("종가 (원)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, cum_i, label="개인", color="#1f77b4", lw=1.6)
    ax2.plot(x, cum_f, label="외국인", color="#d62728", lw=1.6)
    ax2.plot(x, cum_o, label="기관", color="#2ca02c", lw=1.6)
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_ylabel("누적 순매수 (주)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    step = max(1, len(dates) // 10)
    ticks = list(x)[::step]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([dates[i] for i in ticks], rotation=45, ha="right", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="종목 수급 차트 생성 (DB 기반)")
    parser.add_argument("--code", required=True, help="종목코드 (예: 005930)")
    parser.add_argument("--db", default=str(default_db_path()))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    out = args.out or str(default_db_path().parent / f"chart_{args.code}.png")
    con = connect(args.db)
    path = build_chart(con, args.code, out)
    con.close()
    print(f"차트 저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
