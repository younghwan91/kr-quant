"""Generate the README cover figures from the collected DB.

Produces three PNGs under ``docs/images/``:
  * ``ranking.png``   — top accumulation candidates by score
  * ``backtest.png``  — accumulation score vs forward return (signal validation)
  * ``candidate.png`` — supply/demand chart of the #1 ranked candidate

Run after ``kq-collect``::

    .venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kr_quant.storage import connect, default_db_path  # noqa: E402
from kr_quant.strategies.accumulation import load_frame, screen  # noqa: E402
from kr_quant.strategies.backtest import backtest  # noqa: E402
from kr_quant.viz.portfolio import plot_ranking, plot_score_vs_return  # noqa: E402
from kr_quant.viz.supply_demand_chart import build_chart  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"


def main() -> int:
    con = connect(str(default_db_path()))
    df = load_frame(con)

    result = screen(df, min_days=8, max_range_pct=0.15)
    if result.empty:
        print("후보 없음 — 먼저 kq-collect 로 데이터를 수집하세요.")
        return 1

    print("→ ranking.png")
    plot_ranking(result, OUT_DIR / "ranking.png", top=15)

    print("→ backtest.png")
    merged, summary = backtest(df, formation_days=12, min_days=8, max_range_pct=0.15)
    plot_score_vs_return(merged, summary, OUT_DIR / "backtest.png")
    print(f"   Spearman={summary['spearman']:.3f}  n={summary['n']}  "
          f"전체평균={summary['universe_mean']:+.2%}")

    top_code = result.iloc[0]["code"]
    print(f"→ candidate.png ({top_code} {result.iloc[0]['name']})")
    build_chart(con, top_code, OUT_DIR / "candidate.png")

    con.close()
    print(f"\n완료: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
