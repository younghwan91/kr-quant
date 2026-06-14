"""Portfolio-cover figures: screener ranking and accumulation-score validation.

These render the headline visuals shown in the README. Functions take plain
DataFrames (from the screener / backtest) so they stay decoupled from the DB,
and save PNGs that work headless via the Agg backend. Korean labels render if
NanumGothic is installed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

for _fp in (Path.home() / ".local/share/fonts").glob("NanumGothic*.ttf"):
    fm.fontManager.addfont(str(_fp))
if any("NanumGothic" in f.name for f in fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

_FOREIGN = "#d62728"
_INST = "#2ca02c"


def plot_ranking(result: pd.DataFrame, out_path: str | Path, *, top: int = 15) -> Path:
    """Horizontal bar chart of the top-N accumulation candidates by score."""
    top_df = result.head(top).iloc[::-1]  # highest score at the top of the chart
    labels = [f"{n}\n{c}" for n, c in zip(top_df["name"], top_df["code"])]
    # market is stored in Korean: '거래소' (KOSPI) / '코스닥' (KOSDAQ).
    colors = ["#ff7f0e" if "코스닥" in str(m) else "#1f77b4" for m in top_df["market"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(range(len(top_df)), top_df["score"], color=colors)
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("매집 점수 (유동성 대비 매집 강도 ÷ 변동범위)")
    ax.set_title(f"매집 후보 상위 {len(top_df)}종목", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, top_df["score"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {val:.2f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
               plt.Rectangle((0, 0), 1, 1, color="#ff7f0e")]
    ax.legend(handles, ["거래소(KOSPI)", "코스닥(KOSDAQ)"], loc="lower right")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_score_vs_return(merged: pd.DataFrame, summary: dict, out_path: str | Path) -> Path:
    """Two-panel validation: score-vs-return scatter + per-quantile mean bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: scatter of accumulation score vs holdout return.
    ax1.scatter(merged["score"], merged["fwd_ret"] * 100, s=28, alpha=0.6,
                color="#1f77b4", edgecolor="white", linewidth=0.4)
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.axhline(summary["universe_mean"] * 100, color="#d62728", lw=1.0, ls=":",
                label=f"전체 평균 {summary['universe_mean']:+.1%}")
    ax1.set_xscale("log")
    ax1.set_xlabel("매집 점수 (log)")
    ax1.set_ylabel("보유구간 수익률 (%)")
    ax1.set_title(f"매집 점수 vs 후속 수익률  (Spearman={summary['spearman']:.2f}, n={summary['n']})",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Right: mean forward return per score quantile (Q1 = highest score).
    b = summary["buckets"]
    qcolors = ["#2ca02c" if v > 0 else "#d62728" for v in b["mean_fwd"]]
    ax2.bar([f"Q{int(q)}" for q in b["quantile"]], b["mean_fwd"] * 100, color=qcolors)
    ax2.axhline(summary["universe_mean"] * 100, color="gray", lw=1.0, ls=":")
    ax2.set_xlabel("점수 분위 (Q1=최고점)")
    ax2.set_ylabel("평균 수익률 (%)")
    ax2.set_title("점수 분위별 평균 수익률", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(b["mean_fwd"]):
        ax2.text(i, v * 100, f"{v:+.1%}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=8)

    fig.suptitle("매집 스크리너 신호 검증 (in-sample 예시)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_path)


def _save(fig: plt.Figure, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
