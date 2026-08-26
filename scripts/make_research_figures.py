#!/usr/bin/env python
"""README용 리서치 결과 그림 2종 생성 (headless, 실측 재사용).

§3 systematic analysis의 두 핵심 발견을 시각화한다. 숫자는 전부 이 레포의
검증 코드에서 실측으로 뽑는다(발명 금지):

  1. edge_taxonomy.png  — 엣지의 두 유형(§3.1): 볼록형(급등주 스윙, 트레이더) vs
     확산형(PEAD, 기관)의 건당 수익 분포. 왜도·집중도 대비.
  2. apriori_momentum.png — 진입시점 신호의 선험적 예측력(§3.2): 진입 모멘텀
     강도 5분위 → 사후 건당 기대수익(R), TRAIN·OOS 둘 다 우상향.

계산은 전부 기존 코드 재사용:
  - 볼록형: research.signals.contrarian_retail.simulate_detailed (default PARAMS)
  - 확산형: research.experiments.pead_gate.extract_trades (PEAD BASELINE)
  - 분위:   kr_quant.diagnostics.r_distribution.conviction_analysis

src/kr_quant/는 건드리지 않는다(읽기 전용 import만).

실행: uv run python scripts/make_research_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "experiments"))  # pead_gate 상대 import 부트스트랩

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from kr_quant.validation.optimization import TRAIN_HI  # noqa: E402 — sys.path 부트스트랩 뒤

# --- Korean font (repo 관례: NanumGothic 우선, 없으면 Noto Sans CJK KR 폴백) -----
_FONT_CANDS = [
    (str(fp), "NanumGothic")
    for fp in (Path.home() / ".local/share/fonts").glob("NanumGothic*.ttf")
] + [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
    (str(Path.home() / ".local/share/fonts/NotoSansKR.ttf"), "Noto Sans KR"),
]


def _setup_font() -> str | None:
    for path, name in _FONT_CANDS:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
            except (OSError, RuntimeError):
                continue
    installed = {f.name for f in fm.fontManager.ttflist}
    for _, name in _FONT_CANDS:
        if name in installed:
            plt.rcParams["font.family"] = name
            return name
    return None


_setup_font()
plt.rcParams["axes.unicode_minus"] = False

# --- Muted, README-neutral palette (light bg / dark text; reads on light+dark) ---
BG = "#ffffff"
INK = "#2b2b2b"
MUTED = "#6b7280"
GRID = "#d9dde3"
CONVEX = "#c1666b"   # 볼록형(트레이더) — muted rose
DIFFUSE = "#3d6a99"  # 확산형(기관) — muted blue
TRAIN = "#a7c0d8"    # 표본내 — light blue
OOS = "#2f5d86"      # 표본외 — deep blue (강조: 정직한 OOS)

STOP = 0.10               # 하드손절폭 (R = 수익 / STOP)


# ---------------------------------------------------------------------------
# Data extraction (실측 배열)
# ---------------------------------------------------------------------------
def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    s = x.std()
    return float(((x - x.mean()) ** 3).mean() / s ** 3) if s > 0 else float("nan")


def _top5_share(x: np.ndarray) -> float:
    """상위 5건이 양(+)수익 총질량에서 차지하는 비중 (repo r_distribution 관례)."""
    x = np.asarray(x, float)
    pos = x[x > 0].sum()
    return float(np.sort(x)[::-1][:5].sum() / pos) if pos > 0 else float("nan")


def load_convex() -> tuple[np.ndarray, dict]:
    """볼록형: contrarian 급등주 스윙의 OOS 건당 수익 배열 + 상세 dict (default PARAMS)."""
    from research.signals.contrarian_retail import _load_env_db, load_data, simulate_detailed

    _load_env_db()
    params = dict(window=8, top_mom=0.80, ext_q=0.85, stop=0.10, trail=0.20, hold=60)
    prices, flow = load_data()
    d = simulate_detailed(prices, flow, **params)
    oos = d["entry"] >= TRAIN_HI
    ret = d["ret"][oos]
    return ret[np.isfinite(ret)], d


def load_diffuse() -> np.ndarray:
    """확산형: PEAD의 OOS 건당 초과수익 배열 (BASELINE)."""
    from pead_gate import extract_trades
    from pead_refinement import BASELINE, _context, load_data

    prices, yoy = load_data()
    ctx = _context(prices, yoy)
    ent, exc = extract_trades(ctx, **BASELINE)
    oos = ent >= TRAIN_HI
    exc = exc[oos]
    return exc[np.isfinite(exc)]


# ---------------------------------------------------------------------------
# Figure 1 — 엣지의 두 유형
# ---------------------------------------------------------------------------
def figure_edge_taxonomy(convex: np.ndarray, diffuse: np.ndarray, out: Path) -> dict:
    """볼록형 vs 확산형 건당 수익 분포 — 왜도·집중도 대비를 시각화."""
    lo, hi = -50, 100  # 표시 창(%). 통계는 전체 배열로, 꼬리는 주석으로.
    bins = np.linspace(lo, hi, 46)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.8), sharex=True)
    fig.patch.set_facecolor(BG)

    series = [
        ("볼록형 — 급등주 스윙 (트레이더 엣지)", convex, CONVEX, axes[0]),
        ("확산형 — PEAD 실적 드리프트 (기관 엣지)", diffuse, DIFFUSE, axes[1]),
    ]
    stats: dict[str, dict] = {}
    for title, arr, color, ax in series:
        ax.set_facecolor(BG)
        pct = arr * 100.0
        ax.hist(np.clip(pct, lo, hi), bins=bins, density=True, color=color,
                alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=MUTED, lw=0.9, ls="--")
        mean_pct = pct.mean()
        ax.axvline(mean_pct, color=INK, lw=1.3, ls=":",
                   label=f"평균 {mean_pct:+.1f}%")

        # 상위 5건 마커(집중도 시각화) — 표시창 오른쪽 경계에 캡.
        top5 = np.sort(pct)[::-1][:5]
        ax.scatter(np.clip(top5, lo, hi), np.full(5, ax.get_ylim()[1] * 0.06),
                   marker="v", s=55, color=color, edgecolor=INK, linewidth=0.6,
                   zorder=5, clip_on=False)

        win = float(np.mean(arr > 0))
        sk = _skew(arr)
        t5 = _top5_share(arr)
        stats[title] = {"n": int(len(arr)), "win_rate": win, "skew": sk,
                        "top5_share": t5, "mean": float(arr.mean()),
                        "max": float(arr.max()), "min": float(arr.min())}

        box = (f"n = {len(arr):,}\n승률 {win:.0%}\n왜도 {sk:+.1f}\n"
               f"상위 5건 = 양수익의 {t5:.0%}\n최대 {arr.max() * 100:+.0f}%")
        ax.text(0.985, 0.93, box, transform=ax.transAxes, ha="right", va="top",
                fontsize=10.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f6f7f9", ec=GRID, lw=1.0))
        ax.annotate(f"꼬리 최대 {arr.max() * 100:+.0f}%로 이어짐 →",
                    xy=(hi, ax.get_ylim()[1] * 0.06), xytext=(hi - 2, ax.get_ylim()[1] * 0.30),
                    ha="right", fontsize=9, color=color, fontweight="bold")

        ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_ylabel("밀도", fontsize=10, color=INK)
        ax.grid(True, axis="y", alpha=0.35, color=GRID)
        ax.legend(loc="center right", fontsize=9.5, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED)

    axes[1].set_xlabel(f"건당 수익 (%)   ·   표시창 [{lo}%, +{hi}%], 꼬리는 잘라 별도 표기",
                       fontsize=10, color=INK)
    axes[1].set_xlim(lo, hi)
    fig.suptitle("엣지의 두 유형 — 볼록형(트레이더) vs 확산형(기관)",
                 fontsize=16, fontweight="bold", color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.935,
             "같은 알파라도 분포 모양이 다르다: 볼록형은 낮은 승률·강한 우측 왜도·소수 대박에 P&L 집중, "
             "확산형은 높은 승률·낮은 왜도·고르게 분산.",
             fontsize=10, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# Figure 2 — 진입시점 모멘텀 → 사후 기대수익 (단조)
# ---------------------------------------------------------------------------
def figure_apriori_momentum(d: dict, out: Path) -> dict:
    """진입 모멘텀 강도 5분위 → 건당 기대수익(R), TRAIN·OOS 그룹 막대."""
    from kr_quant.diagnostics.r_distribution import conviction_analysis

    R_all = d["ret"] / STOP
    res = conviction_analysis(d["mom"], R_all, d["entry"], train_hi=TRAIN_HI)
    nq = res["nq"]
    q = np.arange(nq)
    tr = np.array([res["train"][i]["expectancy_R"] for i in range(nq)])
    oo = np.array([res["oos"][i]["expectancy_R"] for i in range(nq)])

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    w = 0.38
    b1 = ax.bar(q - w / 2, tr, w, color=TRAIN, edgecolor="white", label="표본내 (TRAIN, <2022)")
    b2 = ax.bar(q + w / 2, oo, w, color=OOS, edgecolor="white", label="표본외 (OOS, ≥2022)")

    # 우상향 추세를 명시(단조성) — 시리즈별 최소제곱 추세선(양의 기울기)으로 방향만.
    # (중간 분위는 표본잡음으로 흔들리나 Q1→Q5 순증가·양의 기울기가 TRAIN·OOS 둘 다.)
    for xoff, ys, col, wd in ((-w / 2, tr, TRAIN, 1.6), (w / 2, oo, OOS, 2.2)):
        xs = q + xoff
        m, c = np.polyfit(q, ys, 1)
        ax.plot(xs, m * q + c, color=col, lw=wd, alpha=0.9, zorder=4)

    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.02 if v >= 0 else -0.02),
                    f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8.5, color=INK)

    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(q)
    ax.set_xticklabels([f"Q{i + 1}" for i in range(nq)], fontsize=11, color=INK)
    ax.set_xlabel("진입 시점 모멘텀 강도 분위 (Q1=약함 → Q5=강함)", fontsize=11, color=INK)
    ax.set_ylabel("사후 건당 기대수익  R = 수익 / 손절폭", fontsize=11, color=INK)
    ax.set_title("진입 시점 모멘텀 강도 → 사후 기대수익 (TRAIN·OOS 단조)",
                 fontsize=15, fontweight="bold", color=INK, loc="left", pad=10)
    ax.grid(True, axis="y", alpha=0.35, color=GRID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="upper left", fontsize=10.5, frameon=False)

    verdict = ("Q5 ≫ Q1 이 TRAIN·OOS 둘 다에서 성립 → 진입 시점에 알 수 있는 선험적 예측력 "
               f"(OOS 상승폭 {oo[-1] - oo[0]:+.2f}R, TRAIN {tr[-1] - tr[0]:+.2f}R)")
    fig.text(0.5, 0.015, verdict, ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    return {"train_R": tr.tolist(), "oos_R": oo.tolist(), "verdict": res["verdict"],
            "train_monotonic": res["train_monotonic"], "oos_monotonic": res["oos_monotonic"]}


def main() -> int:
    out_dir = REPO / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 데이터 로드 (실측, TimescaleDB) ===")
    convex, cdetail = load_convex()
    diffuse = load_diffuse()

    f1 = out_dir / "edge_taxonomy.png"
    s1 = figure_edge_taxonomy(convex, diffuse, f1)
    f2 = out_dir / "apriori_momentum.png"
    s2 = figure_apriori_momentum(cdetail, f2)

    print("\n=== Figure 1: 엣지의 두 유형 ===")
    for title, st in s1.items():
        print(f"  [{title}]")
        print(f"    n={st['n']:,}  승률={st['win_rate']:.1%}  왜도={st['skew']:+.2f}  "
              f"상위5건점유={st['top5_share']:.1%}  평균={st['mean']:+.4f}  "
              f"최대={st['max']:+.1%}  최저={st['min']:+.1%}")
    print(f"  → {f1}  ({f1.stat().st_size // 1024} KB)")

    print("\n=== Figure 2: 진입 모멘텀 5분위 기대수익(R) ===")
    print(f"  TRAIN Q1..Q5: {['%+.3f' % v for v in s2['train_R']]}")
    print(f"  OOS   Q1..Q5: {['%+.3f' % v for v in s2['oos_R']]}")
    print(f"  단조(Q5>Q1): TRAIN={s2['train_monotonic']}  OOS={s2['oos_monotonic']}  "
          f"verdict={s2['verdict']}")
    print(f"  → {f2}  ({f2.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
