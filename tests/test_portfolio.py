"""Portfolio-cover figures — smoke tests: functions run headless and emit a PNG.

Chart content isn't asserted (impractical/brittle); the contract under test is
"given a well-formed DataFrame, a real image file is written without raising".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.viz.portfolio import plot_ranking, plot_rolling_validation, plot_score_vs_return


def _screen_result(n=5):
    return pd.DataFrame({
        "code": [f"{i:06d}" for i in range(n)],
        "name": [f"종목{i}" for i in range(n)],
        "market": ["코스닥" if i % 2 else "거래소" for i in range(n)],
        "score": [float(n - i) for i in range(n)],
    })


def test_plot_ranking_writes_a_real_png(tmp_path):
    out = plot_ranking(_screen_result(), tmp_path / "ranking.png", top=3)
    assert out.exists()
    assert out.stat().st_size > 0


def _summary(n_buckets=4):
    # mean_fwd/hit_rate must be arrays (not plain lists): plot_score_vs_return
    # does `b["mean_fwd"] * 100` expecting elementwise scaling, and a plain
    # Python list would instead repeat itself 100x under `*`.
    return {
        "spearman": 0.42, "n": 20, "universe_mean": 0.01,
        "buckets": {
            "quantile": list(range(1, n_buckets + 1)),
            "mean_fwd": np.array([0.05, 0.02, -0.01, -0.03][:n_buckets]),
            "hit_rate": np.array([0.6, 0.55, 0.4, 0.3][:n_buckets]),
        },
    }


def test_plot_score_vs_return_writes_a_real_png(tmp_path):
    merged = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0], "fwd_ret": [0.01, -0.02, 0.03, 0.0]})
    out = plot_score_vs_return(merged, _summary(), tmp_path / "score_ret.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_rolling_validation_writes_a_real_png(tmp_path):
    splits = pd.DataFrame({"spearman": [0.3, -0.1, 0.2, 0.5]})
    summary = _summary()
    summary.update({"spearman_mean": 0.225, "n_splits": 4, "frac_positive": 0.75})
    summary["buckets"]["median_fwd"] = summary["buckets"]["mean_fwd"]
    out = plot_rolling_validation(splits, summary, tmp_path / "rolling.png")
    assert out.exists()
    assert out.stat().st_size > 0
