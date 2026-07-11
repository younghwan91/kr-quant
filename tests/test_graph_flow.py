"""Graph-diffusion propagation: adjacency construction + signal propagation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.models.graph_flow import build_sector_adjacency, propagate_signal


def _stocks(pairs):
    return pd.DataFrame(pairs, columns=["code", "sector"])


def test_build_sector_adjacency_is_row_stochastic_within_sector():
    stocks = _stocks(
        [("A", "전기전자"), ("B", "전기전자"), ("C", "전기전자"), ("D", "화학")]
    )
    codes, adj = build_sector_adjacency(stocks)
    assert codes == ["A", "B", "C", "D"]

    # A/B/C share a 3-member sector: each row sums to 1, no self-loop.
    for i in range(3):
        assert adj[i, i] == 0.0
        assert np.isclose(adj[i].sum(), 1.0)
        assert np.isclose(adj[i, (i + 1) % 3], 1 / 2) or np.isclose(adj[i, (i + 2) % 3], 1 / 2)

    # D is alone in "화학" — no peers, all-zero row.
    assert np.allclose(adj[3], 0.0)


def test_build_sector_adjacency_ignores_null_sector():
    stocks = _stocks([("A", "전기전자"), ("B", None), ("C", "전기전자")])
    codes, adj = build_sector_adjacency(stocks)
    assert "B" not in codes
    assert len(codes) == 2


def test_propagate_signal_pulls_toward_sector_peers():
    stocks = _stocks([("A", "S"), ("B", "S"), ("C", "S")])
    codes, adj = build_sector_adjacency(stocks)

    # A starts very high, B and C start low — one round with alpha=0.5
    # should pull A down toward its peers and pull B/C up toward A.
    signal = pd.Series({"A": 10.0, "B": 0.0, "C": 0.0})
    out = propagate_signal(signal, codes, adj, steps=1, alpha=0.5)

    assert out["A"] < 10.0
    assert out["B"] > 0.0
    assert out["C"] > 0.0


def test_propagate_signal_alpha_one_is_identity():
    stocks = _stocks([("A", "S"), ("B", "S")])
    codes, adj = build_sector_adjacency(stocks)
    signal = pd.Series({"A": 5.0, "B": -3.0})

    out = propagate_signal(signal, codes, adj, steps=3, alpha=1.0)
    assert np.isclose(out["A"], 5.0)
    assert np.isclose(out["B"], -3.0)


def test_propagate_signal_preserves_nan_for_unknown_codes():
    stocks = _stocks([("A", "S"), ("B", "S"), ("C", "S")])
    codes, adj = build_sector_adjacency(stocks)
    signal = pd.Series({"A": 1.0, "B": 2.0})  # C missing

    out = propagate_signal(signal, codes, adj, steps=1, alpha=0.5)
    assert pd.isna(out["C"])
    assert not pd.isna(out["A"])
    assert not pd.isna(out["B"])


def test_propagate_signal_singleton_sector_untouched_by_neighbors():
    stocks = _stocks([("A", "solo"), ("B", "S"), ("C", "S")])
    codes, adj = build_sector_adjacency(stocks)
    signal = pd.Series({"A": 7.0, "B": 0.0, "C": 0.0})

    # A has an all-zero adjacency row, so neighbor_avg is 0 and the residual
    # blend still shrinks A toward 0 each round (alpha<1) — but crucially it
    # should NOT get pulled toward B/C's values, only decay toward 0.
    out = propagate_signal(signal, codes, adj, steps=1, alpha=0.5)
    assert np.isclose(out["A"], 3.5)  # 0.5*7 + 0.5*0
