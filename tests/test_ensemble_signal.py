"""Ridge-regression channel ensemble: closed-form solver + walk-forward eval."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.models.ensemble_signal import (
    full_sample_channel_importance,
    lasso_fit,
    ridge_fit,
    ridge_predict,
    walk_forward_ensemble_eval,
)


def test_ridge_fit_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    n, p = 200, 3
    X = rng.normal(size=(n, p))
    true_beta = np.array([0.5, 2.0, -1.0, 0.3])  # intercept + 3 coefs
    y = true_beta[0] + X @ true_beta[1:] + rng.normal(scale=0.01, size=n)

    beta = ridge_fit(X, y, lam=0.01)
    assert np.allclose(beta, true_beta, atol=0.1)


def test_ridge_predict_matches_manual_computation():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    beta = np.array([1.0, 0.5, -0.5])  # intercept, b1, b2
    pred = ridge_predict(X, beta)
    expected = np.array([1.0 + 0.5 * 1.0 - 0.5 * 2.0, 1.0 + 0.5 * 3.0 - 0.5 * 4.0])
    assert np.allclose(pred, expected)


def test_ridge_fit_shrinks_toward_zero_as_lambda_grows():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.1, size=50)

    beta_small_lam = ridge_fit(X, y, lam=0.001)
    beta_large_lam = ridge_fit(X, y, lam=1000.0)

    assert np.sum(beta_small_lam[1:] ** 2) > np.sum(beta_large_lam[1:] ** 2)


def test_lasso_fit_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    n, p = 300, 3
    X = rng.normal(size=(n, p))
    true_beta = np.array([0.5, 2.0, -1.0, 0.0])  # intercept + 3 coefs, last one truly zero
    y = true_beta[0] + X @ true_beta[1:] + rng.normal(scale=0.01, size=n)

    beta = lasso_fit(X, y, lam=0.01)
    assert np.allclose(beta[:3], true_beta[:3], atol=0.15)


def test_lasso_fit_zeros_out_irrelevant_features_at_high_lambda():
    rng = np.random.default_rng(2)
    n = 200
    x_relevant = rng.normal(size=n)
    x_noise = rng.normal(size=n)  # unrelated to y
    X = np.column_stack([x_relevant, x_noise])
    y = 3.0 * x_relevant + rng.normal(scale=0.05, size=n)

    beta = lasso_fit(X, y, lam=5.0)
    # The noise feature's coefficient should be driven to exactly zero (or
    # very close to it) while the relevant one stays non-trivial.
    assert abs(beta[2]) < 0.05
    assert abs(beta[1]) > 0.5


def test_lasso_fit_shrinks_toward_zero_as_lambda_grows():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.1, size=80)

    beta_small_lam = lasso_fit(X, y, lam=0.001)
    beta_large_lam = lasso_fit(X, y, lam=50.0)

    assert np.sum(np.abs(beta_small_lam[1:])) > np.sum(np.abs(beta_large_lam[1:]))


def test_lasso_predict_compatible_with_ridge_predict():
    # lasso_fit's returned coefficients must be usable by ridge_predict
    # (same [intercept, beta...] layout, already in raw-feature units).
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 1.0], [2.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    beta = lasso_fit(X, y, lam=0.01)
    pred = ridge_predict(X, beta)
    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))


def _dates(n, start=1):
    return [f"202601{d:02d}" for d in range(start, start + n)]


def _synthetic_multi_frame(n_codes=8, n_days=30, seed=0):
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}" for i in range(n_codes)]
    dates = _dates(n_days)
    rows, short_rows = [], []
    for code in codes:
        price = 1000.0
        balance = 10000.0
        for date in dates:
            price *= 1 + rng.normal(0, 0.01)
            rows.append(
                {
                    "code": code,
                    "date": date,
                    "close": price,
                    "market_cap": 1_000_000_000,
                    "individual": 0,
                    "foreign_": int(rng.integers(-5000, 5000)),
                    "institution": 0,
                    "fnnc_invt": 0,
                    "insrnc": 0,
                    "invtrt": 0,
                    "bank": 0,
                    "penfnd_etc": int(rng.integers(-5000, 5000)),
                    "samo_fund": int(rng.integers(-5000, 5000)),
                    "natn": 0,
                    "etc_corp": 0,
                }
            )
            balance = max(balance + rng.integers(-500, 500), 0)
            short_rows.append({"code": code, "date": date, "short_balance": balance})
    return pd.DataFrame(rows), pd.DataFrame(short_rows)


def test_walk_forward_ensemble_eval_runs_end_to_end():
    supply_df, short_df = _synthetic_multi_frame(n_codes=10, n_days=30)
    splits, summary = walk_forward_ensemble_eval(
        supply_df, short_df, horizons=(3,), min_formation=8, min_train_rows=10
    )
    assert summary["n_splits"] > 0
    assert 0.0 <= summary["frac_positive"] <= 1.0
    assert "n_skipped_insufficient_train" in summary
    assert set(splits.columns) == {
        "base_date", "eval_date", "horizon", "n", "n_train", "spearman", "sign",
    }


def test_walk_forward_ensemble_eval_skips_when_too_little_training_data():
    supply_df, short_df = _synthetic_multi_frame(n_codes=3, n_days=15)
    # min_train_rows set absurdly high -> every cutoff should be skipped.
    splits, summary = walk_forward_ensemble_eval(
        supply_df, short_df, horizons=(3,), min_formation=8, min_train_rows=10_000
    )
    assert summary["n_splits"] == 0
    assert summary["n_skipped_insufficient_train"] > 0


def test_walk_forward_ensemble_eval_supports_lasso_model():
    supply_df, short_df = _synthetic_multi_frame(n_codes=10, n_days=30)
    # Small training sets + lasso's soft-thresholding can zero out every
    # coefficient (constant prediction -> undefined Spearman corr, split
    # skipped) at ridge_lambda's ridge-scaled default; use a smaller penalty
    # so at least some splits produce a non-degenerate prediction.
    splits, summary = walk_forward_ensemble_eval(
        supply_df, short_df, horizons=(3,), min_formation=8, min_train_rows=10,
        model="lasso", ridge_lambda=0.01,
    )
    assert summary["n_splits"] > 0
    assert 0.0 <= summary["frac_positive"] <= 1.0


def test_walk_forward_ensemble_eval_rejects_unknown_model():
    supply_df, short_df = _synthetic_multi_frame(n_codes=5, n_days=15)
    import pytest
    with pytest.raises(ValueError):
        walk_forward_ensemble_eval(supply_df, short_df, model="not_a_model")


def test_walk_forward_ensemble_eval_supports_accel_and_vector_flow_feature():
    supply_df, short_df = _synthetic_multi_frame(n_codes=10, n_days=30)
    for feature in ("level", "accel", "vector"):
        splits, summary = walk_forward_ensemble_eval(
            supply_df, short_df, horizons=(3,), min_formation=8, min_train_rows=10,
            flow_feature=feature,
        )
        assert summary["n_splits"] > 0, f"flow_feature={feature} produced no splits"


def test_full_sample_channel_importance_vector_has_twice_the_flow_columns():
    supply_df, short_df = _synthetic_multi_frame(n_codes=10, n_days=30)
    level_importance = full_sample_channel_importance(
        supply_df, short_df, horizon=3, flow_feature="level"
    )
    vector_importance = full_sample_channel_importance(
        supply_df, short_df, horizon=3, flow_feature="vector"
    )
    # vector adds one accel column per flow channel on top of level's set,
    # so it should have strictly more channels than level alone.
    assert len(vector_importance) > len(level_importance)


def test_full_sample_channel_importance_supports_lasso():
    supply_df, short_df = _synthetic_multi_frame(n_codes=10, n_days=30)
    importance = full_sample_channel_importance(
        supply_df, short_df, horizon=3, model="lasso", ridge_lambda=1.0
    )
    assert all(np.isfinite(v) for v in importance.values())
