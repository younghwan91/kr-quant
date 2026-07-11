"""Ridge-regression ensemble of the multi-channel supply-flow signals.

:mod:`kr_quant.strategies.multi_signal`'s equal-weight average of channels
(foreign/장기기관 EWMA, avg-cost gap, short-covering) made the real-data
result *worse* than the single-channel Phase 1 signal (49% vs 55% directional
consistency) — a plausible reason: channels don't all carry the same signal-
to-noise ratio, and simple averaging lets noisy channels dilute a better one
equally rather than in proportion to how useful each actually is.

This module replaces the equal-weight average with weights *learned* from
the data via ridge regression — still "simple" (closed-form linear
regression, no gradient descent, no deep learning framework), which keeps
the model appropriately sized for ~95 days of history (Architect review
precedent: don't fit more parameters than the data can support). No new
dependency is introduced; the closed-form solve uses only ``numpy``.

Walk-forward discipline: for each evaluation cutoff, the ridge weights are
fit **only** on (channel snapshot, realized forward return) pairs whose
outcome was already known strictly before the evaluation date — i.e. the
model is re-fit at every step on an expanding window of already-realized
history, never on the future. This mirrors the same lookahead discipline as
:mod:`kr_quant.strategies.supply_wave` and :mod:`kr_quant.models.graph_flow`.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..storage import connect, default_db_path
from ..strategies.backtest import forward_returns, spearman
from ..strategies.multi_signal import (
    DEFAULT_FLOW_CHANNELS,
    build_channel_features,
    load_multi_frame,
)
from ..strategies.supply_wave import assert_no_lookahead


def ridge_fit(X: np.ndarray, y: np.ndarray, *, lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression: minimize ||Xb - y||^2 + lam*||b||^2.

    Args:
        X: ``(n, p)`` feature matrix (an intercept column is added internally
            — callers should NOT prepend a constant column themselves).
        y: ``(n,)`` targets.
        lam: L2 penalty strength. The intercept term is not penalized.

    Returns:
        ``(p + 1,)`` coefficient vector, ``[intercept, beta_1, ..., beta_p]``.
    """
    n, p = X.shape
    X_aug = np.hstack([np.ones((n, 1)), X])
    penalty = lam * np.eye(p + 1)
    penalty[0, 0] = 0.0  # don't shrink the intercept
    beta = np.linalg.solve(X_aug.T @ X_aug + penalty, X_aug.T @ y)
    return beta


def ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apply ridge coefficients from :func:`ridge_fit` to new feature rows."""
    n = X.shape[0]
    X_aug = np.hstack([np.ones((n, 1)), X])
    return X_aug @ beta


def lasso_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lam: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """L1-penalized regression via coordinate descent (no sklearn dependency).

    Minimizes ``(1/2)||Xb - y||^2 + lam*||b||_1`` using the standard
    cyclic-coordinate-descent soft-thresholding update (Friedman et al.'s
    "pathwise coordinate descent", implemented directly since this repo
    avoids adding a full ML framework for a ~10-feature problem).

    Unlike :func:`ridge_fit`, L1 tends to zero out coefficients entirely —
    useful here specifically as a *feature-selection diagnostic*: if lasso
    zeros out a channel that ridge gives non-trivial weight to, that channel
    is a plausible confounder/noise, not a redundant-but-real signal.

    Args:
        X: ``(n, p)`` feature matrix (features are standardized internally
            before penalizing, so the penalty is comparable across channels
            regardless of each channel's raw scale — same rationale as
            :func:`full_sample_channel_importance`'s standardization).
        y: ``(n,)`` targets.
        lam: L1 penalty strength.
        max_iter: Maximum coordinate-descent sweeps.
        tol: Convergence tolerance on the max coefficient change per sweep.

    Returns:
        ``(p + 1,)`` coefficient vector in the same
        ``[intercept, beta_1, ..., beta_p]`` layout as :func:`ridge_fit`, so
        it's a drop-in for :func:`ridge_predict` — coefficients are already
        rescaled back to the original (unstandardized) feature units.
    """
    n, p = X.shape
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_std = (X - mu) / sigma
    y_mean = y.mean()
    y_c = y - y_mean

    beta = np.zeros(p)
    col_sq = (X_std**2).sum(axis=0)
    col_sq[col_sq == 0] = 1.0

    for _ in range(max_iter):
        beta_prev = beta.copy()
        for j in range(p):
            residual = y_c - X_std @ beta + X_std[:, j] * beta[j]
            rho = X_std[:, j] @ residual
            if rho < -lam / 2:
                beta[j] = (rho + lam / 2) / col_sq[j]
            elif rho > lam / 2:
                beta[j] = (rho - lam / 2) / col_sq[j]
            else:
                beta[j] = 0.0
        if np.max(np.abs(beta - beta_prev)) < tol:
            break

    beta_raw = beta / sigma
    intercept = y_mean - mu @ beta_raw
    return np.concatenate([[intercept], beta_raw])


def walk_forward_ensemble_eval(
    supply_df: pd.DataFrame,
    short_df: pd.DataFrame,
    *,
    flow_channels: tuple[str, ...] = DEFAULT_FLOW_CHANNELS,
    cost_gap_channel: str = "penfnd_etc",
    halflife: float = 7.0,
    horizons: tuple[int, ...] = (3, 5),
    min_formation: int = 8,
    min_train_rows: int = 30,
    ridge_lambda: float = 1.0,
    credit_df: pd.DataFrame | None = None,
    flow_feature: str = "level",
    model: str = "ridge",
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward: ridge- or lasso-weighted channel ensemble vs. forward return.

    For each ``(cutoff date t, horizon h)`` split, ridge weights are fit on
    every earlier ``(channel snapshot at s, forward_return[s -> s+h])`` pair
    whose outcome is already realized by ``t`` (i.e. ``s + h <= t`` in the
    trading-day index) — never on ``t``'s own not-yet-known outcome. The
    fitted weights predict a composite score for ``t``'s channel snapshot,
    which is then Spearman-correlated against the (only now, for evaluation)
    realized forward return at ``t + h``.

    Args:
        supply_df, short_df: Same shape as
            :func:`kr_quant.strategies.multi_signal.build_multi_channel_signal`
            expects.
        flow_channels, cost_gap_channel, halflife: Forwarded to
            :func:`kr_quant.strategies.multi_signal.build_channel_features`.
        horizons, min_formation: Forwarded to the walk-forward split logic
            (same meaning as in ``supply_wave``/``graph_flow``).
        min_train_rows: Minimum number of (code, historical-date) training
            rows required before a cutoff is evaluated; cutoffs with fewer
            available training examples are skipped (ridge on too few rows
            is unreliable, and skipping is honest about the data limit
            rather than fitting noise).
        ridge_lambda: L2 (ridge) or L1 (lasso) penalty strength, forwarded to
            whichever of :func:`ridge_fit`/:func:`lasso_fit` ``model``
            selects.
        model: ``"ridge"`` (default) or ``"lasso"`` — which fitter to use at
            every walk-forward step.

    Returns:
        ``(splits, summary)`` — same shape/keys as
        :func:`kr_quant.strategies.supply_wave.walk_forward_supply_wave_eval`
        (``base_date, eval_date, horizon, n, spearman, sign`` / ``n_splits,
        signs, frac_positive``), plus ``summary["n_skipped_insufficient_train"]``
        counting cutoffs skipped for lack of training data.
    """
    if model not in ("ridge", "lasso"):
        raise ValueError(f"model must be 'ridge' or 'lasso', got {model!r}")
    fit_fn = ridge_fit if model == "ridge" else lasso_fit

    signal_df, channel_cols = build_channel_features(
        supply_df,
        short_df,
        flow_channels=flow_channels,
        cost_gap_channel=cost_gap_channel,
        halflife=halflife,
        credit_df=credit_df,
        flow_feature=flow_feature,
    )
    dates = sorted(signal_df["date"].astype(str).unique())
    date_idx = {d: i for i, d in enumerate(dates)}

    # Pre-compute, for every (start date s, horizon h) with s+h in range, the
    # per-code forward return — reused across many training-set builds below
    # instead of recomputing per cutoff.
    fwd_cache: dict[tuple[str, int], pd.Series] = {}

    def _fwd(s: str, h: int) -> pd.Series:
        key = (s, h)
        if key not in fwd_cache:
            j = date_idx[s] + h
            if j >= len(dates):
                fwd_cache[key] = pd.Series(dtype=float)
            else:
                fwd_cache[key] = forward_returns(signal_df, s, dates[j])
        return fwd_cache[key]

    splits: list[dict] = []
    n_skipped = 0

    for h in horizons:
        for t in range(min_formation - 1, len(dates) - h):
            base_date, eval_date = dates[t], dates[t + h]

            # Build training rows from every earlier start date s whose
            # (s, s+h) outcome is already known by t.
            train_X: list[np.ndarray] = []
            train_y: list[np.ndarray] = []
            for s_idx in range(min_formation - 1, t):
                if s_idx + h > t:
                    continue
                s = dates[s_idx]
                snap_s = signal_df[signal_df["date"].astype(str) == s].dropna(
                    subset=channel_cols
                )
                if snap_s.empty:
                    continue
                fwd_s = _fwd(s, h)
                merged_s = snap_s.merge(
                    fwd_s, left_on="code", right_index=True, how="inner"
                ).dropna(subset=["fwd_ret"])
                if merged_s.empty:
                    continue
                train_X.append(merged_s[channel_cols].to_numpy(dtype=float))
                train_y.append(merged_s["fwd_ret"].to_numpy(dtype=float))

            if not train_X:
                n_skipped += 1
                continue
            X_train = np.vstack(train_X)
            y_train = np.concatenate(train_y)
            if len(y_train) < min_train_rows:
                n_skipped += 1
                continue

            snap_t = signal_df[signal_df["date"].astype(str) == base_date].dropna(
                subset=channel_cols
            )
            if len(snap_t) < 2:
                continue

            use_date = dates[t + 1] if t + 1 < len(dates) else eval_date
            guard_df = pd.DataFrame({"date": snap_t["date"], "trade_date": use_date})
            assert_no_lookahead(guard_df, source_date_col="date", use_date_col="trade_date")

            beta = fit_fn(X_train, y_train, lam=ridge_lambda)
            X_test = snap_t[channel_cols].to_numpy(dtype=float)
            snap_t = snap_t.assign(ensemble_score=ridge_predict(X_test, beta))

            fwd_t = _fwd(base_date, h)
            merged_t = snap_t.merge(
                fwd_t, left_on="code", right_index=True, how="inner"
            ).dropna(subset=["fwd_ret"])
            if len(merged_t) < 2:
                continue

            corr = spearman(merged_t["ensemble_score"], merged_t["fwd_ret"])
            if pd.isna(corr):
                continue
            sign = "+" if corr > 0 else ("-" if corr < 0 else "0")
            splits.append(
                {
                    "base_date": base_date,
                    "eval_date": eval_date,
                    "horizon": h,
                    "n": len(merged_t),
                    "n_train": len(y_train),
                    "spearman": corr,
                    "sign": sign,
                }
            )

    splits_df = pd.DataFrame(
        splits,
        columns=["base_date", "eval_date", "horizon", "n", "n_train", "spearman", "sign"],
    )
    if splits_df.empty:
        summary = {
            "n_splits": 0,
            "signs": [],
            "frac_positive": float("nan"),
            "n_skipped_insufficient_train": n_skipped,
        }
    else:
        summary = {
            "n_splits": len(splits_df),
            "signs": splits_df["sign"].tolist(),
            "frac_positive": float((splits_df["spearman"] > 0).mean()),
            "n_skipped_insufficient_train": n_skipped,
        }
    return splits_df, summary


def full_sample_channel_importance(
    supply_df: pd.DataFrame,
    short_df: pd.DataFrame,
    *,
    flow_channels: tuple[str, ...] = DEFAULT_FLOW_CHANNELS,
    cost_gap_channel: str = "penfnd_etc",
    halflife: float = 7.0,
    horizon: int = 5,
    ridge_lambda: float = 1.0,
    credit_df: pd.DataFrame | None = None,
    flow_feature: str = "level",
    model: str = "ridge",
) -> dict[str, float]:
    """Fit one ridge/lasso on the *entire* sample (all dates pooled) to read off
    standardized per-channel coefficients — for interpretability only.

    This is deliberately **not** part of :func:`walk_forward_ensemble_eval`'s
    backtest: fitting on the full sample (including data that would be
    "future" relative to many of the walk-forward splits) is exactly the
    lookahead the walk-forward evaluation is built to avoid. Use this only to
    answer "which channels does the model lean on", never to claim a
    backtested result.

    Returns:
        ``{channel_name: standardized_coefficient}`` — channels are
        z-scored before fitting so coefficients are comparable in magnitude
        regardless of each channel's raw scale; sign indicates the learned
        direction (positive = higher rank predicts higher forward return).
    """
    signal_df, channel_cols = build_channel_features(
        supply_df,
        short_df,
        flow_channels=flow_channels,
        cost_gap_channel=cost_gap_channel,
        halflife=halflife,
        credit_df=credit_df,
        flow_feature=flow_feature,
    )
    dates = sorted(signal_df["date"].astype(str).unique())
    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for i, s in enumerate(dates[: len(dates) - horizon]):
        snap = signal_df[signal_df["date"].astype(str) == s].dropna(subset=channel_cols)
        if snap.empty:
            continue
        fwd = forward_returns(signal_df, s, dates[i + horizon])
        merged = snap.merge(fwd, left_on="code", right_index=True, how="inner").dropna(
            subset=["fwd_ret"]
        )
        if merged.empty:
            continue
        X_rows.append(merged[channel_cols].to_numpy(dtype=float))
        y_rows.append(merged["fwd_ret"].to_numpy(dtype=float))

    if not X_rows:
        return {c: float("nan") for c in channel_cols}

    if model not in ("ridge", "lasso"):
        raise ValueError(f"model must be 'ridge' or 'lasso', got {model!r}")

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)

    if model == "ridge":
        mu, sigma = X.mean(axis=0), X.std(axis=0)
        sigma[sigma == 0] = 1.0
        X_std = (X - mu) / sigma
        beta = ridge_fit(X_std, y, lam=ridge_lambda)
    else:
        # lasso_fit standardizes internally and returns coefficients already
        # rescaled to raw-feature units — don't standardize twice here.
        beta = lasso_fit(X, y, lam=ridge_lambda)
    return dict(zip(channel_cols, beta[1:]))  # beta[0] is the intercept


def main() -> int:
    parser = argparse.ArgumentParser(
        description="채널별 가중치를 릿지회귀로 학습하는 다채널 수급 신호 앙상블 워크포워드 검증"
    )
    parser.add_argument("--db", default=str(default_db_path()))
    parser.add_argument("--flow-channels", nargs="+", default=list(DEFAULT_FLOW_CHANNELS))
    parser.add_argument("--cost-gap-channel", default="penfnd_etc")
    parser.add_argument("--halflife", type=float, default=7.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--min-formation", type=int, default=8)
    parser.add_argument("--min-train-rows", type=int, default=30)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument(
        "--use-credit", action="store_true",
        help="credit_balance.balance_rt 기반 신용잔고 채널도 포함 (기본: 미포함)",
    )
    parser.add_argument(
        "--flow-feature", choices=["level", "accel", "vector"], default="level",
        help="flow_channels 표현: level(EWMA 레벨)/accel(EWMA 1차차분)/vector(둘 다)",
    )
    parser.add_argument(
        "--model", choices=["ridge", "lasso"], default="ridge",
        help="채널 결합 모델: ridge(L2, 축소만) / lasso(L1, 불필요 채널을 0으로)",
    )
    args = parser.parse_args()

    con = connect(args.db)
    supply_df, short_df, credit_df = load_multi_frame(con)
    con.close()

    splits, summary = walk_forward_ensemble_eval(
        supply_df,
        short_df,
        flow_channels=tuple(args.flow_channels),
        cost_gap_channel=args.cost_gap_channel,
        halflife=args.halflife,
        horizons=tuple(args.horizons),
        min_formation=args.min_formation,
        min_train_rows=args.min_train_rows,
        ridge_lambda=args.ridge_lambda,
        credit_df=credit_df if args.use_credit else None,
        flow_feature=args.flow_feature,
        model=args.model,
    )
    if summary["n_splits"]:
        print(
            f"워크포워드 분할 {summary['n_splits']}개 "
            f"(훈련데이터 부족으로 스킵 {summary['n_skipped_insufficient_train']}개) | "
            f"{args.model}앙상블 신호 양(+) 비율 {summary['frac_positive']:.0%}"
        )
    else:
        print(
            f"워크포워드 분할 0개 (스킵 {summary['n_skipped_insufficient_train']}개) — "
            "데이터가 부족합니다."
        )
    if not splits.empty:
        print(splits.to_string(index=False))
    print(
        "\n참고: Phase 1(55%) 및 다채널 동일가중 평균(49%)과 나란히 비교하기 위한 것으로, "
        "이 스크립트는 임계값을 강제하지 않습니다."
    )

    importance = full_sample_channel_importance(
        supply_df,
        short_df,
        flow_channels=tuple(args.flow_channels),
        cost_gap_channel=args.cost_gap_channel,
        halflife=args.halflife,
        horizon=args.horizons[0],
        ridge_lambda=args.ridge_lambda,
        credit_df=credit_df if args.use_credit else None,
        flow_feature=args.flow_feature,
        model=args.model,
    )
    print(f"\n채널별 중요도 (표준화 계수, horizon={args.horizons[0]}일, 전체샘플 fit — 참고용이지 백테스트 아님):")
    for name, coef in sorted(importance.items(), key=lambda kv: abs(kv[1]) if kv[1] == kv[1] else -1, reverse=True):
        print(f"  {name}: {coef:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
