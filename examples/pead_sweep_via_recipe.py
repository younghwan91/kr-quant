"""Define a PEAD surprise-filter sweep with the recipe API — no simulation loop.

Acceptance demo for the backtest-engine migration (Step 5): a new cross-sectional
experiment is defined declaratively (a list of :class:`ArmSpec` + one
:class:`ExperimentConfig`) and run with a single :func:`run_recipe` call. The
surprise filter is the ``fresh_days`` gate (only trade names whose earnings filing
is recent) swept across a grid — the engine's ``rank_tilt_backtest`` accounting
(t+1 entry, ADV floor, turnover cost) is reused, never rewritten.

Run: ``python examples/pead_sweep_via_recipe.py`` (synthetic data, no DB needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kr_quant.engine.recipe import ArmSpec, ExperimentConfig, run_recipe


def _synthetic_pead_panels():
    """Synthetic prices (code/date/close/trade_value) + a YoY earnings signal."""
    dates = pd.bdate_range("2020-01-01", periods=300).strftime("%Y-%m-%d")
    rng = np.random.default_rng(42)
    prices, earnings = [], []
    for j in range(50):
        code = f"{j:06d}"
        drift = 0.0005 * (j - 25)                       # signal-correlated drift
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.02, len(dates)))
        age = 0
        for d, c in zip(dates, close):
            prices.append({"code": code, "date": d, "close": c, "trade_value": 1e5})
            earnings.append({"code": code, "date": d, "yoy": drift * 1000,
                             "age_days": age})
            age = age + 1 if age < 90 else 0            # periodic fresh filings
    return pd.DataFrame(prices), pd.DataFrame(earnings)


def main() -> int:
    prices, earnings = _synthetic_pead_panels()

    # The whole experiment definition — one arm per surprise-filter setting.
    base = dict(horizon=20, adv_floor=0.0, start_index=60, min_names=5,
                long_only=True, top_n=10)
    arms = [ArmSpec(name=f"fresh{fd}",
                    backtest_kwargs={**base, "fresh_days": fd})
            for fd in (0, 5, 15, 45)]
    config = ExperimentConfig(experiment_type="cross_sectional", arms=arms)

    table, _ = run_recipe(config, prices, earnings)   # one call — no loop rewritten
    print("PEAD surprise-filter sweep (fresh_days gate) — recipe API\n")
    print(table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
