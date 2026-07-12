"""kr-quant — Korean equity strategy/feature analysis library.

Data collection lives in the kr-quant-airflow repo (collectors/), not here.
Layers:
    storage      — read-side schema helpers (connect, market_cap_asof)
    strategies/  — screeners/strategies over collected data
    features/    — feature engineering
    viz/         — charts
"""

from __future__ import annotations

__version__ = "0.1.0"
