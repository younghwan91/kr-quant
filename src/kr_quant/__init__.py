"""kr-quant — Korean equity supply/demand collection and quant analysis.

Built on the `kiwoom-client` REST API library. Layers:
    collectors/  — fetch Kiwoom datasets into SQLite
    storage      — schema and persistence helpers
    strategies/  — screeners/strategies over the collected data
    viz/         — charts
"""

from __future__ import annotations

__version__ = "0.1.0"
