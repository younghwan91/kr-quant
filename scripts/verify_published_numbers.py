#!/usr/bin/env python
"""One-shot acceptance check: engine-backed strategies match pre-migration behavior.

Manual step (needs real DB data via KR_QUANT_DB) -- not part of the pytest CI
suite. Compares the CURRENT (engine-backed) ``staggered_backtest`` against the
PRE-MIGRATION version of ``pead.py`` (git commit before the Step 3 engine
migration), run on the exact same live-DB data.

Note: this deliberately does NOT compare against a frozen numeric snapshot
(e.g. research/PEAD_REFINEMENT_RESULTS.md). KR_QUANT_DB is a live TimescaleDB
that production DAGs keep ingesting into daily, so day-over-day numbers
naturally drift as new trading days/earnings land -- that drift is expected
and is not a migration regression. The only valid acceptance check is
old-code-vs-new-code on IDENTICAL data, which is what this script does.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PEAD_PATH = REPO_ROOT / "src" / "kr_quant" / "strategies" / "pead.py"
# Last commit before the Step 3 engine migration touched pead.py.
PRE_MIGRATION_REF = "016f426"


def _run_baseline() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "research"))
    # Force re-import in case a prior call in this process cached the module.
    for mod in list(sys.modules):
        if mod == "pead_refinement" or mod.startswith("kr_quant"):
            del sys.modules[mod]
    from pead_refinement import load_data, run_baseline  # noqa: PLC0415

    prices, yoy_panel = load_data()
    return run_baseline(prices, yoy_panel)


def main() -> int:
    current_backup = PEAD_PATH.read_text()

    print("=== Engine-backed (current) staggered_backtest baseline ===")
    new_summary = _run_baseline()
    print(f"  n={new_summary['n']}  Sharpe={new_summary['sharpe']:.4f}  "
          f"t={new_summary['t_stat']:.4f}  cum_net={new_summary['cum_net']:.4f}  "
          f"hit_rate={new_summary['hit_rate']:.4f}")

    print(f"\n=== Pre-migration ({PRE_MIGRATION_REF}) staggered_backtest, same DB data ===")
    old_source = subprocess.run(
        ["git", "show", f"{PRE_MIGRATION_REF}:src/kr_quant/strategies/pead.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    PEAD_PATH.write_text(old_source)
    try:
        old_summary = _run_baseline()
    finally:
        PEAD_PATH.write_text(current_backup)

    print(f"  n={old_summary['n']}  Sharpe={old_summary['sharpe']:.4f}  "
          f"t={old_summary['t_stat']:.4f}  cum_net={old_summary['cum_net']:.4f}  "
          f"hit_rate={old_summary['hit_rate']:.4f}")

    print("\n=== Comparison (old vs new, same data) ===")
    ok = True
    for key in ("n", "sharpe", "t_stat", "cum_net", "hit_rate"):
        old_v, new_v = old_summary[key], new_summary[key]
        same = old_v == new_v
        ok &= same
        print(f"  {key}: old={old_v!r} new={new_v!r} -> {'MATCH' if same else 'DIVERGENCE'}")

    if ok:
        print("\nByte-identical: engine migration introduced zero behavioral change.")
        return 0
    print("\nDIVERGENCE: engine-backed output differs from pre-migration code on "
          "identical data. This IS a regression -- investigate before trusting the migration.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
