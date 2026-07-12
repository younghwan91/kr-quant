"""SEPA faithful-reproduction experiment — the one-command orchestrator.

Wires every Phase 0/1/2 component into the pre-registered multi-arm comparison
(``SEPA_FAITHFUL_DESIGN.md``): build the panels from real prices + DART earnings +
shares (split-adjusted, point-in-time), run the five arms, and hand aligned monthly
return series to :func:`kr_quant.strategies.sepa_compare.compare_arms` for the
paired-bootstrap verdict.

Arms:
    A            faithful SEPA (small-mid, trend template + RS + Code33 + VCP, 6 names)
    A-diversified  A's signal, deployed-style diversified sizing (isolates concentration)
    A-noVCP      A without the VCP/pivot gate (isolates VCP's contribution)
    B-shell      deployed large-cap breakout shell (no earnings, diversified)
    C-bench      cap-weighted index proxy

``run_experiment`` is pure (DataFrames in → verdict out) so it is synthetic-testable
before the real backfill; ``main`` wires the DB + earnings CSV for the live run.
"""

from __future__ import annotations

import pandas as pd

from ..engine.recipe import ArmSpec, ExperimentConfig, run_recipe, sepa_faithful_config

N_CONCENTRATED = 6      # arm A frozen concentration
N_DIVERSIFIED = 50      # A-diversified / B-shell (deployed over-diversification)
LARGE_CAP_RANK = (0, 100)   # B-shell universe: mega/large (the deployed inversion)


def run_experiment(
    prices: pd.DataFrame,
    earnings: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    adjust: bool = True,
    **boot_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Run all five arms and return ``(comparison_table, verdicts)``.

    Everything downstream of the frozen hyperparameters — no tuning knobs — so a
    live run is a single call once the earnings backfill lands. Delegates to the
    declarative recipe API (:func:`kr_quant.engine.recipe.sepa_faithful_config` +
    :func:`kr_quant.engine.recipe.run_recipe`); the panel build + simulation loops
    now live in ``kr_quant.engine``, not here.
    """
    config = sepa_faithful_config(
        adjust=adjust, n_concentrated=N_CONCENTRATED, n_diversified=N_DIVERSIFIED,
        **boot_kwargs)
    return run_recipe(config, prices, earnings, shares)


def robustness_sweep(
    prices: pd.DataFrame,
    earnings: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    adjust: bool = True,
    concentrations: tuple[int, ...] = (4, 6, 8),
    stops: tuple[float, ...] = (0.04, 0.05, 0.08),
    use_vcp: bool = True,
) -> pd.DataFrame:
    """Curve-fit check: arm A's Sharpe/CAGR across frozen-neighbour hyperparameters.

    The verdict uses the frozen (6 names, 5% stop); this sweeps the *alternates*
    (``SEPA_FAITHFUL_DESIGN.md`` §견고성 부록 — **for robustness only, not selection**)
    to show the frozen point is not a knife-edge. Concentration varies the book slots,
    stop varies the hard stop in :func:`sepa_trades`. Each stop is one summary-only
    recipe run (a config whose arms are the concentrations).

    Returns:
        Long DataFrame ``concentration``/``stop``/``sharpe``/``cagr``/``n_trades`` —
        one row per (concentration, stop). NaN Sharpe means the gate produced no
        trades at that setting (not a failure — the faithful gates are very selective).
    """
    rows: list[dict] = []
    for stop in stops:
        arms = [
            ArmSpec(name=str(n), kind="sepa", universe="smallmid",
                    entry_kwargs={"use_vcp": use_vcp, "use_code33": True},
                    trade_kwargs={"stop_pct": stop},
                    book_kwargs={"n_slots": n, "sized": True})
            for n in concentrations
        ]
        config = ExperimentConfig(experiment_type="event_driven", arms=arms,
                                  adjust=adjust, compare=False)
        table, _ = run_recipe(config, prices, earnings, shares)
        for n in concentrations:
            row = table.loc[str(n)]
            rows.append({"concentration": n, "stop": stop,
                         "sharpe": row["sharpe"], "cagr": row["cagr"],
                         "n_trades": int(row["n_trades"])})
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame (indexed by arm) as a GitHub-flavoured markdown table."""
    cols = list(df.columns)
    head = "| arm | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                cells.append("—" if pd.isna(val) else f"{val:+.3f}")
            else:
                cells.append(str(val))
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


HISTORY_MARKER = "<!-- SEPA_VERDICT_HISTORY: content below this line is preserved across write_verdict reruns -->"


def _write_preserving_history(md: str, out_path: str) -> None:
    """Write ``md`` to ``out_path``, keeping any hand-written content that follows
    :data:`HISTORY_MARKER` in the existing file (interpretation notes, judgment-call
    logs, etc. that ``write_verdict`` itself never generates — see the 2026-07-12
    incident where a rerun silently wiped several rounds of curated analysis)."""
    from pathlib import Path

    p = Path(out_path)
    history = ""
    if p.exists():
        old = p.read_text()
        idx = old.find(HISTORY_MARKER)
        if idx != -1:
            history = old[idx:]
    p.write_text(md + (("\n" + history) if history else f"\n{HISTORY_MARKER}\n"))


def write_verdict(
    table: pd.DataFrame,
    verdicts: dict,
    *,
    out_path: str | None = None,
    focus: str = "A",
    diversified: str = "A-diversified",
) -> str:
    """Apply the pre-registered pass/fail criteria and render the verdict markdown.

    Criteria (``SEPA_FAITHFUL_DESIGN.md`` §판정 기준): (i) paired A−B ΔSharpe CI
    excludes 0, (ii) A−C ΔSharpe CI excludes 0, (iii) regime signs 3+/4, (v) A beats
    A-diversified (concentration justified — point comparison of the table Sharpes).
    A ``focus`` arm that produced no trades (NaN Sharpe) is reported as 평가불가.

    ``out_path`` writes are **history-preserving**: any existing content in the file
    at or after :data:`HISTORY_MARKER` survives a rerun untouched — only the
    auto-generated table/verdict above the marker is replaced. Append curated
    interpretation notes below the marker and they will not be lost on the next run.

    Args:
        table: ``compare_arms`` table (arm × sharpe/cagr/max_dd/pos_regimes).
        verdicts: ``compare_arms`` verdicts dict (paired bootstrap per arm).
        out_path: If given, also write the markdown there (history-preserving).
        focus, diversified: The deploy-candidate arm and its diversified sibling.

    Returns:
        The verdict markdown string (auto-generated portion only, not the history).
    """
    lines = ["# 미너비니 SEPA 충실 재현 — 판정", "",
             "## 다-arm 비교 (분할조정·PIT·페어드 부트스트랩)", "", _md_table(table), "",
             f"## 사전등록 판정 — focus arm: `{focus}`", ""]
    sh = table.loc[focus, "sharpe"] if focus in table.index else float("nan")
    if not (isinstance(sh, (int, float)) and pd.notna(sh)):
        lines.append(f"- **평가불가 (무거래)** — `{focus}`가 거래를 내지 못함(게이트 과선별). 판정 보류.")
        md = "\n".join(lines) + "\n"
        if out_path:
            _write_preserving_history(md, out_path)
        return md

    v = verdicts.get(focus, {})
    ci_b = v.get("vs_deployed", {}).get("d_sharpe_ci", (float("nan"), float("nan")))
    ci_c = v.get("vs_benchmark", {}).get("d_sharpe_ci", (float("nan"), float("nan")))
    crit_i = bool(v.get("beats_b_ci", False))
    crit_ii = bool(v.get("beats_c_ci", False))
    pos, tot = _parse_regimes(table.loc[focus, "pos_regimes"])
    crit_iii = tot > 0 and pos * 4 >= 3 * tot          # ≥ 3/4 of the buckets positive
    sh_div = table.loc[diversified, "sharpe"] if diversified in table.index else float("nan")
    crit_v = bool(pd.notna(sh_div) and sh > sh_div)

    def _mark(ok: bool) -> str:
        return "✅ 승" if ok else "❌ 무"

    lines += [
        f"- (i) A−B ΔSharpe CI [{ci_b[0]:+.2f}, {ci_b[1]:+.2f}] (0 배제) → {_mark(crit_i)} — 배포판 초과",
        f"- (ii) A−C ΔSharpe CI [{ci_c[0]:+.2f}, {ci_c[1]:+.2f}] (0 배제) → {_mark(crit_ii)} — 벤치 초과",
        f"- (iii) 레짐 부호 {pos}/{tot} (3+/4 필요) → {_mark(crit_iii)}",
        f"- (v) A Sharpe {sh:+.2f} vs A-diversified {sh_div:+.2f} → {_mark(crit_v)} — 집중 정당",
        "",
    ]
    overall = crit_i and crit_ii and crit_iii and crit_v
    lines.append(f"## 종합: {'✅ 배포후보 갱신 (A 승)' if overall else '❌ 기존 배포판 유지 (A 미달)'}")
    md = "\n".join(lines) + "\n"
    if out_path:
        _write_preserving_history(md, out_path)
    return md


def _parse_regimes(s: object) -> tuple[int, int]:
    """Parse a ``"3/4"`` regime string → ``(3, 4)``; ``(0, 0)`` if unparseable."""
    try:
        a, b = str(s).split("/")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return 0, 0


def main() -> int:
    """CLI (``kq-sepa``): run the faithful-SEPA arm comparison on real data."""
    import argparse

    import pandas as _pd

    from ..storage import connect, db_default

    ap = argparse.ArgumentParser(description="미너비니 SEPA 충실 재현 — 다-arm 비교 판정")
    ap.add_argument("--db", default=db_default())
    ap.add_argument("--earnings-csv", default=None,
                    help="DART 실적 CSV(dart_earnings.main() 산출물, 10컬럼): code,period,"
                         "avail_date,netinc,netinc_prior,yoy,revenue,revenue_prior,op_income,"
                         "op_income_prior. --earnings-table과 상호배타.")
    ap.add_argument("--earnings-table", default=None,
                    help="실적을 DB 테이블(예: earnings)에서 직접 조회 — 파이프라인 DB화 이후 "
                         "권장 경로(임시 스크립트 불필요). 스키마: code,period,avail_date,"
                         "netinc,netinc_prior,revenue,revenue_prior,op_income,op_income_prior.")
    ap.add_argument("--no-adjust", action="store_true", help="분할조정 생략(디버그용)")
    ap.add_argument("--verdict-out", default=None, help="사전등록 판정 markdown 저장 경로")
    args = ap.parse_args()
    if bool(args.earnings_csv) == bool(args.earnings_table):
        raise SystemExit("--earnings-csv 또는 --earnings-table 중 정확히 하나를 지정하세요")

    con = connect(args.db)
    if args.earnings_table:
        ea = _pd.read_sql_query(f"SELECT * FROM {args.earnings_table}", con)  # noqa: S608 — trusted local config, not user input
        ea["code"] = ea["code"].astype(str)
        ea["period"] = ea["period"].astype(str)
        ea["avail_date"] = ea["avail_date"].astype(str)
    else:
        # dart_earnings.main() CSV 스키마 (10칸, 헤더 없음): yoy가 6번째 — code33_panel엔
        # 불필요하나 위치 정렬을 위해 이름을 준다.
        cols = ["code", "period", "avail_date", "netinc", "netinc_prior", "yoy",
                "revenue", "revenue_prior", "op_income", "op_income_prior"]
        ea = _pd.read_csv(args.earnings_csv, names=cols, dtype={"code": str, "avail_date": str, "period": str})
    codes = sorted(ea["code"].unique())
    prices = _pd.read_sql_query(
        "SELECT code,date,open,high,low,close,volume,trade_value FROM daily_bars "
        "WHERE code = ANY(%(c)s)", con, params={"c": codes})
    shares = _pd.read_sql_query(
        "SELECT code,date,shares_outstanding FROM shares_outstanding_history WHERE code = ANY(%(c)s)",
        con, params={"c": codes})
    con.close()
    prices["date"] = prices["date"].astype(str)

    table, verdicts = run_experiment(prices, ea, shares, adjust=not args.no_adjust)
    print("\n=== SEPA 다-arm 비교 (분할조정·PIT·페어드 부트스트랩) ===")
    print(table.to_string())
    print("\n=== 판정 (ΔSharpe CI가 0 배제해야 승) ===")
    for arm, v in verdicts.items():
        b, c = v["vs_deployed"]["d_sharpe_ci"], v["vs_benchmark"]["d_sharpe_ci"]
        print(f"{arm}: vs B-shell ΔSharpe CI [{b[0]:+.2f},{b[1]:+.2f}] {'승' if v['beats_b_ci'] else '무'} | "
              f"vs C-bench CI [{c[0]:+.2f},{c[1]:+.2f}] {'승' if v['beats_c_ci'] else '무'}")
    if args.verdict_out:
        write_verdict(table, verdicts, out_path=args.verdict_out)
        print(f"\n판정 문서 저장 → {args.verdict_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
