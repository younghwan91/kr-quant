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

from ..features.fundamentals import code33_panel, earnings_yield_panel
from ..features.rs_rating import rs_rating_panel
from ..features.universe import CAP_BAND, smallmid_universe
from ..price_adjust import adjust_prices
from .minervini_sepa import sepa_entries, sepa_trades
from .pead import market_cap_panel
from .sepa_compare import benchmark_returns, book_returns, compare_arms

N_CONCENTRATED = 6      # arm A frozen concentration
N_DIVERSIFIED = 50      # A-diversified / B-shell (deployed over-diversification)
LARGE_CAP_RANK = (0, 100)   # B-shell universe: mega/large (the deployed inversion)


def _adv_panel(prices: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Trailing ``window``-day average trade value → long code/date/adv (as-of)."""
    tv = prices[["code", "date", "trade_value"]].copy()
    tv["trade_value"] = tv["trade_value"].abs()
    tv = tv.sort_values(["code", "date"])
    tv["adv"] = tv.groupby("code")["trade_value"].transform(
        lambda s: s.rolling(window, min_periods=window).mean())
    return tv.dropna(subset=["adv"])[["code", "date", "adv"]].reset_index(drop=True)


def build_panels(
    prices: pd.DataFrame,
    earnings: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    adjust: bool = True,
) -> dict:
    """Assemble every panel the arms need (split-adjusted, point-in-time).

    Args:
        prices: Long ``code``/``date``/``open``/``high``/``low``/``close``/
            ``trade_value``/``volume``.
        earnings: DART rows for :func:`code33_panel` (code, avail_date, netinc,
            netinc_prior, revenue, revenue_prior, op_income, op_income_prior).
        shares: ``code``/``date``/shares-outstanding for market cap.
        adjust: Apply corporate-action back-adjustment (both strategy and bench).
    """
    # Normalize date to string so every downstream to_datetime yields the same
    # resolution — DB columns arrive at mixed datetime64 units and break merge_asof.
    prices = prices.copy()
    prices["date"] = prices["date"].astype(str)
    shares = shares.copy()
    shares["date"] = shares["date"].astype(str)
    if adjust:
        prices = adjust_prices(prices)
    dates = sorted(prices["date"].astype(str).unique())
    cap = market_cap_panel(prices, shares)
    adv = _adv_panel(prices)
    annual = earnings[earnings["period"].astype(str).str.endswith("Q4")]
    ep = earnings_yield_panel(annual, cap)                             # code/date/ep (E/P)
    ep["pe"] = (1.0 / ep["ep"]).where(ep["ep"] > 0)                    # P/E for the sell rule
    return {
        "prices": prices,
        "cap": cap,
        # Absolute cap band, not rank — rank-within-a-liquidity-filtered-universe
        # silently lands on large/mega caps, not small-mid (see universe.py note).
        "smallmid": smallmid_universe(cap, adv, cap_band=CAP_BAND),
        "largecap": smallmid_universe(cap, adv, cap_rank=LARGE_CAP_RANK),  # B-shell
        "rs": rs_rating_panel(prices),
        "code33": code33_panel(earnings, dates),
        "pe": ep[["code", "date", "pe"]],
    }


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
    live run is a single call once the earnings backfill lands.
    """
    p = build_panels(prices, earnings, shares, adjust=adjust)
    px, rs, c33 = p["prices"], p["rs"], p["code33"]

    pe = p["pe"]
    ent_a = sepa_entries(px, p["smallmid"], rs, c33, use_vcp=True, use_code33=True)
    trades_a = sepa_trades(px, ent_a, pe_panel=pe)
    ent_avcp = sepa_entries(px, p["smallmid"], rs, c33, use_vcp=False, use_code33=True)
    ent_b = sepa_entries(px, p["largecap"], rs, c33, use_vcp=False, use_code33=False,
                         use_base_count=False, rs_min=0.0)

    arms = {
        # A / A-noVCP: frozen concentration sizing (6 names, 25/15, pilot).
        "A": book_returns(px, trades_a, n_slots=N_CONCENTRATED, sized=True),
        # A-diversified / B-shell: deployed-style diversified equal-weight (isolates concentration).
        "A-diversified": book_returns(px, trades_a, n_slots=N_DIVERSIFIED, sized=False),
        "A-noVCP": book_returns(px, sepa_trades(px, ent_avcp, pe_panel=pe), n_slots=N_CONCENTRATED, sized=True),
        "B-shell": book_returns(px, sepa_trades(px, ent_b), n_slots=N_DIVERSIFIED, sized=False),
        "C-bench": benchmark_returns(px, p["cap"]),
    }
    # Align every arm on the union of months so the paired bootstrap is well-defined.
    months = sorted(set().union(*[r.index for r in arms.values()]))
    idx = pd.Index(months, name="month")
    arms = {k: r.reindex(idx).fillna(0.0) for k, r in arms.items()}
    return compare_arms(arms, deployed="B-shell", benchmark="C-bench", **boot_kwargs)


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
    stop varies the hard stop in :func:`sepa_trades`.

    Returns:
        Long DataFrame ``concentration``/``stop``/``sharpe``/``cagr``/``n_trades`` —
        one row per (concentration, stop). NaN Sharpe means the gate produced no
        trades at that setting (not a failure — the faithful gates are very selective).
    """
    from .sepa_compare import _ann_sharpe, _cagr, book_returns

    p = build_panels(prices, earnings, shares, adjust=adjust)
    ent = sepa_entries(p["prices"], p["smallmid"], p["rs"], p["code33"],
                       use_vcp=use_vcp, use_code33=True)
    rows: list[dict] = []
    for stop in stops:
        trades = sepa_trades(p["prices"], ent, stop_pct=stop)
        for n in concentrations:
            r = book_returns(p["prices"], trades, n_slots=n, sized=True)
            rows.append({"concentration": n, "stop": stop,
                         "sharpe": _ann_sharpe(r.to_numpy(float)),
                         "cagr": _cagr(r.to_numpy(float)), "n_trades": len(trades)})
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
