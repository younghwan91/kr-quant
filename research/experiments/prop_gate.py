#!/usr/bin/env python
"""Step 0 — 프랍 스윙 셋업을 **동일한 잣대**로 판정하는 공통 게이트 하버스 (REPORTER).

세 프랍 셋업(Minervini 돌파 / 풀백 / PEAD 집중)을 apples-to-apples 로 재려면
개별 트레이드 (진입일, 수익) → 동일 배터리 를 한 곳에서 돌려야 한다. 이 파일이
그 배터리다. **판정기(verdict machine)가 아니라 리포터**다 (Principle 5): PASS/FAIL
bool 도, "monster<50%" 같은 하드코딩 임계값도 두지 않는다. 숫자·분포만 보고하고
배포 여부는 사람이 읽고 판단한다.

배터리(순서대로):
  1. 슬리피지 스윕 — 비용별 OOS(진입≥train_hi) 기대값 R + 폴드 일관성 → 엣지가 죽는 비용.
  2. 폴드 재현성 (frozen FOLDS) — 폴드별 건당 기대값 R, 양수/유효 카운트.
     **R2**: test 창이 train_hi 이전인 폴드(2020·2021)는 INSIDE-TRAIN 으로 명시 표기 —
     raw k/6 과 clean-OOS k/N(test_lo≥train_hi) 을 둘 다 보고한다. 6클린폴드 주장 금지.
  3. 분포 모양(dist_shape) + 취약성(fragility_report: monster top-5/20, 최장연패, 꼬리제거).
  4. **미접촉 최종창** (R1) — untouched_lo≤진입<untouched_hi 트레이드의 기대값 R·n 을
     walk-forward 폴드와 **별개로** 보고(탐색 중 절대 안 본 held-out 확인 슬라이스).
  5. gate_report(oos_R, fold_expectancies, entry, cost_curve) 호출 → 배포신호 dict.

원칙(비협상):
  - R-멀티플 = 트레이드수익 ÷ 하드손절폭(stop). 개별 트레이드가 표본. 복리·포트프레이밍 없음.
  - no-lookahead: TRAIN = 진입 < train_hi(=2022-01-01), OOS = 그 이후. FOLDS 는 frozen.
  - 비용은 각 트레이드 수익에서 상수 차감(선택·손절에 영향 없다고 가정) — gross 1회 → 사후 차감.

라이브러리 경계: kr_quant.validation.* · kr_quant.diagnostics.* 만 import 한다.
research/ 로부터 아무것도 import 하지 않고 src/kr_quant 를 수정하지 않는다.

실행(합성 스모크): uv run python research/experiments/prop_gate.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kr_quant.diagnostics.fragility import fragility_report, monster_share
from kr_quant.diagnostics.trials import count_trials, record_trial
from kr_quant.diagnostics.gate_report import gate_report
from kr_quant.diagnostics.r_distribution import dist_shape, r_multiples
from kr_quant.validation.walkforward import FOLDS, _expectancy, entry_mask, slice_by_entry

COSTS = (0.0046, 0.008, 0.010, 0.015)   # 왕복비용 스윕: 46 / 80 / 100 / 150 bp
TRAIN_HI = "2022-01-01"                  # no-lookahead 경계 (진입 < 이 날짜 = TRAIN)
UNTOUCHED_LO = "2025-07-01"              # R1 held-out 최종창 하한 (탐색 중 미접촉)
UNTOUCHED_HI = "2026-07-01"              # R1 held-out 최종창 상한


def _ledger_target(label: str, log_dir: str | None) -> tuple[str, "str | None"]:
    """(원장 label, 상위 디렉터리). ``log_dir`` 을 주면 그 경로의 마지막 조각이 label 이 된다.

    ``research/logs/pullback_prop`` → ``("pullback_prop", "research/logs")`` — 원장이
    VERDICT.md 와 같은 디렉터리에 놓이게 하기 위한 변환이다.

    **결과적으로 원장의 키는 label 이 아니라 VERDICT 디렉터리다.** 의도한 의미다 —
    시행 수 N 은 "이 판정문이 근거로 삼은 config 를 몇 개나 시도했나"이지 "이 파일이
    몇 번 돌았나"가 아니다. 그래서 한 알파를 여러 러너가 나눠 재는 경우(pead_gate 는
    top_n=40 분산형, pead_concentrated_gate 는 top_n=3 집중형, 판정은 하나로 통합)
    두 config 가 같은 원장에 쌓여 N=2 가 된다.
    """
    if log_dir is None:
        return label, None
    p = Path(log_dir)
    return p.name, str(p.parent)


def _fold_rows(entry: np.ndarray, R: np.ndarray, folds, train_hi: str) -> list[dict]:
    """폴드별 TEST 슬라이스 통계(재최적화 없음, 고정 R). R2 의 inside-train 플래그 부착.

    각 dict: test_window, n, expectancy_R, positive(bool), inside_train(bool).
    inside_train = fold.test_lo < train_hi → 이 폴드는 clean OOS 가 아니다(R2).
    """
    rows: list[dict] = []
    for f in folds:
        sl, sh = f.test_lo, f.test_hi
        r = slice_by_entry(R, entry, sl, sh)
        r = r[np.isfinite(r)]
        rows.append({
            "test_window": f"{sl[:7]}~{sh[:7]}",
            "test_lo": sl,
            "n": int(len(r)),
            "expectancy_R": _expectancy(r),
            "positive": bool(len(r) and r.mean() > 0),
            "inside_train": bool(sl < train_hi),
        })
    return rows


def _consistency(rows: list[dict], *, clean_only: bool, min_n: int = 1) -> tuple[int, int]:
    """폴드 rows → (양수 폴드, 유효 폴드). clean_only 면 inside-train 폴드 제외(R2).

    유효 = n>=min_n 인 폴드(희소 폴드는 세지 않음). 판정 아님 — 카운트 리포트.
    """
    pos = valid = 0
    for row in rows:
        if clean_only and row["inside_train"]:
            continue
        if row["n"] < min_n:
            continue
        valid += 1
        pos += int(row["positive"])
    return pos, valid


def prop_gate(
    entry_dates,
    returns,
    stop: float,
    *,
    label: str = "setup",
    costs: tuple = COSTS,
    folds=FOLDS,
    train_hi: str = TRAIN_HI,
    untouched_lo: str = UNTOUCHED_LO,
    untouched_hi: str = UNTOUCHED_HI,
    config: dict | None = None,
    log_dir: str | None = None,
    n_trials: int | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """프랍 스윙 셋업의 개별 트레이드를 STRICT 배터리에 태우고 신호를 dict 로 보고한다.

    **이 함수는 REPORTER 다 (Principle 5: reporter, not verdict machine).** bool
    PASS/FAIL 키를 담지 않으며 "monster<50%"·"5/6폴드 합격" 같은 임계값을 하드코딩하지
    않는다. 확률적(stochastic) 데이터에 결정론적 판정을 박는 것 자체가 과최적이기
    때문이다. 반환된 숫자·분포를 읽고 배포를 결정하는 것은 연구자의 몫이다.

    Args:
        entry_dates: 트레이드별 진입일 ISO 문자열 배열(len == returns).
        returns: 트레이드별 gross 수익(소수, 예: 0.12=+12%). R 아님 — 여기서 ÷stop.
        stop: 진입시 하드손절 폭(양수 소수). R 정규화 분모.
        label: 로깅용 이름.
        costs: 왕복비용 스윕(오름차순). 각 비용은 수익에서 상수 차감.
        folds: 워크포워드 fold 세트(기본 frozen FOLDS). fold-shopping 금지.
        train_hi: no-lookahead 경계. 진입 < train_hi = TRAIN(=OOS 아님). R2 clean 판별에도 사용.
        untouched_lo/untouched_hi: R1 held-out 최종 확인창 [lo, hi).
        config: 이 호출의 사전등록 config(옵션). 주면 **다중검정 원장에 자동 기록**되고
            ``n_trials`` 를 원장에서 읽는다 — 연구자가 시행 수를 손으로 세지 않는다.
            같은 config 재실행은 시행으로 안 센다(config 지문 중복 제거).
        log_dir: 원장을 둘 디렉터리(예: ``"research/logs/pullback_prop"``). 생략하면
            ``research/logs/<label>`` — 다만 label 과 로그 디렉터리가 다른 게이트가 있어
            (pullback → pullback_prop) 그런 경우 반드시 지정해야 한다. **원장은 그 알파의
            VERDICT.md 옆에 있어야 한다** — 따로 놀면 둘이 어긋나도 아무도 모른다.
        n_trials: 시행 수를 직접 지정(옵션). 주면 원장 대신 이 값을 쓴다 — 원장 이전에
            돌린 시행을 소급 반영하는 등 예외적 용도. 주어지면 gate_report 가 Deflated
            Sharpe·t-haircut 로 선택편향을 깎아 보고한다(§11). 둘 다 None 이면 생략.
        n_boot, seed: gate_report 부트스트랩 파라미터.
        verbose: True 면 사람이 읽는 요약을 print(판정줄 없음). 음성대조는 False.

    Returns:
        dict — label, stop, n_total, entry_range, primary_cost, cost_sweep(1),
        cost_edge_dies, folds(2: rows·raw/clean 카운트), distribution(3), fragility(3),
        untouched(4), gate_report(5). **bool 판정 키는 없다.**
    """
    if config is not None:
        # 게이트가 직접 적는다 — record_trial 을 연구자가 부를 일이 없게(§0 "부탁 금지").
        led_label, led_dir = _ledger_target(label, log_dir)
        record_trial(led_label, config, logs_dir=led_dir)
        if n_trials is None:
            n_trials = count_trials(led_label, logs_dir=led_dir)

    entry = np.asarray(entry_dates)
    ret = np.asarray(returns, float)
    if len(entry) != len(ret):
        raise ValueError(f"entry_dates({len(entry)}) 와 returns({len(ret)}) 길이 불일치")

    primary_cost = costs[0]  # 상세 섹션(2·3·4·5)의 기준 비용 = 가장 현실적인 최저 비용
    fin_all = np.isfinite(ret)
    n_total = int(fin_all.sum())
    edates_fin = entry[fin_all]
    entry_range = (str(min(edates_fin))[:10], str(max(edates_fin))[:10]) if n_total else (None, None)

    # === (1) 슬리피지 스윕 — 비용별 OOS 기대값 R + 폴드 일관성 ===
    oos_mask = entry_mask(entry, lo=train_hi)
    cost_sweep: list[dict] = []
    cost_curve: dict[float, float] = {}
    for c in costs:
        R_c = (ret - c) / stop                      # 정렬 유지(진입일과 1:1) → 폴드 슬라이스 가능
        oos_R = r_multiples(ret[oos_mask] - c, stop)  # OOS 집계는 r_multiples(비유한 제거)
        oos_exp = _expectancy(oos_R)
        rows_c = _fold_rows(entry, R_c, folds, train_hi)
        raw_pos, raw_val = _consistency(rows_c, clean_only=False)
        clean_pos, clean_val = _consistency(rows_c, clean_only=True)
        cost_sweep.append({
            "cost": float(c),
            "oos_n": int(len(oos_R)),
            "oos_expectancy_R": oos_exp,
            "raw_fold_positive": raw_pos, "raw_fold_valid": raw_val,
            "clean_fold_positive": clean_pos, "clean_fold_valid": clean_val,
            "fold_expectancies": [row["expectancy_R"] for row in rows_c],
        })
        cost_curve[float(c)] = oos_exp

    # === 기준 비용에서의 정렬된 R (섹션 2·3·4·5 공용) ===
    R_primary = (ret - primary_cost) / stop
    prim_rows = _fold_rows(entry, R_primary, folds, train_hi)
    raw_pos, raw_val = _consistency(prim_rows, clean_only=False)
    clean_pos, clean_val = _consistency(prim_rows, clean_only=True)

    # === (2) 폴드 재현성 — R2: raw k/6 과 clean-OOS k/N 둘 다 ===
    folds_block = {
        "primary_cost": float(primary_cost),
        "rows": prim_rows,
        "raw_positive": raw_pos, "raw_valid": raw_val,
        "clean_oos_positive": clean_pos, "clean_oos_valid": clean_val,
        "inside_train_windows": [r["test_window"] for r in prim_rows if r["inside_train"]],
    }

    # === (3) 분포 모양 + 취약성 (기준 비용, OOS 트레이드) ===
    oos_R_prim = r_multiples(ret[oos_mask] - primary_cost, stop)
    oos_entry_prim = entry[oos_mask & fin_all]      # 시간순 연패용(finite 정렬)
    distribution = dist_shape(oos_R_prim)
    fragility = fragility_report(oos_R_prim, entry=oos_entry_prim,
                                 monster_k=5, tail_k=20)
    # monster top-5 & top-20 을 명시(fragility["monster_share"] 는 monster_k=5).
    fragility["monster_share_top5"] = monster_share(oos_R_prim, k=5)
    fragility["monster_share_top20"] = monster_share(oos_R_prim, k=20)

    # === (4) 미접촉 최종창 (R1) — 폴드와 별개로 보고 ===
    unt_mask = entry_mask(entry, lo=untouched_lo, hi=untouched_hi)
    unt_R = r_multiples(ret[unt_mask] - primary_cost, stop)
    untouched = {
        "lo": untouched_lo, "hi": untouched_hi,
        "n": int(len(unt_R)),
        "expectancy_R": _expectancy(unt_R),
        "dist": dist_shape(unt_R),
    }

    # === (5) gate_report — 배포신호 REPORTER ===
    gr = gate_report(
        oos_R_prim,
        fold_expectancies=[r["expectancy_R"] for r in prim_rows],
        entry=oos_entry_prim,
        monster_k=5,
        cost_curve=cost_curve,
        n_trials=n_trials,
        n_boot=n_boot,
        seed=seed,
    )

    report = {
        "label": label,
        "stop": float(stop),
        "n_total": n_total,
        "entry_range": entry_range,
        "primary_cost": float(primary_cost),
        "cost_sweep": cost_sweep,
        "cost_edge_dies": gr.get("cost_edge_dies"),
        "folds": folds_block,
        "distribution": distribution,
        "fragility": fragility,
        "untouched": untouched,
        "gate_report": gr,
    }
    if verbose:
        _print_summary(report)
    return report


def _print_summary(rep: dict) -> None:
    """사람이 읽는 요약 — 숫자만, **판정줄 없음** (Principle 5)."""
    print(f"\n{'=' * 72}")
    print(f"  prop_gate: {rep['label']}   stop={rep['stop']:.0%}   "
          f"n={rep['n_total']}   진입 {rep['entry_range'][0]}~{rep['entry_range'][1]}")
    print("  (리포터 — PASS/FAIL·하드코딩 임계값 없음. 아래 숫자를 읽고 사람이 판단)")
    print("=" * 72)

    print("\n[1] 슬리피지 스윕 — 비용별 OOS(진입≥train_hi) 기대값 R + 폴드일관")
    print(f"  {'cost':>6} | {'oos_n':>6} {'oos_expR':>9} | "
          f"{'raw k/6':>8} {'clean k/N':>10}")
    for cs in rep["cost_sweep"]:
        bp = int(round(cs["cost"] * 1e4))
        raw_k = f"{cs['raw_fold_positive']}/{cs['raw_fold_valid']}"
        clean_k = f"{cs['clean_fold_positive']}/{cs['clean_fold_valid']}"
        print(f"  {bp:>4}bp | {cs['oos_n']:>6} {cs['oos_expectancy_R']:>+9.3f} | "
              f"{raw_k:>8} {clean_k:>10}")
    edies = rep["cost_edge_dies"]
    print(f"  → 엣지 사망 비용(OOS expR 첫 ≤0): "
          f"{'전 구간 생존' if edies is None else f'{int(round(edies * 1e4))}bp'}")

    fb = rep["folds"]
    print(f"\n[2] 폴드 재현성 (기준 {int(round(fb['primary_cost'] * 1e4))}bp, 재최적화 없음)")
    print(f"  {'TEST창':>16} {'n':>5} {'expR':>8} {'양수?':>5}  구분")
    for row in fb["rows"]:
        tag = "INSIDE-TRAIN" if row["inside_train"] else "clean OOS"
        mark = "●" if row["positive"] else "○"
        print(f"  {row['test_window']:>16} {row['n']:>5} {row['expectancy_R']:>+8.3f} "
              f"{mark:>5}  {tag}")
    print(f"  → raw {fb['raw_positive']}/{fb['raw_valid']}  |  "
          f"clean-OOS {fb['clean_oos_positive']}/{fb['clean_oos_valid']}  "
          f"(R2: 6클린폴드 주장 금지 — inside-train {fb['inside_train_windows']} 제외)")

    d = rep["distribution"]
    fr = rep["fragility"]
    print("\n[3] 분포 모양 + 취약성 (기준 비용, OOS 트레이드)")
    print(f"  n={d['n']}  expR={d['expectancy_R']:+.3f}  승률={d['win_rate']:.0%}  "
          f"손익비={d['payoff']:.2f}  왜도={d['skew']:+.2f}  왼꼬리={d['left_tail_share']:.0%}")
    tr = fr["tail_removal"]
    print(f"  monster top5={fr['monster_share_top5']:.0%}  top20={fr['monster_share_top20']:.0%}"
          f"  최장연패={fr['max_loss_streak']}  중앙값R={fr['median_trade']:+.3f}")
    print(f"  꼬리제거(상위20): 기대값 {tr['expectancy_full']:+.3f} → {tr['expectancy_ex']:+.3f} "
          f"({'생존' if tr['expectancy_ex'] > 0 else '붕괴'})")

    u = rep["untouched"]
    print(f"\n[4] 미접촉 최종창 (R1 held-out) [{u['lo']}~{u['hi']}) — 폴드와 별개")
    print(f"  n={u['n']}  기대값R={u['expectancy_R']:+.3f}  "
          f"승률={u['dist']['win_rate']:.0%}  왜도={u['dist']['skew']:+.2f}")

    gr = rep["gate_report"]
    ci = gr["expectancy_ci"]
    print("\n[5] gate_report 배포신호 (REPORTER)")
    print(f"  OOS n={gr['n']}  기대값R={gr['expectancy_R']:+.3f}  "
          f"95%CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]")
    fc = gr["fold_consistency"]
    print(f"  폴드 {fc['n_positive']}/{fc['n_folds']} 양수  monster={gr['monster_share']:.0%}  "
          f"최장연패={gr['max_loss_streak']}")
    df = gr.get("deflation")
    if df is not None:
        th = df["t_haircut"]
        print(f"  [다중검정 보정 — 시도 config N={df['n_trials']}] (선택편향 깎기, 판정 아님)")
        print(f"    건당 Sharpe={df['observed_sharpe']:+.3f}  "
              f"E[maxSharpe|H0]={df['expected_max_sharpe_h0']:.3f}  "
              f"deflated={df['deflated_sharpe']:+.3f}")
        print(f"    P(Sharpe>0)={df['prob_sharpe_gt0']:.1%}  "
              f"P(Sharpe>SR0)={df['prob_deflated_sharpe']:.1%}  "
              f"t-haircut ×{th['haircut_multiple']:.2f} (문턱 t {th['base_hurdle_t']:.2f}→{th['adjusted_hurdle_t']:.2f})")


def random_entry_control(
    all_entry_dates,
    returns_pool,
    stop: float,
    *,
    n_per_draw: int,
    n_draws: int = 200,
    seed: int = 0,
    costs: tuple = COSTS,
    train_hi: str = TRAIN_HI,
    untouched_lo: str = UNTOUCHED_LO,
    untouched_hi: str = UNTOUCHED_HI,
    verbose: bool = True,
) -> dict:
    """랜덤 음성대조 (R1) — 게이트의 **자체 위양성률**을 보정한다.

    한 셋업의 marginal 분포/타이밍을 흉내낸 무작위 트레이드를 게이트에 태운다:
    진입일은 ``all_entry_dates`` 에서, 수익은 ``returns_pool`` 에서 **독립적으로**
    복원추출해 무작위 페어링한다(진입→수익의 실제 관계를 파괴). 이러면 진입일의
    시간분포와 수익의 marginal 분포는 보존되지만 신호는 없다. 이 draw 들이 얼마나
    자주 "5/6 폴드" 같은 바를 넘는지 = 게이트가 순수 노이즈를 통과시키는 빈도.

    무작위가 clean-OOS 폴드 바를 실제 셋업 못지않게 자주 넘으면 바가 너무 느슨하다.

    Returns:
        dict — n_draws·n_per_draw, draw별 raw/clean 폴드양수 카운트·OOS 기대값 리스트,
        raw≥5 도달 비율, clean 전폴드 도달 비율, 분위수. (판정 없음 — 보정 숫자.)
    """
    all_entry_dates = np.asarray(all_entry_dates)
    pool = np.asarray(returns_pool, float)
    pool = pool[np.isfinite(pool)]
    rng = np.random.default_rng(seed)

    raw_pos_list: list[int] = []
    clean_pos_list: list[int] = []
    clean_val_list: list[int] = []
    oos_exp_list: list[float] = []
    for _ in range(n_draws):
        ed = rng.choice(all_entry_dates, size=n_per_draw, replace=True)
        rr = rng.choice(pool, size=n_per_draw, replace=True)  # 진입과 독립 → 신호 파괴
        rep = prop_gate(ed, rr, stop, label="rand", costs=costs, train_hi=train_hi,
                        untouched_lo=untouched_lo, untouched_hi=untouched_hi,
                        n_boot=200, verbose=False)
        fb = rep["folds"]
        raw_pos_list.append(fb["raw_positive"])
        clean_pos_list.append(fb["clean_oos_positive"])
        clean_val_list.append(fb["clean_oos_valid"])
        oos_exp_list.append(rep["cost_sweep"][0]["oos_expectancy_R"])

    raw_arr = np.array(raw_pos_list)
    clean_arr = np.array(clean_pos_list)
    clean_val = int(np.median(clean_val_list)) if clean_val_list else 0
    exp_arr = np.array(oos_exp_list, float)
    exp_fin = exp_arr[np.isfinite(exp_arr)]
    out = {
        "n_draws": n_draws, "n_per_draw": n_per_draw,
        "raw_fold_positive": raw_pos_list,
        "clean_fold_positive": clean_pos_list,
        "clean_fold_valid_median": clean_val,
        "oos_expectancy_R": oos_exp_list,
        "raw_ge5_frac": float((raw_arr >= 5).mean()) if len(raw_arr) else float("nan"),
        "clean_all_positive_frac": (
            float((clean_arr >= clean_val).mean()) if clean_val and len(clean_arr) else float("nan")),
        "oos_expectancy_R_mean": float(exp_fin.mean()) if len(exp_fin) else float("nan"),
        "oos_expectancy_R_p95": float(np.percentile(exp_fin, 95)) if len(exp_fin) else float("nan"),
    }
    if verbose:
        print(f"\n{'=' * 72}\n  랜덤 음성대조 — {n_draws} draws × {n_per_draw} trades "
              f"(게이트 위양성률 보정)\n{'=' * 72}")
        print(f"  raw ≥5/6 폴드 도달: {out['raw_ge5_frac']:.1%} of draws")
        print(f"  clean-OOS 전폴드(≥{clean_val}/{clean_val}) 양수 도달: "
              f"{out['clean_all_positive_frac']:.1%} of draws")
        print(f"  OOS 기대값R: 평균 {out['oos_expectancy_R_mean']:+.3f}  "
              f"p95 {out['oos_expectancy_R_p95']:+.3f}")
        print("  → 무작위가 이 바를 자주 넘으면 게이트가 너무 느슨하다(사람이 판단).")
    return out


def _synthetic_setup(seed: int = 0, edge: float = 0.0):
    """스모크·자체검증용 합성 (entry_dates, returns). edge>0 이면 양(+)의 진짜 엣지."""
    rng = np.random.default_rng(seed)
    n = 600
    years = rng.integers(2018, 2027, size=n)
    months = rng.integers(1, 13, size=n)
    days = rng.integers(1, 29, size=n)
    entry = np.array([f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)])
    # 손절 -10% 왼꼬리 절단 + edge 만큼 양(+) 쉬프트된 오른꼬리
    ret = np.where(rng.random(n) < 0.45, -0.10, rng.exponential(0.08, n) + edge)
    return entry, ret


def main() -> int:
    """DB 불요 합성 스모크 — 하버스가 end-to-end 로 도는지 확인(엣지 있는 셋업 + 음성대조)."""
    print("=== prop_gate 합성 스모크 (DB 불요) ===")
    entry, ret = _synthetic_setup(seed=1, edge=0.02)
    rep = prop_gate(entry, ret, stop=0.10, label="synthetic_edge")
    print(f"\n  [반환 dict 키] {sorted(rep)}")
    random_entry_control(entry, ret, stop=0.10, n_per_draw=len(ret), n_draws=50, seed=7)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
