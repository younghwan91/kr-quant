"""Walk-forward validation — folds, no-lookahead slicing, fold-consistency.

Signal-agnostic extraction from ``research/bo_validate.py`` (FOLDS, _wf_oos,
run_walkforward) and ``research/selective_walkforward.py`` (_fold_slices, _rdist,
fold-consistency; 두 원본 모두 추출과 함께 삭제돼 git 이력에만 있다).
Everything here takes GENERIC inputs: returns/entry-date arrays
and a ``simulate(params) -> (returns, entry_dates)`` callable. No research imports.

Two design rules encode the "no-lookahead is sacred" principle:

1. **Folds are a frozen default, not a per-experiment knob.** ``FOLDS`` is produced
   once by ``rolling_folds()`` and shared. Handing each experiment a mutable fold
   config invites fold-shopping — silently widening/shifting windows until the OOS
   number looks good, which is a lookahead violation. Regenerate with different
   arguments only for a *documented* reason, never per run.
2. **TRAIN-only fit, OOS-only eval.** ``walk_forward`` fits on ``[train_lo, train_hi)``
   via a caller-supplied ``fit`` callable and evaluates on ``[test_lo, test_hi)``.
   Each fold's ``train_hi <= test_lo``, so the fit never sees its own test window.
   Default TRAIN boundary: entry < 2022-01-01.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

Simulate = Callable[[dict], "tuple[np.ndarray, np.ndarray]"]


class Fold(NamedTuple):
    """하나의 walk-forward fold. TRAIN[lo,hi)에서 학습, TEST[lo,hi)에서 평가.

    NamedTuple이라 평범한 4-튜플과 == 비교된다(기존 FOLDS 튜플과 호환)."""

    train_lo: str
    train_hi: str
    test_lo: str
    test_hi: str


def rolling_folds(
    *,
    first_test_year: int = 2020,
    last_test_year: int = 2025,
    train_years: int = 3,
    test_years: int = 1,
    final_test_hi: str | None = "2027-01-01",
) -> "tuple[Fold, ...]":
    """롤링 fold 생성기: train ``train_years``년 → test ``test_years``년, test 연도 이동.

    기본 인자는 기존 연구 스크립트의 frozen 6-fold와 정확히 일치한다(회귀 고정).
    마지막 fold는 잔여(부분 연도) 데이터를 모두 담도록 test 상한을 ``final_test_hi``로
    연장한다 — 마지막 test 연도의 이후 데이터가 dangling partial fold로 새지 않게."""
    folds = []
    for y in range(first_test_year, last_test_year + 1):
        folds.append(
            Fold(f"{y - train_years}-01-01", f"{y}-01-01",
                 f"{y}-01-01", f"{y + test_years}-01-01")
        )
    if final_test_hi is not None and folds:
        f = folds[-1]
        folds[-1] = Fold(f.train_lo, f.train_hi, f.test_lo, final_test_hi)
    return tuple(folds)


# frozen 기본 fold 세트(fold-shopping 방지). 실험마다 새로 만들지 말 것.
FOLDS: "tuple[Fold, ...]" = rolling_folds()


def _expectancy(rets: np.ndarray) -> float:
    """유한수익의 건당 기대값(=평균). 비어있으면 NaN. 기본 fold 통계."""
    r = rets[np.isfinite(rets)] if len(rets) else rets
    return float(r.mean()) if len(r) else float("nan")


def entry_mask(edates: np.ndarray, lo=None, hi=None) -> np.ndarray:
    """[lo, hi) 진입일 마스크. lo/hi는 ISO 날짜 문자열(사전식 비교 = 시간 비교)."""
    m = np.ones(len(edates), bool)
    if lo is not None:
        m &= edates >= lo
    if hi is not None:
        m &= edates < hi
    return m


def slice_by_entry(rets: np.ndarray, edates: np.ndarray, lo=None, hi=None) -> np.ndarray:
    """진입일이 [lo, hi)인 수익만 반환. no-lookahead fold 슬라이스의 기본 연산."""
    return rets[entry_mask(edates, lo, hi)]


class FoldMask(NamedTuple):
    """한 fold의 최종 TRAIN/TEST 소속 마스크(트레이드 배열에 대한 bool).

    ``purge_embargo``의 반환형. ``rets[mask.train]`` / ``rets[mask.test]`` 로 슬라이스한다.
    기본(purge/embargo 미적용) 시 각각 ``entry_mask(edates, train_lo, train_hi)`` /
    ``entry_mask(edates, test_lo, test_hi)`` 와 정확히 동일하다(회귀 고정)."""

    train: np.ndarray
    test: np.ndarray


def resolve_exit_dates(entry_dates, *, exit_dates=None, max_hold=None) -> np.ndarray:
    """트레이드의 실현(청산)일을 ``datetime64[D]`` 로 정한다 — purge의 라벨 종료시점.

    실현시점을 두 방식으로 받는다(둘 중 하나만):
      - ``exit_dates``: 명시적 청산일 배열(우리 트레이드는 ``exit_date`` 컬럼을 가진다).
      - ``max_hold``: 보유 상한 → 청산 ≈ 진입 + ``max_hold`` **달력일**. 트레이딩일 상한을
        쓰는 호출자는 넉넉히(예: 주말 포함) 올린 값을 넣거나 명시적 ``exit_dates``를 쓴다.
        (달력일 근사는 트레이딩일보다 짧게 잡힐 수 있으니 상향 근사로 보수적으로 쓰라.)

    둘 다 없으면 청산=진입(보유 0 근사) → purge가 진입일 기준으로 퇴화(=미적용).
    둘 다 주면 ``ValueError``."""
    if exit_dates is not None and max_hold is not None:
        raise ValueError("exit_dates 와 max_hold 는 동시 지정 불가 — 하나만.")
    if exit_dates is not None:
        return np.asarray(exit_dates, dtype="datetime64[D]")
    entry = np.asarray(entry_dates, dtype="datetime64[D]")
    if max_hold is not None:
        if int(max_hold) < 0:
            raise ValueError("max_hold 는 음수 불가.")
        return entry + np.timedelta64(int(max_hold), "D")
    return entry


def purge_embargo(
    entry_dates,
    fold,
    *,
    exit_dates=None,
    max_hold=None,
    embargo_days: int = 0,
) -> FoldMask:
    """Purged K-fold + Embargo (López de Prado, *AFML* §7.4) — 보유기간 라벨 인접-누출 차단.

    frozen fold는 ``train_hi == test_lo`` 로 인접한다. 경계 직전 진입해 며칠 보유한
    스윙 트레이드는 진입일로는 TRAIN이지만 **실현은 TEST 창 안**이라 라벨이 겹친다
    (§4 최우선 공백 #1). 이 함수가 그 겹치는 라벨을 TRAIN에서 걷어낸다.

    우리 walk-forward는 TRAIN이 시간상 TEST **앞**에 온다(``train_hi <= test_lo``)이라
    AFML을 이 기하에 맞춰 구현한다:

      - **PURGE(라벨 겹침)**: 실현(청산)이 TEST 창으로 넘어가는 TRAIN 트레이드 제거 —
        즉 ``exit_date >= test_lo`` 이면 (진입이 TRAIN이라도) TRAIN에서 뺀다. AFML의
        "테스트창과 겹치는 라벨을 TRAIN에서 purge"를 진입-앞-테스트 기하로 특수화한 것.
      - **EMBARGO(직렬상관 완충)**: TEST 시작(``test_lo``) 직전 ``embargo_days`` 달력일에
        진입한 TRAIN 트레이드 제거 — 명시적 라벨 종료가 test_lo 전이어도 직렬상관으로
        새는 경계-인접 표본을 버린다. AFML의 embargo(테스트셋 **뒤** 완충을 이후 TRAIN에서
        purge)를 TRAIN-앞-TEST 기하의 대칭형(테스트셋 **앞** 완충)으로 옮긴 것.

    embargo는 **TRAIN에서만** 제거하고 TEST 평가창은 절대 줄이지 않는다(OOS 지표 불변).

    실현시점은 ``resolve_exit_dates`` 로 정한다 — 명시적 ``exit_dates`` 또는 ``max_hold``
    (달력일) 중 하나. 둘 다 없으면 청산=진입 → PURGE는 no-op.

    **기본값은 무개입**: ``exit_dates``/``max_hold`` 없고 ``embargo_days=0`` 이면 반환 마스크는
    ``entry_mask`` 소속과 **정확히 동일**하다(회귀 고정). purge/embargo는 opt-in.

    반환: ``FoldMask(train, test)`` — 트레이드 배열에 대한 bool 마스크."""
    if int(embargo_days) < 0:
        raise ValueError("embargo_days 는 음수 불가.")
    train = entry_mask(entry_dates, fold.train_lo, fold.train_hi)
    test = entry_mask(entry_dates, fold.test_lo, fold.test_hi)
    test_lo = np.datetime64(fold.test_lo, "D")

    # PURGE: 실현이 TEST 창으로 넘어가는(exit >= test_lo) TRAIN 트레이드 제거.
    exits = resolve_exit_dates(entry_dates, exit_dates=exit_dates, max_hold=max_hold)
    train = train & ~(exits >= test_lo)

    # EMBARGO: [test_lo - embargo_days, test_lo) 진입 TRAIN 제거(테스트셋 앞 완충).
    if int(embargo_days) > 0:
        entry_dt = np.asarray(entry_dates, dtype="datetime64[D]")
        embargo_lo = test_lo - np.timedelta64(int(embargo_days), "D")
        train = train & ~(entry_dt >= embargo_lo)

    return FoldMask(train, test)


def oos_fixed(
    folds,
    simulate: Simulate,
    params: dict,
    *,
    stat: Callable[[np.ndarray], float] = _expectancy,
) -> np.ndarray:
    """고정 params를 각 fold의 TEST 구간에서 평가(재최적화 없음) → OOS 지표 배열.

    ``research.bo_validate._wf_oos`` 일반화. params가 고정이라 sim은 한 번만 돌리고
    fold별로 슬라이스한다(결과 동일, 재실행 절약)."""
    rets, edates = simulate(params)
    return np.array([stat(slice_by_entry(rets, edates, f.test_lo, f.test_hi))
                     for f in folds])


def walk_forward(
    folds,
    simulate: Simulate,
    fit: Callable[[str, str], dict],
    *,
    stat: Callable[[np.ndarray], float] = _expectancy,
) -> list:
    """fold마다 TRAIN서 fit → TEST서 평가. no-lookahead 롤링 검증.

    ``fit(train_lo, train_hi) -> params`` 는 TRAIN 경계만 본다(예: optimization.mini_bo
    래퍼). optuna 의존은 fit 콜러블에 있어 이 모듈에는 없다. IS≫OOS 반복이면 과최적.

    반환: fold별 ``{"fold", "params", "is", "oos"}`` 레코드 리스트."""
    records = []
    for f in folds:
        params = fit(f.train_lo, f.train_hi)  # TRAIN-only fit
        rets, edates = simulate(params)
        records.append({
            "fold": f,
            "params": params,
            "is": stat(slice_by_entry(rets, edates, f.train_lo, f.train_hi)),
            "oos": stat(slice_by_entry(rets, edates, f.test_lo, f.test_hi)),
        })
    return records


def rdist(R: np.ndarray) -> dict:
    """개별표본 R-분포 요약: n·기대값R·≥3R빈도·손익비. R = 수익 ÷ 하드손절폭.

    ``selective_walkforward._rdist`` 이식. 복리 없음 — 개별 트레이드가 표본."""
    if len(R) == 0:
        return dict(n=0, expR=float("nan"), tail3=float("nan"), payoff=float("nan"))
    wins, losses = R[R > 0], R[R < 0]
    payoff = (wins.mean() / -losses.mean()) if len(wins) and len(losses) else float("nan")
    return dict(n=len(R), expR=float(R.mean()), tail3=float(np.mean(R >= 3.0)), payoff=payoff)


def fold_slices(
    entry: np.ndarray,
    feature: np.ndarray,
    value: np.ndarray,
    fold,
    frac: float,
    *,
    min_train: int = 40,
    exit_dates=None,
    max_hold=None,
    embargo_days: int = 0,
):
    """TRAIN에서 feature의 (1-frac) 분위 θ 학습 → TEST에 그대로 적용. look-ahead 없음.

    진입시점에 이미 아는 θ로 "잡을지 말지"를 정하므로 배포 가능. ``value``는 보통
    R-멀티플(수익÷손절폭). ``selective_walkforward._fold_slices`` 일반화(시그널 무관).

    **여기가 purge 가 필요한 자리다.** θ 를 TRAIN 에서 *학습*하므로, 경계 직전 진입해
    TEST 창 안에서 실현되는 트레이드가 TRAIN 에 남아 있으면 그 라벨이 θ 에 스며든다
    (AFML §7.4). ``exit_dates`` 또는 ``max_hold`` 를 주면 :func:`purge_embargo` 로
    그 트레이드를 TRAIN 에서 걷어낸다. 아무것도 안 주면 기존 동작 그대로다 —
    발표 수치가 그 경로에 고정돼 있어 opt-in 으로 둔다.

    TEST 슬라이스는 어떤 경우에도 줄이지 않는다(OOS 지표 불변).

    반환: ``(theta, base_values, selected_values)`` 또는 TRAIN 부족시 ``None``."""
    fin = np.isfinite(feature) & np.isfinite(value)
    mask = purge_embargo(entry, fold, exit_dates=exit_dates, max_hold=max_hold,
                         embargo_days=embargo_days)
    tr = mask.train & fin
    te = mask.test & fin
    if tr.sum() < min_train:
        return None
    theta = np.quantile(feature[tr], 1.0 - frac)  # TRAIN 분위 = 고정 임계값
    base = value[te]
    sel = value[te & (feature >= theta)]  # TEST에 θ 적용(진입시점 결정 가능)
    return theta, base, sel


def fold_consistency(
    entry: np.ndarray,
    feature: np.ndarray,
    value: np.ndarray,
    folds,
    frac: float,
    *,
    min_train: int = 40,
    min_sel: int = 15,
    exit_dates=None,
    max_hold=None,
    embargo_days: int = 0,
) -> dict:
    """여러 fold에서 "선별이 R-분포를 굽히나"를 재현성으로 판정(결정론 아님).

    fold마다 base(전부) vs 선별(feature≥θ) R-분포를 비교. 굽힘 = 선별 기대값R >
    베이스 기대값R. 유효폴드 중 몇 폴드에서 재현되나(부호검정 서술). 선별표본<min_sel
    이면 폴드 무효(희소성 정직 처리).

    ``exit_dates``/``max_hold``/``embargo_days`` 는 :func:`fold_slices` 로 그대로
    넘어간다(라벨 겹침 purge). 안 주면 기존 동작 그대로.

    반환: ``{"valid", "bent", "rows"}`` — rows는 fold별 상세 dict."""
    valid = bent = 0
    rows = []
    for f in folds:
        fs = fold_slices(entry, feature, value, f, frac, min_train=min_train,
                         exit_dates=exit_dates, max_hold=max_hold,
                         embargo_days=embargo_days)
        if fs is None:
            rows.append({"fold": f, "status": "train_short"})
            continue
        theta, base, sel = fs
        if len(sel) < min_sel:
            rows.append({"fold": f, "status": "sparse", "theta": float(theta),
                         "base": rdist(base)})
            continue
        valid += 1
        b, s = rdist(base), rdist(sel)
        up = s["expR"] > b["expR"]
        bent += int(up)
        rows.append({"fold": f, "status": "ok", "theta": float(theta),
                     "base": b, "sel": s, "bent": bool(up)})
    return {"valid": valid, "bent": bent, "rows": rows}
