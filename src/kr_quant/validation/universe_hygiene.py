"""생존편향(survivorship) 스멜테스트 — 유니버스가 PIT·상장폐지 포함인지 리포트.

가장 큰 숨은 인플레이터는 생존편향이다(GUARDRAILS §3): 실패·상장폐지 종목을
유니버스에서 빼면 수익이 부풀려진다. 검증 스택(walk-forward·fragility)은 넘겨받은
트레이드만 믿기 때문에, 애초에 유니버스가 살아남은 종목만 담고 있으면 아무 게이트도
그걸 잡지 못한다(GUARDRAILS §4 공백 2).

이 모듈은 code×date 유니버스/가격 패널에서 **생존편향 스멜**을 리포트한다:
얼마나 많은 종목이 패널 끝 전에 사라지는가(상장폐지 대리지표), 전 구간을 살아남은
종목의 비율(너무 높으면 스멜), first/last-seen 분포.

⚠️ **정직한 한계.** 이건 스멜테스트지 보증이 아니다. 진짜 방어는 (1) 시점정합(PIT)
유니버스 스냅샷과 (2) **상장폐지 종목의 폐지수익(delisting return)** 데이터다. 밑단
DB가 폐지수익을 갖고 있지 않으면, 여기서 "종목이 사라진다"는 것은 거래정지·상장폐지가
패널에 **흔적으로** 남아있다는 것만 말해줄 뿐, 그 종목의 폐지 시점 손익이 트레이드
표본에 올바르게 반영됐다는 보증은 되지 못한다. 그래서 아래 :func:`assert_point_in_time`
은 명백한 생존필터(다년 구간에서 단 한 종목도 안 사라짐)만 잡는 스멜 게이트다.

리포터-not-판정기(GUARDRAILS §8): PASS/FAIL bool 을 내지 않는다. 숫자만 리포트하고
사람이 읽는다. 유일한 예외는 명시적 strict 헬퍼 :func:`assert_point_in_time` 로,
"이건 스멜이다"라고 요청받았을 때만 RAISE 한다.

순수함수: (codes, dates, presence-or-close) in → dict out. DB 불요.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_presence(
    codes: list,
    dates: list,
    panel,
) -> np.ndarray:
    """(codes × dates) 불리언 존재행렬로 정규화.

    ``panel`` 은 다음 중 하나:
      - 존재행렬(bool/0-1) ndarray 또는 DataFrame — truthy = 존재.
      - 종가(close) 패널 ndarray 또는 DataFrame — 유한·양수 = 존재(NaN·0·음수 = 부재).

    DataFrame 이면 (codes, dates) 로 reindex 한다. ndarray 이면 shape 가
    ``(len(codes), len(dates))`` 와 일치해야 한다.
    """
    if isinstance(panel, pd.DataFrame):
        arr = panel.reindex(index=codes, columns=dates).to_numpy(dtype=float)
    else:
        arr = np.asarray(panel, dtype=float)
        if arr.shape != (len(codes), len(dates)):
            raise ValueError(
                f"panel shape {arr.shape} != (len(codes), len(dates)) "
                f"({len(codes)}, {len(dates)})"
            )
    # 존재 = 유한 & 양수. 존재행렬(0/1)이든 종가든 동일하게 동작(0/NaN = 부재).
    return np.isfinite(arr) & (arr > 0)


def survivorship_report(codes: list, dates: list, presence_matrix_or_close) -> dict:
    """code×date 패널의 생존편향 스멜을 리포트(판정 없음).

    Args:
        codes: 종목코드 시퀀스(행 라벨).
        dates: 정렬된 날짜 시퀀스(열 라벨, 오름차순 가정).
        presence_matrix_or_close: (codes × dates) 존재행렬 또는 종가 패널.
            ndarray 또는 DataFrame(:func:`_to_presence` 규칙). 종가면 NaN/≤0 = 부재.

    Returns:
        dict — 전부 리포트용 숫자(합격선 없음):
          - ``n_codes`` / ``n_dates`` — 패널 크기.
          - ``n_ever_present`` — 구간 내 한 번이라도 존재한 종목 수.
          - ``n_disappear_before_end`` — 마지막 날짜 이전에 자취를 감춘(last-seen <
            마지막) 종목 수. **상장폐지·거래정지의 대리지표**.
          - ``frac_disappear_before_end`` — 위 비율(0 이면 아무도 안 사라짐 = 생존필터 스멜).
          - ``n_full_span`` / ``frac_full_span`` — 첫 날짜와 마지막 날짜에 **둘 다** 존재한
            종목 수/비율. 이 비율이 과도하게 높으면 생존편향 스멜.
          - ``n_present_at_start`` — 첫 날짜에 존재한 종목 수.
          - ``survival_rate`` — 첫 날짜 존재 종목 중 마지막 날짜에도 존재한 비율
            (없으면 ``nan``). 이것이 1.0 에 가까우면 폐지가 지워졌다는 강한 스멜.
          - ``frac_first_seen_at_start`` — first-seen 이 첫 날짜인 종목 비율.
          - ``frac_last_seen_at_end`` — last-seen 이 마지막 날짜인 종목 비율.
          - ``first_seen_idx`` / ``last_seen_idx`` — 종목별 first/last-seen 날짜 인덱스
            요약 ``{"min","median","max"}`` (분포 확인용; 부재 종목은 제외).
    """
    codes = list(codes)
    dates = list(dates)
    n_codes, n_dates = len(codes), len(dates)
    pres = _to_presence(codes, dates, presence_matrix_or_close)

    ever = pres.any(axis=1)                       # 구간 내 한 번이라도 존재
    n_ever = int(ever.sum())

    # first/last-seen 인덱스(부재 종목은 -1 센티넬 → 이후 마스킹).
    first_seen = np.where(ever, pres.argmax(axis=1), -1)
    # 마지막 True 위치 = 뒤집어서 argmax.
    last_seen = np.where(ever, n_dates - 1 - pres[:, ::-1].argmax(axis=1), -1)

    at_start = pres[:, 0]
    at_end = pres[:, -1]
    n_at_start = int(at_start.sum())
    n_full_span = int((at_start & at_end).sum())

    # 마지막 날짜 이전에 사라진 종목(대리 상폐): 한 번이라도 존재했으나 마지막날 부재.
    disappeared = ever & ~at_end
    n_disappear = int(disappeared.sum())

    def _summ(v: np.ndarray) -> dict:
        v = v[ever]
        if v.size == 0:
            return {"min": None, "median": None, "max": None}
        return {"min": int(v.min()), "median": float(np.median(v)), "max": int(v.max())}

    return {
        "n_codes": n_codes,
        "n_dates": n_dates,
        "n_ever_present": n_ever,
        "n_disappear_before_end": n_disappear,
        "frac_disappear_before_end": (n_disappear / n_ever) if n_ever else float("nan"),
        "n_full_span": n_full_span,
        "frac_full_span": (n_full_span / n_ever) if n_ever else float("nan"),
        "n_present_at_start": n_at_start,
        "survival_rate": (int((at_start & at_end).sum()) / n_at_start) if n_at_start else float("nan"),
        "frac_first_seen_at_start": (int((first_seen == 0).sum()) / n_ever) if n_ever else float("nan"),
        "frac_last_seen_at_end": (int((last_seen == n_dates - 1).sum()) / n_ever) if n_ever else float("nan"),
        "first_seen_idx": _summ(first_seen),
        "last_seen_idx": _summ(last_seen),
    }


def assert_point_in_time(
    codes: list,
    dates: list,
    presence_matrix_or_close,
    *,
    min_disappear_frac: float = 0.01,
    min_span_dates: int = 250,
) -> dict:
    """명백한 생존필터면 RAISE 하는 strict 스멜 게이트(옵션·비-기본).

    다년(≥ ``min_span_dates`` 날짜) 구간에서 **단 한 종목도 (거의) 사라지지 않으면**
    유니버스가 생존필터된 것으로 보고 ``AssertionError`` 를 낸다. 짧은 구간에서는
    통과시킨다(사라질 시간 자체가 부족하므로 스멜을 주장할 수 없다).

    ⚠️ **정직한 한계.** 통과해도 PIT 를 보증하지 않는다. 이건 명백한 생존필터만
    잡는 스멜테스트다. 진짜 무생존편향은 (1) 시점정합 유니버스 스냅샷과 (2) **상장폐지
    종목의 폐지수익** 데이터로만 보장된다. 밑단 DB 에 폐지수익이 없으면, 종목이
    '사라지는' 흔적이 있더라도 폐지 손익이 트레이드에 반영됐다는 보증은 안 된다.
    그래서 이 게이트는 *부재*(자취를 감춤)만 확인하지 폐지수익 정합성은 확인 못 한다.

    Args:
        min_disappear_frac: 다년 구간에서 최소 이 비율만큼은 사라져야 통과.
            (기본 1% — 정상 시장에서 다년간 상폐·거래정지가 최소 이 정도는 난다.)
        min_span_dates: 이 날짜 수 미만이면 검사 건너뜀(구간이 짧아 스멜 주장 불가).

    Returns:
        :func:`survivorship_report` 의 dict(리포트 겸용).

    Raises:
        AssertionError: 다년 구간인데 ``frac_disappear_before_end`` 가 문턱 미만.
    """
    rep = survivorship_report(codes, dates, presence_matrix_or_close)
    if rep["n_dates"] < min_span_dates:
        return rep  # 구간이 짧다 — 스멜을 주장하지 않는다.
    frac = rep["frac_disappear_before_end"]
    if not np.isfinite(frac) or frac < min_disappear_frac:
        raise AssertionError(
            "생존편향 스멜: {n_dates}개 날짜(다년) 구간에서 "
            "종목의 {pct:.2%}만 패널 끝 전에 사라짐 (문턱 {thr:.2%}). "
            "PIT 유니버스 + 상장폐지 폐지수익이 빠졌을 가능성 — "
            "살아남은 종목만 담긴 유니버스는 수익을 부풀린다. "
            "이건 스멜테스트지 보증이 아님(폐지수익 정합성은 별도 확인 필요).".format(
                n_dates=rep["n_dates"],
                pct=(0.0 if not np.isfinite(frac) else frac),
                thr=min_disappear_frac,
            )
        )
    return rep
