"""ML-강화 PEAD 신호 (GOAL 루프 76-85).

## 핵심 발견 — ML은 매매 목표로 학습해야 한다

LightGBM으로 다중 피처(실적 YoY + 모멘텀 + 수급 + 사이즈)를 결합하면 횡단면 rank-IC가
PEAD 단일 YoY 대비 2배(+0.029→+0.063)로 오른다. **그러나 IC를 목표로 학습하면 그 예측력이
사이즈·모멘텀역전 팩터로 새서 실제 매매(집중 롱)엔 무익하다** (top-15 롱 Sharpe 0.44→0.04,
롱숏도 음수). IC는 거래하지 않는 분포 중간의 순위 상관이기 때문.

**해법(사용자 통찰: "예측력이 올랐다면 활용을 못한 것"):** 학습 목표를 IC가 아니라 **매매 목표
(top-20% 승자 분류)**로 두면 예측력이 tradeable 승자에 실려 복리 수익으로 추출된다:

| top-15 롱, 60일 비중첩, 비용 0.7% | CAGR | Sharpe |
|---|---|---|
| PEAD(YoY 룰) | +35.0% | 1.13 |
| ML-순위IC 학습 | +31.1% | 0.98 |
| **ML-top20% 분류** | **+40.1%** | 1.11 |

집중 롱온리 ML-PEAD는 CAGR +36%/DD −49%. **마켓뉴트럴 excess 슬리브(Sharpe 1.15)로**
멀티-알파 북(IDX/ML-PEAD/미너비니)에 넣으면 북 Sharpe 1.06→1.57, CAGR +19%→+29%, DD −20%.
상세: research/MULTI_ALPHA.md.

## 재현

컨테이너에서 피처행렬을 덤프(daily_bars 분할조정 + dart earnings 병합) 후 mlenv(LightGBM)에서
walk-forward. 데이터 누수 방지: 매년 과거만 학습해 그 해를 예측(expanding window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATS = ["yoy", "age", "r20", "r60", "r120", "accum", "logmc"]


def walk_forward_ml_score(df: pd.DataFrame, *, target: str = "top20") -> np.ndarray:
    """연도별 expanding-window walk-forward로 ML 점수 산출 (룩어헤드 없음).

    Args:
        df: 컬럼 ``date``(YYYY-MM-DD), ``code``, FEATS, ``fwd``(H일 forward 수익).
        target: ``"top20"``(권장, 매매목표 분류) 또는 ``"rank"``(IC목표, 비권장) 또는
            ``"fwd"``(forward 수익 회귀).

    Returns:
        len(df) 길이의 ML 점수 배열(학습 이전 연도는 NaN).
    """
    import lightgbm as lgb

    d = df.copy()
    d["yr"] = d["date"].str[:4].astype(int)
    if target == "top20":
        y = (d.groupby("date")["fwd"].rank(pct=True) > 0.8).astype(int)
        Model, kw = lgb.LGBMClassifier, {}
    elif target == "rank":
        y = d.groupby("date")["fwd"].rank(pct=True)
        Model, kw = lgb.LGBMRegressor, {}
    else:
        y = d["fwd"]
        Model, kw = lgb.LGBMRegressor, {}
    d["_y"] = y
    pred = np.full(len(d), np.nan)
    for ty in sorted(d["yr"].unique()):
        tr = d[d["yr"] < ty]
        te = d[d["yr"] == ty]
        if len(tr) < 3000 or len(te) < 50:
            continue
        med = tr[FEATS].median()
        m = Model(n_estimators=300, num_leaves=15, learning_rate=0.03,
                  min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                  verbose=-1, **kw)
        m.fit(tr[FEATS].fillna(med), tr["_y"])
        Xte = te[FEATS].fillna(med)
        pred[te.index] = (m.predict_proba(Xte)[:, 1] if target == "top20"
                          else m.predict(Xte))
    return pred


def ml_pead_excess(df: pd.DataFrame, *, top_n: int = 15) -> pd.Series:
    """ML 점수 상위 N종목의 유니버스 대비 excess(마켓뉴트럴 슬리브용) 시계열."""
    d = df.copy()
    d["ml"] = walk_forward_ml_score(d, target="top20")
    d = d.dropna(subset=["ml"])
    rows = []
    for dt, g in d.groupby("date"):
        if len(g) < 40:
            continue
        rows.append((dt, g.nlargest(top_n, "ml")["fwd"].mean() - g["fwd"].mean()))
    s = pd.DataFrame(rows, columns=["date", "ex"]).set_index("date")["ex"]
    s.index = pd.to_datetime(s.index)
    return s
