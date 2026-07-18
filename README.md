# kr-quant

[![CI](https://github.com/younghwan91/kr-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/younghwan91/kr-quant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

**한국 주식(KOSPI·KOSDAQ)에 대한 반증 중심(falsification-first) 알파 리서치 프레임워크.**

이 저장소의 목적은 "돈 버는 전략"의 나열이 아니라, **가설을 엄격히 기각하는 검증 절차**를 코드로 강제하는 데 있다. 백테스트는 거의 언제나 우상향 곡선을 만들어낸다. 문제는 그 대부분이 과거 표본에 맞춰 깎아낸 과최적(overfitting)이라는 점이다. 따라서 이 프로젝트는 확증(confirmation)이 아니라 **반증**을 1차 목표로 삼고, 반증에 살아남은 소수의 알파만을 "검증됨"으로 인정하며, 기각된 가설은 근거와 함께 **정직한 부정 결과(negative result)**로 공개한다.

- **검증된 알파:** PEAD(실적 서프라이즈 후 드리프트)가 단독 robust한 유일한 수익원이며, 무상관 슬리브와 결합한 멀티-알파 북이 비용·분할조정 후 시총가중 인덱스를 파레토 지배한다(§3).
- **기각된 알파:** 개미 투매 역발상, 미너비니식 돌파, 눌림목 평균회귀, 실적 서프라이즈 집중 스윙 — 개별 트레이드 관점에서 재현되지 않아 전부 기각(§4).
- **방법론:** 개별 트레이드 분포·walk-forward 재현성·랜덤 음성대조·손 안 댄 최종 구간·비용 스트레스·취약성 진단을 하나의 게이트로 강제([`docs/GUARDRAILS.md`](docs/GUARDRAILS.md), §2).

> **데이터 수집은 이 저장소에 없다.** 수집 로직(DART·키움·KRX·네이버 커넥터, TimescaleDB 적재)은 [kr-quant-airflow](https://github.com/younghwan91/kr-quant-airflow)로 분리되어 있다. 본 저장소는 TimescaleDB(또는 로컬 SQLite)를 **읽기 전용**으로 사용해 전략을 검증한다. 분석 세션에서 수집기를 오작동시켜 DB 정합성을 훼손하는 사고를 구조적으로 차단하기 위함이다.

---

## 1. 연구 원칙

1. **확률론적 최적화, 결정론이 아니다.** 한 번의 백테스트 경로는 분포에서 뽑은 하나의 표본이다. 그 경로의 극값(파라미터 미니마)을 추구하는 것은 곧 표본 노이즈에 과최적하는 것이다. 판정은 효과가 여러 폴드·표본에서 같은 방향으로 재현되는지로 내린다.
2. **분석 단위는 개별 트레이드다.** 평균이나 복리 자본곡선이 아니라 개별 트레이드의 R-멀티플 분포로 본다. 한 종목에서 발생한 복수의 매매는 각각 독립 표본이다.
3. **취약성은 1급 지표다.** 양(+)의 기대값이 소수의 꼬리 사건에 의존한다면 통계적 신뢰는 낮다. 상위 k건 제거 생존·최장 연패·집중도를 함께 본다.
4. **룩어헤드는 절대 금기다.** 지표는 t 시점까지의 정보로만 계산하고 진입은 t+1, 임계값은 TRAIN(진입일 < 2022-01-01)에서만 학습해 전진 적용한다.
5. **게이트는 리포터이지 판정기가 아니다.** 검증 함수는 숫자만 산출하고 배포 결정은 사람이 내린다. 하드코딩된 합격선은 그 자체가 결정론적 과최적이다.

## 2. 검증 방법론 (Definition of Done)

후보 알파는 페이퍼트레이딩으로 넘어가기 전에 동일한 게이트를 통과해야 한다. 게이트는 [`research/experiments/prop_gate.py`](research/experiments/prop_gate.py)에 하나의 하버스로 구현되어 있고, 규칙과 근거는 [`docs/GUARDRAILS.md`](docs/GUARDRAILS.md)에 정리되어 있다.

| 관문 | 질문 | 라이브러리 |
|---|---|---|
| Walk-forward 재현성 | 여러 시기에서 반복되는가 (clean-OOS 폴드 부호) | `kr_quant.validation.walkforward` |
| 랜덤 음성대조 | 신호를 파괴한 널(null)을 이기는가 | `prop_gate.random_entry_control` |
| 손 안 댄 최종 구간 | 탐색에 한 번도 쓰지 않은 구간에서 사는가 | `prop_gate` (untouched window) |
| 비용·슬리피지 | 현실 비용의 2배에도 사는가 | `prop_gate` (cost sweep) |
| 취약성 | 상위 몇 건을 제거해도 사는가 | `kr_quant.diagnostics.fragility` |
| 다중검정 보정 | 시도한 config 수를 반영한 Deflated Sharpe | `kr_quant.diagnostics.gate_report` |

**음성대조의 중요성.** 폴드 성공 개수만으로 판정하면 순수 노이즈도 "6폴드 중 5폴드 양수"를 약 46% 확률로 통과한다. 따라서 진짜 판별 기준은 폴드 수가 아니라 "전략이 자기 자신의 랜덤 버전을 유의하게 이기는가"이다.

정석 개념은 문헌을 따른다 — Deflated Sharpe Ratio 및 Probability of Backtest Overfitting(Bailey & López de Prado, 2014; Bailey et al., CSCV), 다중검정 t-haircut(Harvey & Liu, 2016), purged/embargo 교차검증(López de Prado, *Advances in Financial Machine Learning*, 2018). §7 참고.

## 3. 검증된 결과 — 멀티-알파 북

서로 상관이 낮은 세 수익원 — **PEAD**(실적 YoY 드리프트, 대형주 스태거드 롱온리 excess), **미너비니**(추세템플릿+VCP 돌파+breadth 레짐의 리더주), **인덱스**(시총가중 베타) — 를 고정 가중으로 결합한다. 월수익 상관은 IDX–PEAD −0.19, IDX–MNV −0.04, MNV–PEAD +0.05로, 롤링 24개월에서도 |상관| < 0.25로 안정적이다. 어느 슬리브도 단독으로 인덱스를 압도하지 못하지만(미너비니는 오히려 뒤진다), 무상관 결합의 분산 이득이 지속된다.

| 분할조정·비용 반영, n≈87–96개월 | CAGR | Sharpe | MaxDD |
|---|---:|---:|---:|
| 시총가중 KOSPI100 프록시 (벤치마크) | +15.5% | 0.66 | −35% |
| **멀티-알파 북 (IDX 50 / PEAD 25 / MNV 25)** | **+19.3%** | **1.06** | **−24%** |

- **PEAD는 세 슬리브 중 유일하게 단독 robust**하다 — 유니버스 대비 초과수익 약 +8%p/년, t ≈ 2.16–2.97(2016–2026), 비용후 단독 Sharpe 0.65(t 2.37). 표본을 2016–2017로 확장해도 유지된다.
- **과최적 경고(저자 명시).** in-sample 최대-Sharpe 가중(IDX25/PEAD65/MNV10, Sharpe 1.36)은 사용하지 않는다 — PEAD 과적합이다. 고정·단순 가중이 OOS에서 견고하다. LightGBM 강화판(북 Sharpe 1.37–1.57)도 표본을 확장하면 plain PEAD와 동급으로 수렴하므로 robust한 우위로 인정하지 않는다.
- **개별 트레이드 관점의 한계.** PEAD의 높은 Sharpe는 개별 트레이드가 두꺼워서가 아니라 매 기간 다수 종목을 평균하는 **분산(diversification) 효과**에서 나온다. 개별 트레이드로 벗기면 건당 초과수익은 +0.46% 수준으로 얇다. 즉 PEAD는 집중형 트레이더 엣지가 아니라 **기관형 분산 알파**다([`research/experiments/pead_gate.py`](research/experiments/pead_gate.py)).

전체 규칙·가중 근거·caveat은 [`research/logs/MULTI_ALPHA.md`](research/logs/MULTI_ALPHA.md).

## 4. 기각된 결과 — 정직한 부정

부정 결과는 실패가 아니라 정당한 과학적 산출이다. 아래 네 가설은 위 게이트에서 기각되었으며, 각 부검은 근거 숫자와 재현 스크립트와 함께 [`research/logs/`](research/logs/)에 있다.

| 가설 | 유형 | 결정적 지표 | 기록 |
|---|---|---|---|
| 개미 투매 역발상 | 볼록·급등주 스윙 | walk-forward 2/6 폴드; 상위 1%가 P&L의 84% | [logs](research/logs/) |
| 미너비니식 돌파 | 추세추종 | OOS −0.029R; 랜덤 널(+0.078R)에 열위 | [VERDICT](research/logs/minervini_prop/VERDICT.md) |
| 눌림목 평균회귀 | 고승률 가설 | OOS −0.334R; 승률 34%; 널(−0.267R)에 열위 | [VERDICT](research/logs/pullback_prop/VERDICT.md) |
| 실적 서프라이즈 집중 | 이벤트 스윙 | OOS −0.208R; 집중+손절이 드리프트를 파괴 | [VERDICT](research/logs/pead_concentrated/VERDICT.md) |

**해석.** 네 가설 모두 전 세계가 오래 차익거래해 온 가격 패턴이며, 유동 대형주에서 개별 트레이드 단위로 재현되는 롱온리 엣지를 남기지 않았다. 종합은 [`research/logs/prop_swing_search/SUMMARY.md`](research/logs/prop_swing_search/SUMMARY.md).

## 5. 아키텍처

```
src/kr_quant/
├── storage.py           # 읽기 전용 DB 접근 (connect / market_cap_asof)
├── price_adjust.py      # 기업행동 백조정 (airflow DAG가 in-place 실행)
├── engine/              # 백테스트 엔진: cross-sectional·event-driven sim, 패널, 메트릭, recipe
├── validation/          # walk-forward·민감도·BO 목적함수·purge/embargo·생존편향 검사
├── diagnostics/         # R-멀티플 분포·취약성·gate_report(리포터)
├── strategies/          # pead · accumulation · minervini_sepa · supply_wave · multi_signal
├── models/              # graph_flow(그래프 확산) · ensemble_signal(릿지 앙상블)
├── features/            # fundamentals · rs_rating · vcp · short_flow · sector_flow · universe
└── viz/                 # 수급 시각화

research/                # 리서치 실험 (라이브러리를 호출하는 얇은 스크립트)
├── signals/             # 신호 정의 (contrarian_retail · operator_flow/minervini)
├── experiments/         # 실험 러너 — prop_gate 게이트 하버스로 심사
└── logs/                # VERDICT · SUMMARY (검증·부정 결과 기록)
```

설계 원칙: `src/kr_quant`는 순수 라이브러리(numpy/pandas)로 `research/`를 import하지 않는다(경계는 `scripts/check_guardrails.py`가 CI에서 강제). 새 알파는 `research/signals/`에 신호를 정의하고 `research/experiments/`에서 게이트로 심사한 뒤 `research/logs/`에 결과를 남긴다 — 표준 흐름은 [`research/TEMPLATE.md`](research/TEMPLATE.md).

## 6. 재현

```bash
git clone https://github.com/younghwan91/kr-quant
cd kr-quant
uv venv && uv pip install -e ".[viz,dev]"
cp .env.example .env          # KR_QUANT_DB에 TimescaleDB 접속정보 (비우면 로컬 SQLite)

uv run pytest                 # 네트워크 불요 — 검증·진단 라이브러리 회귀 테스트
uv run ruff check .
python scripts/check_guardrails.py   # 경계·리포터 규율 검사
```

분석 CLI(모두 DB 읽기 전용):

```bash
kq-pead --earnings-csv earnings.csv --horizon 60   # PEAD 백테스트 (핵심 알파)
kq-sepa                                            # SEPA/minervini 멀티암 비교
kq-screen --top 30                                 # 매집 스크리너 (§8)
kq-chart --code 005930                             # 종목 수급 차트
```

## 7. 참고 문헌

- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. *The Probability of Backtest Overfitting* (CSCV).
- Harvey, C. R., & Liu, Y. (2016). *…and the Cross-Section of Expected Returns.* Review of Financial Studies (다중검정 t-haircut).
- López de Prado, M. (2018). *Advances in Financial Machine Learning* (purged K-fold + embargo).

## 8. 부록 — 매집 스크리너

주가가 좁은 범위에서 횡보하는 동안 외국인·기관이 순매수하고 개인이 순매도하는 와이코프식 매집 구간을 점수화하는 1차 스크리너다. 점수 = (외국인+기관 누적 순매수 ÷ 평균거래량) ÷ 변동범위. 이는 검증된 알파가 아니라 **탐색용 스크리너**이며, 아래 그림은 단일 짧은 윈도우(2026-05-15~06-12, 19거래일)에서 뽑은 **in-sample 예시**로 성과 주장이 아니다.

![매집 후보 상위 종목](docs/images/ranking.png)
![매집 점수 vs 후속 수익률](docs/images/backtest.png)

> 매집 신호는 돌파를 보장하지 않는다. 실전 적용 시 기관 세부주체(연기금·투신 vs 프로그램), 거래량 추세, 펀더멘털을 함께 검토해야 하며, 롤링 아웃오브샘플·거래비용·생존편향을 통제한 정식 백테스트가 아니다. 재생성: `python scripts/make_figures.py`.

## 9. 데이터 스키마 (읽기 전용)

| 테이블 | 용도 |
|---|---|
| `stocks` | code, name, market, sector, kind |
| `supply_demand` | 투자자별 수급 (foreign/institution + 기관 세부 8종, PK: code+date) |
| `daily_bars` / `daily_bars_adjusted` | OHLCV; 분할조정본은 `price_adjust.py` 생성 |
| `earnings` | DART 실적 (avail_date 기준 룩어헤드 없음) — PEAD 소스 |
| `consensus` | 목표주가·투자의견·forward EPS 일별 축적 |
| `shares_outstanding_history` | 시가총액용 발행주식수 이력 |
| `short_selling` / `credit_balance` / `sector_index` | 공매도·신용잔고·업종지수 |
| `delisted_stocks` | 상장폐지 종목 (생존편향 보정용) |

테이블 소유·마이그레이션은 [kr-quant-airflow](https://github.com/younghwan91/kr-quant-airflow)가 담당한다.

## 10. 범위와 한계

- **단일 시장·일봉 지평.** 결과는 한국 유동 대형·중형주와 일봉 데이터에 한정된다. 장중·틱·대체데이터는 다루지 않는다.
- **용량(capacity).** 대형주 스태거드 롱온리 기준이며, 소형·급등주로 확장할 때의 슬리피지·용량은 별도 통제가 필요하다.
- **생존편향.** `delisted_stocks`로 부분 보정하나, 진정한 상장폐지 수익(delisting return)의 완전성은 데이터 벤더에 의존한다(`universe_hygiene`는 스멜테스트이지 보증이 아니다).
- **과거는 미래를 보장하지 않는다.** 모든 수치는 백테스트 결과다. 이 저장소의 목적은 수익 보장이 아니라 **자기기만을 줄이는 검증 규율의 공유**다.

## 라이선스

MIT
