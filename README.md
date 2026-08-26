# kr-quant

[![CI](https://github.com/younghwan91/kr-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/younghwan91/kr-quant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

**한국 주식(KOSPI·KOSDAQ) 알파를 개별 트레이드 분포로 심사하는 검증 프레임워크 — 그리고 그 프레임워크가 내린 정직한 판정들.**

백테스트는 거의 언제나 우상향 곡선을 그린다. 손절을 걸면 평균회귀가, 풀면 추세가 아름답게 나온다. **그래서 이 저장소의 산출물은 전략이 아니라 판별 능력이다** — 어떤 곡선이 구조적 엣지이고 어떤 곡선이 표본 노이즈인지 가려내는 기계, 그리고 그 기계를 실제 가설 5건과 위험 오버레이 1건에 돌려 남긴 기록.

**심사 결과: 알파 가설 5건 중 4건 기각, 위험 오버레이 1건 기각.** 기각은 실패 목록이 아니라 이 저장소의 주된 산출물이다 — 재현되지 않는 전략을 일찍 접는 것이 자본과 시간의 절약이고, *왜* 안 되는지의 메커니즘은 다음 탐색으로 이전(transfer)된다.

**이 프레임워크가 실제로 잡아낸 것들**

- **포트폴리오 지표가 개별 엣지를 과대평가한다** — 미너비니 돌파는 자본곡선이 훌륭했지만 건당 기대값은 랜덤 널 아래였다. 레버리지·복리 효과였다(§5).
- **분산 엣지를 집중하면 죽는다** — PEAD를 소수 종목에 몰아 담자 손절·보유상한이 승자를 잘라 엣지가 사라졌다(§5).
- **생존편향은 전략이 아니라 벤치마크를 더 크게 왜곡한다** — 상장폐지 종목을 복원하자 PEAD 초과수익이 오히려 *올라갔다*(§4). 통념과 반대 방향이며 상대수익 전략의 구조적 성질이다.
- **순수 노이즈가 "6폴드 중 5폴드 양수"를 46% 확률로 통과한다** — 그래서 폴드 수가 아니라 자기 자신의 랜덤 버전을 이기는지가 판별 기준이다(§2).
- **노출을 줄이면 드로다운은 저절로 줄어든다** — 국면 오프스위치의 MDD 개선은 듀티사이클을 맞춘 랜덤 널과 구분되지 않았다. 위험 오버레이에는 알파와 다른 널이 필요하다(§5.1).
- 통과한 것은 **PEAD 하나**다(§4). 다만 이건 소수 꼬리에 의존하는 트레이더형이 아니라 다수 종목에 분산된 기관형 엣지다 — 둘의 차이는 §3에서 해부한다.

> **왜 신뢰할 수 있나.** 위 규율은 부탁이 아니라 CI 린트로 강제된다 — 게이트가 공용 하버스를 안 쓰거나, 음성대조 없이 판정하거나, 시행 수를 원장에 안 남기면 빌드가 깨진다(§7).

> **데이터 수집은 이 저장소에 없다.** 수집 로직(DART·키움·KRX·네이버 커넥터, TimescaleDB 적재)은 [quant-airflow](https://github.com/younghwan91/quant-airflow)로 분리되어 있다. 본 저장소는 DB를 **읽기 전용**으로 사용한다.

## 지금 바로 확인 — API 키도 DB도 필요 없다

위의 "노이즈가 46% 확률로 통과한다"는 주장을 합성 데이터로 직접 재현할 수 있다. 1초 안에 끝난다.

```bash
uv run python research/experiments/prop_gate.py     # 심사 배터리 전체 + 랜덤 음성대조
uv run python examples/pead_sweep_via_recipe.py     # 레시피로 PEAD 파라미터 스윕
```

실제 출력 마지막 블록:

```text
  랜덤 음성대조 — 50 draws × 600 trades (게이트 위양성률 보정)
========================================================================
  raw ≥5/6 폴드 도달: 46.0% of draws
  clean-OOS 전폴드(≥4/4) 양수 도달: 22.0% of draws
  OOS 기대값R: 평균 +0.076  p95 +0.185
  → 무작위가 이 바를 자주 넘으면 게이트가 너무 느슨하다(사람이 판단).
```

**신호가 아예 없는 데이터로 만든 전략이 "6폴드 중 5폴드 양수"를 46% 확률로 달성한다.** 백테스트 자본곡선이 왜 증거가 못 되는지가 이 한 줄에 들어 있다. 같은 배터리가 §3–§5의 실제 가설 5건에 그대로 돌아간다.

---

## 1. 연구 원칙

1. **확률론적 최적화, 결정론이 아니다.** 한 번의 백테스트 경로는 분포에서 뽑은 하나의 표본이다. 그 경로의 극값(파라미터 미니마)을 추구하는 것은 곧 표본 노이즈에 과최적하는 것이다. 판정은 효과가 여러 폴드·표본에서 같은 방향으로 재현되는지로 내린다.
2. **분석 단위는 개별 트레이드다.** 평균이나 복리 자본곡선이 아니라 개별 트레이드의 R-멀티플 분포로 본다. 한 종목에서 발생한 복수의 매매는 각각 독립 표본이다.
3. **취약성은 1급 지표다.** 양(+)의 기대값이 소수의 꼬리 사건에 의존한다면 통계적 신뢰는 낮다. 상위 k건 제거 생존·최장 연패·집중도를 함께 본다.
4. **룩어헤드는 절대 금기다.** 지표는 t 시점까지의 정보로만 계산하고 진입은 t+1, 임계값은 TRAIN(진입일 < 2022-01-01)에서만 학습해 전진 적용한다.
5. **게이트는 리포터이지 판정기가 아니다.** 검증 함수는 숫자만 산출하고 배포 결정은 사람이 내린다. 하드코딩된 합격선은 그 자체가 결정론적 과최적이다.

## 2. 심사 기계 (Definition of Done) — 이 저장소의 본체

후보 알파는 배포 전 동일한 게이트를 통과해야 한다. 게이트는 [`research/experiments/prop_gate.py`](research/experiments/prop_gate.py)에 하나의 하버스로 구현되어 있고, 규칙과 문헌 근거는 [`docs/GUARDRAILS.md`](docs/GUARDRAILS.md)에 있다.

| 관문 | 질문 | 라이브러리 |
|---|---|---|
| Walk-forward 재현성 | 여러 시기에서 반복되는가 (clean-OOS 폴드 부호) | `kr_quant.validation.walkforward` |
| 랜덤 음성대조 | 신호를 파괴한 널(null)을 이기는가 | `prop_gate.random_entry_control` |
| 손 안 댄 최종 구간 | 탐색에 한 번도 쓰지 않은 구간에서 사는가 | `prop_gate` (untouched window) |
| 비용·슬리피지 | 현실 비용의 2배에도 사는가 | `prop_gate` (cost sweep) |
| 취약성 | 상위 몇 건을 제거해도 사는가 | `kr_quant.diagnostics.fragility` |
| 다중검정 보정 | 시도한 config 수를 반영한 Deflated Sharpe·t-haircut | `diagnostics.gate_report` + `diagnostics.trials`(원장) |
| 라벨 누출 차단 | 보유기간이 TEST로 넘어간 표본을 TRAIN에서 걷어냈나 | `validation.walkforward.purge_embargo` |
| 유니버스 무결성 | 상장폐지 종목이 유니버스에 들어있나 | `storage.read_prices`(로딩 시점 assert) |

**시행 수는 손으로 세지 않는다.** Deflated Sharpe의 입력 N을 사람이 적으면 규율이 아니라 부탁이다 — 첫 config가 죽고 조용히 두 번째를 돌려도 아무도 못 잡는다. 게이트가 사전등록 config를 `research/logs/<alpha>/TRIALS.jsonl`에 append하고 N을 거기서 읽는다. 같은 config 재실행은 시행으로 세지 않는다(시행 = *다르게* 시도한 횟수). **민감도 격자의 각 칸도 시행이다** — 승자를 뽑지 않더라도 시도한 것이므로 N에 들어간다(GUARDRAILS §6 사전등록 템플릿).

> **이 규율이 실제로는 작동하지 않고 있었다(2026-08-16 발견·수정).** 세 군데가 동시에 끊겨 있었다 — ① 린트 (e)가 파일 단위라 스윕 칸들이 원장을 우회했고, ② 사전등록 게이트가 격자보다 **먼저** 실행돼 자기 자신만 센 N=1을 읽었으며, ③ `prop_gate`가 보정값을 stdout으로 출력만 하고 VERDICT 렌더러는 그 블록을 통째로 빠뜨렸다. 결과적으로 **어떤 판정문에도 다중검정 보정이 남아 있지 않았고**, 남았더라도 `n_trials ≤ 1`이라 haircut이 0이었다. 셋 다 고치고 두 게이트를 재실행한 결과:
>
> | | N (전→후) | E[maxSharpe\|H0] | t-haircut | 문턱 t |
> |---|---:|---:|---:|---:|
> | 눌림목 | 1 → **19** | 0 → 0.061 | ×1.00 → **×1.53** | 1.96 → 3.01 |
> | PEAD 집중 | 2 → **29** | 0 → 0.160 | ×1.00 → **×1.60** | 1.96 → 3.13 |
>
> 둘 다 이미 기각이라 **결론은 뒤집히지 않고 강화**됐다(deflated Sharpe −0.364 / −0.329). 통과 판정이었다면 이야기가 달랐을 것이다 — 그게 이 보정을 두는 이유다.

**음성대조가 핵심인 이유.** 폴드 성공 개수만으로 판정하면 순수 노이즈도 "6폴드 중 5폴드 양수"를 약 46% 확률로 통과한다. 따라서 진짜 판별 기준은 폴드 수가 아니라 "전략이 자기 자신의 랜덤 버전을 유의하게 이기는가"이다. 정석 개념은 문헌을 따른다(§9).

## 3. 심사에서 드러난 구조 — 엣지의 해부

### 3.1 두 유형의 엣지

동일한 한국 주식 데이터에서 성격이 상반된 두 종류의 엣지가 관찰된다. 이 구분은 전략을 고르기 전에 **어떤 종류의 게임을 하는지**를 규정한다.

| 축 (개별 트레이드, OOS) | 볼록형 (트레이더 엣지) | 확산형 (기관 엣지) |
|---|---|---|
| 대표 사례 | 급등주 스윙(개미 투매 역발상) | PEAD(실적 드리프트) |
| 승률 | ≈35% | ≈45% |
| 분포 왜도 | +5.9 (강한 우측 꼬리) | +1.7 |
| 최대 단일 트레이드 | +521% | +132% |
| 엣지의 출처 | 소수 대박 꼬리 | 수많은 얇은 엣지의 평균 |
| walk-forward 재현성 | 깨지기 쉬움 | 안정적 |

볼록형은 손절로 왼꼬리를 자르고 승자를 태워 오른꼬리를 노리지만, 수익이 소수 사건에 집중될수록 재현이 불안정하다. 확산형은 개별 트레이드가 얇고 승률도 압도적이지 않으나(≈45%), 상관이 낮은 다수 종목을 묶으면 분산으로 변동성이 낮아져 book 단위에서 안정적 Sharpe가 나온다(§4) — 이것이 PEAD가 트레이더 엣지가 아니라 **기관 엣지**인 이유다.

![엣지의 두 유형 — 개별 트레이드 수익 분포](docs/images/edge_taxonomy.png)
*개별 트레이드 수익 분포. 볼록형(급등주 스윙)은 왜도가 높고 소수 꼬리에 P&L이 집중되는 반면, 확산형(PEAD)은 대칭에 가깝고 넓게 분산된다.*

### 3.2 진입 시점 신호의 선험적 예측력

개별 트레이드 분포를 진입 시점 특성으로 층화하면 구조적 규칙성이 드러난다. 급등주 스윙에서 **진입 시점의 모멘텀 강도**로 5분위를 나누면 사후 기대수익이 전반적으로 우상향하고, **최상위 분위(Q5)가 최하위(Q1)를 크게 웃도는 관계가 TRAIN과 OOS 양쪽에서 성립**한다. 개별 분위는 표본 노이즈로 다소 비단조적이지만 방향성 자체는 견고하다. 즉 "어느 트레이드가 대박이 될지"는 순전한 운이 아니라 진입 시점에 어느 정도 예측된다. 다만 이 예측력만으로는 집중·비용을 반영한 배포 기준을 넘지 못한다 — 신호가 정보를 담는 것과 그 정보가 거래비용을 이기는 것은 별개다.

![진입 시점 모멘텀 강도 → 사후 기대수익](docs/images/apriori_momentum.png)
*진입 시점 모멘텀 강도로 나눈 5분위별 사후 기대수익. Q5가 Q1을 크게 상회하는 우상향 관계가 TRAIN·OOS 둘 다에서 성립한다(개별 분위는 노이즈로 다소 비단조적이다).*

분석 코드: [`research/experiments/contrarian_distribution.py`](research/experiments/contrarian_distribution.py)(계산은 `kr_quant.diagnostics.r_distribution.conviction_analysis`), 게이트 하버스: `prop_gate`.

## 4. 통과 사례 — PEAD와 멀티-알파 북

> 5건 중 유일하게 게이트를 통과한 가설이다. **이 저장소가 파는 것은 이 전략이 아니라 §2의 잣대다** — 아래 수치는 그 잣대를 통과하면 어떤 모양인지 보여주는 사례로 읽는 게 맞다.

상관이 낮은 세 수익원 — **PEAD**(실적 YoY 드리프트, 대형주 스태거드 롱온리 excess), **미너비니**(추세템플릿+VCP 돌파+breadth 레짐의 리더주), **인덱스**(시총가중 베타) — 를 고정 가중으로 결합한다. 월수익 상관은 IDX–PEAD −0.19, IDX–MNV −0.04, MNV–PEAD +0.05로 롤링 24개월에서도 |상관| < 0.25이다. 어느 슬리브도 단독으로 인덱스를 압도하지 못하지만, 무상관 결합의 분산 이득이 지속된다.

| 분할조정·비용 반영, n≈87–96개월 | CAGR | Sharpe | MaxDD |
|---|---:|---:|---:|
| 시총가중 KOSPI100 프록시 (벤치마크) | +15.5% | 0.66 | −35% |
| **멀티-알파 북 (IDX 50 / PEAD 25 / MNV 25)** | **+19.3%** | **1.06** | **−24%** |

- **PEAD는 세 슬리브 중 유일하게 단독 robust**하다 — 유니버스 대비 초과수익 약 +8%p/년, t ≈ 2.16–2.97(2016–2026), 비용후 단독 Sharpe 0.65(t 2.37). 표본을 2016–2017로 확장해도 유지된다.
- **이 수치는 생존편향 때문에 보수적이었다(2026-08-15 측정).** 상장폐지 종목을 유니버스에 넣자 같은 베이스라인의 Sharpe가 1.20 → 1.49, t가 3.47 → 4.27로 올랐다. 편향이 부풀린 것은 전략이 아니라 **벤치마크**다 — PEAD는 유니버스 대비 초과수익이라, 죽은 회사를 빼면 유니버스 평균이 실제보다 좋아 보여 초과수익이 과소 계상된다(벤치 −0.391% vs 실제 −0.606%, 기간당). 북 자체는 예상대로 −0.053%p 나빠지며, 벤치마크 왜곡이 그 4.1배다. 근거·한계: [VERDICT](research/logs/survivorship_bias/VERDICT.md).
- **과최적 방지(저자 명시).** in-sample 최대-Sharpe 가중(IDX25/PEAD65/MNV10, Sharpe 1.36)은 채택하지 않는다 — PEAD 과적합이다. 고정·단순 가중이 OOS에서 견고하다. LightGBM 강화판(북 Sharpe 1.37–1.57)도 표본을 확장하면 plain PEAD와 동급으로 수렴하므로 robust한 우위로 인정하지 않는다.

전체 규칙·가중 근거·caveat: [`research/logs/MULTI_ALPHA.md`](research/logs/MULTI_ALPHA.md).

> **2026-08-13 기준 재현성 고지.** MNV 슬리브의 구현 코드(추세템플릿·VCP·SEPA 백테스트 하네스)는
> §5의 개별-트레이드 기각 판정에 따라 레포에서 삭제했다. 위 표의 수치는 삭제 시점까지 검증된
> 기록으로 남기며, 지금 레포에서 그대로 재현되지는 않는다 — 코드는 git 이력
> (`chore/remove-minervini` 이전)에 있다. PEAD·인덱스 슬리브는 영향받지 않는다.

## 5. 기각 사례 — 메커니즘 분석

방법론(§2)을 여러 볼록형 스윙 가설에 적용한 결과다. 각 항목은 "죽었다"가 아니라 **왜 개별-트레이드 단위에서 재현되지 않는지의 메커니즘**으로 읽어야 한다 — 이것이 스크리너 수준의 백테스트가 놓치는 지점이다.

| 가설 | 유형 | 관측된 메커니즘 | 기록 |
|---|---|---|---|
| 개미 투매 역발상 | 볼록·급등주 스윙 | 수익이 상위 1%(순P&L의 84%)에 집중 → 단일 국면 의존, walk-forward 2/6 | [logs](research/logs/) |
| 미너비니식 돌파 | 추세추종 | 포트폴리오 성과는 레버리지·복리 효과였고, 개별 트레이드 기대값은 랜덤 널(+0.078R) 아래(−0.029R) | [VERDICT](research/logs/minervini_prop/VERDICT.md) |
| 눌림목 평균회귀 | 고승률 가설 | 가설의 전제인 고승률이 성립 안 함(35%); 널(−0.267R)보다도 낮은 −0.329R | [VERDICT](research/logs/pullback_prop/VERDICT.md) |
| 실적 서프라이즈 집중 | 이벤트 스윙 | PEAD는 느린 60일 드리프트 → 손절·보유상한이 승자를 잘라 확산 엣지를 파괴(−0.227R, 널 −0.042R) | [VERDICT](research/logs/pead_concentrated/VERDICT.md) |

> **2026-08-15 생존편향 보정 후 재측정.** 상장폐지 종목을 유니버스에 넣고 세 게이트를 다시 돌렸다. **기각은 전부 유지**된다 — 눌림목 −0.334R → −0.329R(트레이드 1,810 → 1,894건), PEAD집중 −0.208R → −0.227R(랜덤 널과의 격차는 오히려 벌어졌다), 개미투매 역발상은 폴드일관 2/6·46bp 사망·monster-share 104%가 그대로다. 미너비니는 러너를 삭제해 재측정 대상이 아니다. §4의 PEAD와 달리 이들은 **절대수익 R-멀티플** 기준이라, 벤치마크 왜곡이 아니라 트레이드 자체로 평가된다.

**분석적 결론.** 네 가설은 모두 오래전부터 차익거래로 닳아온 가격 패턴이며, 유동 대형주에서 개별 트레이드 단위로 재현되는 롱온리 엣지를 남기지 않았다. 이 발견의 값어치는 방향 자체가 아니라 **메커니즘의 이해**에 있다 — 분산 엣지를 집중하면 왜 죽는지(PEAD 집중), 포트폴리오 지표가 왜 개별 엣지를 과대평가하는지(미너비니). 종합: [`research/logs/prop_swing_search/SUMMARY.md`](research/logs/prop_swing_search/SUMMARY.md).

### 5.1 위험 오버레이 심사 — 국면 오프스위치

위 네 건은 *알파 가설*이다. 성격이 다른 심사도 한 건 있다 — **이미 검증된 북에 씌우는 위험
오버레이**. 재량 트레이더들의 "지수 역배열이면 매매 안 한다"를 기계화해, 배포형 PEAD 북
(롱온리 + 인버스 ETF 헤지)에 지수가 200일 이평 아래인 달마다 북을 내려두는 스위치를 붙였다.

**여기서는 랜덤 진입 널이 판별기가 못 된다.** 노출을 줄이는 *어떤* 스위치든 드로다운은
줄어들기 때문에, "always-on보다 MDD가 낮다"는 아무것도 증명하지 않는다. 그래서 **회전 널**을
썼다 — 같은 상태 수열을 시간축으로 돌려 듀티사이클·런렝스·스위치 횟수를 그대로 두고
수익과의 *정렬만* 깨뜨린 널이다. 노출 축소 효과는 회전에도 살아남고 타이밍 효과만 죽는다.

| 관측 | 값 |
|---|---|
| MDD 개선 | −9.4% → −9.1% (**+0.3%p**) |
| 회전 널 안에서의 위치 | **53 퍼센타일** (널 중앙값 −9.4%, p90 −6.0%) |
| 대가 | Sharpe 0.85 → 0.56, NW-t 2.14 → 1.38 |
| 시장노출 북에 적용 시 | MDD **−9.6%p 악화**, 널의 **4 퍼센타일** |

그 0.3%p는 타이밍이 아니라 36%의 시간을 쉰 효과다. 사전등록 합격 바 5개 중 4개 실패로
기각했다. 민감도 격자에서 MA 100~300과 3단 변형이 전부 널 백분위 52~64%에 머물렀다는 점도
같이 남긴다 — 격자만 봤으면 "3단이 답"이라고 썼을 것이다.

기록: [VERDICT](research/logs/regime_switch/VERDICT.md) · 재사용 가능한 널:
`kr_quant.strategies.regime.rotation_null`

## 6. 이 저장소가 남기는 것

알파 탐색은 대부분 부정으로 끝난다. 그래서 오래 남는 것은 특정 전략이 아니라 아래 다섯이다.

- **재사용 가능한 검증 프레임워크.** walk-forward·음성대조·손 안 댄 구간·취약성·Deflated Sharpe·purge/embargo를 하나의 게이트로 묶은 `kr_quant.validation`·`kr_quant.diagnostics`. 어떤 새 가설이든 동일한 잣대로 몇 시간 만에 심사한다 — 특정 알파보다 오래 남는 자산이다.
- **만드는 것과 연결하는 것은 다른 일이다.** 이 레포에서 반복된 실패 모드는 기능 부재가 아니라 **배선 누락**이었다 — purge/embargo·Deflated Sharpe·음성대조 규약이 전부 구현된 채 한 번도 호출되지 않고 있었다. 그래서 지금은 배선이 빠지면 CI가 실패한다(§7).
- **분포적 사고.** 평균이나 자본곡선이 아니라 개별 트레이드 분포로 보면 같은 데이터가 다르게 읽힌다. 볼록형/확산형 구분, 꼬리 집중도, 선험적 예측력은 이 렌즈에서만 드러난다.
- **음성대조라는 규율.** "내 전략이 랜덤보다 나은가"는 단순하지만 대부분의 자기기만을 잡아낸다(순수 노이즈가 폴드 기준을 46% 통과한다).
- **엔지니어링 토대.** PIT 데이터·무룩어헤드·읽기전용 분리·경계 린트·CI. 알파와 무관하게 재현 가능한 리서치를 떠받치는 하부구조다.
- **무엇을 하지 않을지 아는 것.** 재현되지 않는 전략을 일찍 접는 것은 실패가 아니라 자본과 시간의 절약이다.
- **메커니즘 이해.** "왜 되는가/왜 안 되는가"는 다음 탐색의 지도다 — 분산 엣지를 집중하면 죽는다, 포트폴리오 지표는 개별 엣지를 과대평가한다. 이런 통찰은 특정 전략보다 이전(transfer)된다.

검증하는 법을 갖추면 다음 알파는 더 빠르고 정직하게 찾을 수 있다. 이 저장소가 공유하려는 것은 결국 그 **역량**이며, §4·§5의 판정들은 그 역량이 실제로 작동한 기록이다.

## 7. 아키텍처

**심사 기계 (src/kr_quant) — 알파와 무관하게 재사용된다.**

```
src/kr_quant/
├── storage.py           # 읽기 전용 DB 접근. read_prices/read_earnings가 유일한 정문 —
│                        #   폐지종목 포함·정정공시 버전을 로딩 시점에 검사한다
├── price_adjust.py      # 기업행동 백조정 (airflow DAG가 in-place 실행)
├── engine/              # 백테스트 회계: cross-sectional sim, 패널, 메트릭, recipe
├── validation/          # walk-forward·민감도·BO 목적함수·purge/embargo·생존편향 검사
├── diagnostics/         # R-멀티플 분포·취약성·gate_report(리포터)·trials(다중검정 원장)
├── features/            # fundamentals(실적 YoY·정정공시 bitemporal) · universe(PIT) ·
│                        #   volatility(직전 60일 실현변동성 — 저변동 팩터 원재료)
└── strategies/          # pead.py = 통과한 유일한 알파의 DataFrame 어댑터 (회계는 engine/).
                         #   lowvol·combo·hedge 는 scalp-it 에서 이관한 **후보** 어댑터로
                         #   정식 게이트 배터리(prop_gate)로는 아직 재심사되지 않았다
                         #   (docs/lowvol-strategy.md; 결합 북 수치는 비공개)
```

**심사 대상과 판정 기록 (research/) — 기계에 태워진 가설들.**

```
research/
├── signals/             # 신호 정의 (contrarian_retail · pullback_swing)
├── experiments/         # 실험 러너 — prop_gate 게이트 하버스로 심사
└── logs/                # VERDICT · SUMMARY · TRIALS.jsonl (판정·분석·시행 원장)
```

> **2026-08-16 정리.** 스크리너·ML·차트 계열(accumulation·backtest·supply_wave·
> multi_signal·graph_flow·ensemble_signal·viz + 그것만 쓰던 수급/신용/공매도/섹터 피처)
> 4,400여 줄을 삭제했다. 삭제 이유는 "안 쓰여서"가 아니다 — 전부 `supply_demand JOIN
> stocks`(상장 마스터와의 INNER JOIN이라 **폐지 종목이 구조적으로 빠진다**) 위에서
> **분할 미조정** 종가로 돌고 있었다. 즉 이 저장소가 §2에서 강제한다고 적어둔 두 규율을
> src/ 안에서 스스로 어기던 코드다. GUARDRAILS §0 원칙대로 부탁 대신 **제거**로 막았다.

설계 원칙: `src/kr_quant`는 순수 라이브러리(numpy/pandas)로 `research/`를 import하지 않는다. 새 알파의 표준 흐름은 [`research/TEMPLATE.md`](research/TEMPLATE.md).

**규율은 부탁이 아니라 린트다.** `scripts/check_guardrails.py`가 CI에서 8가지를 막는다 — (a) src→research import, (b) VERDICT 없는 `*_gate.py`, (c) 하드코딩 "PASS"/"FAIL" 판정 문자열, (d) `storage` 밖에서 raw `SELECT ... FROM earnings`(정정공시 버전이 중복 행으로 샌다), (e) `prop_gate`/`gate_sim` **호출 지점마다** `config=` 누락(원장에 시행이 안 남아 DSR이 계산되지 않는다), (f) `*_gate.py`가 공용 하버스를 안 쓰는 것(음성대조·비용2배·R분포가 조용히 빠진다), (g) 가격 테이블 직접 SELECT + `supply_demand JOIN stocks`(유니버스에서 폐지 종목이 빠진다), (h) **코드 없는 VERDICT**((b)의 역방향 — 러너가 등록·실재하거나 재현불가를 명시해야 한다).

> **그리고 린트를 검사하는 린트가 있다.** `tests/test_check_guardrails.py`가 각 규칙에 위반을 주입해 *실제로 실패하는지* 확인한다. 이게 없어서 실제로 일이 났다 — 규칙 (e)가 **파일 단위**로 구현돼 있어, 파일 어딘가에 `config=`가 한 번만 있으면 같은 파일의 나머지 게이트 호출이 전부 면제됐다. 그 탓에 민감도 스윕 18~27칸이 원장을 통째로 우회했고 `n_trials ≤ 1`이라 **Deflated Sharpe의 haircut이 0**이었는데 CI는 초록이었다. 규칙이 막으려던 실패 모드를 규칙이 못 보고 있었다(2026-08-16 수정).

린트를 이렇게까지 겹쳐 두는 이유는 하나다. 이 저장소에서 반복된 실패 모드가 기능 부재가 아니라 **배선 누락**이었기 때문이다 — purge/embargo·Deflated Sharpe·음성대조 규약이 전부 구현된 채로 한 번도 호출되지 않고 있었다.

## 8. 재현

```bash
git clone https://github.com/younghwan91/kr-quant
cd kr-quant
uv venv && uv pip install -e ".[viz,dev]"
cp .env.example .env          # KR_QUANT_DB에 TimescaleDB 접속정보 (비우면 로컬 SQLite)

uv run pytest                 # 네트워크 불요 — 검증·진단 라이브러리 회귀 테스트
uv run ruff check .
python scripts/check_guardrails.py   # 경계·판정·정문·하버스 규율 검사 (a)~(h)
```

분석 CLI(DB 읽기 전용): `kq-pead` — 게이트를 통과한 유일한 알파의 재현용 백테스트. 이
저장소는 스크리너 제품이 아니라 검증 프레임워크라 설치되는 CLI 엔트리포인트는 이것
하나다. 후보 전략(저변동·결합·인버스헤지)의 **배포북** — 재현 성적표와 적합된 비중 —
은 판정문이 아니라 따라 할 수 있는 레시피라 이 공개 저장소에 싣지 않는다. 라이브러리
(`strategies/{lowvol,combo,hedge}`·`features/volatility`)와 테스트는 여기 그대로 있다.

## 9. 참고 문헌

- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. *The Probability of Backtest Overfitting* (CSCV).
- Harvey, C. R., & Liu, Y. (2016). *…and the Cross-Section of Expected Returns.* Review of Financial Studies (다중검정 t-haircut).
- López de Prado, M. (2018). *Advances in Financial Machine Learning* (purged K-fold + embargo).

## 10. 데이터 스키마 (읽기 전용)

| 테이블 | 용도 |
|---|---|
| `stocks` | code, name, market, sector, kind |
| `supply_demand` | 투자자별 수급 (foreign/institution + 기관 세부 8종, PK: code+date) |
| `daily_bars` / `daily_bars_adjusted` | OHLCV; 분할조정본은 `price_adjust.py` 생성. `source`가 `'kiwoom'`(상장 종목, 보고된 거래대금) / `'naver'`(폐지 종목 백필, 거래대금은 `close×volume` 근사)를 구분한다 |
| `earnings` | DART 실적 (avail_date 기준 룩어헤드 없음) — PEAD 소스. PK는 `(code, period, knowledge_date)`로, 정정공시는 기존 행을 덮어쓰지 않고 새 버전으로 쌓인다 — 읽을 때 `knowledge_date <= t`로 그 시점에 알 수 있었던 값을 고른다 |
| `consensus` | 목표주가·투자의견·forward EPS 일별 축적 |
| `shares_outstanding_history` | 시가총액용 발행주식수 이력 |
| `short_selling` / `credit_balance` / `sector_index` | 공매도·신용잔고·업종지수 |
| `delisted_stocks` | 상장폐지 종목 마스터. 과거 시세는 `daily_bars`에 `source='naver'`로 들어간다 |

테이블 소유·마이그레이션은 [quant-airflow](https://github.com/younghwan91/quant-airflow)가 담당한다.

## 11. 범위와 한계

> 이 저장소는 **수익을 파는 곳이 아니다.** §4의 수치도 백테스트 결과이며, 기여는 그 수치가 아니라 그것을 얻고 검증한 절차에 있다.

- **단일 시장·일봉 지평.** 결과는 한국 유동 대형·중형주와 일봉 데이터에 한정된다. 장중·틱·대체데이터는 다루지 않는다.
- **용량(capacity).** 대형주 스태거드 롱온리 기준이며, 소형·급등주 확장 시 슬리피지·용량은 별도 통제가 필요하다.
- **생존편향.** 상장폐지 종목을 네 층 모두 복원했고, 유니버스 무결성은 `storage.read_prices`가 로딩 시점에 검사한다 — 시세 460 / 실적 364 / 상장주식수 418(2,013행, 417종목이 마지막 거래일 이전에 점을 가져 생애 중반 시총이 실제로 계산된다) / 수급 460(448,237행). 다만 **수급은 부분이다** — 키움은 폐지 코드에 성공 응답 + 0행을 주고, 네이버는 폐지분을 주지만 기관·외국인 순매매만 있어 11개 투자자 분류 중 2개만 채워진다. 나머지는 `NULL`로 남긴다(0은 "순매매 없음", NULL은 "모름"이라 구분해야 한다). **개인 순매매를 신호로 쓰는 연구는 이 데이터로 재현할 수 없다.** 폐지 손실 실현은 `delisting_exit=True`로 선택 가능하며 실측 영향은 미미했다(정리매매 구간이 이미 가격에 들어 있어서).
- **과거는 미래를 보장하지 않는다.** 모든 수치는 백테스트 결과이며, 이 저장소의 기여는 수익 보장이 아니라 **체계적 검증 방법론과 그 분석 결과의 공유**다.

## 라이선스

MIT


---

## ⭐ 도움이 되셨다면

이 프로젝트가 유용했다면 우측 상단 **[⭐ Star](https://github.com/younghwan91/kr-quant)** 를 눌러주세요. 검색·추천 노출이 올라가 더 많은 분들이 찾을 수 있습니다.

- 🐛 버그·질문 → [Issues](https://github.com/younghwan91/kr-quant/issues)
- 📈 업데이트 소식 → [팔로우 @younghwan91](https://github.com/younghwan91)

## 관련 프로젝트 — 오픈소스 퀀트 스택

한국·미국 주식과 암호화폐를 아우르는 오픈소스 스택입니다. 각 저장소는 독립적으로 쓸 수 있습니다.

| 축 | 프로젝트 | 설명 |
|---|---|---|
| 🇰🇷 한국 주식 | **[kiwoom-rest-api](https://github.com/younghwan91/kiwoom-rest-api)** | 키움증권 REST API Python 라이브러리 — 국내주식 엔드포인트 전수·실시간 WebSocket, sync + async (`pip install kiwoom-client`) |
| 🇰🇷 한국 주식 | **[krx-fundamentals-api](https://github.com/younghwan91/krx-fundamentals-api)** | 국내 기업 펀더멘탈 REST API — 재무제표·투자지표·배당·종목 스크리닝 (DART + KRX + 네이버) |
| 🇰🇷 한국 주식 | **[krx-news-rest-api](https://github.com/younghwan91/krx-news-rest-api)** | 한국 주식 뉴스·공시 수집 REST API (FastAPI + Redis) |
| 🇰🇷 한국 주식 | **[quant-airflow](https://github.com/younghwan91/quant-airflow)** | 시세·수급·실적을 TimescaleDB 로 수집하는 Airflow 파이프라인 — 상장폐지 종목까지 담아 생존편향을 막는다 |
| 🇺🇸 미국 주식 | **[portfolio-research](https://github.com/younghwan91/portfolio-research)** | 미국주식 팩터 엔진 — point-in-time·생존편향 보정 데이터 위에서 walk-forward 를 Deflated Sharpe·PBO 로 게이팅 (+ ETF 전술배분 TAA — 9개 사전등록, 채택 0) |
| 🇺🇸 미국 주식 | **[automated-stock-trading-systems](https://github.com/younghwan91/automated-stock-trading-systems)** | Bensdorp 의 7개 비상관 트레이딩 시스템 백테스터 (교육용 재구현) |
| ₿ 암호화폐 | **[quantbox-engine](https://github.com/younghwan91/quantbox-engine)** | 암호화폐 선물 백테스트·실행 엔진 — 룩어헤드 0, 백테스트↔실거래 일체화 |

## 만든 사람

**채영환 (Younghwan Chae)** · [GitHub @younghwan91](https://github.com/younghwan91) · [LinkedIn](https://www.linkedin.com/in/younghwan-chae/)

전체 오픈소스 퀀트 스택은 [프로필](https://github.com/younghwan91)에서 한눈에 볼 수 있습니다.
