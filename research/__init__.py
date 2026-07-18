"""kr-quant 알파 리서치 — 사적 실험 계층(오픈소스 라이브러리 src/kr_quant/ 위에서 동작).

파이프라인 강제 구조(디렉터리 = 단계):
    signals/      "무엇에 베팅하나" — 신호 빌더 + load_data + 시뮬레이터(신호 특정).
    experiments/  "가설 → 결과" — 라이브러리(kr_quant.validation/diagnostics)를 호출하는
                  얇은 러너. 검증·진단 로직을 재구현하지 않는다(단일 소스 = 라이브러리).
    logs/         구조화된 실험 결과(VERDICT.md · best params · 서사 md).

경계: src/kr_quant/ 는 research/ 로부터 아무것도 import 하지 않는다. 반대로 research/ 는
kr_quant.{engine,validation,diagnostics} 를 자유로이 import 한다. 새 알파 흐름은 TEMPLATE.md.
"""
