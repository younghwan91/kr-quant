"""통과한 단 하나의 알파 — PEAD 참조 구현.

이 패키지는 '전략 모음'이 아니다. 심사를 통과한 유일한 가설(§4 PEAD)의 DataFrame 레벨
어댑터만 남아 있고, 회계 자체는 전부 :mod:`kr_quant.engine` 에 있다. 나머지 스크리너·
ML·차트 전략(accumulation·backtest·supply_wave·multi_signal·graph_flow·ensemble_signal)은
2026-08-16 에 삭제했다 — 생존자 전용 유니버스 위에서 분할 미조정 종가로 돌아, 이
저장소가 강제한다고 적어둔 규율을 스스로 어기고 있었기 때문이다(git 이력에 있다).
"""
