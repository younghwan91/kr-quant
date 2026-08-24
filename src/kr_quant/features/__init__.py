"""신호 원재료 — 순수 DataFrame in/out 피처 계산.

``fundamentals`` (실적 YoY·정정공시 bitemporal 선택)가 PEAD 신호의 소스이고,
``universe`` 는 PIT 소형·중형주 유니버스 구성기다(현재 휴면 — 모듈 독스트링 참조).
``volatility`` (직전 60일 실현변동성)는 저변동 팩터의 원재료다(scalp-it 노트 #31 이관).
수급·신용·공매도·섹터 피처는 그것을 쓰던 스크리너와 함께 2026-08-16 에 삭제했다.
"""
