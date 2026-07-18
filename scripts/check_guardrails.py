#!/usr/bin/env python
"""가드레일 경계 린트 — 순수 파이썬(무의존), CI 스텝.

GUARDRAILS §5 로드맵 item 5 를 코드로 강제한다. CI 는 특정 연구가 가드레일을
'썼는지'는 못 막지만, 아래 세 가지 **구조적 경계**는 기계적으로 지킬 수 있다:

  (a) 경계 위반 — ``src/kr_quant/`` 가 ``research/`` 를 import 하면 실패.
      (TEMPLATE 1단계: src 는 research 를 import 하지 않는다. 반대만 허용.)
  (b) 판정 유실 — ``research/experiments/*_gate.py`` 에 대응 VERDICT.md 가 없으면 실패.
      (정직한 부정 결과도 산출물. 게이트만 돌리고 로깅 안 하면 기록이 사라진다.)
  (c) 하드코딩 판정 — 실험이 리터럴 "PASS"/"FAIL" 판정 문자열을 값으로 박으면 실패.
      (리포터-not-판정기, GUARDRAILS §8. 하드코딩 합격선 자체가 결정론적 과최적.)

무의존·표준 라이브러리만. exit 0 = 위반 없음, exit 1 = 위반.
실행: ``python scripts/check_guardrails.py``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "kr_quant"
EXPERIMENTS = REPO / "research" / "experiments"
LOGS = REPO / "research" / "logs"

# --- (b) 게이트 → 로그 디렉터리 매핑 규약 --------------------------------------
# 기본 규약: ``<name>_gate.py`` → ``research/logs/<name>/VERDICT.md``.
# 아래는 그 규약의 명시적 예외다(현실 반영 — 하드코딩된 합격선이 아니라 규약 맵).
#
# EXEMPT: 알파가 아니라 공용 하버스/라이브러리라 자체 VERDICT 를 갖지 않는 게이트.
GATE_EXEMPT = {
    # prop_gate 는 세 프랍 셋업을 동일 잣대로 재는 공용 리포터 배터리(알파 아님).
    "prop_gate",
}
# OVERRIDE: 로그 디렉터리 이름이 기본 규약과 다른 게이트(판정이 통합·개명된 경우).
GATE_LOG_OVERRIDE = {
    # 베이스 PEAD 게이트. 판정은 pead_concentrated/VERDICT.md 로 통합됨
    # (+ research/logs/PEAD_REFINEMENT_RESULTS.md).
    "pead_gate": "pead_concentrated",
}

# (a) src import 경계 위반 패턴 — ``import research`` / ``from research``.
_IMPORT_RESEARCH = re.compile(r"^\s*(?:import\s+research\b|from\s+research\b)")
# (c) 값으로 박힌 리터럴 판정 문자열(주석·독스트링 문구는 제외하려 따옴표를 요구).
_LITERAL_VERDICT = re.compile(r"""(?<![#])["'](PASS|FAIL|PASSED|FAILED)["']""")


def _iter_py(root: Path):
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


def check_src_no_research_import() -> list[str]:
    """(a) src/kr_quant 가 research 를 import 하면 위반."""
    out = []
    for p in _iter_py(SRC):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _IMPORT_RESEARCH.match(line):
                rel = p.relative_to(REPO)
                out.append(f"[boundary] {rel}:{i} — src must not import research: {line.strip()}")
    return out


def check_gates_have_verdict() -> list[str]:
    """(b) 각 *_gate.py 에 대응 VERDICT.md 가 있어야 한다(EXEMPT/OVERRIDE 규약 적용)."""
    out = []
    for p in sorted(EXPERIMENTS.glob("*_gate.py")):
        stem = p.stem                       # 예: minervini_prop_gate
        if stem in GATE_EXEMPT:
            continue
        log_dir = GATE_LOG_OVERRIDE.get(stem, stem[: -len("_gate")])  # 기본: _gate 제거
        verdict = LOGS / log_dir / "VERDICT.md"
        if not verdict.exists():
            rel = p.relative_to(REPO)
            out.append(
                f"[verdict] {rel} — 대응 VERDICT 없음: "
                f"research/logs/{log_dir}/VERDICT.md 를 만들거나 규약 맵을 갱신하라."
            )
    return out


def check_no_literal_verdict() -> list[str]:
    """(c) 실험이 리터럴 "PASS"/"FAIL" 판정 문자열을 값으로 박으면 위반.

    주석 라인은 통째로 제외한다(예: prop_gate 독스트링의 'PASS/FAIL bool 도 두지 않는다'
    같은 서술은 따옴표가 없어 애초에 매칭 안 되지만, 안전하게 주석도 건너뛴다).
    """
    out = []
    for p in _iter_py(EXPERIMENTS):
        for i, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.split("#", 1)[0]     # 인라인 주석 제거
            if _LITERAL_VERDICT.search(line):
                rel = p.relative_to(REPO)
                out.append(
                    f"[reporter] {rel}:{i} — 리터럴 판정 문자열(리포터-not-판정기 위반): {raw.strip()}"
                )
    return out


def main() -> int:
    violations: list[str] = []
    violations += check_src_no_research_import()
    violations += check_gates_have_verdict()
    violations += check_no_literal_verdict()

    if violations:
        print("가드레일 린트 실패 — 위반 %d건:\n" % len(violations))
        for v in violations:
            print("  " + v)
        print("\n(GUARDRAILS.md §5 — 경계·판정·리포터 규칙)")
        return 1

    print("가드레일 린트 통과 — 경계·VERDICT·리포터 위반 없음.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
