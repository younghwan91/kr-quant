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
    # 생존편향 측정 스크립트. 알파를 심사하는 게 아니라 유니버스 보정이 기존 측정에
    # 미친 영향을 재는 리포터라, 트레이드 표본도 사전등록 config 도 없다.
    "survivorship_bias_gate",
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
        stem = p.stem                       # 예: pullback_prop_gate
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


# (d) earnings 는 정정공시가 새 행으로 쌓이는 버전 테이블이라(PK 에 knowledge_date),
# 그냥 SELECT 하면 (code, period)당 행이 여럿이 되어 하류 패널이 조용히 중복된다.
# storage.read_earnings() 가 as-of 로 한 버전만 고르는 유일한 정문이다.
# SELECT ... FROM earnings 한 줄 형태만 잡는다 — "from earnings drift" 같은 산문이
# 걸리면 규칙이 소음이 되어 아무도 안 본다.
_RAW_EARNINGS = re.compile(r"\bSELECT\b.*\bFROM\s+earnings\b", re.IGNORECASE)
_EARNINGS_READ_EXEMPT = {
    "src/kr_quant/storage.py",          # 정문 자신 + 스키마 문자열
    "tests/test_earnings_asof.py",      # 정문의 테스트 — 버전이 실제로 쌓이는지 직접 확인해야 한다
}


def check_no_raw_earnings_select() -> list[str]:
    """(d) storage 밖에서 earnings 를 직접 SELECT 하면 위반."""
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        if rel in _EARNINGS_READ_EXEMPT or rel.startswith(".venv/"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _RAW_EARNINGS.search(line):
                out.append(
                    f"[earnings] {rel}:{i} — earnings 를 직접 SELECT 했다. "
                    f"kr_quant.storage.read_earnings(con, asof=...) 를 쓸 것 "
                    f"(정정공시 버전이 중복 행으로 새어나온다)."
                )
    return out


# (e) *_gate.py 는 prop_gate 에 config= 를 넘겨 다중검정 원장에 시행을 남겨야 한다.
# 안 넘기면 DSR/t-haircut 이 계산되지 않고, 두 번째 config 를 조용히 돌려도 아무도
# 못 잡는다(GUARDRAILS §4 공백 5). 이 레포에서 반복된 "기능은 있고 쓰는 쪽이 안 씀"을
# 코드로 막는다.
_PROP_GATE_CALL = re.compile(r"\bprop_gate\s*\(")
_HAS_CONFIG_ARG = re.compile(r"\bconfig\s*=")


def check_gates_record_trials() -> list[str]:
    """(e) prop_gate 를 호출하는 *_gate.py 는 config= 를 함께 넘겨야 한다."""
    out = []
    for p in sorted(EXPERIMENTS.glob("*_gate.py")):
        stem = p.stem
        if stem in GATE_EXEMPT:
            continue
        text = p.read_text(encoding="utf-8")
        if not _PROP_GATE_CALL.search(text) or _HAS_CONFIG_ARG.search(text):
            continue
        rel = p.relative_to(REPO)
        out.append(
            f"[trials] {rel} — prop_gate 에 config= 를 안 넘겼다. 사전등록 config 를 "
            f"넘겨야 다중검정 원장(TRIALS.jsonl)에 시행이 남고 DSR 이 계산된다."
        )
    return out


# (f) *_gate.py 는 공용 하버스(prop_gate)를 재사용해야 한다. 자체 배터리를 새로 짜면
# 음성대조·비용 2배 스트레스·손안댄창·R분포·fragility 중 무엇이 빠졌는지 아무도 모른다
# (GUARDRAILS §4 공백 6·7 — 실제로 pead_gate 가 그 상태였다). 동일 잣대 강제.
_USES_HARNESS = re.compile(r"\bprop_gate\s*\(")


def check_gates_use_shared_harness() -> list[str]:
    """(f) *_gate.py 는 prop_gate 하버스를 통과시켜야 한다."""
    out = []
    for p in sorted(EXPERIMENTS.glob("*_gate.py")):
        if p.stem in GATE_EXEMPT:
            continue
        if _USES_HARNESS.search(p.read_text(encoding="utf-8")):
            continue
        rel = p.relative_to(REPO)
        out.append(
            f"[harness] {rel} — prop_gate 를 안 쓴다. 자체 배터리를 짜면 음성대조·"
            f"비용2배·손안댄창·R분포·fragility 중 빠진 게 드러나지 않는다."
        )
    return out


def main() -> int:
    violations: list[str] = []
    violations += check_src_no_research_import()
    violations += check_gates_have_verdict()
    violations += check_no_literal_verdict()
    violations += check_no_raw_earnings_select()
    violations += check_gates_record_trials()
    violations += check_gates_use_shared_harness()

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
