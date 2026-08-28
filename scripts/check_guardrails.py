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
  (d) raw earnings SELECT — 정정공시 버전이 중복 행으로 새어나오는 걸 막는다.
  (e) 원장 우회 — prop_gate/gate_sim **호출 지점마다** config= 를 요구한다(다중검정 N).
  (f) 자체 배터리 — *_gate.py 가 공용 하버스를 안 쓰면 실패.
  (g) raw 가격 SELECT — 유니버스에서 폐지 종목이 조용히 빠지는 걸 막는다.
  (g2) 생존자 전용 조인 — ``supply_demand JOIN stocks`` 는 WHERE 없이도 폐지분을 떨군다.
  (h) 코드 없는 판정 — (b)의 역방향. VERDICT 에 러너가 등록·실재하거나 재현불가 선언 필요.

**이 린트 자체는 tests/test_check_guardrails.py 가 검사한다.** 각 규칙에 위반을 주입해
실제로 실패하는지 확인한다 — 규칙이 죽어도 초록인 상태를 실제로 겪었기 때문이다((e) 참조).

무의존·표준 라이브러리만. exit 0 = 위반 없음, exit 1 = 위반.
실행: ``python scripts/check_guardrails.py``
"""

from __future__ import annotations

import ast
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


#: 스캔에서 뺄 디렉터리. `.claude/worktrees` 는 에이전트용 **레포 복사본**이라
#: 그냥 두면 같은 위반이 두 번 세어지고, 남의 작업 중인 코드로 CI 가 깨진다.
_SKIP_DIRS = {"__pycache__", ".venv", "node_modules", ".claude", ".git"}


def _iter_py(root: Path):
    for p in sorted(root.rglob("*.py")):
        # **루트 기준** 상대경로로 판단한다. 절대경로를 보면 레포가
        # `.claude/...` 아래 놓였을 때 스캔이 통째로 조용히 비어버린다.
        if _SKIP_DIRS & set(p.relative_to(root).parts):
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


# (e) 원장을 남기는 진입점(prop_gate·gate_sim)은 **호출 지점마다** config= 를 받아야 한다.
# 안 넘기면 DSR/t-haircut 이 계산되지 않고, 두 번째 config 를 조용히 돌려도 아무도
# 못 잡는다(GUARDRAILS §4 공백 5). 이 레포에서 반복된 "기능은 있고 쓰는 쪽이 안 씀"을
# 코드로 막는다.
#
# 2026-08-16 재작성. 이전 구현은 **파일 단위**였다 — 파일 어딘가에 `config=` 가 한 번만
# 있으면 같은 파일의 나머지 prop_gate 호출이 전부 면제됐다. 그래서 실제로 다음이 CI 를
# 통과하고 있었다: 사전등록 1회는 config= 를 넘기고, 민감도 스윕 18~27칸은 config 없이
# gate_sim 을 거쳐 원장을 우회 → TRIALS.jsonl 1~2줄 → `n_trials <= 1` → **deflation 0**.
# 즉 린트가 막으려던 바로 그 실패 모드를 린트가 못 봤다. 이제 ast 로 호출 노드를 세고
# 파일 전체 텍스트가 아니라 그 호출의 키워드만 본다. 정규식은 다인자 줄바꿈 호출에서
# 괄호 범위를 못 잡으므로 쓰지 않는다.
_LEDGER_ENTRYPOINTS = ("prop_gate", "gate_sim")


def _called_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def check_gates_record_trials() -> list[str]:
    """(e) prop_gate/gate_sim 호출은 각각 config= 를 동반해야 한다(호출 지점 단위)."""
    out = []
    for p in _iter_py(EXPERIMENTS):
        if p.stem in GATE_EXEMPT:
            continue
        text = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:                      # 파싱 실패는 침묵시키지 않는다
            out.append(f"[trials] {p.relative_to(REPO)} — 파싱 실패: {exc}")
            continue
        # 하버스 자신의 정의부(def prop_gate / def gate_sim)는 대상이 아니다.
        defined_here = {
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in _LEDGER_ENTRYPOINTS
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in _LEDGER_ENTRYPOINTS or name in defined_here:
                continue
            if any(kw.arg == "config" for kw in node.keywords):
                continue
            out.append(
                f"[trials] {p.relative_to(REPO)}:{node.lineno} — {name}() 에 config= 를 "
                f"안 넘겼다. 사전등록 config 를 넘겨야 다중검정 원장(TRIALS.jsonl)에 "
                f"시행이 남고 DSR 이 계산된다(민감도 격자 셀도 시행이다 — GUARDRAILS §6)."
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


# (g) 가격 테이블도 storage 정문(read_prices)으로만 읽는다. 직접 SELECT 하면 유니버스를
# 좁히는 WHERE 가 조용히 들어가고, 생존편향은 검증 스택이 못 잡는다 — walk-forward·
# 음성대조·fragility 는 전부 넘겨받은 트레이드만 보기 때문이다(GUARDRAILS §4 공백 2).
_RAW_PRICES = re.compile(r"\bSELECT\b.*\bFROM\s+(daily_bars_adjusted|daily_bars)\b", re.I)
_PRICE_READ_EXEMPT = {
    "src/kr_quant/storage.py",        # 정문 자신 + 스키마 문자열
    "src/kr_quant/price_adjust.py",   # 조정가 테이블을 *만드는* 쪽 — 정문의 상류
    # 2026-08-16: tests/ 디렉터리 일괄 면제를 파일 단위로 좁혔다. 일괄 면제는
    # 어떤 픽스처 빌더든 생존자 전용 유니버스를 무검사로 스위트에 들일 수 있게 했고,
    # 규칙 (d)가 테스트 파일을 하나씩 명시하는 것과도 비대칭이었다.
    "tests/test_read_prices_guard.py",  # 정문의 가드 자체를 검사 — 원자료를 봐야 한다
    "tests/test_price_adjust.py",       # 조정 로직 검사 — 조정 전/후를 직접 비교한다
    # 2026-08-27: 무결성 감사기. 이 규칙이 막으려는 것은 "유니버스를 조용히 좁히는
    # 로더"인데, 이 파일은 유니버스를 **만들지 않는다** — count(*) 로 중복·OHLC
    # 불변식·단위 정합만 센다. 수익률 시계열을 만드는 코드가 이 파일에 들어오면
    # 면제를 거둬야 한다.
    "scripts/verify_report.py",
}

# 상장 마스터(stocks)와의 INNER JOIN 은 **WHERE 절 없이도** 폐지 종목을 떨군다.
# stocks 는 현재 상장 종목만 담는 마스터이고 폐지분은 delisted_stocks 에 따로 있기
# 때문이다. 즉 지울 조건절이 없고 편향이 조인 자체에 박혀 있어, 읽는 사람 눈에 잘 안
# 띈다. 실제로 이 모양의 로더 3개가 src/ 안에서 생존자 전용 유니버스로 수익률을
# 계산하고 있었다(2026-08-16 에 구현째 삭제). 다시 들어오면 여기서 막는다.
#
# 여러 줄로 쪼갠 SQL 을 잡아야 하므로 줄 단위가 아니라 **파일 전체**에서 찾는다 —
# 규칙 (g)의 한 줄 정규식이 놓치던 지점이 정확히 이것이었다.
_SD_JOIN_STOCKS = re.compile(
    r"\bFROM\s+supply_demand\b[\s\S]{0,200}?\bJOIN\s+stocks\b", re.I
)
_JOIN_RULE_EXEMPT = {
    "scripts/check_guardrails.py",        # 규칙 자신(위 정규식·주석)
    "tests/test_check_guardrails.py",    # 규칙의 회귀 테스트 — 위반 모양을 픽스처로 담는다
}


def check_no_raw_price_select() -> list[str]:
    """(g) storage/price_adjust 밖에서 가격 테이블을 직접 SELECT 하면 위반."""
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        if rel in _PRICE_READ_EXEMPT or rel.startswith(".venv/"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _RAW_PRICES.search(line):
                out.append(
                    f"[universe] {rel}:{i} — 가격 테이블을 직접 SELECT 했다. "
                    f"kr_quant.storage.read_prices() 를 쓸 것 (상장폐지 종목 포함을 "
                    f"로딩 시점에 검사한다)."
                )
    return out


def check_no_survivor_only_join() -> list[str]:
    """(g2) supply_demand 를 상장 마스터(stocks)와 INNER JOIN 하면 위반."""
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        # 규칙 자신과 그 회귀 테스트는 위반 모양을 문자열로 들고 있어야 한다.
        if rel in _JOIN_RULE_EXEMPT or rel.startswith(".venv/"):
            continue
        m = _SD_JOIN_STOCKS.search(p.read_text(encoding="utf-8"))
        if m is None:
            continue
        line_no = p.read_text(encoding="utf-8")[: m.start()].count("\n") + 1
        out.append(
            f"[universe] {rel}:{line_no} — supply_demand 를 stocks 와 INNER JOIN 했다. "
            f"stocks 는 현재 상장 마스터라 이 조인만으로 폐지 종목이 유니버스에서 "
            f"통째로 빠진다(WHERE 절이 없어도 그렇다). 폐지분이 필요하면 "
            f"delisted_stocks 를 함께 읽거나 조인을 걷어내라."
        )
    return out


# (h) 규칙 (b)의 **역방향** — 모든 VERDICT 에는 그 숫자를 만든 러너가 있어야 한다.
#
# (b)는 "이 코드에 판정이 있나"만 묻는다. 그래서 러너가 삭제되면 판정만 홀로 남고
# 아무도 못 잡는다 — 실제로 `research/logs/minervini_prop/VERDICT.md` 가 601트레이드
# 사전등록 배터리 전문을 싣고 있는데 그걸 만든 코드는 레포에 없다. 재현 불가는 그 자체로
# 죄가 아니지만(기각된 알파의 구현을 지우는 건 정당하다) **선언되지 않은** 재현 불가는
# 독자를 속인다. 그래서 둘 중 하나를 요구한다: 등록된 러너가 실재하거나, VERDICT 가
# 재현 불가를 명시하거나.
VERDICT_RUNNER = {
    "pead_concentrated": "pead_concentrated_gate.py",
    "pullback_prop": "pullback_prop_gate.py",
    "survivorship_bias": "survivorship_bias_gate.py",
    # 게이트 접미사를 안 쓰는 러너도 등록하면 된다 — 규칙의 목적은 파일명 규약이
    # 아니라 "판정에 코드가 붙어 있는가"다.
    "contrarian_retail": "contrarian_validate.py",
    "regime_switch": "regime_switch.py",
    "inst_flow_accel": "inst_flow_accel_gate.py",
}
# VERDICT 가 러너 없이 존재해도 되는 유일한 조건: 아래 문구로 재현 불가를 선언한다.
_IRREPRODUCIBLE_MARKER = "재현 불가 고지"


# (i) 커밋 신원이 레포 설정과 다르면 실패.
#
# 커밋 시 `git -c user.email=...` 로 신원을 덮어쓰면 의도치 않은 주소가 공개
# 이력에 박히고, 이력은 되돌리기 어렵다. "푸시 전에 확인하는 습관" 은 부탁이지
# 규칙이 아니다(§0) — 레포 설정을 상속하게 두고 어긋나면 여기서 막는다.
def check_commit_identity() -> list[str]:
    """(i) 최근 커밋의 author/committer 이메일이 레포 설정과 같은가."""
    import subprocess  # noqa: PLC0415 — 이 규칙에서만 쓴다

    def _git(*args: str) -> str:
        try:
            return subprocess.run(("git", *args), cwd=REPO, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001 — git 이 없거나 저장소가 아니면 검사 생략
            return ""

    want = _git("config", "--get", "user.email")
    if not want:
        return []                     # 설정이 없으면 비교할 기준이 없다
    out = _git("log", "-30", "--format=%h %ae %ce")
    bad = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        sha, ae, ce = parts
        if ae != want or ce != want:
            bad.append(
                f"[identity] {sha} — 커밋 신원이 레포 설정과 다르다 "
                f"(author={ae} committer={ce}, 설정={want}). "
                f"`git -c user.email=...` 로 덮어쓰지 말 것 — 레포 설정을 그대로 쓴다."
            )
    return bad


def check_verdicts_have_runner() -> list[str]:
    """(h) 각 research/logs/<alpha>/VERDICT.md 에 러너가 등록·실재하거나 재현불가 선언이 있어야 한다."""
    out = []
    if not LOGS.is_dir():
        return out
    for verdict in sorted(LOGS.glob("*/VERDICT.md")):
        alpha = verdict.parent.name
        runner = VERDICT_RUNNER.get(alpha)
        if runner and (EXPERIMENTS / runner).exists():
            continue
        if _IRREPRODUCIBLE_MARKER in verdict.read_text(encoding="utf-8"):
            continue
        rel = verdict.relative_to(REPO)
        detail = (
            f"등록된 러너 {runner} 가 없다" if runner
            else "VERDICT_RUNNER 에 등록되지 않았다"
        )
        out.append(
            f"[orphan-verdict] {rel} — {detail}. 판정을 만든 러너를 등록하거나, "
            f"러너를 지웠다면 VERDICT 에 '{_IRREPRODUCIBLE_MARKER}'를 적어 "
            f"재현 불가를 명시하라(선언 없는 재현 불가는 독자를 속인다)."
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
    violations += check_no_raw_price_select()
    violations += check_no_survivor_only_join()
    violations += check_verdicts_have_runner()
    violations += check_commit_identity()

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
