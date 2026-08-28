#!/usr/bin/env python3
"""워크트리 머지 게이트 — "검사가 통과했다" 를 믿지 않고 **검사가 자기 버그를 잡는가**를 본다.

이 저장소가 실제로 겪은 일이 이 도구의 존재 이유다.

* 열 정렬 버그 5종을 주입했더니 기존 검사 21개가 **전부 초록**이었다. 렌더가 마지막에
  줄 전체를 화면 폭으로 패딩하므로 셀이 통째로 빠져도 모든 행의 표시 폭은 여전히
  정확히 ``width`` 다 — "모든 행의 폭이 같은가" 는 그 부류를 구조적으로 못 잡는다.
* 그 뒤 새로 쓴 pty 검사 셋이 **자기 버그를 못 잡았다.** 좀비 검사는 배수 스레드와
  fd 경합으로 앱이 아니라 테스트 사정으로 끝났고, 폭 검사는 잡아야 할 케이스를 스스로
  ``continue`` 로 건너뛰었고, 리사이즈 검사는 ``ioctl`` 만으로는 SIGWINCH 가 안 가서
  ``KEY_RESIZE`` 가 아예 도달하지 않았다.
* 별개 세션이 **독립적으로 같은 폭 불변식 함정을 밟았다.**

즉 "테스트가 초록" 의 정보량은 거의 0 이다. 그래서 이 하네스의 중심은 §3 주입 검증이다 —
워크트리가 **자기 변이 목록을 선언**하고, 하네스가 그 변이를 소스에 실제로 넣어
선언한 검사가 **정말 빨개지는지** 확인한다. GUARDRAILS §5 각주의 교훈과 같다:
"규칙을 추가할 때는 그 규칙이 잡아야 할 위반을 실제로 주입해봐야 한다."

리포터-not-판정기(GUARDRAILS §8)의 예외로 이 도구는 PASS/REJECT 를 낸다 — 알파 판정이
아니라 머지 판정이고, 근거를 항목별로 다 적는다. 애매한 것은 ``WARN`` 으로 남겨
사람이 읽게 한다.

사용::

    python scripts/verify_merge.py <워크트리 경로>
    python scripts/verify_merge.py <워크트리> --only mutations,render
    python scripts/verify_merge.py <워크트리> --mutations /경로/보고서에서_옮긴.toml
    python scripts/verify_merge.py --self-test          # 하네스가 정말 REJECT 를 내는가

변이 선언 형식 — 워크트리의 ``tests/mutations.toml`` (또는 ``--mutations``)::

    [[mutation]]
    id      = "drop-accel-cell"
    why     = "가속 셀을 통째로 빼면 그 뒤 열이 한 칸씩 밀린다"
    guards  = ["tests/test_tui_flow_view.py::test_data_cells_sit_under_their_headers"]
    file    = "src/kr_quant/tui/flow_view.py"
    find    = "pad(fmt_pct(r.get(\"accel\")), 9, True),"
    replace = ""

``find`` 는 파일 안에서 **정확히 한 번** 나와야 한다(여러 번이면 무엇을 바꿨는지
모호해진다). 주입 후에는 원본 바이트로 되돌리고 ``git status`` 로 확인한다.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:                                    # 3.11+ 는 표준, 3.10 은 tomli (pytest 의존성)
    import tomllib
except ModuleNotFoundError:             # pragma: no cover — 인터프리터에 달렸다
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_BASE = "feat/sector-flow"
DEFAULT_REPORT = "~/Documents/kr-quant-reports/latest"

PASS, REJECT, WARN, SKIP = "PASS", "REJECT", "WARN", "SKIP"


class Section:
    """항목 하나의 판정과 근거.

    dataclass 를 쓰지 않는다 — ``spec_from_file_location`` 으로 이 파일을 불러오면
    (레포의 `test_research_imports` 가 그렇게 한다) 모듈이 `sys.modules` 에 없어
    dataclasses 가 `AttributeError: 'NoneType' object has no attribute '__dict__'`
    로 죽는다. 하네스가 레포 자신의 검사를 깨면 그 하네스는 못 쓴다.
    """

    def __init__(self, name: str, status: str = PASS,
                 lines: list[str] | None = None):
        self.name = name
        self.status = status
        self.lines: list[str] = lines if lines is not None else []

    def say(self, msg: str) -> None:
        self.lines.append(msg)

    def fail(self, msg: str) -> None:
        self.status = REJECT
        self.lines.append("REJECT — " + msg)

    def warn(self, msg: str) -> None:
        if self.status != REJECT:
            self.status = WARN
        self.lines.append("WARN — " + msg)

    def skip(self, msg: str) -> None:
        if self.status == PASS:
            self.status = SKIP
        self.lines.append("SKIP — " + msg)


# --------------------------------------------------------------------------- 공통

def run(cmd: list[str], cwd: Path, env: dict | None = None,
        timeout: int = 1200) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout, env=env)


def git(cwd: Path, *args: str) -> str:
    p = run(["git", *args], cwd, timeout=60)
    return p.stdout.strip()


def tail(text: str, n: int = 2000) -> str:
    text = text.strip()
    return text if len(text) <= n else "…\n" + text[-n:]


def py_env(worktree: Path) -> dict:
    """워크트리 소스를 **확실히** 임포트하게 하는 환경.

    함정: 이 저장소의 ``.venv`` 는 ``.pth`` 로 메인 저장소의 ``src`` 를 가리킨다.
    그대로 돌리면 워크트리 안에서 테스트를 돌려도 임포트되는 건 **메인 소스**라,
    워크트리를 한 줄도 안 고쳐도 초록이 나온다(이 저장소의 상습 실패 모드 —
    "기능은 있고 쓰는 쪽이 안 씀" 의 검증판). PYTHONPATH 가 site-packages 보다
    앞서므로 이걸로 이기고, §임포트출처 검사가 실제로 이겼는지 확인한다.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src") + os.pathsep + str(worktree)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONHASHSEED", None)
    return env


def interpreter(worktree: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    for cand in (REPO / ".venv/bin/python", worktree / ".venv/bin/python"):
        if cand.exists():
            return str(cand)
    return sys.executable


# --------------------------------------------------------------------------- 1. 워크트리

def sec_worktree(wt: Path, base: str, py: str) -> tuple[Section, list[str]]:
    """워크트리가 검증 가능한 상태인가 + 임포트가 정말 워크트리에서 오는가."""
    s = Section("worktree")
    if not (wt / ".git").exists():
        s.fail(f"{wt} 는 git 워크트리가 아니다")
        return s, []
    dirty = git(wt, "status", "--porcelain")
    if dirty:
        s.warn("작업 트리가 깨끗하지 않다 — 커밋 안 된 변경 위에서 판정한다:\n"
               + "\n".join("      " + ln for ln in dirty.splitlines()[:20]))
    head = git(wt, "rev-parse", "--short", "HEAD")
    mb = git(wt, "merge-base", base, "HEAD")
    if not mb:
        s.fail(f"기준 {base} 와의 merge-base 를 못 찾았다")
        return s, []
    commits = git(wt, "log", "--format=%H", f"{mb}..HEAD").splitlines()
    s.say(f"HEAD={head}  기준={base}({git(wt, 'rev-parse', '--short', mb)})  "
          f"새 커밋 {len(commits)}건")
    for line in git(wt, "log", "--format=  %h %s", f"{mb}..HEAD").splitlines():
        s.say(line)
    changed = git(wt, "diff", "--name-only", f"{mb}..HEAD").splitlines()
    if dirty:
        changed += [ln[3:] for ln in dirty.splitlines()]
    s.say(f"바뀐 파일 {len(set(changed))}개: " + ", ".join(sorted(set(changed))[:12]))

    # 임포트 출처 — 여기서 틀리면 아래 모든 초록이 거짓말이다.
    p = run([py, "-c", "import kr_quant, sys; print(kr_quant.__file__)"],
            wt, py_env(wt), timeout=120)
    got = p.stdout.strip()
    if not got.startswith(str(wt.resolve())):
        s.fail(f"kr_quant 가 워크트리 밖에서 임포트된다: {got or p.stderr.strip()}\n"
               f"      (이 상태로 돌린 테스트는 워크트리가 아니라 다른 소스를 검사한다)")
    else:
        s.say(f"임포트 출처 확인: {got}")
    return s, commits


# --------------------------------------------------------------------------- 2. 신원

def sec_identity(wt: Path, commits: list[str]) -> Section:
    """새 커밋의 author/committer 가 레포 설정과 같은가.

    공개 레포다. 2026-08-29 에 ``git -c user.email=<회사주소>`` 로 찍힌 커밋 52건이
    올라간 적이 있다. check_guardrails 규칙 (i) 와 같은 검사지만, 여기서는 **기준
    브랜치 이후의 새 커밋만** 본다(규칙 (i) 는 최근 30건 고정이라 새 커밋이 30건을
    넘으면 오래된 것부터 시야에서 사라진다).
    """
    s = Section("identity")
    want_mail = git(wt, "config", "--get", "user.email")
    want_name = git(wt, "config", "--get", "user.name")
    if not want_mail:
        s.warn("레포에 user.email 설정이 없어 비교 기준이 없다")
        return s
    if not commits:
        s.say("새 커밋 없음 — 검사할 대상이 없다")
        return s
    s.say(f"기준 신원: {want_name} <{want_mail}>")
    for sha in commits:
        line = git(wt, "log", "-1", "--format=%h|%an|%ae|%cn|%ce", sha)
        h, an, ae, cn, ce = line.split("|")
        if ae != want_mail or ce != want_mail:
            s.fail(f"{h} 신원이 다르다 (author={ae} committer={ce}) — "
                   f"`git -c user.email=…` 금지, 레포 설정을 그대로 쓴다")
        elif an != want_name or cn != want_name:
            s.warn(f"{h} 이름이 다르다 (author={an} committer={cn})")
        else:
            s.say(f"  {h} ok")
    return s


# --------------------------------------------------------------------------- 3. 기본

#: CI 와 **같은 범위**를 본다. `src/ tests/` 만 보고 "통과" 라고 적었다가 CI 가
#: 빨간 채로 남은 적이 있다 — 범위가 다르면 통과 보고가 거짓말이 된다.
RUFF_PATHS = ("src/", "tests/", "research/", "scripts/", "examples/")


def _ruff(py: str, cwd: Path, env: dict) -> set[str]:
    ruff = Path(py).parent / "ruff"
    cmd = ([str(ruff)] if ruff.exists() else [py, "-m", "ruff"]) + \
        ["check", "--output-format", "concise", *RUFF_PATHS]
    p = run(cmd, cwd, env, timeout=300)
    out = set()
    for ln in p.stdout.splitlines():
        m = re.match(r"^(\S+?):\d+:\d+: (\S+) (.*)$", ln.strip())
        if m:                      # 줄번호는 뺀다 — 무관한 편집으로도 밀린다
            out.add(f"{m.group(1)}: {m.group(2)} {m.group(3)}")
    return out


def sec_lint(wt: Path, py: str, base_sha: str, scratch: Path) -> Section:
    """ruff — **이 브랜치가 새로 넣은** 위반만 REJECT.

    기준이 이미 빨간 상태일 수 있다(실제로 그랬다: 79e14df 에 8건이 남아 있었고
    뒤늦게 별도 커밋으로 고쳤다). 그걸 워크트리에 물리면 남의 빚으로 남을
    REJECT 하게 된다 — 판정이 틀리는 것보다 나쁜 건, 판정을 안 믿게 되는 것이다.
    """
    s = Section("lint")
    env = py_env(wt)
    now = _ruff(py, wt, env)
    base_dir = scratch / "lint_base"
    base_dir.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "archive", base_sha], cwd=str(wt),
                         capture_output=True, timeout=120)
    subprocess.run(["tar", "-x", "-C", str(base_dir)], input=tar.stdout, timeout=120)
    was = _ruff(py, base_dir, py_env(base_dir))
    new, gone, kept = now - was, was - now, now & was
    if new:
        s.fail(f"이 브랜치가 새로 넣은 ruff 위반 {len(new)}건\n"
               + "\n".join("      " + v for v in sorted(new)))
    else:
        s.say(f"새 위반 없음 (검사 경로: {' '.join(RUFF_PATHS)})")
    if kept:
        s.warn(f"기준에서 물려받은 위반 {len(kept)}건 — 이 워크트리의 책임이 "
               f"아니지만 머지 후에도 남는다:\n"
               + "\n".join("      " + v for v in sorted(kept)))
    if gone:
        s.say(f"고쳐진 위반 {len(gone)}건")
    return s


def sec_guardrails(wt: Path, py: str) -> Section:
    s = Section("guardrails")
    p = run([py, "scripts/check_guardrails.py"], wt, py_env(wt), timeout=300)
    if p.returncode != 0:
        s.fail("check_guardrails 실패\n" + tail(p.stdout + p.stderr))
    else:
        s.say(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "ok")
    return s


def sec_tests(wt: Path, py: str) -> Section:
    s = Section("tests")
    p = run([py, "-m", "pytest", "-q"], wt, py_env(wt), timeout=1800)
    last = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
    if p.returncode != 0:
        s.fail("pytest 실패\n" + tail(p.stdout + p.stderr, 3000))
    else:
        s.say(last[0])
    return s


# --------------------------------------------------------------------------- 4. 주입 검증

def new_test_functions(wt: Path, base_sha: str) -> set[str]:
    """기준 이후 **추가·수정된** 검사 함수의 nodeid 집합.

    diff 의 변경 줄 번호를 새 파일의 AST 에 겹쳐 그 줄을 품은 ``def test_*`` 를 찾는다.
    파일 단위로 세면 "한 줄 고친 파일의 검사 40개" 가 전부 미선언으로 뜬다.
    """
    out: set[str] = set()
    files = [f for f in git(wt, "diff", "--name-only", f"{base_sha}..HEAD").splitlines()
             if f.startswith("tests/") and f.endswith(".py")]
    files += [ln[3:] for ln in git(wt, "status", "--porcelain").splitlines()
              if ln[3:].startswith("tests/") and ln[3:].endswith(".py")]
    for f in sorted(set(files)):
        path = wt / f
        if not path.exists():
            continue
        diff = git(wt, "diff", "-U0", base_sha, "--", f)
        touched: set[int] = set()
        for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff, re.M):
            start, n = int(m.group(1)), int(m.group(2) or 1)
            touched.update(range(start, start + max(n, 1)))
        if not touched:                      # 새 파일이면 전부 새 검사다
            touched = set(range(1, len(path.read_text("utf-8").splitlines()) + 2))
        try:
            tree = ast.parse(path.read_text("utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name.startswith("test"):
                lo = min([node.lineno] + [d.lineno for d in node.decorator_list])
                if touched & set(range(lo, (node.end_lineno or lo) + 1)):
                    out.add(f"{f}::{node.name}")
    return out


def sec_mutations(wt: Path, py: str, base_sha: str, spec_path: Path | None) -> Section:
    """**핵심.** 워크트리가 선언한 변이를 실제로 주입해 선언한 검사가 빨개지는가."""
    s = Section("mutations")
    spec = spec_path or (wt / "tests/mutations.toml")
    new_tests = new_test_functions(wt, base_sha)
    if new_tests:
        s.say(f"기준 이후 추가·수정된 검사 {len(new_tests)}개")

    if not spec.exists():
        if new_tests:
            s.warn(f"변이 선언({spec}) 이 없는데 새 검사가 {len(new_tests)}개 있다 — "
                   f"이 검사들이 자기 버그를 잡는지 아무도 확인하지 않았다:\n"
                   + "\n".join("      " + t for t in sorted(new_tests)))
        else:
            s.say("변이 선언도 없고 새 검사도 없다")
        return s

    try:
        muts = tomllib.loads(spec.read_text("utf-8")).get("mutation", [])
    except tomllib.TOMLDecodeError as e:
        s.fail(f"{spec} 파싱 실패: {e}")
        return s
    if not muts:
        s.fail(f"{spec} 에 [[mutation]] 이 하나도 없다")
        return s

    before = git(wt, "status", "--porcelain")
    guarded: set[str] = set()
    env = py_env(wt)

    for m in muts:
        mid = m.get("id", "?")
        # 한 버그가 두 군데를 동시에 바꿔야 재현되는 경우가 있다(예: 리사이즈가
        # 도움말을 닫던 버그는 `else: help=False` 와 KEY_RESIZE 가드가 한 쌍이다).
        # 한 곳만 바꾸면 다른 곳이 막아서 "검사가 못 잡는다" 는 **거짓 판정**이 난다.
        edits = m.get("edits") or [{"file": m.get("file"), "find": m.get("find"),
                                    "replace": m.get("replace", "")}]
        if "guards" not in m or any(not e.get("file") or e.get("find") is None
                                    for e in edits):
            s.fail(f"[{mid}] 필드 누락 — guards 와 (file, find) 가 필요하다")
            continue
        saved: dict[Path, bytes] = {}
        skip = False
        for e in edits:
            target = wt / e["file"]
            if not target.exists():
                s.fail(f"[{mid}] 대상 파일이 없다: {e['file']}")
                skip = True
                break
            n = target.read_text("utf-8").count(e["find"])
            if n == 0:
                # 애매한 경우다 — 사람이 봐야 한다(리포터 원칙). 자기가 선언한
                # 변이라면 오타이고, 기준 브랜치에서 물려받은 변이라면 리팩터로
                # 앵커가 사라진 것이다. 둘은 코드를 봐야 갈린다.
                s.warn(f"[{mid}] 앵커가 {e['file']} 에 없어 이 변이는 "
                       f"**돌지 않았다**. 자기 선언이면 오타이고, 기준에서 "
                       f"물려받은 것이면 리팩터로 앵커가 사라진 것이다 — 그 회귀 "
                       f"검사가 아직 무는지 사람이 확인하라 ({m.get('why', '')})")
                skip = True
                break
            if n != 1:
                s.fail(f"[{mid}] find 가 {e['file']} 에서 {n}번 나온다 — "
                       f"정확히 1번이어야 무엇을 바꿨는지가 확정된다")
                skip = True
                break
        if skip:
            continue
        guards = list(m["guards"])
        guarded.update(guards)

        # ① 깨끗한 상태에서 가드가 초록인가 — 늘 빨간 검사는 아무것도 증명하지 않는다.
        clean = run([py, "-m", "pytest", "-q", *guards], wt, env, timeout=900)
        if clean.returncode != 0:
            if "no tests ran" in clean.stdout or "ERROR" in clean.stdout:
                # 기준에서 물려받은 변이인데 워크트리가 그보다 먼저 갈라져 나가
                # 그 검사가 아직 없는 것일 수 있다. 오타와 구분이 안 되므로 경고다.
                s.warn(f"[{mid}] 가드를 이 워크트리에서 **찾지 못했다** — nodeid "
                       f"오타이거나, 기준이 그 검사를 워크트리 분기 뒤에 추가한 "
                       f"것이다. 머지하면 붙으므로 머지 후 다시 돌려라: "
                       f"{', '.join(guards)}")
            else:
                s.fail(f"[{mid}] 가드가 **주입 전에도** 실패한다 (이미 깨진 "
                       f"검사다)\n" + tail(clean.stdout, 1200))
            continue

        # ② 주입 → 빨개져야 한다.
        try:
            for e in edits:
                target = wt / e["file"]
                # 한 파일을 두 번 고칠 수 있다 — **첫 스냅샷만** 원본이다.
                # 편집마다 새로 읽어 쌓으면 두 번째 스냅샷이 이미 변이본이라
                # 복구가 변이를 되살린다(원상복구 검사가 실제로 이걸 잡았다).
                saved.setdefault(target, target.read_bytes())
                target.write_text(
                    target.read_text("utf-8").replace(e["find"], e["replace"], 1),
                    encoding="utf-8")
            hit = run([py, "-m", "pytest", "-q", *guards], wt, env, timeout=900)
        finally:
            for target, original in saved.items():   # 무슨 일이 있어도 원상복구
                target.write_bytes(original)
        if hit.returncode == 0:
            s.fail(f"[{mid}] 변이를 넣었는데 가드가 **초록이다** — 이 검사는 "
                   f"{m.get('why', '이 버그')} 를 못 잡는다\n"
                   f"      guards: {', '.join(guards)}")
        else:
            first = next((ln for ln in hit.stdout.splitlines()
                          if ln.startswith("FAILED") or " failed" in ln), "")
            s.say(f"  [{mid}] 잡힘 — {m.get('why', '')} ({first.strip()})")

    after = git(wt, "status", "--porcelain")
    if after != before:
        s.fail("주입 후 워크트리가 원상복구되지 않았다:\n"
               + "\n".join("      " + ln for ln in after.splitlines()[:20]))
    else:
        s.say("원상복구 확인 (git status 가 주입 전과 같다)")

    uncovered = {t for t in new_tests
                 if not any(t == g or t.split("::")[-1] == g.split("::")[-1]
                            for g in guarded)}
    if uncovered:
        s.warn(f"변이를 하나도 선언하지 않은 새 검사 {len(uncovered)}개 — "
               f"자기 버그를 잡는지 미확인:\n"
               + "\n".join("      " + t for t in sorted(uncovered)))
    return s


# --------------------------------------------------------------------------- 5. 렌더 스모크

RENDER_SMOKE = r'''
"""실데이터 렌더 스모크 — 예외·행폭·헤더정렬·숫자잘림."""
import bisect, pathlib, sys
from kr_quant.tui.flow_app import load
from kr_quant.tui import flow_view as V

report = sys.argv[1]
d = load(report)
bad = []

# 값 기대표 — 헤더 이름 → (원본 키, 서식). **선언이지 구현이 아니다.**
# 열의 뜻이 바뀌면(예: 미실현이 숫자에서 발산 막대로) 이 표가 낡는 것이지
# 화면이 틀린 게 아니다. 그래서 표에 없는 헤더는 **구조 검사만** 받는다
# (폭·구분칸·자릿수 중간 잘림). 워크트리별 갱신은 --mutations 파일의
# [render.expect] 로 넣는다.
EXPECT = {
    "섹터": ("sector", "str"), "종목[수]": ("n_all", "int"),
    "임펄스[억]": ("flow", "amt"), "가속[%p]": ("accel", "pct2"),
    "수익률[%]": ("ret", "pct2"), "미실현[%p]": ("x", "pct1"),
    "G[0~1]": ("G", "f2"),
    "종목": ("name", "str"), "코드": ("code", "raw"),
    "순매수[억]": ("flow", "amt"), "시총대비[%p]": ("a", "pct2"),
}
if len(sys.argv) > 2 and sys.argv[2]:
    import json
    over = json.loads(pathlib.Path(sys.argv[2]).read_text("utf-8"))
    for k, v in over.items():
        if v is None:
            EXPECT.pop(k, None)          # 이 워크트리에서 뜻이 바뀐 열
        else:
            EXPECT[k] = (v[0], v[1])


def want(r, name):
    spec = EXPECT.get(name)
    if spec is None:
        return None                      # 선언이 없으면 구조 검사만
    key, how = spec
    v = r.get(key)
    if how == "str":
        return v if v is not None else "—"
    if how == "raw":
        return v if v is not None else ""
    if how == "int":
        return str(v if v is not None else "—")
    if how == "amt":
        return V.fmt_amt(v)
    if how == "pct1":
        return V.fmt_pct(v, 1) if v is not None else "—"
    if how == "pct2":
        return V.fmt_pct(v) if v is not None else "—"
    if how == "f2":
        return f"{v:.2f}" if v is not None else "—"
    return None

def W(s):
    return sum(V.cell_width(c) for c in s)

_CELLS = {}

def cell_starts(line):
    """문자 i 의 시작 표시칸. 줄마다 한 번만 만든다 — 열마다 줄을 다시 훑으면
    조합 4,620개 × 열 12개 × 줄길이 200 이라 검증이 분 단위로 늘어진다."""
    v = _CELLS.get(line)
    if v is None:
        st, c = [], 0
        for ch in line:
            st.append(c)
            c += V.cell_width(ch)
        st.append(c)
        v = _CELLS[line] = st
        if len(_CELLS) > 20000:
            _CELLS.clear()
    return v

def slice_cells(line, start, width, starts=None):
    """[start, start+width) 표시칸에 **시작점이 들어오는** 문자들."""
    st = starts if starts is not None else cell_starts(line)
    lo = bisect.bisect_left(st, start, 0, len(line))
    hi = bisect.bisect_left(st, start + width, 0, len(line))
    return line[lo:hi]

def col3(c):
    """열 정의에서 (헤더, 폭, 우측정렬) 만 꺼낸다.

    워크트리마다 열 표현이 다르다 — 튜플 3칸일 수도, `Col(header,width,right,fn)`
    처럼 필드가 더 붙은 namedtuple 일 수도 있다. 3칸으로 언패킹하면 하네스가
    **워크트리의 리팩터 때문에** 터진다(실제로 터졌다). 앞 세 칸만 본다.
    """
    return c[0], c[1], c[2]

def check_block(tag, lines, width, cols, nhead, rows, want_of):
    for i, ln in enumerate(lines):
        if cell_starts(ln)[-1] != width:
            bad.append(f"{tag} 행{i} 표시폭 {cell_starts(ln)[-1]} != {width}: "
                       f"{ln[:60]!r}")
            return
    head = lines[nhead - 1]
    for c in cols:
        name, w, right = col3(c)
        span = V.col_span(cols, name)
        if span is None or span[0] + w > width:
            continue
        if slice_cells(head, *span) != V.pad(name, w, right):
            bad.append(f"{tag} 헤더 '{name}' 이 자기 칸에 없다")
    spans = {c[0]: V.col_span(cols, c[0]) for c in cols}
    for r, ln in zip(rows, lines[nhead:]):
        st = cell_starts(ln)
        for c in cols:
            name, w, right = col3(c)
            span = spans[name]
            if span is None:
                continue
            start = span[0]
            # ① 열 사이 구분 공백이 살아 있는가 — 셀이 하나 빠지면 여기가 먼저 깨진다.
            if 0 < start <= width and slice_cells(ln, start - 1, 1, st) not in ("", " "):
                bad.append(f"{tag} '{name}' 앞 구분칸이 공백이 아니다: {ln[:70]!r}")
            got = slice_cells(ln, start, min(w, max(width - start, 0)), st)
            val = want_of(r, name)
            if val is None:
                continue
            if start + w <= width:
                # ② 헤더와 셀이 같은 칸에 있는가 (strip 금지 — 여백이 어긋남을 먹는다)
                if got != V.pad(val, w, right):
                    bad.append(f"{tag} '{name}' 셀이 헤더와 어긋났다: "
                               f"{got!r} != {V.pad(val, w, right)!r}")
            elif got.strip():
                # ③ 온전히 안 들어가는 열은 **아무것도** 안 보여야 한다.
                #    반쪽 숫자는 안 보이는 것보다 나쁘다(-1,360 이 -1 로 보였다).
                bad.append(f"{tag} '{name}' 이 {got!r} 로 잘려 보인다 (참값 {val!r})")

st = V.State(d)
widths = [40, 80, 100, 132, 150, 170, 200]
n = 0
seen_sort_keys = set()

# SORT_COL·NAME_SORT_COL 이 실제 열 이름과 이어져 있는가. 헤더 이름 한 글자만
# 바꿔도 여기가 끊기고, 그래도 모든 행의 폭은 그대로라 폭 검사는 초록이다.
all_headers = set()
for _w in widths:
    for _wi in range(len(V.WINDOWS)):
        st.wi = _wi
        all_headers |= {c[0] for c in V.table_cols(st, _w)}
st.wi = 0
for _k, _h in V.SORT_COL.items():
    if _h not in all_headers:
        bad.append(f"SORT_COL[{_k}]={_h!r} 라는 열이 어느 폭에도 없다")
for _k, _h in V.NAME_SORT_COL.items():
    if _h not in {c[0] for c in V.names_cols()}:
        bad.append(f"NAME_SORT_COL[{_k}]={_h!r} 라는 열이 종목 목록에 없다")
for width in widths:
    for wi in range(len(V.WINDOWS)):
        for mi in range(len(st.markets)):
            for ai in range(len(V.ACTORS)):
                for si in range(len(V.SORTS)):
                    st.wi, st.mi, st.ai, st.si = wi, mi, ai, si
                    st.row = 0
                    n += 1
                    try:
                        rows = st.rows()
                        lines, thin, nhead = V.table_lines(st, width, 40)
                        hdr = V.header_lines(st, width)
                        det = V.detail_lines(st, width)
                    except Exception as e:
                        bad.append(f"w{width} {V.WINDOWS[wi]}/{st.market}/"
                                   f"{V.ACTORS[ai][1]}/{V.SORTS[si][1]} 예외 {e!r}")
                        continue
                    if len(thin) != len(lines):
                        bad.append(f"w{width} thin 길이 {len(thin)} != 행 {len(lines)}")
                    for ln in hdr + det:
                        if W(ln) != width:
                            bad.append(f"w{width} 헤더/상세 폭 {W(ln)} != {width}")
                    cols = V._fit(V.table_cols(st, width), width)
                    tag = (f"w{width} {V.WINDOWS[wi]}/{st.market}/"
                           f"{V.ACTORS[ai][1]}/{V.SORTS[si][1]}")
                    check_block(tag, lines, width, cols, nhead, rows, want)
                    # 주체 일관성 — 화면의 임펄스·가속이 **선택된 주체**의 숫자인가.
                    # 위 want() 는 렌더가 쓴 것과 같은 투영 행에서 값을 다시 만드므로
                    # "기관 값이 그대로 남았다" 부류에는 동어반복이다. 그래서 여기서는
                    # **생 페이로드**로 되짚는다(투영을 거치지 않은 원본).
                    raw = ((d.get("blocks") or {}).get(
                        f"{V.WINDOWS[wi]}|{st.market}") or {}).get("rows") or []
                    by_sec = {x.get("sector"): x for x in raw}
                    akey = V.ACTORS[ai][0]
                    for r in rows:
                        src = by_sec.get(r.get("sector"))
                        if src is None:
                            continue
                        if r.get("flow") != src.get(akey):
                            bad.append(f"{tag} '{r.get('sector')}' 임펄스가 "
                                       f"{r.get('flow')} — 이 주체의 값은 "
                                       f"{src.get(akey)} 다 (다른 주체 값이 남았나)")
                            break
                        cap, fl = src.get("cap"), src.get(akey)
                        exp = (fl / cap * 100) if (cap and fl is not None) else None
                        if r.get("accel") != exp:
                            bad.append(f"{tag} '{r.get('sector')}' 가속이 "
                                       f"{r.get('accel')} != {exp}")
                            break
                    # 정렬 하이라이트가 **실제 정렬 열**을 가리키는가
                    span = V.sort_span(st, width)
                    key, _f = st.effective_sort(rows)
                    name = V.SORT_COL.get(key)
                    seen_sort_keys.add(key)
                    fitted = [c[0] for c in cols]
                    if span is None and name in fitted and V.WINDOWS[wi] != "종합":
                        # 헤더 이름만 바뀌어도 SORT_COL 조회가 조용히 깨진다 —
                        # 그러면 "무엇으로 줄세웠나" 표시가 사라지는데 폭 검사는
                        # 전부 초록이다(열은 여전히 제자리다).
                        bad.append(f"{tag} 정렬 열 '{name}' 이 화면에 있는데 "
                                   f"하이라이트가 없다 (SORT_COL 이 끊겼나)")
                    if span is not None:
                        if name is None:
                            bad.append(f"{tag} 정렬키 {key} 에 SORT_COL 항목이 없다")
                        elif slice_cells(lines[0], *span).strip() != name.strip():
                            bad.append(f"{tag} 정렬 하이라이트가 다른 열을 가리킨다")
                    # 종목 목록 화면
                    if si == 0:
                        for nsi in range(len(V.NAME_SORTS)):
                            st.nsi = nsi
                            try:
                                nl, nh = V.names_lines(st, width)
                            except Exception as e:
                                bad.append(f"{tag} 종목목록 예외 {e!r}")
                                continue
                            ncols = V._fit(V.names_cols(), width)
                            names = st.names()
                            check_block(tag + " 종목", nl, width, ncols, nh,
                                        names, want)

for line in bad[:40]:
    print("  " + line)
print(f"조합 {n}개 · 정렬키 {len(seen_sort_keys)}종 · 위반 {len(bad)}건")
sys.exit(1 if bad else 0)
'''


def sec_render(wt: Path, py: str, report: str, scratch: Path,
               spec: Path | None) -> Section:
    """실데이터 렌더 스모크.

    값 기대표는 선언이라 낡는다 — 열의 뜻이 바뀌면(미실현이 숫자에서 발산 막대로
    바뀐 것처럼) 화면이 아니라 표가 틀린 것이다. 워크트리는 변이 선언 파일의
    ``[render.expect]`` 로 갱신한다: ``"미실현[%p]" = []`` 는 "이 열은 이제
    뜻이 달라 값 검사에서 빼라"(구조 검사는 그대로 받는다)는 뜻이다.
    """
    s = Section("render")
    rpt = Path(os.path.expanduser(report))
    if not (rpt / "numbers.html").exists():
        s.skip(f"실데이터가 없다: {rpt}/numbers.html — 렌더 스모크를 못 돌린다")
        return s
    over: dict = {}
    if spec and spec.exists():
        try:
            raw = tomllib.loads(spec.read_text("utf-8")).get("render", {})
            for k, v in (raw.get("expect") or {}).items():
                over[k] = None if not v else list(v)
        except tomllib.TOMLDecodeError:
            pass                       # §mutations 가 같은 파일을 파싱하며 보고한다
    over_path = ""
    if over:
        over_path = str(scratch / "expect.json")
        Path(over_path).write_text(json.dumps(over, ensure_ascii=False), "utf-8")
        s.say(f"값 기대표 갱신 {len(over)}건: " + ", ".join(sorted(over)))
    script = scratch / "render_smoke.py"
    script.write_text(RENDER_SMOKE, encoding="utf-8")
    p = run([py, str(script), str(rpt), over_path], wt, py_env(wt), timeout=1200)
    if p.returncode != 0:
        s.fail("실데이터 렌더 위반\n" + tail(p.stdout + p.stderr, 3000))
    else:
        s.say(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "ok")
    return s


# --------------------------------------------------------------------------- 6. pty 스모크

PTY_SMOKE = r'''
"""실제 curses 경로 — 뜨는가·q 로 나가는가·stderr 가 비었는가·40x10 에서
사는가·터미널이 사라지면 나가는가."""
import os, pty, subprocess, sys, threading, time

report = sys.argv[1]
LAUNCH = "from kr_quant.tui.flow_app import main; main()"
bad = []


def drain(primary, box):
    """pty 를 계속 읽어 버린다. 안 읽으면 버퍼(~64KB)가 차는 순간 앱이
    refresh() 의 write 에서 막히고, 앱은 q 를 못 읽는다 — 그러면 '앱이 종료하지
    않았다' 는 **앱을 가리키는 엉뚱한 메시지**로 실패한다."""
    buf = []
    try:
        while True:
            c = os.read(primary, 65536)
            if not c:
                break
            buf.append(c)
    except OSError:
        pass
    box.append(b"".join(buf).decode("utf-8", "replace"))


def launch(lines, cols, stderr=subprocess.PIPE):
    primary, replica = pty.openpty()
    env = {**os.environ, "TERM": "xterm-256color", "LINES": lines, "COLUMNS": cols}
    proc = subprocess.Popen([sys.executable, "-c", LAUNCH, "--dir", report],
                            stdin=replica, stdout=replica, stderr=stderr, env=env)
    os.close(replica)
    return primary, proc


def drive(tag, keys, lines="24", cols="120", want_screen=None):
    primary, proc = launch(lines, cols)
    box = []
    t = threading.Thread(target=drain, args=(primary, box), daemon=True)
    t.start()
    try:
        os.write(primary, keys)
        _o, err = proc.communicate(timeout=40)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.communicate(); os.close(primary); t.join(timeout=2)
        bad.append(f"{tag}: q 를 눌렀는데 종료하지 않았다")
        return
    os.close(primary); t.join(timeout=2)
    screen = box[0] if box else ""
    if proc.returncode != 0:
        bad.append(f"{tag}: 종료코드 {proc.returncode}\n"
                   + err.decode("utf-8", "replace")[-1200:])
    if err.strip():
        bad.append(f"{tag}: stderr 가 비어 있지 않다\n"
                   + err.decode("utf-8", "replace")[-1200:])
    if want_screen and want_screen not in screen:
        bad.append(f"{tag}: 화면에 {want_screen!r} 가 없다")


drive("기본 80x24", b"wmas\x1b[B\n\x1b" b"qq", cols="80", want_screen="섹터 자금")
drive("좁은 40x10", b"wmas?\x1b\n\x1bqq", lines="10", cols="40")
drive("넓은 200x50", b"wwmmaassjjkk\n\x1b" b"?" b"q" b"q", lines="50", cols="200")
# 버퍼 압력 — 키를 많이 넣으면 출력이 64KB 를 넘긴다.
drive("긴 키 시퀀스", b"wwwwmmmaaaassss" * 8 + b"?\x1b" + b"\n\x1b" + b"q")

# 터미널이 사라지면(SSH 끊김) 나가는가. 여기서는 **배수 스레드를 쓰지 않는다** —
# 읽는 중에 같은 fd 를 닫으면 경합이라 앱이 아니라 테스트 사정으로 끝날 수 있다.
primary, proc = launch("24", "120", stderr=subprocess.DEVNULL)
time.sleep(0.6)
os.close(primary)
try:
    proc.communicate(timeout=12)
except subprocess.TimeoutExpired:
    ticks = "?"
    try:
        stat = open(f"/proc/{proc.pid}/stat").read().split()
        ticks = int(stat[13]) + int(stat[14])
    except OSError:
        pass
    proc.kill(); proc.communicate()
    bad.append(f"터미널 소멸: 앱이 계속 돈다 — CPU 좀비 ({ticks} 틱)")

for b in bad:
    print("  " + b)
print(f"pty 시나리오 5개 · 위반 {len(bad)}건")
sys.exit(1 if bad else 0)
'''


def sec_pty(wt: Path, py: str, report: str, scratch: Path) -> Section:
    s = Section("pty")
    if sys.platform == "win32":
        s.skip("pty 는 POSIX 전용")
        return s
    rpt = Path(os.path.expanduser(report))
    if not (rpt / "numbers.html").exists():
        s.skip(f"실데이터가 없다: {rpt}/numbers.html")
        return s
    script = scratch / "pty_smoke.py"
    script.write_text(PTY_SMOKE, encoding="utf-8")
    p = run([py, str(script), str(rpt)], wt, py_env(wt), timeout=600)
    if p.returncode != 0:
        s.fail("pty 스모크 위반\n" + tail(p.stdout + p.stderr, 3000))
    else:
        s.say(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "ok")
    return s


# --------------------------------------------------------------------------- 판정

ALL = ("worktree", "identity", "lint", "guardrails", "tests", "mutations",
       "render", "pty")


def verify(wt: Path, base: str, report: str, only: set[str], py: str,
           spec: Path | None) -> tuple[str, list[Section]]:
    secs: list[Section] = []
    scratch = Path(tempfile.mkdtemp(prefix="verify_merge."))
    try:
        s0, commits = sec_worktree(wt, base, py)
        secs.append(s0)
        if s0.status == REJECT:
            return REJECT, secs
        base_sha = git(wt, "merge-base", base, "HEAD")
        table = {
            "identity": lambda: sec_identity(wt, commits),
            "lint": lambda: sec_lint(wt, py, base_sha, scratch),
            "guardrails": lambda: sec_guardrails(wt, py),
            "tests": lambda: sec_tests(wt, py),
            "mutations": lambda: sec_mutations(wt, py, base_sha, spec),
            "render": lambda: sec_render(wt, py, report, scratch, spec),
            "pty": lambda: sec_pty(wt, py, report, scratch),
        }
        for name in ALL[1:]:
            if name in only:
                secs.append(table[name]())
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    verdict = REJECT if any(s.status == REJECT for s in secs) else PASS
    return verdict, secs


def render_report(wt: Path, verdict: str, secs: list[Section]) -> str:
    out = ["", "=" * 78, f"머지 검증 — {wt}", "=" * 78]
    for s in secs:
        out.append("")
        out.append(f"[{s.status:6}] {s.name}")
        for ln in s.lines:
            out.append("    " + ln.replace("\n", "\n    "))
    warns = [s.name for s in secs if s.status == WARN]
    skips = [s.name for s in secs if s.status == SKIP]
    out += ["", "-" * 78]
    if warns:
        out.append(f"애매한 항목(사람이 읽어야 함): {', '.join(warns)}")
    if skips:
        out.append(f"못 돌린 항목: {', '.join(skips)}")
    out.append(f"판정: {verdict}")
    out.append("주의 — 이 하네스는 '선언된 변이가 잡히는가' 까지만 본다. "
               "선언 자체가 얕으면(자기 구현을 베낀 동어반복 검사) 통과한다. "
               "코드와 보고서는 사람이 읽어야 한다.")
    out.append("-" * 78)
    return "\n".join(out)


# --------------------------------------------------------------------------- 자기 검증

SELF_SCENARIOS = """일부러 깨진 워크트리를 만들어 하네스가 정말 REJECT 를 내는지 본다.
늘 PASS 하는 하네스는 장식이다."""


def _scratch_repo(dst: Path, base: str, mail: str, name: str) -> None:
    """기준 커밋의 트리를 새 git 저장소로 복제한다(원본 워크트리를 안 건드린다)."""
    dst.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "archive", base], cwd=str(REPO),
                         capture_output=True, timeout=120)
    subprocess.run(["tar", "-x", "-C", str(dst)], input=tar.stdout, timeout=120)
    for a in (["init", "-q", "-b", "wt"], ["config", "user.email", mail],
              ["config", "user.name", name], ["add", "-A"],
              ["commit", "-q", "-m", "base"]):
        subprocess.run(["git", *a], cwd=str(dst), capture_output=True, timeout=120)
    subprocess.run(["git", "branch", "-f", base.replace("/", "-") or "base"],
                   cwd=str(dst), capture_output=True, timeout=60)


def self_test(py: str, report: str) -> int:
    print(SELF_SCENARIOS)
    mail = git(REPO, "config", "--get", "user.email") or "a@b.c"
    name = git(REPO, "config", "--get", "user.name") or "x"
    root = Path(tempfile.mkdtemp(prefix="verify_merge.selftest."))
    results: list[tuple[str, str, str, bool]] = []

    def scenario(tag: str, only: set[str], want: str, setup,
                 by: str | None = None) -> None:
        """`by` = 그 판정을 내야 하는 항목. 엉뚱한 이유로 난 REJECT 는 통과가 아니다
        (예: 하네스 자신이 레포 검사를 깨서 pytest 가 빨개진 것)."""
        wt = root / tag
        _scratch_repo(wt, "HEAD", mail, name)
        base_branch = "base"
        subprocess.run(["git", "branch", "-f", base_branch], cwd=str(wt),
                       capture_output=True, timeout=60)
        setup(wt)
        got, secs = verify(wt, base_branch, report, only, py, None)
        detail = "; ".join(f"{s.name}={s.status}" for s in secs)
        right_place = by is None or any(
            s.name == by and s.status in (REJECT, WARN) for s in secs)
        ok = got == want and right_place
        results.append((tag, f"{want}@{by or '*'}", f"{got}@{detail}", ok))
        print(f"  {tag}: 기대 {want}({by or '전체'}) · 실제 {got}  [{detail}]")
        if not ok:
            for s in secs:
                for ln in s.lines:
                    print("      " + ln[:200])

    def commit(wt: Path, msg: str, mail_override: str | None = None) -> None:
        subprocess.run(["git", "add", "-A"], cwd=str(wt), capture_output=True)
        env = dict(os.environ)
        if mail_override:
            env |= {"GIT_AUTHOR_EMAIL": mail_override,
                    "GIT_COMMITTER_EMAIL": mail_override}
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=str(wt),
                       capture_output=True, env=env)

    # ① 대조군 — 손 안 댄 기준 트리는 PASS 여야 한다(늘 REJECT 도 장식이다).
    scenario("clean", {"identity", "lint", "guardrails", "tests", "mutations"},
             PASS, lambda wt: None)

    # ② 테스트 실패
    def break_test(wt: Path) -> None:
        (wt / "tests/test_zz_broken.py").write_text(
            "def test_broken():\n    assert False, '일부러 깨뜨림'\n", "utf-8")
        commit(wt, "broken test")
    scenario("broken-test", {"tests"}, REJECT, break_test, by="tests")

    # ③ 신원 위반 — 회사 이메일이 새어 들어간 커밋
    def bad_identity(wt: Path) -> None:
        (wt / "README.md").write_text(
            (wt / "README.md").read_text("utf-8") + "\n", "utf-8")
        commit(wt, "leak", mail_override="someone@corp.example.com")
    scenario("bad-identity", {"identity"}, REJECT, bad_identity,
             by="identity")

    # ④ 변이를 못 잡는 검사 — 이 하네스의 존재 이유
    def blind_guard(wt: Path) -> None:
        (wt / "tests/test_zz_blind.py").write_text(
            '"""폭만 보는 검사 — 셀이 통째로 빠져도 초록이다."""\n'
            "from kr_quant.tui.flow_view import cell_width\n\n\n"
            "def test_blind():\n"
            "    assert cell_width('가') == 2\n", "utf-8")
        (wt / "tests/mutations.toml").write_text(
            '[[mutation]]\n'
            'id = "drop-accel-cell"\n'
            'why = "가속 셀을 빼면 그 뒤 열이 밀린다"\n'
            'guards = ["tests/test_zz_blind.py::test_blind"]\n'
            'file = "src/kr_quant/tui/flow_view.py"\n'
            'find = "pad(fmt_pct(r.get(\\"accel\\")), 9, True),"\n'
            'replace = ""\n', "utf-8")
        commit(wt, "blind guard")
    scenario("blind-guard", {"mutations"}, REJECT, blind_guard,
             by="mutations")

    # ⑤ 진짜 잡는 검사 — 같은 변이를 기존 정렬 검사에 걸면 REJECT 가 아니어야 한다.
    def real_guard(wt: Path) -> None:
        (wt / "tests/mutations.toml").write_text(
            '[[mutation]]\n'
            'id = "drop-accel-cell"\n'
            'why = "가속 셀을 빼면 그 뒤 열이 밀린다"\n'
            'guards = ["tests/test_tui_flow_view.py::'
            'test_data_cells_sit_under_their_headers"]\n'
            'file = "src/kr_quant/tui/flow_view.py"\n'
            'find = "pad(fmt_pct(r.get(\\"accel\\")), 9, True),"\n'
            'replace = ""\n', "utf-8")
        commit(wt, "real guard")
    scenario("real-guard", {"mutations"}, PASS, real_guard)

    # ⑥ 변이 미선언 새 검사 → 경고(REJECT 아님)이고, 판정은 PASS 여야 한다.
    def undeclared(wt: Path) -> None:
        (wt / "tests/test_zz_new.py").write_text(
            "def test_new_but_undeclared():\n    assert True\n", "utf-8")
        commit(wt, "new test, no mutation")
    scenario("undeclared-test", {"mutations"}, PASS, undeclared,
             by="mutations")   # 경고는 나야 한다

    shutil.rmtree(root, ignore_errors=True)
    ok = all(r[3] for r in results)
    print("\n자기 검증: " + ("전 시나리오 기대대로" if ok else "**기대와 다름**"))
    for tag, want, got, good in results:
        print(f"  {'ok ' if good else 'BAD'} {tag}: 기대 {want} 실제 {got}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("worktree", nargs="?", help="검증할 워크트리 경로")
    ap.add_argument("--base", default=DEFAULT_BASE, help="기준 브랜치/커밋")
    ap.add_argument("--report-dir", default=DEFAULT_REPORT, help="실데이터 리포트 폴더")
    ap.add_argument("--only", default="", help="돌릴 항목 (쉼표): " + ",".join(ALL[1:]))
    ap.add_argument("--python", default=None, help="쓸 인터프리터")
    ap.add_argument("--mutations", default=None,
                    help="변이 선언 파일 (기본: 워크트리의 tests/mutations.toml)")
    ap.add_argument("--self-test", action="store_true",
                    help="일부러 깨진 워크트리로 하네스가 REJECT 를 내는지 확인")
    a = ap.parse_args()

    if a.self_test:
        return self_test(interpreter(REPO, a.python), a.report_dir)
    if not a.worktree:
        ap.error("워크트리 경로가 필요하다 (또는 --self-test)")

    wt = Path(a.worktree).resolve()
    only = set(x.strip() for x in a.only.split(",") if x.strip()) or set(ALL[1:])
    unknown = only - set(ALL)
    if unknown:
        ap.error(f"모르는 항목: {', '.join(sorted(unknown))}")
    spec = Path(a.mutations).resolve() if a.mutations else None
    verdict, secs = verify(wt, a.base, a.report_dir, only,
                           interpreter(wt, a.python), spec)
    print(render_report(wt, verdict, secs))
    return 0 if verdict == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
