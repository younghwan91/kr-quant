"""다중검정 원장 — 한 알파에 **실제로 시도한 config 수**를 세는 장치.

Deflated Sharpe / t-haircut(``gate_report``)은 시행 수 ``N`` 을 입력으로 받는다. 그 N 을
사람이 손으로 적으면 규율이 아니라 부탁이 된다 — 첫 config 가 죽고 조용히 두 번째를
돌려도 아무도 못 잡는다(GUARDRAILS §4 공백 5). 그래서 게이트를 통과시킬 때마다
config 를 원장에 적고, N 은 그 원장에서 **읽는다**.

설계 원칙(§0 "기억하라고 부탁하지 말 것"):

- **게이트가 직접 적는다.** 연구자가 ``record_trial`` 을 호출할 일이 없다.
  ``prop_gate(config=...)`` 에 사전등록 config 를 넘기면 그 호출 자체가 기록된다.
- **config 지문으로 중복 제거.** 같은 config 를 재실행해도 N 이 늘지 않는다 —
  안 그러면 오늘 두 번 돌린 게 시행 2회로 잡혀 N 이 거짓으로 부푼다. 시행은
  "다르게 시도한 횟수"지 "실행 횟수"가 아니다.
- **append-only JSONL.** 사람이 읽을 수 있고, git 이력이 증인이 된다. 줄여 적으려면
  파일을 지워야 하는데 그건 커밋에 남는다.

원장 위치: ``research/logs/<label>/TRIALS.jsonl`` — 그 알파의 VERDICT.md 옆.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

LEDGER_NAME = "TRIALS.jsonl"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ledger_path(label: str, *, logs_dir: "Path | str | None" = None) -> Path:
    """``research/logs/<label>/TRIALS.jsonl`` 경로 (파일이 없어도 경로만 반환)."""
    base = Path(logs_dir) if logs_dir is not None else _repo_root() / "research" / "logs"
    return base / label / LEDGER_NAME


def config_fingerprint(config: dict) -> str:
    """config 의 안정적 지문. 키 순서·부동소수 표기에 흔들리지 않게 정규화한다."""
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def record_trial(label: str, config: dict, *, note: str | None = None,
                 logs_dir: "Path | str | None" = None) -> str:
    """config 를 원장에 append 하고 지문을 반환. 같은 지문이 이미 있으면 다시 안 적는다.

    호출자가 직접 쓸 일은 거의 없다 — ``prop_gate(config=...)`` 가 대신 부른다.
    """
    fp = config_fingerprint(config)
    path = ledger_path(label, logs_dir=logs_dir)
    if fp in _fingerprints(path):
        return fp
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"fingerprint": fp, "config": config}
    if note:
        row["note"] = note
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return fp


def _fingerprints(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.add(json.loads(line)["fingerprint"])
        except (json.JSONDecodeError, KeyError):
            continue          # 손상된 줄은 세지 않는다(원장을 못 읽는 게 더 나쁘다)
    return out


def count_trials(label: str, *, logs_dir: "Path | str | None" = None) -> int:
    """이 알파에 기록된 **서로 다른** config 수. 원장이 없으면 0."""
    return len(_fingerprints(ledger_path(label, logs_dir=logs_dir)))


def read_trials(label: str, *, logs_dir: "Path | str | None" = None) -> list[dict[str, Any]]:
    """원장 전체를 순서대로 반환(사람이 읽거나 문서에 싣기 위한 것)."""
    path = ledger_path(label, logs_dir=logs_dir)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
