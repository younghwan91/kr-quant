"""research/·scripts/·examples/ 의 모든 모듈이 import 되는지.

이 스위트는 `src/kr_quant` 라이브러리만 덮고 있어서, 라이브러리에서 심볼을 지워도
그걸 import 하던 리서치 러너는 아무도 안 깨뜨렸다 — ruff 는 크로스모듈 심볼을 풀지
않으므로 린트도 통과한다. 실제로 미너비니 제거 때 `fundamentals._yoy_vec` 이 이 경로로
사라져 pead_gate·pead_concentrated_gate·pead_refinement·prop_feasibility 4개가
동시에 깨졌고, 전체 테스트는 초록이었다.

각 모듈을 실제로 exec 해서 import 시점 오류만 잡는다(러너 본문은 `main()` 뒤에 있어
실행되지 않는다). DB·네트워크 불요.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DIRS = ("research", "scripts", "examples")
# 러너들은 서로를 top-level 이름으로 import 한다(from prop_swing_common import ...).
EXTRA_PATH = [str(REPO / "research" / "experiments"), str(REPO / "research" / "signals")]


def _modules() -> list[Path]:
    out: list[Path] = []
    for d in DIRS:
        out += sorted(p for p in (REPO / d).rglob("*.py") if "__pycache__" not in p.parts)
    return out


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(REPO)))
def test_module_imports(path: Path):
    added = [p for p in EXTRA_PATH if p not in sys.path]
    sys.path[:0] = added
    try:
        spec = importlib.util.spec_from_file_location(f"_smoke_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass  # argparse 를 모듈 레벨에서 도는 스크립트
    except (ImportError, AttributeError, NameError) as e:
        pytest.fail(f"{path.relative_to(REPO)} import 실패: {type(e).__name__}: {e}")
    finally:
        for p in added:
            sys.path.remove(p)
