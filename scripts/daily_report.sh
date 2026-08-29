#!/usr/bin/env bash
# 섹터 자금흐름 일일 리포트 — 날짜 폴더에 재워둔다.
#
# 폴더 이름은 **데이터 기준일**(supply_demand 의 MAX(date))이지 실행일이 아니다.
# 휴장일에 돌면 전 거래일 폴더가 이미 있으므로 건너뛴다 — 같은 데이터로 폴더가
# 여러 개 생기는 걸 막는다.
#
# 리포트는 저장소 밖에 쌓는다(생성물이지 소스가 아니다).
set -euo pipefail

REPO="/home/young/Documents/git/kr-quant"
OUT_ROOT="${KR_QUANT_REPORTS:-$HOME/Documents/kr-quant-reports}"
DAYS="${KR_QUANT_REPORT_DAYS:-260}"
UV="/home/young/.local/bin/uv"
# ⚠️ extra 를 명시한다. `uv run` 은 기본 의존성만 맞추므로, 누가 `uv sync --extra dev`
# 를 돌리면 psycopg2(pg extra)가 사라지고 이 배치가 조용히 죽는다 — 실제로 그렇게
# 한 번 빠졌다. 여기서 못박아 두면 실행 시점에 항상 복원된다.
UVR=("$UV" run --quiet --extra pg)

cd "$REPO"
set -a; . ./.env; set +a

ASOF="$("${UVR[@]}" python - <<'PY' 2>/dev/null
import os
from kr_quant.storage import connect, db_default
con = connect(os.environ.get("KR_QUANT_DB") or db_default())
cur = con.cursor(); cur.execute("SELECT max(date) FROM supply_demand")
print(cur.fetchone()[0]); con.close()
PY
)"

if [ -z "$ASOF" ]; then
  echo "[$(date '+%F %T')] 기준일을 못 읽었다 — DB 접속 실패로 본다. 중단." >&2
  exit 1
fi

DEST="$OUT_ROOT/$ASOF"
# 건너뛰기는 **데이터**가 그대로일 때를 위한 것이다(휴장일 재실행). 리포트에 실리는
# 값이 늘어났을 때는 데이터가 같아도 다시 만들어야 하는데, 그럴 방법이 없었다 —
# 열을 추가하고 나서 사용자 화면에 새 열이 통째로 비어 있었고, 원인은 코드가 아니라
# 폴더가 이미 있어서 이 배치가 조용히 건너뛴 것이었다. 다음 거래일까지 그 상태로
# 남는다. KR_QUANT_FORCE=1 로 그 문을 연다.
if [ -d "$DEST" ] && [ -f "$DEST/numbers.html" ] && [ -z "${KR_QUANT_FORCE:-}" ]; then
  echo "[$(date '+%F %T')] $ASOF 리포트가 이미 있다 — 건너뛴다(휴장일이거나 재실행)."
  echo "    리포트 형식이 바뀌어 다시 만들려면: KR_QUANT_FORCE=1 $0"
  exit 0
fi

# 강제 재생성은 **옆에 짓고 마지막에 바꿔 끼운다.** 제자리에 덮어쓰면 5분 남짓
# 동안 폴더 안에 새 payload 와 옛 numbers.html 이 섞여 있고, 그 사이에 kq-flow 를
# 띄운 사람은 어느 쪽도 아닌 화면을 본다. 실패하면 옛 리포트가 그대로 남는다.
BUILD="$DEST"
SWAP=""
if [ -d "$DEST" ]; then
  BUILD="$DEST.building.$$"
  SWAP="1"
  rm -rf "$BUILD"
  echo "[$(date '+%F %T')] $ASOF 리포트를 **다시** 만든다(KR_QUANT_FORCE) — 옆에 짓고 바꿔 낀다"
fi
DEST_FINAL="$DEST"
DEST="$BUILD"

mkdir -p "$DEST"
echo "[$(date '+%F %T')] 생성 시작 — 기준일 $ASOF, 구간 ${DAYS}거래일"
# 페이로드를 **한 번만** 만들고 둘 다 그걸 쓴다. 월별 시총 계산이 수 분이라
# 예전처럼 세 번 돌리면 배치가 15분을 넘고 그만큼 실패 창이 넓어진다.
"${UVR[@]}" python scripts/sector_flow.py    --days "$DAYS" --json "$DEST/payload.json"
"${UVR[@]}" python scripts/sector_flow.py    --from-json "$DEST/payload.json" --html "$DEST/viewer.html"
"${UVR[@]}" python scripts/sector_numbers.py --payload "$DEST/payload.json" --html "$DEST/numbers.html"

# 저장한 것을 그 자리에서 검증한다 — 실패하면 폴더에 VERIFY_FAILED 를 남긴다.
# 조용히 틀린 리포트가 쌓이는 것이 검증 없이 도는 것보다 나쁘다.
if "${UVR[@]}" python scripts/verify_report.py --dir "$DEST" --db-check \
     > "$DEST/VERIFY.txt" 2>&1; then
  echo "[$(date '+%F %T')] 검증 통과 — $DEST/VERIFY.txt"
else
  cp "$DEST/VERIFY.txt" "$DEST/VERIFY_FAILED.txt"
  echo "[$(date '+%F %T')] ⚠️ 검증 실패 — $DEST/VERIFY_FAILED.txt 확인" >&2
  tail -5 "$DEST/VERIFY.txt" >&2
fi

if [ -n "$SWAP" ]; then
  OLD="$DEST_FINAL.replaced.$$"
  mv "$DEST_FINAL" "$OLD"
  mv "$DEST" "$DEST_FINAL"
  rm -rf "$OLD"
  DEST="$DEST_FINAL"
  echo "[$(date '+%F %T')] 바꿔 끼웠다 — $DEST"
fi

ln -sfn "$DEST" "$OUT_ROOT/latest"
echo "[$(date '+%F %T')] 완료 — $DEST"
ls -lh "$DEST" | tail -n +2 | awk '{print "   ", $9, $5}'
