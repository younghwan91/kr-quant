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

cd "$REPO"
set -a; . ./.env; set +a

ASOF="$("$UV" run --quiet python - <<'PY' 2>/dev/null
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
if [ -d "$DEST" ] && [ -f "$DEST/numbers.html" ]; then
  echo "[$(date '+%F %T')] $ASOF 리포트가 이미 있다 — 건너뛴다(휴장일이거나 재실행)."
  exit 0
fi

mkdir -p "$DEST"
echo "[$(date '+%F %T')] 생성 시작 — 기준일 $ASOF, 창 ${DAYS}거래일"
# 페이로드를 **한 번만** 만들고 둘 다 그걸 쓴다. 월별 시총 계산이 수 분이라
# 예전처럼 세 번 돌리면 배치가 15분을 넘고 그만큼 실패 창이 넓어진다.
"$UV" run --quiet python scripts/sector_flow.py    --days "$DAYS" --json "$DEST/payload.json"
"$UV" run --quiet python scripts/sector_flow.py    --from-json "$DEST/payload.json" --html "$DEST/viewer.html"
"$UV" run --quiet python scripts/sector_numbers.py --payload "$DEST/payload.json" --html "$DEST/numbers.html"

# 저장한 것을 그 자리에서 검증한다 — 실패하면 폴더에 VERIFY_FAILED 를 남긴다.
# 조용히 틀린 리포트가 쌓이는 것이 검증 없이 도는 것보다 나쁘다.
if "$UV" run --quiet python scripts/verify_report.py --dir "$DEST" --db-check \
     > "$DEST/VERIFY.txt" 2>&1; then
  echo "[$(date '+%F %T')] 검증 통과 — $DEST/VERIFY.txt"
else
  cp "$DEST/VERIFY.txt" "$DEST/VERIFY_FAILED.txt"
  echo "[$(date '+%F %T')] ⚠️ 검증 실패 — $DEST/VERIFY_FAILED.txt 확인" >&2
  tail -5 "$DEST/VERIFY.txt" >&2
fi

ln -sfn "$DEST" "$OUT_ROOT/latest"
echo "[$(date '+%F %T')] 완료 — $DEST"
ls -lh "$DEST" | tail -n +2 | awk '{print "   ", $9, $5}'
