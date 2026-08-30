"""자금 원장 TUI 의 **순수 렌더 로직** — curses 를 import 하지 않는다.

설계 근거는 ``docs/superpowers/specs/2026-08-29-money-flow-ledger.md``.
요지 한 줄: **관측된 것은 (날짜, 시장, 섹터, 주체)의 순매수 금액뿐이다.**

그래서 이 모듈이 그리는 것은 **주체 ↔ 섹터 이분 그래프**(간선 하나하나가 실측)이고,
그리지 않는 것은 **섹터 → 섹터** 와 **주체 → 주체** 다. 둘 다 같은 이유로 미식별이다 —
주변합(marginal)은 쌍별 흐름을 결정하지 않는다. 4주체의 주변합 4개(독립 3개)로는
C(4,2)=6개의 쌍별 이전량이 3차원만큼 남는다.

``flow_view`` 와 나란히 산다. 폭 계산(``cell_width``·``cell_len``·``pad``)과
폭 단계 고르기(``tier_for``), 시장 순서(``MARKET_ORDER``)는 거기 것을
**재사용**한다 — 한글 2칸·U+2212 1칸을, 그리고 "m 두 번 = 코스닥" 을 두 화면이
각자 정의하면 반드시 갈라진다.

``flow_view`` 가 고친 세 가지(줄을 통째로 잘라 숫자가 자릿수 중간에서 끊김 ·
얇은 섹터 경고를 색에만 실음 · 터미널이 사라지면 100% CPU 좀비)는 이 화면에도
**같은 모양으로** 있었다. 셋 다 여기서 같이 고쳤다 — 한쪽만 고치면 다른 쪽이
조용히 남는다.

그림 글자는 **전부 브라유**(U+28xx)다 — 스파크라인·히트맵 농도·발산 막대. 블록
문자(``▁▂▃█``·``·░▒▓█``·``▌``)는 East Asian Width 가 'A' 라 터미널이 2칸으로 그릴
수 있는데, 이 화면들은 그림 글자가 **1칸이라고 가정하고 칸을 센다.** 어긋나면
스파크라인이 잘려 구간의 일부를 전부인 척하고, 히트맵 27열이 54칸을 먹는다.
``kq-flow`` 가 먼저 옮겼고 원장만 남아 있었다.
"""

from __future__ import annotations

import json
import math
import os
import random

from kr_quant.tui.flow_view import (
    HELP_TITLE_TIERS, MARKET_ORDER, cell_len, cell_width, fit_widths, fmt_amt, help_desc,
    help_lines, hint_line, pad, tier_for)

ACTORS = (("indiv", "개인"), ("forgn", "외국인"), ("inst", "기관"), ("etc", "기타법인"))
ACTOR_KEYS = tuple(k for k, _ in ACTORS)
WINDOWS = (5, 20, 60, 120, 260)
VIEWS = (("ledger", "원장"), ("timeline", "전개"), ("comove", "동시성"), ("limits", "한계"))
SORTS = (("abs", "절대크기"), ("actor", "선택주체"), ("name", "섹터명"),
         ("spike", "최대일몫"), ("n", "종목수"))

#: 모든 화면 최하단에 상주하는 한 줄. 각주가 아니라 본문이다.
#: 폭에 맞춰 단계별로 줄인다 — 잘라내면 "미관측" 이 사라지는 자리라, 한 줄
#: 고정이면 좁은 화면에서 **관측된 것만 남고 경고가 없어진다**.
BANNER_TIERS = (
    "관측: 주체×섹터 순매수.  미관측: 섹터→섹터 이동 · 주체→주체 이전(주변합만 있다).",
    "관측: 주체×섹터 순매수 · 미관측: 섹터→섹터 · 주체→주체",
    "미관측: 섹터→섹터 이동 · 주체→주체 이전",
    "미관측: 섹터→섹터 · 주체→주체",
    "미관측: 이동·이전",
)
BANNER = BANNER_TIERS[0]


def banner_for(width: int) -> str:
    """그 폭에 온전히 들어가는 가장 긴 배너. 앞에 공백 한 칸이 붙는다."""
    return tier_for(BANNER_TIERS, width - 1)

#: 전개 스파크라인의 글자 — **브라유**(U+28xx) 다. 세로 한가운데가 **0** 이고,
#: 위 두 단계(``⠒⠛``)가 순매수, 아래 두 단계(``⠤⣤``)가 순매도, ``⠐`` 이 정확히 0 이다.
#:
#: 두 가지를 한꺼번에 고친다.
#:
#: 1. **폭.** 예전 글자표 ``▁▂▃▄▅▆▇█`` 는 East Asian Width 가 ``A``(Ambiguous) 라
#:    한글 로케일 터미널이 2칸으로 그릴 수 있다. 20점이 40칸이 되면 ``pad`` 가
#:    뒤를 잘라내고 **구간의 절반만 그린 채 전부인 척한다.** 브라유는 ``N`` 이라
#:    어떤 로케일에서도 1칸이다. ``kq-flow`` 가 같은 이유로 먼저 옮겼다.
#: 2. **기준선.** 예전에는 ``lo, hi = min(values), max(values)`` 라 ``▁`` 이 행마다
#:    다른 뜻이었다 — 전기/전자에서는 −24,304, 건설에서는 ≈0. 그래서 20일 내내
#:    판 섹터가 ``████████…▁▁`` 로 나와 "높다가 없어졌다"로 읽혔다(사실은 계속
#:    팔아 내려간 것이다). 0 을 **세로 한가운데에 고정**하면 그 오독이 표현
#:    불가능해진다: 계속 판 섹터는 처음부터 끝까지 중앙선 아래에 있다.
#:
#: ⚠️ ``kq-flow`` 의 처방(브라유 + 누적 + ``min(0,·)``)만으로는 **모자란다.**
#: 0 을 범위에 접어 넣어도 눈금은 여전히 바닥 채우기라, 계속 판 섹터는 hi=0 에서
#: 시작해 ``⣿⣿⣿…⣀⣀`` 가 된다 — 같은 오독이다(실측으로 확인했다). flow 는 크기
#: 순위 화면이라 부호가 부차적이지만 **원장은 회계 화면이라 부호가 전부**다.
#: 대신 크기 해상도가 한쪽에 두 단계뿐인데, 그건 ``눈금[억]`` 열이 메운다.
SPARK_SIGNED = {2: "⠛", 1: "⠒", 0: "⠐", -1: "⠤", -2: "⣤"}
#: 검사·렌더가 "이게 스파크라인 글자냐"를 물을 때 쓰는 문자열.
SPARK = "".join(SPARK_SIGNED[lv] for lv in (2, 1, 0, -1, -2))
#: 반칸 막대 — **브라유**다. 양수는 오른쪽으로 자라며 꼬리가 `⡇`(칸의 왼쪽 절반),
#: 음수는 왼쪽으로 자라며 꼬리가 `⢸`(칸의 오른쪽 절반). 두 글자로 양쪽 모두
#: **반칸 해상도**가 나온다. 브라유는 4행 × 2열이라 한 열을 통째로 채우면
#: 반칸 블록과 같은 그림이 된다.
#:
#: ⚠️ 예전엔 `█▌▐` 였다. `█`·`▌` 는 East Asian Width 가 'A' 라 터미널이 2칸으로
#: 그릴 수 있는데, :func:`signed_bar` 는 **1칸이라고 가정하고 칸을 센다** —
#: `KQ_AMBIGUOUS_WIDE=1` 을 켜면(그리고 실제 한글 로케일 터미널에서는) 막대가
#: 자기 폭의 두 배를 먹고 뒤 열을 밀어냈다. 브라유(EAW 'N')는 어느 쪽이든 1칸이다.
BLOCK_FULL, BLOCK_L, BLOCK_R = "⣿", "⡇", "⢸"

#: 이보다 종목이 적은 (시장,섹터)는 '섹터'로 읽으면 안 된다 — 거래소/부동산은 3종목이다.
THIN_N = 10
_MIN_CORR_N = 20        # 이보다 짧은 구간에서는 상관을 내지 않는다

#: 비율 열(``잔여몫``·``최대일몫``)의 **분모 하한** [억]. 이보다 작으면 값을 안 낸다.
#:
#: 두 열은 몫이라 분모가 작으면 뜻을 잃는다. 표의 금액 열은 **억 단위 정수**로
#: 찍히므로 분모가 1억 미만이면 그 비율을 이루는 금액이 화면에 하나도 안 보인다
#: — 독자는 ``33.3!`` 만 보고 검산할 방법이 없다. 실측(2026-08-28, 1,620행):
#: 코스닥/출판·매체복제 20일 기관은 구간 gross 가 **0.09억**(9백만원)인데
#: ``최대일몫 33.3!`` 이라 적혀 있었다 — 균등(5%)의 3배를 넘었다는 `!` 까지 달고서.
#: 8칸이 여기 걸리고 전부 2~4종목짜리 얇은 칸이다. ``kq-flow`` 가 ``½kx²`` 에서
#: 같은 판단을 했다: **뜻을 잃은 값은 안 보인다**(고쳐 쓰지 않고 결측으로 둔다).
_MIN_RATIO_GROSS = 1.0
_NULL_SHIFTS = 20       # 순환이동 널의 반복 수


# ---------------------------------------------------------------- 데이터 로딩

def load(report_dir: str) -> dict:
    """리포트 폴더의 ``payload.json`` 을 읽는다 — DB 에 붙지 않는다.

    ``scripts/sector_flow.py --json`` 산출물이다. 같은 폴더의 ``numbers.html``·
    ``viewer.html`` 과 **같은 숫자**를 보게 된다.
    """
    path = os.path.join(os.path.expanduser(report_dir), "payload.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"페이로드를 찾을 수 없다: {path}\n"
            f"  먼저 scripts/daily_report.sh 를 돌리거나 --dir 로 지정하라.")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"{path} 의 JSON 이 깨졌다 ({e.lineno}행 {e.colno}칸): {e.msg}\n"
            f"  리포트를 다시 생성하라 — scripts/daily_report.sh") from None
    # 방어가 행 수준(.get)에만 있으면 문서가 통째로 다른 형식일 때 curses 안에서
    # 생 트레이스백으로 터진다. 문 앞에서 잡는다.
    # ⚠️ 여기서 안 보는 키는 나중에 **대괄호로** 집힌다 — `cap`·`n_by_sector` 가
    # 없으면 curses 안에서 생 KeyError 로 터진다. 문 앞 검사는 대괄호로 집는
    # 것들과 같은 목록이어야 한다.
    missing = [k for k in ("dates", "sectors", "markets", "flows", "cap",
                           "n_by_sector") if not d.get(k)]
    if missing:
        raise SystemExit(f"{path} 의 데이터에 없는 키: {', '.join(missing)} — "
                         f"리포트 형식이 바뀌었나?\n  있는 키: {sorted(d)}")
    no_flow = [m for m in d["markets"] if m not in d["flows"]]
    if no_flow:
        raise SystemExit(f"{path}: 시장 {', '.join(no_flow)} 의 흐름이 없다 — "
                         f"markets 와 flows 가 어긋난다.")
    return d


# ---------------------------------------------------------------- 파생 계산

def residual(cell: dict, i: int) -> float:
    """``잔여 = −Σ(4주체)`` — 오차가 아니라 **미분류 주체의 순매수 추정치**.

    KRX 주체 분류에는 우리가 안 싣는 항목(기타외국인 등)이 있다. 전 주체의 합은
    수량 기준으로 정확히 0 이므로, 우리가 가진 4주체 합의 부호를 뒤집으면 나머지
    주체의 순매수가 된다. 여기에 결측·종가환산 오차가 섞여 있다.

    ⚠️ 이걸 4주체에 안분해 0 으로 만들면 **측정되지 않은 주체가 측정된 주체의 옷을
    입는다.** 그래서 별도 값으로 남긴다.
    """
    return -sum(cell[k][i] for k in ACTOR_KEYS)


def spark_scale(values: list[float]) -> float:
    """스파크라인의 세로 눈금 — ``⠛``/``⣤`` 끝 칸에 해당하는 값.

    **그려지는 점들로 잰다.** 원본 점들의 최고−최저를 옆에 적어 두던 예전
    ``진폭[억]`` 열은 그림이 다운샘플된 점으로 정규화된다는 사실을 무시했다.
    실측(2026-08-28, 폭 80·260일, 시장 3 × 주체 4 전수): 두 값이 최대 **59.3%**
    어긋난다(코스닥·개인 일반서비스 — 원본 28,588억 vs 그림 11,630억).
    "숫자가 그림의 눈금을 알려준다"는 계약은 **같은 점들**을 봐야 참이 된다.
    """
    return max((abs(v) for v in values), default=0.0)


def spark(values: list[float]) -> str:
    """0 을 세로 한가운데 고정한 **부호 스파크라인**(:data:`SPARK_SIGNED`).

    ``values`` 는 **누적 경로**다. 중앙선 위면 그 시점까지 순매수, 아래면 순매도.
    크기는 그 행의 ``max|누적|``(= :func:`spark_scale`) 을 1 로 두고 반으로 가른다 —
    ``⠛``/``⣤`` 는 절반 너머, ``⠒``/``⠤`` 는 절반 이내, ``⠐`` 은 정확히 0 이다.
    """
    if not values:
        return ""
    peak = spark_scale(values)
    out = []
    for v in values:
        if peak <= 0 or v == 0:
            lv = 0
        else:
            r = v / peak
            lv = 2 if r > 0.5 else 1 if r > 0 else -1 if r >= -0.5 else -2
        out.append(SPARK_SIGNED[lv])
    return "".join(out)


def _fit(cols: list[tuple[str, int, bool]], width: int) -> list[tuple[str, int, bool]]:
    """폭에 **온전히** 들어가는 열까지만 남긴다. 첫 열은 잘려도 남긴다
    (섹터 이름은 잘려도 뜻이 남지만, 숫자는 잘리면 다른 값이 된다).

    규칙 자체는 ``flow_view.fit_widths`` 한 곳에만 있다 — 예전엔 같은 로직이
    여기와 흐름 화면에 두 벌 살았고, 둘을 잇는 검사가 없어 한쪽만 고쳐도
    초록이었다.
    """
    return cols[:fit_widths([c[1] for c in cols], width)]


def downsample(values: list[float], n: int) -> list[float]:
    """구간이 화면 칸보다 길 때 **묶어서** 줄인다 — 앞을 잘라내지 않는다.

    ``values[-n:]`` 로 뒤만 남기면 누적 곡선이 시작점을 잃고, 그림이 "이 구간에
    아무 일도 없다가 최근에 움직였다"고 거짓말한다. 각 묶음의 **끝점**을 뽑는다.

    ⚠️ 경계는 **정수 산술**로 낸다. 예전엔 ``int((i + 1) * len/n) - 1`` 이라
    부동소수 반올림이 마지막 묶음의 끝을 한 칸 앞으로 밀었다 — ``n=11, len=60``
    에서 ``(11) * (60/11) = 60.000000000000007`` 이 아니라 ``59.99999999999999``
    가 되어 **구간의 마지막 날이 안 그려졌다.** 실측(폭 51·60일, 전체·외국인·
    전기/전자): ``누적[억] −533,125`` 인데 옆의 ``눈금[억]`` 은 516,107 이었다 —
    한 줄 안의 두 숫자가 서로를 부정했고, 그림의 오른쪽 끝은 "구간의 끝" 이
    아니었다. 폭 48~400 × 5구간 중 20개 조합이 그랬다.
    """
    if n <= 0 or len(values) <= n:
        return list(values)
    k = len(values)
    return [values[(i + 1) * k // n - 1] for i in range(n)]


def signed_bar(value: float, scale: float, half: int) -> str:
    """0 을 가운데 둔 부호 막대. 표시 폭은 **항상** ``2 * half``.

    ``scale`` 은 막대 한쪽 끝(= ``half`` 칸)에 해당하는 값이다. 넘치면 잘리되
    끝 칸을 `⣿` 로 채워 "넘쳤다"가 보이게 둔다.

    ⚠️ 이 함수는 글자가 **1칸이라고 가정하고 칸을 센다**(`" " * half` 처럼).
    그래서 글자표(:data:`BLOCK_FULL` 외)는 EAW 'N' 이어야 한다.
    """
    if half <= 0:
        return ""
    if scale <= 0 or value == 0 or value != value:
        return " " * (2 * half)
    units = min(round(abs(value) / scale * half * 2), half * 2)   # 반칸 단위
    full, rem = divmod(int(units), 2)
    if value > 0:
        bar = BLOCK_FULL * full + (BLOCK_L if rem else "")
        return " " * half + pad(bar, half)
    bar = (BLOCK_R if rem else "") + BLOCK_FULL * full
    return pad(bar, half, right=True) + " " * half


def _corr(a: list[float], b: list[float]) -> float:
    """피어슨 상관. **분산이 0 이면 ``nan``** 이다 — 0.0 이 아니다.

    한쪽이 상수(그 구간에 흐름이 아예 없는 섹터)면 상관은 **정의되지 않는다.**
    예전엔 0.0 을 돌려줬는데, 히트맵에서 그건 ``|r|<0.05`` 와 **같은 빈칸**이라
    "무상관"과 "잴 수 없다"가 구분되지 않았다. ``nan`` 은 화면에서 ``?`` 로 나온다.
    """
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)


def neg_frac(values) -> float | None:
    """**음의 상관 쌍**의 비율 — `nan` 은 분자에서도 분모에서도 뺀다.

    `nan` 은 분산 0 이라 **잴 수 없는** 쌍이다. `nan < 0` 은 False 라 그냥 두면
    "음수가 아닌 쌍" 으로 세어져 비율이 조용히 낮아진다 — 그리고 이 비율은
    "로테이션은 우연보다 드물다" 는 판정의 분자다.

    관측(`obs_neg_frac`)·널(`null_neg_frac`)·근거 스크립트(`scripts/ledger_numbers.py`)
    가 **같은 함수**를 봐야 한다. 스크립트는 화면이 인용하는 실측치를 만드는
    곳이라, 다른 자로 재면 근거와 주장이 다른 수가 된다.
    """
    vals = [v for v in values if v == v]
    return sum(1 for v in vals if v < 0) / len(vals) if vals else None


class Model:
    """화면 상태 + 페이로드에서 뽑은 파생 계열."""

    def __init__(self, data: dict):
        self.d = data
        self.dates: list[str] = data["dates"]
        self.sectors: list[str] = list(data["sectors"])
        # set/파일 순서에 기대지 않는다 — "m 두 번 = 코스닥" 손버릇이 조용히
        # 깨지면 사용자는 다른 시장을 보면서 같은 시장이라고 믿는다.
        seen = set(data["markets"])
        self.markets = (["전체"] + [m for m in MARKET_ORDER if m in seen]
                        + sorted(seen - set(MARKET_ORDER)))
        self.wi = WINDOWS.index(20)
        self.mi = 0
        self.ai = 0
        self.si = 0
        self.vi = 0
        self.row = 0
        self.hrow = 0
        self.detrend = False
        # 도움말은 **화면이 아니라 겹쳐 뜨는 모달**이다. VIEWS 에 넣으면 v 로
        # 순환할 때 끼어들어 표 사이를 오가는 손버릇을 망가뜨리고, 닫았을 때
        # 어디로 돌아갈지도 정해지지 않는다.
        self.help = False
        self.help_row = 0
        self._corr_cache: dict = {}
        self._null_cache: dict = {}
        # ⚠️ 캐시 키에 **정렬이 의존하는 것까지** 넣는다. `sort_key == "actor"` 도,
        # `spike`·`최대일몫` 도 고른 주체의 값이라 **actor 가 키에 있어야 한다.**
        # `kq-flow` 가 바로 이 자리에서 물렸다(역순이 키에 빠져 `r` 이 안 먹었다).
        self._rows_cache: dict = {}
        # 종목 수도 (시장,구간)마다 2,645개 이름을 훑는다 — 행마다 부르면 표 한
        # 판에 27번이다. 창 키는 페이로드가 어떤 구간을 실었는지의 사실이라
        # 한 번만 모은다.
        self._n_cache: dict = {}
        self._win_keys = {k for r in (data.get("names") or {}).values()
                          for k in (r.get("win") or {})}

    # --- 선택 ---
    @property
    def window(self) -> int:
        return min(WINDOWS[self.wi], len(self.dates))

    @property
    def market(self) -> str:
        return self.markets[self.mi]

    @property
    def actor(self) -> str:
        return ACTORS[self.ai][0]

    @property
    def actor_ko(self) -> str:
        return ACTORS[self.ai][1]

    @property
    def view(self) -> str:
        return VIEWS[self.vi][0]

    @property
    def sort_key(self) -> str:
        return SORTS[self.si][0]

    @property
    def has_cursor(self) -> bool:
        """이 화면에 **행 커서가 있는가** — 원장·전개만 그렇다.

        동시성은 행·열 순서가 평균상관이고 스크롤만 하며(`screen()` 이 커서를
        안 낸다), 한계는 산문이다. 그런데 상태줄은 그 두 화면에서도
        ``rows()[row]`` 의 섹터를 이름까지 대고 있었다 — 동시성의 `row` 는 스크롤
        위치라 화면의 그 행조차 아니다. **화면이 없는 선택을 보고했다.**

        판정을 여기 한 곳에 둔다. 커서를 그리는 쪽과 상태줄이 다른 답을 내면
        그 순간 다시 거짓말이 된다(``flow_view.State.sortable`` 과 같은 규율).
        """
        return self.view in ("ledger", "timeline")

    @property
    def sortable(self) -> bool:
        """이 화면에 **정렬이 있는가.**

        동시성의 순서는 평균상관이라 `s` 를 눌러도 행렬이 안 바뀌고(검사가
        그 사실을 붙들고 있다), 한계는 표가 아니다. 그런데 `s` 는 눌리기는 해서
        `si` 가 돌았고, 다른 화면으로 돌아가면 정렬이 바뀌어 있었다 — 누른 사람은
        방금 이 화면에서 무슨 일이 일어났다고 믿는다.

        헤더는 이미 사실을 적고 있었다(`순서[평균상관]`). 그 판단을 키까지 밀고
        간다 — ``kq-flow`` 가 종합 화면에서 밟은 그 자리다.
        """
        return self.has_cursor

    def cycle(self, what: str, step: int = 1) -> None:
        if what == "w":
            self.wi = (self.wi + step) % len(WINDOWS)
        elif what == "m":
            self.mi = (self.mi + step) % len(self.markets)
        elif what == "a":
            self.ai = (self.ai + step) % len(ACTORS)
        elif what == "s":
            self.si = (self.si + step) % len(SORTS)
        elif what == "v":
            self.vi = (self.vi + step) % len(VIEWS)
            self.row = 0
            return
        if what in ("w", "m", "s"):
            self.row = 0

    # --- 계열 ---
    def _mkts(self) -> list[str]:
        return self.d["markets"] if self.market == "전체" else [self.market]

    def daily(self, sector: str, key: str) -> list[float]:
        """(선택 시장) 섹터의 일별 순매수. ``key`` 가 ``resid`` 면 잔여."""
        n = len(self.dates)
        mk = self._mkts()
        cells = [self.d["flows"][m][sector] for m in mk if sector in self.d["flows"][m]]
        if not cells:
            return [0.0] * n
        if key == "resid":
            return [sum(residual(c, i) for c in cells) for i in range(n)]
        return [sum(c[key][i] for c in cells) for i in range(n)]

    def n_stocks(self, sector: str) -> int:
        """그 (시장,섹터)에서 **이 구간에 거래된** 종목 수.

        '상장 종목 수' 가 아니다. ``kq-flow`` 가 같은 열에서 그 이름을 버렸다 —
        벤더 마스터(``stocks``)에는 수급 보고가 두 달 전에 끊긴 이름이 남아 있고
        화면은 그 차이를 알 방법이 없다. 원장만 옛 정의(``n_by_sector`` =
        페이로드 260일 창에 한 번이라도 거래된 종목)로 남아 있어서, **같은
        (시장,섹터)를 두 앱이 다른 수로 말했다**: 실측(2026-08-28) 324칸 중
        49칸이 어긋났고, 그 중 2칸은 ``~``(얇은 섹터) 판정까지 갈렸다
        (코스닥/종이/목재 5·20일 — 원장 10 이라 경고가 없고 flow 9 라 경고가 있다).

        구간별 종목 집계(``names[code]["win"][창]``)가 있으면 그것으로 센다.
        없는 구간(260일)은 페이로드 전 구간이 곧 그 창이므로 **그 (시장,섹터)의
        종목을 다 센다** — 정의가 갈리는 게 아니라 같은 정의의 같은 답이다.
        ``names`` 자체가 없는 페이로드(합성·구버전)만 ``n_by_sector`` 로 떨어진다.
        """
        cnt = self._n_cache.get((self.market, self.window))
        if cnt is None:
            cnt = self._count_names(self.window)
            self._n_cache[(self.market, self.window)] = cnt
        return cnt.get(sector, 0)

    def _count_names(self, win: int) -> dict[str, int]:
        """(시장,구간) 하나의 섹터별 종목 수. :meth:`n_stocks` 가 캐시한다."""
        names = self.d.get("names") or {}
        mks = set(self._mkts())
        if not names:
            return {s: sum(int(self.d["n_by_sector"].get(m, {}).get(s, 0))
                           for m in mks) for s in self.sectors}
        w = str(win)
        need = w in self._win_keys
        out: dict[str, int] = {}
        for nm in names.values():
            if nm.get("market") not in mks:
                continue
            if need and not (nm.get("win") or {}).get(w):
                continue
            s = nm.get("sector")
            out[s] = out.get(s, 0) + 1
        return out

    def cap(self, sector: str) -> float:
        return sum(float(self.d["cap"].get(m, {}).get(sector, 0.0)) for m in self._mkts())

    def uniform_spike(self) -> float:
        """매일 고르게 들어왔을 때의 ``최대일몫`` — ``100 ÷ 구간일수``.

        앵커가 없으면 17.9% 가 큰지 작은지 알 수 없다. 20일이면 5.0% 다.
        """
        return 100.0 / self.window if self.window else 0.0

    def rows(self) -> list[dict]:
        """표 한 판. **캐시한다** — 한 번 그릴 때 세 곳에서 부른다(1회 23ms).

        키는 결과를 바꾸는 것 전부다: ``(시장, 구간, 정렬, 주체)``. 주체가 빠지면
        ``a`` 를 눌러도 ``최대일몫`` 과 ``선택주체`` 정렬이 안 바뀐다.
        """
        key = (self.market, self.window, self.sort_key, self.actor)
        hit = self._rows_cache.get(key)
        if hit is not None:
            return hit
        w = self.window
        ai = ACTOR_KEYS.index(self.actor)
        out = []
        for s in self.sectors:
            # 4주체 일별을 **한 번만** 뽑아 합계·잔여·몫·최대일을 여기서 다 낸다.
            # 예전엔 net×4 + net("resid") + spike 가 각각 daily 를 다시 돌았다.
            d4 = [self.daily(s, k)[-w:] for k in ACTOR_KEYS]
            r: dict = {"sector": s, "n": self.n_stocks(s)}
            for k, arr in zip(ACTOR_KEYS, d4):
                r[k] = sum(arr)
            r["resid"] = -sum(r[k] for k in ACTOR_KEYS)
            r["abs"] = sum(abs(r[k]) for k in ACTOR_KEYS)
            # ⚠️ 잔여 몫은 **일별 절댓값**으로 잰다. 구간 합계로 재면 부호가
            # 엇갈려 상쇄된다 — 실측(20일·전시장) 전기/전자는 합계 −175억인데
            # 일별 |잔여| 합이 1,934억(11배), IT 서비스는 48배다. 한계 §5 가
            # 경고한 "전체로 뭉개면 한 칸의 잔여가 지워진다"를 화면이 그대로
            # 저지르고 있었다.
            # ⚠️ 분모 하한(:data:`_MIN_RATIO_GROSS`)은 **두 비율에 같이** 건다.
            # 몫은 분모가 작으면 뜻을 잃고, 그 자리에서 오히려 커 보인다.
            gross = sum(abs(x) for arr in d4 for x in arr)
            rabs = sum(abs(sum(arr[i] for arr in d4)) for i in range(len(d4[0])))
            r["resid_pct"] = (rabs / gross * 100) if gross >= _MIN_RATIO_GROSS else None
            av = d4[ai]
            agross = sum(abs(x) for x in av)
            # 구간 중 **최대 하루**가 그 구간 총거래(gross)에서 차지하는 비중(%).
            # 구간 합만 보면 하루짜리 블록딜이 안 보인다. 실측: 최근 20일
            # 전기/전자는 2026-07-31 하루가 개인 gross 의 17.9% 였다.
            r["spike"] = (max(abs(x) for x in av) / agross * 100
                          if agross >= _MIN_RATIO_GROSS else None)
            out.append(r)
        key_s = self.sort_key
        if key_s == "name":
            out.sort(key=lambda r: r["sector"])
        elif key_s == "actor":
            out.sort(key=lambda r: -r[self.actor])
        else:
            k = key_s if key_s in ("abs", "spike", "n") else "abs"
            out.sort(key=lambda r: -(r[k] if r[k] is not None else -1e18))
        self._rows_cache[key] = out
        return out

    def selected(self) -> dict | None:
        rows = self.rows()
        return rows[min(self.row, len(rows) - 1)] if rows else None

    # --- 동시성 ---
    def _norm_series(self) -> dict[str, list[float]]:
        """섹터 시총으로 정규화한 일별 흐름(%p). 시총 0 인 섹터는 뺀다."""
        out = {}
        for s in self.sectors:
            c = self.cap(s)
            if c > 0:
                out[s] = [v / c * 100 for v in self.daily(s, self.actor)[-self.window:]]
        return out

    def _detrended(self, ser: dict[str, list[float]]) -> dict[str, list[float]]:
        """시장 공통요인을 β 회귀로 뺀 잔차.

        ⚠️ 이 데이터에서 β 제거 후 가장 음의 상관을 보이는 쌍은 전부
        ``전기/전자 ↔ 나머지`` 다. 전기/전자가 시총의 대부분이라 **구성적 인공물**과
        구분되지 않는다. 그래서 기본값은 끈 상태이고, 켜면 화면이 경고를 띄운다.
        """
        keys = list(ser)
        n = len(ser[keys[0]]) if keys else 0
        caps = {s: self.cap(s) for s in keys}
        tot = sum(caps.values()) or 1.0
        mkt = [sum(ser[s][i] * caps[s] for s in keys) / tot for i in range(n)]
        mm = sum(mkt) / n if n else 0.0
        vx = sum((v - mm) ** 2 for v in mkt)
        out = {}
        for s in keys:
            y = ser[s]
            my = sum(y) / n
            b = 0.0 if vx <= 0 else sum((mkt[i] - mm) * (y[i] - my) for i in range(n)) / vx
            out[s] = [y[i] - my - b * (mkt[i] - mm) for i in range(n)]
        return out

    def corr_matrix(self) -> tuple[list[str], list[list[float]]]:
        """상관행렬. 행·열 순서는 **평균상관 내림차순**이지 가나다순이 아니다.

        27×27 을 가나다로 늘어놓으면 무늬가 없다 — 섹터 이름의 첫 글자와 상관은
        아무 관계가 없으니 당연하다. 이 화면이 답해야 하는 질문은 하나뿐이고
        (로테이션은 이미 기각됐다) 그건 **"어느 섹터가 시장 공통요인과 따로
        노는가"** 다. 평균상관으로 줄세우면 그 답이 그림에 나온다 — 왼쪽 위에
        ``+⣿`` 덩어리, 오른쪽 아래로 ``-`` 띠, 그리고 맨 아래 네 줄이 답이다.

        ⚠️ ``s 정렬`` 은 이 화면에 **안 듣는다.** 안 듣는 값을 헤더 칩이 계속
        표시하지 않도록 :func:`header_lines` 가 여기서는 ``순서[평균상관]`` 을
        찍는다.
        """
        keys, m, _mean = self._corr_ordered()
        return keys, m

    def corr_means(self) -> dict[str, float]:
        """섹터 → 나머지와의 **평균 상관**. 그림을 못 봐도 결론이 남는 열이다."""
        return self._corr_ordered()[2]

    def _corr_ordered(self):
        key = (self.market, self.actor, self.window, self.detrend)
        if key in self._corr_cache:
            return self._corr_cache[key]
        ser = self._norm_series()
        if self.detrend:
            ser = self._detrended(ser)
        keys = [s for s in self.sectors if s in ser]
        raw = {a: {b: (1.0 if a == b else _corr(ser[a], ser[b])) for b in keys}
               for a in keys}
        mean = {}
        for a in keys:
            vals = [raw[a][b] for b in keys if b != a and raw[a][b] == raw[a][b]]
            mean[a] = sum(vals) / len(vals) if vals else float("nan")
        # 못 잰 섹터(분산 0)는 맨 뒤로 — nan 은 어떤 비교에도 False 라 정렬이
        # 조용히 뒤엉킨다. 그래서 정렬 키에서 nan 을 **명시적으로** 걷어낸다.
        keys.sort(key=lambda s: (mean[s] != mean[s], -mean[s]
                                 if mean[s] == mean[s] else 0.0, s))
        m = [[raw[a][b] for b in keys] for a in keys]
        self._corr_cache[key] = (keys, m, mean)
        return keys, m, mean

    def null_neg_frac(self) -> float | None:
        """순환이동 널에서 **음의 상관 쌍**이 차지하는 비율.

        순수 노이즈라면 0.5 다. 관측값이 이보다 낮으면 "섹터가 서로 반대로 갔다"는
        서사가 우연보다도 **드물다**는 뜻이다 — 실측이 정확히 그랬다(§3.3).
        """
        key = (self.market, self.actor, self.window, self.detrend)
        if key in self._null_cache:
            return self._null_cache[key]
        if self.window < _MIN_CORR_N:
            self._null_cache[key] = None
            return None
        ser = self._norm_series()
        if self.detrend:
            ser = self._detrended(ser)
        keys = list(ser)
        n = self.window
        if n < _MIN_CORR_N or len(keys) < 2:
            self._null_cache[key] = None
            return None
        rnd = random.Random(11)
        vals = []
        for _ in range(_NULL_SHIFTS):
            sh = {}
            for s in keys:
                t = rnd.randrange(n)
                sh[s] = ser[s][t:] + ser[s][:t]
            vals += [_corr(sh[a], sh[b])
                     for i, a in enumerate(keys) for b in keys[i + 1:]]
        # 잴 수 없는 쌍(nan)을 빼는 것은 `neg_frac` 한 곳이 한다 — 관측 쪽과
        # 근거 스크립트가 같은 함수를 본다.
        out = neg_frac(vals)
        self._null_cache[key] = out
        return out

    def obs_neg_frac(self) -> float | None:
        if self.window < _MIN_CORR_N:
            return None                  # 헛계산 방지 — 27×27 을 만들고 버리지 않는다
        keys, m = self.corr_matrix()
        if len(keys) < 2:
            return None
        return neg_frac(m[i][j] for i in range(len(keys))
                        for j in range(i + 1, len(keys)))


# ---------------------------------------------------------------- 렌더

def header_lines(mo: Model, width: int) -> list[str]:
    d = mo.d
    chip = "확정" if d.get("finalized") else "장중·미확정"
    l1 = (f" 자금 원장 · {mo.dates[-1]} {chip} · "
          f"{mo.dates[max(0, len(mo.dates) - mo.window)]}~{mo.dates[-1]} "
          f"({mo.window}거래일)")
    # ⚠️ 동시성 화면에는 **정렬이 없다**(순서는 평균상관이다). 칩이 정렬을 계속
    # 표시하면 화면이 안 듣는 값을 광고하는 것이고, 실제로 `s` 가 라벨만 바꾸던
    # 시절이 있었다. 판정은 `Model.sortable` **한 곳**이다 — 키를 받는 쪽과
    # 화면에 적는 쪽이 다른 답을 내면 그 순간 다시 거짓말이 된다.
    order = (f"정렬[{SORTS[mo.si][1]}]" if mo.sortable else "순서[평균상관]")
    l2 = (f" 화면[{VIEWS[mo.vi][1]}] 구간[{mo.window}일] 시장[{mo.market}]"
          f" 주체[{mo.actor_ko}] {order}")
    return [pad(l1, width), pad(l2, width)]


def _col(name: str, want: int, right: bool = True) -> tuple[str, int, bool]:
    """열 하나 — 폭은 **헤더가 잘리지 않는** 최소값 이상으로 잡는다.

    ``기타법인[억]`` 은 표시 폭 12 인데 10 을 주면 ``기타법인[`` 로 잘린다.
    이 저장소는 잘린 헤더를 한 번 겪었고(`test_headers_are_not_truncated_...`),
    폭을 손으로 세는 대신 여기서 계산한다.
    """
    return (name, max(want, cell_len(name)), right)


#: ``최대일몫`` 이 균등(=100÷구간일수)의 몇 배를 넘으면 `!` 를 붙이나.
#: 3배면 20일 구간에서 15% 다 — "추세가 아니라 사건" 쪽으로 넘어가는 자리다.
SPIKE_MARK_MULT = 3

#: 원장의 최소 폭. 4주체 + 잔여가 **한 덩어리**라 이보다 좁으면 표를 안 그린다.
LEDGER_MIN_W = 71
#: 전개의 최소 폭. 스파크라인이 잘리면 구간의 일부만 보여주면서 전부인 척한다.
TIMELINE_MIN_W = 48


def ledger_cols() -> list[tuple[str, int, bool]]:
    """원장 표의 **전체** 열 정의. 폭에 맞춰 줄이는 것은 :func:`_fit` 하나가 한다.

    한때 여기서 ``width >= 96`` 같은 문턱으로 열을 걸렀다. 그러면 자르는 규칙이
    두 군데(문턱과 ``_fit``)가 되고, 문턱이 이미 딱 맞게 걸러 주는 바람에 ``_fit``
    은 아무 일도 안 하는 **장식**이 된다 — 지워도 아무 검사가 안 깨지는 가드는
    가드가 아니다. 문턱을 없애 ``_fit`` 을 유일한 기제로 만든다.

    순서가 곧 우선순위다. 뒤에서부터 떨어지므로 **4주체와 잔여를 앞에** 둔다.
    ⚠️ 잔여를 부분적으로 보여주는 건 안 보여주는 것보다 나쁘다. ``잔여 = −Σ4주체``
    라서, 넷 중 하나라도 화면 밖이면 독자가 검산할 수 없는데 숫자는 검산 가능한
    얼굴로 앉아 있다. 그래서 다섯을 못 담는 폭에서는 표 대신 안내를 낸다
    (:data:`LEDGER_MIN_W`).
    """
    # 얇은 섹터 표시(~)는 **별도 1칸 열**이다. 색(A_DIM/파랑)에만 실으면
    # 무색 터미널·색맹·`--dump` 에서 경고가 통째로 사라진다. 값에 붙이면 열이 밀린다.
    return ([_col("섹터", 13, False), _col("", 1, False)]
            + [_col(f"{ko}[억]", 10) for _, ko in ACTORS]
            + [_col("잔여[억]", 9), _col("잔여몫[%]", 9),
               _col("최대일몫[%]", 10), _col("종목[수]", 8)])


#: 좁아서 표를 못 그릴 때의 안내 — 넓은 것부터. 잘린 안내는 안내가 아니다.
#: 예전엔 한 줄 고정이라 `pad` 가 잘랐고, 하필 **전개 안내는 어떤 폭에서도**
#: 온전히 안 나왔다(54칸이 필요한데 48칸 미만에서만 뜨는 문구였다).
LEDGER_NARROW_TIERS = (
    " 4주체와 잔여는 함께 봐야 한다(잔여 = −Σ4주체). 창을 넓혀라.",
    " 4주체와 잔여는 함께 봐야 한다. 창을 넓혀라.",
    " 잔여는 4주체와 같이 봐야 한다.",
    " 창을 넓혀라",
    " 좁다",
)
TIMELINE_NARROW_TIERS = (
    " 스파크라인이 잘리면 구간의 일부를 전부인 척 보여준다.",
    " 잘린 스파크라인은 일부를 전부인 척한다.",
    " 잘린 그림은 거짓말을 한다.",
    " 창을 넓혀라",
    " 좁다",
)


#: 좁아서 표를 못 그릴 때 **이 폭에서도 되는 것**. 넓은 것부터.
#: "창을 넓혀라"는 SSH 로 붙은 창에 자주 막다른 길이다 — 폭 70 에서 원장만
#: 안 그려질 뿐 전개(48칸)·동시성·한계는 멀쩡히 돌고, 그때 화면의 75%가 빈 줄이다.
#: 되는 것을 한 줄로 말하면 좁은 창이 쓸모를 잃지 않는다.
NARROW_ALSO_TIERS = (
    " 이 폭에서도 볼 수 있다 → v 로 전개 · 동시성 · 한계, ? 로 도움말",
    " 이 폭에서도 → v 전개 · 동시성 · 한계 · ? 도움말",
    " v 전개 · 동시성 · 한계 · ? 도움말",
    " v 다른 화면 · ? 도움말",
    " v 화면 · ? 도움",
)
#: 전개조차 안 되는 폭(<48)용 — 없는 것을 있다고 하면 안내가 아니라 오도다.
NARROW_ALSO_NO_TIMELINE_TIERS = (
    " 이 폭에서도 볼 수 있다 → v 로 동시성 · 한계, ? 로 도움말",
    " 이 폭에서도 → v 동시성 · 한계 · ? 도움말",
    " v 동시성 · 한계 · ? 도움말",
    " v 다른 화면 · ? 도움말",
    " v 화면 · ? 도움",
)


def narrow_also(width: int) -> str:
    """좁은 화면 안내의 **셋째 줄** — 여기서도 되는 것."""
    return tier_for(NARROW_ALSO_TIERS if width >= TIMELINE_MIN_W
                    else NARROW_ALSO_NO_TIMELINE_TIERS, width)


def _too_narrow(what: str, width: int, minw: int) -> str:
    """"안 그리는 이유" 첫 줄 — 폭에 맞춰 단계별로."""
    return tier_for((f" 폭이 {width}칸이라 {what} 안 그린다 — 최소 {minw}칸.",
                     f" {what} 안 그린다 — 최소 {minw}칸",
                     f" {what} 안 그린다({minw}칸)",
                     f" 최소 {minw}칸", " 좁다"), width)


def ledger_lines(mo: Model, width: int) -> tuple[list[str], list[bool], int]:
    """(행, 얇은섹터 여부, 헤더 줄 수).

    이건 **주체 × 섹터 이분 그래프의 인접행렬**이다. 노드-링크로 안 그린다 —
    4×27=108 간선을 선으로 그리면 교차 때문에 못 읽고, 선 굵기는 눈으로 비교가
    안 된다. 행렬은 같은 정보를 주면서 정렬·비교·정확한 수치를 동시에 준다.
    """
    if width < LEDGER_MIN_W:
        return ([pad(_too_narrow("원장을", width, LEDGER_MIN_W), width),
                 pad(tier_for(LEDGER_NARROW_TIERS, width), width),
                 pad(narrow_also(width), width)],
                [False, False, False], 3)
    # 폭이 모자라면 **열 경계에서** 떨어뜨린다. 줄을 통째로 자르면 숫자가
    # 자릿수 중간에서 끊겨 -1,360 이 -1 로 보인다 — 안 보이는 것보다 나쁘다.
    cols = _fit(ledger_cols(), width)
    used = sum(c[1] + 1 for c in cols)
    half = max(0, min(12, (width - used - 2) // 2))
    head = " ".join(pad(c[0], c[1], c[2]) for c in cols)
    if half:
        head += " " + pad(f"−  {mo.actor_ko}  +", 2 * half)
    out = [pad(head, width)]
    thin = [False]

    rows = mo.rows()
    scale = max((abs(r[mo.actor]) for r in rows), default=0.0)
    unif = mo.uniform_spike()
    for r in rows:
        sp, rp = r["spike"], r["resid_pct"]
        # 문턱 초과는 **글자**로 표시한다(`~` 와 같은 이유 — 색은 무색 터미널·
        # 색맹·`--dump` 에서 사라진다). 마커는 늘 1칸이라 자릿수는 안 밀린다.
        mark = "!" if sp is not None and sp >= SPIKE_MARK_MULT * unif else " "
        vals = ([r["sector"], "~" if r["n"] < THIN_N else ""]
                + [fmt_amt(r[k]) for k in ACTOR_KEYS] + [fmt_amt(r["resid"])]
                + [f"{rp:.1f}" if rp is not None else "—",
                   f"{sp:.1f}{mark}" if sp is not None else "—", str(r["n"])])
        cells = [pad(v, c[1], c[2]) for v, c in zip(vals, cols)]
        line = " ".join(cells)
        if half:
            line += " " + signed_bar(r[mo.actor], scale, half)
        out.append(pad(line, width))
        thin.append(r["n"] < THIN_N)
    return out, thin, 1


#: 스파크라인 읽는 법 — 넓은 것부터. :func:`tier_for` 가 폭에 맞춰 고른다.
#: **가운데가 0** 이라는 문장이 어느 단계에서도 안 사라진다. 그게 이 그림의
#: 전부이고, 예전 그림이 못 하던 것이다.
TIMELINE_NOTE_TIERS = (
    " ↑ 세로 한가운데가 0 이다 — 위 ⠒⠛ 는 순매수, 아래 ⠤⣤ 는 순매도."
    " 크기 눈금만 행마다 다르다(눈금[억] 열이 그 값이다).",
    " ↑ 가운데가 0 · 위 ⠒⠛ 순매수, 아래 ⠤⣤ 순매도 · 크기는 눈금[억] 열",
    " ↑ 가운데가 0 · 위 순매수 · 아래 순매도",
    " ↑ 가운데가 0",
)


def timeline_cols() -> list[tuple[str, int, bool]]:
    """전개 표의 **전체** 열 정의. 폭에 맞춰 줄이는 것은 :func:`_fit` 이 한다.

    함수로 뽑아 둔 이유는 상태줄이 **"지금 화면에 무슨 열이 있나"** 를 물어야 하기
    때문이다(:func:`visible_columns`) — 표에 이미 있는 값을 상태줄이 또 적던 자리다.
    """
    return [_col("섹터", 13, False), _col("", 1, False),
            _col("누적[억]", 11), _col("눈금[억]", 11)]


def timeline_lines(mo: Model, width: int) -> tuple[list[str], list[bool], int]:
    """섹터별 **누적** 순매수 스파크라인 — 회전을 보여주되 화살표는 없다.

    순위가 교차하는 게 눈에 보이면 그게 전부다. "A 에서 B 로 옮겨갔다"는 이
    화면에서 **읽을 수 없다** — 돈에 꼬리표가 없기 때문이다.

    스파크라인의 **0 은 어느 행에서나 세로 한가운데**다(:func:`spark`). 크기만
    행마다 정규화되므로 3종목짜리 부동산의 그림이 387종목짜리 전기/전자만큼
    극적으로 보일 수 있다 — 그래서 ``눈금[억]`` 열을 옆에 둔다. 그 값은
    **그려지는 점들**로 잰 것이라 그림과 정확히 맞는다(:func:`spark_scale`).
    """
    if width < TIMELINE_MIN_W:
        return ([pad(_too_narrow("전개를", width, TIMELINE_MIN_W), width),
                 pad(tier_for(TIMELINE_NARROW_TIERS, width), width),
                 pad(narrow_also(width), width)],
                [False, False, False], 3)
    cols = _fit(timeline_cols(), width)
    used = sum(c[1] + 1 for c in cols)
    # 남는 칸이 곧 스파크라인 폭이다. 최소값으로 **올려** 잡으면 줄이 폭을 넘어
    # pad 가 뒤를 잘라내고, 구간의 일부만 그려진 채 전부인 척한다.
    sw = min(mo.window, width - used)
    head = " ".join(pad(c[0], c[1], c[2]) for c in cols)
    # 헤더가 자기 열 폭보다 길면 잘린다("누적추이[" 처럼). 짧은 이름을 **고른다**.
    sh = tier_for((f"누적추이[{sw}점]", f"추이[{sw}]", "추이", ""), sw)
    head += " " + pad(sh, sw)
    # 경고는 **둘째 헤더 줄**이다. 스파크라인 열 폭(구간 길이)에 밀어 넣으면 짧은
    # 구간에서 잘려 사라지는데, 하필 그때 가장 필요한 문장이다. 줄로 빼도 폭이
    # 좁으면 또 잘리므로 폭에 맞는 표현을 **고른다** — 잘린 경고는 경고가 아니다.
    out = [pad(head, width), pad(tier_for(TIMELINE_NOTE_TIERS, width), width)]
    thin = [False, False]
    for r in mo.rows():
        v = mo.daily(r["sector"], mo.actor)[-mo.window:]
        cum, acc = [], 0.0
        for x in v:
            acc += x
            cum.append(acc)
        show = downsample(cum, sw)
        # ⚠️ 눈금은 **그려지는 점들**(show)로 잰다. 원본 260점으로 재면 그림의
        # 정규화와 어긋난다 — 폭 80·260일에서 최대 59.3% 다(실측).
        # 늘 양수라 부호를 붙이지 않는다 — `+` 를 달면 순매수처럼 읽힌다.
        vals = [r["sector"], "~" if r["n"] < THIN_N else "",
                fmt_amt(acc), f"{spark_scale(show):,.0f}"]
        line = " ".join(pad(x, c[1], c[2]) for x, c in zip(vals, cols))
        out.append(pad(line + " " + pad(spark(show), sw), width))
        thin.append(r["n"] < THIN_N)
    return out, thin, 2


#: 히트맵 한 칸 = 2글자. 첫 글자가 **부호**(ASCII `+`/`-`), 둘째가 **세기**.
#: 색만으로 부호를 표현하면 8색 터미널·평문 덤프·색맹에서 그림이 통째로 사라진다.
#: 글자에 부호를 박아두면 색은 **보조**가 되고 정보는 색 없이도 남는다.
#:
#: 세기는 **브라유 점 개수**로 낸다 — `⡀`1 `⣀`2 `⣤`4 `⣶`6 `⣿`8 점. 예전 `·░▒▓█`
#: 은 다섯 중 넷(`·▒▓█`)이 EAW 'A' 라, 터미널이 2칸으로 그리면 27열 히트맵이
#: 54칸을 먹고 행이 통째로 밀렸다. 폭이 생명인 화면이라 블록을 지키고 폭 계산만
#: 고치는 길(= 켜면 폭 예산 2배)은 못 간다 — 폭 80 에서 아예 못 그리게 된다.
#: 브라유는 EAW 'N' 이라 모드와 무관하게 1칸이고, 점 개수가 단조라 농도가 남는다.
HEAT_RAMP = " ⡀⣀⣤⣶⣿"


def heat_level(c: float) -> int:
    """상관을 −5..+5 의 정수 등급으로. 색은 앱이 고른다(여기는 curses 를 모른다)."""
    if c != c:
        return 0
    return max(-5, min(5, int(round(c * 5 / 0.5))))


#: 못 잰 쌍(분산 0). 빈칸으로 두면 ``|r|<0.05`` 와 구분되지 않는다 —
#: "무상관" 과 "잴 수 없다" 는 다른 말이다.
HEAT_UNDEF = " ?"


def heat_cell(c: float) -> str:
    if c != c:
        return HEAT_UNDEF
    lv = heat_level(c)
    if lv == 0:
        return "  "
    return ("+" if lv > 0 else "-") + HEAT_RAMP[abs(lv)]


def heat_diag() -> str:
    """대각선(자기상관 1.0) 자리. **비운다.**

    27칸을 ``+⣿`` 로 채워 봐야 아는 사실("자기 자신과는 상관 1")만 되풀이하고,
    무늬를 읽을 때 눈이 걸린다. 비우면 그 빈 줄이 곧 눈이 따라갈 기준선이 된다.
    """
    return "  "


def comove_verdict_tiers(obs: float, nul: float | None) -> tuple[str, ...]:
    """첫 줄 — 폭에 맞춰 **단계별로**. 잘린 결론은 뜻이 뒤집힌다.

    폭 80 에서 예전 한 줄은 ``→ 관측이 널보다 낮다 → 로테이션`` 에서 끝났다.
    원문은 "로테이션은 우연보다 **드물다**" 인데, 잘린 자리에서 읽으면 정확히
    반대로 읽힌다. 이 저장소는 배너·푸터·도움말 제목에 전부 단계를 깔아 뒀는데
    **가장 결론적인 이 줄에만** 없었다.
    """
    low = nul is not None and obs < nul - 0.03
    long = ("로테이션은 우연보다 드물다" if low else "널과 다르지 않다 — 무늬를 읽지 마라")
    short = ("로테이션은 드물다" if low else "무늬를 읽지 마라")
    o, n = obs * 100, (nul * 100 if nul is not None else float("nan"))
    return (
        f" 음의 상관 쌍  관측 {o:.0f}%  널(순환이동×{_NULL_SHIFTS}) {n:.0f}%  → {long}",
        f" 음상관 관측 {o:.0f}% · 널 {n:.0f}% → {long}",
        f" 관측 {o:.0f}% · 널 {n:.0f}% → {short}",
        f" 관측 {o:.0f}%/널 {n:.0f}% → {short}",
        f" → {short}",
    )


def comove_outliers(keys: list[str], mean: dict[str, float], n: int = 4) -> tuple[str, ...]:
    """"따로 노는 섹터" 한 줄 — 폭에 맞춰 이름을 하나씩 뺀다.

    ``keys`` 는 이미 평균상관 내림차순이라 **뒤에서** 집으면 그게 답이다.
    그림을 못 보는 사람(좁은 창·평문 덤프·스크롤 아래)에게도 결론이 남아야 한다.
    """
    tail = [k for k in reversed(keys) if mean[k] == mean[k]][:n]
    if not tail:
        return (" 평균상관을 낼 수 있는 섹터가 없다.",)
    names = [f"{k} {mean[k] * 100:+.0f}" for k in tail]
    return tuple(
        [" 시장 공통요인과 따로 논다(평균 낮은 순): " + " · ".join(names[:i])
         for i in range(len(names), 0, -1)]
        + [" 따로 논다: " + tail[0]]
    )


#: 행렬 아래 범례 — 각각 폭에 맞춰 단계별로. **척도를 밝힌다.**
#: 예전 범례는 "부호 + 세기(·░▒▓█)" 라고만 적어서 `▓` 가 0.3 인지 0.7 인지
#: 알 수 없었다 — 글자에서 r 을 역산할 수 없으면 그건 범례가 아니다. 그리고
#: 빈칸이 "무상관" 인지 "잴 수 없음" 인지도 구분되지 않았다 — 실측(2026-08-28,
#: 시장 3 × 구간 4 × 주체 4 전수)으로 빈칸은 쌍의 **2.6~65.5%** 다. 한 숫자로
#: 적어 두면 그건 어느 한 조합에서만 참인 수가 된다.
COMOVE_LEGEND_TIERS = (
    (" 세기: |r| 0.1↑ ⡀  0.2↑ ⣀  0.3↑ ⣤  0.4↑ ⣶  0.5↑ ⣿ (0.5 에서 포화).",
     " 세기: 0.1↑ ⡀  0.2↑ ⣀  0.3↑ ⣤  0.4↑ ⣶  0.5↑ ⣿",
     " ⡀ ⣀ ⣤ ⣶ ⣿ = |r| 0.1·0.2·0.3·0.4·0.5↑",
     " 세기 = |r| 0.1 단계"),
    (" 빈칸 = |r| 0.05 미만 · ? = 잴 수 없다(분산 0) · 대각선은 비웠다.",
     " 빈칸 = |r|<0.05 · ? = 잴 수 없다 · 대각선은 비웠다",
     " 빈칸 |r|<0.05 · ? 잴 수 없다",
     " 빈칸 = 무상관"),
    (" 평균 = 그 섹터와 나머지의 평균 상관 ×100 — 행·열이 그 내림차순이다.",
     " 평균 = 나머지와의 평균 상관 ×100 (행·열 순서)",
     " 평균 = 평균 상관 ×100",
     " 평균 ×100"),
)


def comove_lines(mo: Model, width: int) -> tuple[list[str], list[tuple], int]:
    """섹터쌍 상관 히트맵 + **널 대조**. 행·열은 **평균상관 내림차순**이다.

    이 화면이 답하는 질문은 하나다 — **"어느 섹터가 시장 공통요인과 따로 노는가."**
    로테이션("A 에서 B 로 옮겨갔다")은 여기서 이미 기각됐고(그게 첫 줄의 판정이다),
    남는 것이 그 질문이다. 가나다순으로 늘어놓으면 무늬가 없지만 평균상관으로
    줄세우면 답이 그림에 나온다: 왼쪽 위 ``+⣿`` 덩어리, 오른쪽 아래로 ``-`` 띠.
    그림을 못 봐도 ``평균`` 열과 셋째 줄이 같은 결론을 글자로 남긴다.

    헤더가 스스로 답한다 — "이 무늬가 우연보다 두드러지나". 순수 노이즈면 음수쌍이
    50% 다. 개인·기관·외국인은 **2~38%**(β제거 끈 상태, 시장 3 × 구간 4 = 12조합
    전수, 2026-08-28) 라 섹터들이 서로 **반대가 아니라 같이** 움직인다.
    **기타법인은 41~52% 로 널과 구분되지 않는다** — 그리고 ``d`` 로 β제거를 켜면
    셋도 17~53% 라 판정이 뒤집히는 조합이 있다. 주체마다·설정마다 다르므로 범위
    하나로 단정하지 말고 헤더의 판정줄을 읽어야 한다.

    ⚠️ 번호→이름 범례를 행렬 **아래**에 12줄로 두던 것을 없앴다. 폭 80·높이 24
    에서는 행이 15개만 보이는데 범례가 화면 밖이라, 정작 행렬을 보는 동안 열이
    무엇인지 알 수 없었다. 행·열 순서가 같으므로 **행 이름이 곧 범례**이고,
    그 사실을 둘째 줄이 말한다 — 없애도 정보가 안 준다.

    반환하는 두 번째 값은 색칠 지시 ``(y, x, width, level)`` 이다 — 문자열만으로는
    발산 색상을 표현할 수 없어서, 무엇을 어디에 칠할지만 알려주고 색은 앱이 정한다.
    """
    # ⚠️ 구간 검사가 **먼저**다. 예전엔 `corr_matrix()` 를 부른 뒤에 버려서
    # 5·10일 구간에서 27×27 을 헛계산했다.
    obs = mo.obs_neg_frac()
    if obs is None:
        note = f" 구간이 {mo.window}일이라 상관을 내지 않는다(최소 {_MIN_CORR_N}일)."
        return [pad(note, width), pad(" " + banner_for(width), width)], [], 1
    keys, m = mo.corr_matrix()
    mean = mo.corr_means()
    nul = mo.null_neg_frac()

    lines = [pad(tier_for(comove_verdict_tiers(obs, nul), width), width)]
    l2 = tier_for((
        " 동시에 갔다는 사실이다 — 옮겨갔다는 뜻이 아니다."
        "  행·열 순서는 같다(열 N = 행 N).",
        " 동시에 갔다는 사실이다 — 옮겨갔다는 뜻이 아니다. 열 N = 행 N.",
        " 옮겨갔다는 뜻이 아니다 · 열 N = 행 N",
        " 옮겨갔다는 뜻이 아니다",
    ), width)
    if mo.detrend and cell_len(l2) + 46 <= width:
        l2 += "  ⚠β제거: 전기/전자발 음상관은 인공물과 구분 안 된다"
    lines.append(pad(l2, width))
    lines.append(pad(tier_for(comove_outliers(keys, mean), width), width))
    marks: list[tuple] = []

    lab = 13
    avg = 4                       # `평균` 열(상관 ×100). 그림을 못 봐도 남는 결론.
    x0 = lab + 1 + avg + 1
    ncol = max(0, min(len(keys), (width - x0) // 2))
    pre = pad("", lab) + " " + pad("", avg) + " "
    tens = pre + "".join(f"{(i + 1) // 10 or ' ':>2}" for i in range(ncol))
    ones = (pad("", lab) + " " + pad("평균", avg) + " "
            + "".join(f"{(i + 1) % 10:>2}" for i in range(ncol)))
    lines += [pad(tens, width), pad(ones, width)]
    head = len(lines)

    for i, a in enumerate(keys):
        mv = mean[a]
        row = (pad(f"{i + 1:>2} {a}", lab) + " "
               + pad(f"{mv * 100:+.0f}" if mv == mv else "?", avg) + " ")
        for j in range(ncol):
            if i == j:
                # 대각선은 비운다 — 자기상관 1.0 이 27칸을 먹으면 무늬를 가린다.
                # 색칠 지시도 **안 낸다**: 등급 0 짜리 지시는 앱이 어차피 안
                # 칠하는데, 있으면 "지시한 자리에 부호 글자가 있다"는 계약이 깨진다.
                row += heat_diag()
                continue
            row += heat_cell(m[i][j])
            lv = heat_level(m[i][j])
            # 등급 0(빈칸·`?`)은 칠할 것이 없다. 지시를 내면 앱은 안 칠하는데
            # "지시한 자리에 부호 글자가 있다"는 계약만 거짓이 된다 — 그 계약을
            # 검사가 붙들고 있으므로 여기서 안 낸다.
            if lv:
                marks.append((head + i, x0 + 2 * j, 2, lv))
        lines.append(pad(row, width))

    lines.append(pad("", width))
    if ncol < len(keys):
        lines.append(pad(f" 폭이 좁아 {ncol}/{len(keys)} 열만 보인다 — 창을 넓혀라.",
                         width))
    # 척도를 밝힌다 — 글자에서 r 을 역산할 수 있어야 범례다. 예전 범례는
    # "부호 + 세기" 라고만 적어 `▓` 가 0.3 인지 0.7 인지 알 수 없었고,
    # 빈칸이 "무상관" 인지 "잴 수 없음" 인지도 구분이 안 됐다.
    for tiers in COMOVE_LEGEND_TIERS:
        lines.append(pad(tier_for(tiers, width), width))
    return lines, marks, head


LIMITS = [
    "한계 — 이 화면이 주장하지 않는 것",
    "",
    "1. 관측되는 것은 (날짜, 시장, 섹터, 주체)의 **순매수 금액**뿐이다.",
    "   섹터 A 에서 빠진 돈이 섹터 B 로 갔는지는 **관측되지 않는다**. 돈에 꼬리표가 없다.",
    "",
    "2. 주체 → 주체 전이도 그리지 않는다. 4주체의 주변합(독립 3개)으로는 쌍별 이전량",
    "   C(4,2)=6 개가 결정되지 않는다 — 해공간이 3차원 남는다. 섹터→섹터와 같은 미식별이다.",
    "",
    "3. '순매수'는 소유권 이전이지 시장에 **자금이 들어온 것이 아니다**. 모든 체결에는",
    "   같은 크기의 매수와 매도가 있다.",
    "",
    "4. 금액은 **종가 환산 근사**다. DB 는 순매매 수량만 준다(참값은 VWAP 가중).",
    "   방향과 상대 크기를 보는 용도이며 원 단위 정확도를 주장하지 않는다.",
    "",
    "5. '잔여' 열 = −Σ(4주체). KRX 의 나머지 주체(기타외국인 등) + 결측 + 환산오차의",
    "   혼합이다. 순수한 한 주체가 아니다.",
    "   실측(260일) — **크기는 분모를 정해야 말할 수 있다**: 거래대금(9,554조) 대비",
    "   0.0076%, gross 순매수(1,813조) 대비 0.0402%, 일별 전시장 중앙값 0.099%,",
    "   그리고 이 화면의 한 칸인 **(시장,섹터,일) 셀 단위로는 0.330%** 다 — 다만",
    "   이건 gross 가중 **평균**이고, **gross 100억 이상인 칸의 최악은 37.7%** 다",
    "   (거래소/금속/2026-04-07, 잔여 −237억 / gross 630억).",
    "   하한을 밝히는 이유: 하한 없이 세면 최악은 gross 0.01억짜리 칸의 **100%** 이고,",
    "   그건 잔여의 크기가 아니라 분모의 작음을 재는 수다. 몫은 분모를 정해야 뜻이 선다",
    "   — 그래서 표의 잔여몫·최대일몫도 분모가 1억 미만이면 값을 내지 않는다.",
    "   전체로 뭉갠 숫자는 '닫힌다'는 안심을 주지만 한 칸에서 잔여가 눈에 보일 만큼",
    "   크다는 사실을 지운다 — 그래서 별도 열로 남긴다.",
    "",
    "6. 섹터쌍 상관은 **동시성**이지 이동이 아니다. 그리고 이 데이터에서는 동시성조차",
    "   널을 못 이긴다 — 개인·기관·외국인은 음의 상관 쌍이 관측 **2~38%** 로",
    "   널(50%)보다 **드물다**. 섹터 자금흐름은 공통요인(위험선호)이 지배한다.",
    "   범위는 β제거 끈 상태로 시장 3 × 구간 4 = **12조합 전수**를 잰 값이다(2026-08-28).",
    "   β제거(d)를 켜면 셋도 17~53% 라 판정이 뒤집히는 조합이 있다.",
    "   예전엔 17~34% 라 적혀 있었는데 그건 잰 적 없는 좁은 범위였다 —",
    "   실측 주장은 **잰 범위와 같이** 적어야 한다.",
    "   **기타법인은 예외다** — 41~52% 라 구간을 어떻게 잡아도 널과 구분되지 않는다.",
    "   그 주체에 대해서는 '같이 움직인다'도 '반대로 움직인다'도 말할 수 없다.",
    "   화면 상단 판정줄이 고른 주체·구간에 대해 그때그때 답한다 — 그걸 읽어라.",
    "",
    "7. **생존편향.** 이 페이로드는 require_delisted=False 로 만들어졌고 키움 수급은",
    "   폐지 종목에 빈 응답을 준다. 폐지 종목의 마지막 자금흐름이 빠져 있다 —",
    "   섹터 총량이 과소 측정되는 방향이다. 크기는 실측했다(2026-08-28, 72종목):",
    "   거래대금의 **0.151%**, 기관 gross 의 **0.203%**(빠진 종목의 기관 gross ÷",
    "   실린 종목의 기관 gross). 그리고 **코스닥이 아니라 거래소에서 크다** —",
    "   거래소 0.195% vs 코스닥 0.041% 로 5배다. 종목 수는 코스닥이 많지만",
    "   (52 vs 20) 빠진 금액은 거래소가 크다. 수를 크기로 착각하지 마라.",
    "   숫자는 날마다 조금씩 바뀐다 — verify_report 의 F 층이 매일 다시 재고,",
    "   바뀌지 않아야 하는 것(빠진 폭이 작다 · 거래소가 코스닥보다 크다 ·",
    "   종목 수는 코스닥이 많다)만 검사한다.",
    "",
    "8. 주체는 4분류뿐이다. '외국인' 한 칸에 롱온리 자금과 헤지 북이 같이 들어 있다.",
    "",
    "키:  v 화면  w 구간  m 시장  a 주체  s 정렬  d β제거  ↑↓ 이동  ? 이 화면  q 종료",
]


def wrap_cells(text: str, width: int) -> list[str]:
    """표시 칸 기준 줄바꿈. 이어지는 줄은 원래 들여쓰기를 유지한다.

    한계 문단은 **표를 대신하는 문장**이라 잘리면 뜻이 바뀐다. 그런데 curses
    경로는 `pad` 로 잘라 문장을 버렸고, `--dump` 경로는 아무것도 안 해서
    폭을 넘겼다(폭 80 에서도 두 줄). 같은 글을 두 경로가 다르게 다뤘다.

    표의 숫자는 열 경계에서 떨어뜨리는 게 맞지만(잘린 숫자는 다른 값이 된다),
    산문은 접는 게 맞다 — 버리는 것보다 낫다.
    """
    # 소스의 **강조** 는 읽는 사람 눈에 띄라고 쓴 표기지 화면에 나갈 글자가 아니다.
    text = text.replace("**", "")
    if width < 8:
        return [text[:1]]
    lead = " " * (len(text) - len(text.lstrip()))
    out, cur, used = [], "", 0
    for word in text.split(" "):
        w = sum(cell_width(c) for c in word)
        if cur and used + 1 + w > width:
            out.append(cur)
            cur, used = lead + word, sum(cell_width(c) for c in lead) + w
        else:
            cur = (cur + " " + word) if cur else word
            used += (1 if used else 0) + w
    if cur:
        out.append(cur)
    # 한 낱말이 폭보다 길면(긴 URL·수식) 칸 단위로 쪼갠다.
    fixed = []
    for ln in out:
        while sum(cell_width(c) for c in ln) > width:
            cut, u = "", 0
            for ch in ln:
                cw = cell_width(ch)
                if u + cw > width:
                    break
                cut += ch
                u += cw
            fixed.append(cut)
            ln = lead + ln[len(cut):]
        fixed.append(ln)
    return fixed


def limits_body(width: int) -> list[str]:
    """한계 화면의 본문 — curses 와 `--dump` 가 **같은 것**을 쓴다."""
    return [ln for t in LIMITS for ln in (wrap_cells(t, width) if t.strip() else [""])]


def limits_lines(width: int, top: int, height: int) -> tuple[list[str], int]:
    body = [pad(t, width) for t in limits_body(width)]
    top = max(0, min(top, max(0, len(body) - height)))
    return body[top:top + height], len(body)


# ---------------------------------------------------------------- 도움말 · 힌트

#: 라벨 열의 표시 폭. ``기타법인[억]``(12칸)이 잘리지 않는 최소값이다.
#: ⚠️ ``flow_view.HELP`` 는 아직 10 이라 ``순매수상위[억]`` 이 ``순매수상위`` 로
#: 잘려 있다 — 그쪽은 이어쓰기 들여쓰기가 10 에 맞춰져 있어 같이 옮겨야 한다.
HELP_LABEL_W = 12

#: 원장 도움말 — **키와 열의 뜻**. 렌더는 ``flow_view.help_lines`` 가 한다.
#: 형식은 ``(이름, 설명)``, 이름이 비면 앞 항목의 이어쓰기다(``kq-flow`` 와 같다).
#:
#: 한계(:data:`LIMITS`)는 여기 안 옮긴다. 둘은 답하는 질문이 다르다 —
#: 도움말은 **"이 숫자가 뭐냐"**, 한계는 **"이 숫자가 무엇을 주장하지 않느냐"** 다.
#: 한 모달에 합치면 60줄이 넘어 키를 찾으러 온 사람이 스크롤을 세 번 해야 하고,
#: 한계는 지금 **스스로 화면 하나**(v 로 도달)라서 옮기면 그 화면이 빈다.
LEDGER_HELP = [
    ("", "── 키 ──"),
    ("↑↓ j k", "한 줄 이동. Home 맨 위 · Space·PgDn 한 화면 아래 · PgUp 위."),
    ("v V", "화면 바꾸기 — 원장·전개·동시성·한계. 대문자는 역방향."),
    ("w W", "구간 5·20·60·120·260 거래일. 대문자는 역방향."),
    ("m M", "시장 전체·거래소·코스닥. 대문자는 역방향."),
    ("a A", "주체 개인·외국인·기관·기타법인. 대문자는 역방향."),
    ("", "           고른 주체만 막대·최대일몫·전개·동시성에 쓰인다. 원장 표의"),
    ("", "           금액 열 넷은 주체 선택과 무관하게 늘 넷 다 나온다."),
    ("s S", "정렬 절대크기·선택주체·섹터명·최대일몫·종목수. 대문자는 역방향."),
    ("", "           동시성 화면에는 **안 듣는다** — 거기 순서는 평균상관이라서,"),
    ("", "           헤더 칩도 그 화면에서는 순서[평균상관] 으로 바뀐다."),
    ("", "           역순 토글은 없다 — 섹터명만 오름차순, 나머지는 내림차순이다."),
    ("d", "동시성 화면의 β제거 토글. 다른 화면에서는 아무 일도 하지 않는다."),
    ("? F1", "이 도움말. 안에서 ↑↓·PgUp/PgDn·g·G 로 훑고, q·Esc·?·Enter 로"),
    ("", "           닫는다. 도움말은 겹쳐 뜨는 모달이라 뒤의 화면은 그대로 있다."),
    ("q", "종료. 단 도움말 안에서는 **닫기만** 한다(한 번 더 눌러야 종료)."),
    ("Esc", "종료가 **아니다** — 느린 SSH 에서 방향키가 Esc 와 나머지로 쪼개져"),
    ("", "           도착하면(ESCDELAY 기본 1초) ↓ 를 눌렀을 뿐인데 앱이 끝난다."),
    ("", "한영 상태에서도 위 키가 그대로 듣는다. 다만 자판이 Shift 를 구분하지"),
    ("", "           않는 자리가 있어 **한글에서 역방향은 W(구간)뿐**이다 — 나머지는"),
    ("", "           소문자 동작으로 떨어진다(v·m·a·s 는 계속 눌러 한 바퀴 돌면 된다)."),
    ("", ""),
    ("", "── 화면 (v) ──"),
    ("원장", "주체 × 섹터 순매수 표. 행이 섹터, 열이 주체다. 간선 하나하나가"),
    ("", "           실측인 **이분 그래프의 인접행렬**이다."),
    ("전개", "섹터별 **누적** 순매수 스파크라인 — 구간 안에서 언제 들어왔나."),
    ("동시성", "섹터쌍 상관 히트맵 + 순환이동 널. 같이 움직였나만 본다."),
    ("한계", "이 화면이 주장하지 않는 것 여덟 가지. ↑↓ 로 스크롤한다."),
    ("", "           도움말과 답하는 질문이 다르다 — 여기는 뜻, 저기는 주장의 범위다."),
    ("", ""),
    ("", "── 원장 열 ──"),
    ("섹터", "벤더 분류(stocks.sector). KRX 업종 분류와 다르다."),
    ("~", "그 (시장,섹터)의 종목이 10개 미만 — '섹터'로 읽으면 안 된다"),
    ("", "           (부동산 3종목). 색이 없어도 남도록 **글자로 둔** 1칸 열이다."),
    ("개인[억]", "구간 누적 순매수 [억원] = Σ(그날 순매매 수량 × 그날 종가)."),
    ("", "           DB 는 **수량만** 주므로 금액은 종가 환산 근사다(참값은 VWAP"),
    ("", "           가중). 방향과 상대 크기를 보는 값이지 원 단위 정확도가 아니다."),
    ("외국인[억]", "같은 계산. 롱온리 자금과 헤지 북이 이 한 칸에 같이 들어 있다."),
    ("기관[억]", "같은 계산. 금투·보험·투신·은행·연기금·사모가 이 안에 들어 있다."),
    ("", "           **그 여섯의 합은 아니다** — 벤더가 주는 기관 합계를 그대로"),
    ("", "           쓴다. 실측(260일): 674,480행 중 60,021행(8.9%)에서 둘이"),
    ("", "           다르고, Σ|차이| 가 Σ|기관| 의 2.9% 다. 세부에 없는 항목이"),
    ("", "           합계에는 들어 있다는 뜻이라 세부를 더해 검산하면 안 맞는다."),
    ("기타법인[억]", "같은 계산. 일반 법인이다 — 빼면 그만큼 잔여로 넘어간다."),
    ("잔여[억]", "**오차가 아니라 다섯 번째 주체다.** 이름에 속지 마라 — 한국어에서"),
    ("", "           '잔여'는 나머지·오차로 읽히지만 이 값은 KRX 주체 분류 중 우리가"),
    ("", "           안 싣는 주체(기타외국인 등)의 **순매수 추정치**다."),
    ("", "           식은 −(개인+외국인+기관+기타법인). 전 주체의 순매수 합은 수량"),
    ("", "           기준으로 정확히 0 이라, 가진 넷의 합을 뒤집으면 나머지가 된다."),
    ("", "           결측·종가환산 오차도 여기 섞여 든다."),
    ("", "           4주체에 안분해 0 으로 만들지 않는다 — 그러면 **측정되지 않은**"),
    ("", "           **주체가 측정된 주체의 옷을 입는다.**"),
    ("", "           크기는 분모를 정해야 말할 수 있다(260일 실측): 거래대금 대비"),
    ("", "           0.0076%, (시장,섹터,일) 칸의 gross 가중 평균 0.330%. 다만 그건"),
    ("", "           평균이고, **gross 100억 이상인 칸의 최악은 37.7%** 다"),
    ("", "           (거래소/금속/2026-04-07, 잔여 −237억 / gross 630억). 하한을"),
    ("", "           밝혀야 하는 수다 — 하한 없이 세면 최악은 gross 0.01억짜리"),
    ("", "           칸의 **100%** 이고, 그건 잔여가 아니라 분모를 재는 것이다."),
    ("잔여몫[%]", "잔여가 그 섹터 순매매에서 차지한 몫 = Σ|일별 잔여| ÷ Σ|일별 4주체|"),
    ("", "           ⚠️ **합계가 아니라 일별 절댓값으로** 잰다. 구간 합계로 재면"),
    ("", "           부호가 엇갈려 상쇄된다 — 20일·전시장 전기/전자는 합계가"),
    ("", "           −175억인데 일별 |잔여| 합은 1,934억으로 **11배**, IT 서비스는"),
    ("", "           **48배** 다. 한계 §5 가 경고한 '전체로 뭉개면 한 칸의 잔여가"),
    ("", "           지워진다'를 이 화면이 그대로 저지르고 있었다."),
    ("", "           `기타제조 -1억` 은 무시하게 생겼지만 그 섹터 순매매의 0.81% 다."),
    ("", "           **분모가 1억 미만이면 값을 안 낸다**(`—`) — 그 아래에서는"),
    ("", "           비율을 이루는 금액이 억 단위 표에 하나도 안 보인다."),
    ("최대일몫[%]", "그 구간 총량 중 **하루가 차지한 몫** = max|일별| ÷ Σ|일별| × 100."),
    ("", "           분자·분모 모두 **고른 주체(a 로 바뀐다)·고른 시장**의 값이고"),
    ("", "           부호는 절댓값으로 죽인다. 매일 고르게 들어왔다면 100 ÷ 구간일수"),
    ("", "           다 — 20일이면 5%. 그 앵커는 상태줄이 행마다 같이 찍는다."),
    ("", "           그래서 17.9%(20일 전기/전자 개인, 실측)는 **추세가 아니라 사건**"),
    ("", "           이라는 뜻이다. 하루짜리 블록딜 하나로 구간 합이 만들어진다."),
    ("", "           잔여몫과 같은 하한을 쓴다 — 구간 gross 가 **1억 미만이면 `—`**"),
    ("", "           다. 실측에서 gross 0.09억짜리 칸이 `33.3!` 이라 적고 있었다."),
    ("!", "최대일몫이 균등의 3배를 넘었다는 표시. 20일이면 15% 다."),
    ("", "           `~` 와 같은 이유로 **색이 아니라 글자**다 — 무색 터미널·색맹·"),
    ("", "           `--dump` 에서 색은 통째로 사라진다. 구간 총량이 0 이면 '—' 다."),
    ("종목[수]", "그 (시장,섹터)에서 **이 구간에 거래된** 종목 수 — kq-flow 의 같은"),
    ("", "           열과 같은 집합이다. '상장 종목 수' 가 아니다: 벤더 마스터에는"),
    ("", "           수급 보고가 두 달 전에 끊긴 이름이 남아 있어서, 그걸 세면 두"),
    ("", "           화면이 같은 칸을 다른 수로 말한다(실측 324칸 중 49칸)."),
    ("", "           10 미만이면 ~ 마커가 붙고, 그 판정도 이 수로 한다."),
    ("절대크기", "정렬 전용 값 — Σ|4주체|. 열로는 안 보인다. 부호를 지우고 더하므로"),
    ("", "           서로 반대로 큰 주체가 있는 섹터가 위로 온다."),
    ("막대", "오른쪽 끝의 '−  주체  +' 열 — 고른 주체의 순매수. 한쪽 끝이 지금"),
    ("", "           화면 안의 최대 절댓값이라 **화면이 바뀌면 눈금도 바뀐다.**"),
    ("", "           반칸(⡇⢸) 해상도이고 넘치면 끝 칸이 ⣿ 로 찬다. 폭이 좁으면"),
    ("", "           아예 안 그린다."),
    ("", ""),
    ("", "── 전개 열 ──"),
    ("누적[억]", "구간 누적 순매수 [억원] — 고른 주체. 원장의 그 주체 칸과 같다."),
    ("눈금[억]", "스파크라인의 **세로 눈금** [억원] — ⠛ 과 ⣤ 끝 칸이 이 값이다."),
    ("", "           늘 양수라 부호를 안 붙인다. **그려지는 점들로 잰다** — 원본"),
    ("", "           260점으로 재던 예전 진폭[억] 은 그림이 다운샘플된 점으로"),
    ("", "           정규화된다는 걸 무시해 폭 80·260일에서 최대 59.3% 어긋났다"),
    ("", "           (실측 2026-08-28, 시장 3 × 주체 4 전수)."),
    ("추이", "구간 동안 순매수가 **누적된 경로**. 왼쪽이 구간 시작, 오른쪽이 구간의"),
    ("", "           **마지막 날**이다 — 묶어서 줄일 때도 마지막 점은 안 버린다"),
    ("", "           (예전엔 부동소수 반올림 때문에 폭 51·60일 같은 자리에서 하루가"),
    ("", "           통째로 안 그려졌고, 그때 누적[억] 이 눈금[억] 보다 컸다)."),
    ("", "           **세로 한가운데가 0 이다** — 위 ⠒⠛ 는 그때까지 순매수, 아래"),
    ("", "           ⠤⣤ 는 순매도, ⠐ 은 정확히 0. 0 이 어느 행에서나 같은 자리라"),
    ("", "           `계속 팔았다` 가 `높다가 없어졌다` 로 안 읽힌다(예전 ▁▂▃█ 은"),
    ("", "           행마다 바닥이 달라서 정확히 그렇게 읽혔다)."),
    ("", "           올라가면 들어오는 중 · 내려가면 빠져나가는 중 · 평평하면 멈췄다."),
    ("", "           크기는 **그 행 안에서** 정규화되므로 행끼리 비교되지 않는다 —"),
    ("", "           크기는 눈금[억] 열이 말한다. 글자는 브라유라 어떤 로케일에서도"),
    ("", "           1칸이다(▁▂▃█ 은 한글 로케일에서 2칸이 될 수 있다)."),
    ("", "           구간이 칸보다 길면 묶어서 각 묶음의 **끝점**을 찍는다(앞을 안 버린다)."),
    ("", ""),
    ("", "── 동시성 ──"),
    ("칸", "두 섹터의 상관 = 부호 + 세기(⡀⣀⣤⣶⣿). +⣿ 은 같이 샀다/팔았다,"),
    ("", "           -⣿ 은 반대로 갔다. 세기는 |r| 0.1 마다 한 단계, 0.5 이상이 ⣿ 다."),
    ("", "           색은 **보조**다 — 부호를 글자에 박아 무색·평문에서도 남는다."),
    ("", "           빈칸은 |r| 0.05 미만, `?` 는 **잴 수 없다**(구간 분산 0)."),
    ("", "           대각선(자기상관 1.0)은 비운다 — 27칸을 먹고 아는 것만 되풀이한다."),
    ("평균", "그 섹터와 나머지의 **평균 상관** ×100. 행·열이 이 값의 내림차순이라"),
    ("", "           왼쪽 위가 같이 가는 덩어리, 오른쪽 아래가 따로 노는 섹터다."),
    ("", "           이 화면이 답하는 질문이 그거다 — 로테이션은 판정줄이 이미"),
    ("", "           기각했고, 남는 것은 **누가 시장 공통요인과 따로 노는가** 다."),
    ("", "           행렬 셋째 줄이 그 답을 글자로도 남긴다(그림을 못 봐도 된다)."),
    ("상관", "섹터 시총으로 나눈 일별 흐름(%p)의 피어슨 상관. 고른 주체·구간."),
    ("", "           구간이 20일 미만이면 상관을 아예 내지 않는다."),
    ("널", "각 계열을 **무작위로 순환이동**해 20번 다시 잰 음의 상관 쌍 비율."),
    ("", "           순수 노이즈면 50% 다. **주체마다 다르다** — 개인·기관·외국인은"),
    ("", "           **2~38%** 로 널보다 낮다(섹터는 반대가 아니라 **같이** 움직인다)."),
    ("", "           범위는 β제거 끈 상태로 12조합 전수다(시장 3 × 구간 4, 08-28)."),
    ("", "           β제거(d)를 켜면 셋도 17~53% 라 판정이 뒤집히는 조합이 있다."),
    ("", "           실측 주장은 **잰 범위와 같이** 적어야 한다 — 예전의 17~34% 는"),
    ("", "           그러지 않았다(근거 스크립트의 260일 값을 화면의 범위로 적었다)."),
    ("", "           기타법인은 41~52% 로 **널과 구분되지 않는다.** 판정줄이 화면에서"),
    ("", "           그때그때 말해 주니 숫자를 외우지 말고 그 줄을 읽어라."),
    ("", "           그리고 상관은 **동시성이지 이동이 아니다.** 돈에 꼬리표가 없다."),
    ("β제거 (d)", "시총가중 시장요인을 회귀로 뺀 잔차. 기본은 꺼둔다 — 켜면 가장 음인"),
    ("", "           쌍이 전부 전기/전자↔나머지라 **구성적 인공물**과 구분이 안 된다."),
    ("", ""),
    ("", "── 알아둘 것 ──"),
    ("", "· 관측되는 것은 (날짜, 시장, 섹터, 주체)의 순매수 금액뿐이다."),
    ("", "  섹터→섹터 이동과 주체→주체 이전은 **그리지 않는다** — 주변합으로는"),
    ("", "  쌍별 흐름이 결정되지 않는다. v 로 '한계' 화면에 자세히 적었다."),
    ("", "· '순매수'는 소유권 이전이지 시장에 자금이 들어온 것이 아니다."),
    ("", "· 생존편향 — 폐지 종목의 마지막 흐름이 빠져 있다. 크기는 거래대금의"),
    ("", "  0.151%(기관 gross 의 0.203%)이고, **코스닥이 아니라 거래소에서 크다**"),
    ("", "  (거래소 0.195% vs 코스닥 0.041%). 종목 수는 코스닥이 많지만(52 vs 20)"),
    # ⚠️ 여기 "한계 화면 7번은 아직 옛 문장이다" 가 붙어 있었다. 한계 §7 은 이미
    # 같은 실측치로 고쳐져 있는데도 그렇게 적혀 있어서, **화면이 옆 화면의 상태를
    # 잘못 보고**했다. 두 화면이 같은 숫자를 적고 있는지는 검사가 본다.
    ("", "  빠진 **금액**은 거래소가 5배다. v 로 한계 화면 7번에 더 적었다."),
    ("", "· 숫자는 리포트 폴더의 payload.json 그대로다. numbers.html 과 같은 값이다."),
]


def help_body(width: int, offset: int = 0, height: int = 10**6):
    """도움말 본문 — 렌더는 ``flow_view.help_lines`` 를 **그대로** 쓴다.

    ``kq-flow`` 와 ``kq-ledger`` 는 같은 제품이다. 도움말 렌더가 두 벌이면
    라벨 폭·이어쓰기 들여쓰기·``**`` 떼기·스크롤 하한이 반드시 갈라진다 —
    이 저장소는 오늘 폭 단계 고르기가 네 벌로 갈라져 있던 걸 하나로 합쳤다.
    **내용만** 다르고 기제는 하나다.
    """
    return help_lines(width, offset, height, LEDGER_HELP, HELP_TITLE_TIERS,
                      label_w=HELP_LABEL_W)


def help_total() -> int:
    """도움말 전체 줄 수 — 스크롤 하한. 폭과 무관한 항목 수다."""
    return len(LEDGER_HELP)


HELP_FOOT_TIERS = (
    " ↑↓/PgDn:스크롤  Home:처음  q·Esc·?·Enter:닫기 (종료는 닫은 뒤 q 를 한 번 더)",
    " ↑↓/PgDn:스크롤  q·Esc·?:닫기 (종료는 한 번 더)",
    " ↑↓:스크롤  q:닫기(종료는 한 번 더)",
    " q:닫기(종료는 한 번 더)",
    " q:닫기",
)


def help_screen(mo: "Model", width: int, height: int) -> list[str]:
    """전면 도움말 — 정확히 ``height`` 줄. 제목·본문·위치표시 푸터."""
    lines, total = help_body(width, mo.help_row, max(1, height - 1))
    shown = max(0, len(lines) - 1)
    pos = f"  {mo.help_row + 1}-{mo.help_row + shown} / {total}"
    foot = tier_for(HELP_FOOT_TIERS, max(0, width - cell_len(pos))) + pos
    out = lines + [pad("", width)] * max(0, height - 1 - len(lines))
    return out[:height - 1] + [pad(foot, width)]


#: 정렬 키 → 도움말 항목 이름. 열이 없는 정렬(절대크기)도 항목이 있다.
_SORT_HELP = {"abs": "절대크기", "name": "섹터",
              "spike": "최대일몫[%]", "n": "종목[수]"}

#: 힌트바 전용 **짧은 설명** — 열 이름 → 한 줄. 없으면 :data:`LEDGER_HELP` 로 떨어진다.
#:
#: 왜 두 벌인가: 같은 문장을 길이 제약이 **정반대**인 두 자리에 쓰고 있었다.
#: 도움말은 전면 모달이라 유래·경고·실측치가 다 들어가도 되지만, 힌트바는 한 줄이라
#: 그 뒤가 `…` 로 잘려나간다 — 그리고 잘린 설명은 설명이 아니다. 실측(폭 120,
#: 절대크기 정렬)에서 화면에 늘 떠 있던 줄이 이랬다:
#: ``정렬 절대크기▼ · 정렬 전용 값 — Σ|4주체|. 열로는 안 보인다. 부호를 지우고 더하므로``
#:
#: 규칙: **그 숫자를 읽는 데 필요한 것만** — 분자·분모와 단위까지. 유래·경고·실측치는
#: `?` 의 일이다. 길이는 폭 80 에서 `…` 로 잘리지 않는 선을 지킨다(검사가 전
#: 화면 × 전 정렬 × 전 주체 × 폭 80 을 돌며 확인한다). ``kq-flow`` 의
#: ``HINT_DESC`` 와 같은 규칙이고 고르는 함수도 같은 모양이다.
LEDGER_HINT_DESC = {
    "절대크기": "Σ|4주체| — 부호를 지우고 더한 정렬 전용 값. 열로는 없다",
    "섹터": "벤더 분류(stocks.sector). 가나다순",
    "최대일몫[%]": "max|일별| ÷ Σ|일별| × 100 [%]. 균등은 100÷구간일수",
    "종목[수]": "이 구간에 거래된 종목 수. ~ 는 10개 미만",
    # 주체 열 넷 — `s` 가 `선택주체` 면 힌트가 가리키는 헤더가 `a` 따라 바뀐다.
    "개인[억]": "구간 누적 순매수 [억원] — 수량 × 그날 종가",
    "외국인[억]": "구간 누적 순매수 [억원] — 롱온리와 헤지 북이 한 칸이다",
    "기관[억]": "구간 누적 순매수 [억원] — 벤더 기관 합계(세부 합이 아니다)",
    "기타법인[억]": "구간 누적 순매수 [억원] — 일반 법인",
}


def hint_desc(header: str) -> str:
    """힌트바 한 줄 설명 — 짧은 것이 있으면 그것, 없으면 도움말의 긴 설명.

    폴백을 남기는 이유: 열이 하나 늘었는데 여기 안 적히면 힌트바가 **빈다.**
    긴 설명은 잘리기라도 하지만 빈 줄은 아무 말도 안 한다.
    """
    return LEDGER_HINT_DESC.get(header) or help_desc(LEDGER_HELP, header)

#: 동시성 화면의 힌트 — 정렬이 없는 화면이라 열 대신 **읽는 법**을 말한다.
HINT_COMOVE_TIERS = (
    " 동시성이지 이동이 아니다 — 같이 갔다는 사실일 뿐이다 · d 로 β제거 · ? 로 설명",
    " 동시성이지 이동이 아니다 · d 로 β제거 · ? 로 설명",
    " 동시성 ≠ 이동 · ? 로 설명",
    " 동시성 ≠ 이동 · ?",
    " ? 로 설명",
)
#: 한계 화면의 힌트.
HINT_LIMITS_TIERS = (
    " 한계 — ↑↓ 로 스크롤 · v 로 표로 돌아간다 · ? 는 키와 열의 뜻(다른 화면이다)",
    " 한계 — ↑↓ 스크롤 · v 로 표 · ? 는 키와 열",
    " 한계 — ↑↓ 스크롤 · ? 는 키와 열",
    " ↑↓ 스크롤 · ? 키와 열",
    " ? 키와 열",
)


def hint_text(mo: "Model", width: int) -> str:
    """푸터 위 **항상 보이는 한 줄** — 지금 줄세운 열의 뜻.

    ``?`` 는 전면 모달이라 "이 숫자가 뭐냐" 를 물은 사람이 **그 숫자를 보면서**
    답을 읽을 수 없다. ``kq-flow`` 가 같은 이유로 힌트바를 뒀다. 원장에도 맞는가 —
    맞다. 상태줄은 커서가 놓인 **행의 값**을 말하지 그 열이 **무슨 뜻인지**는
    말하지 않는데, 원장에서 설명이 가장 필요한 두 열(``잔여``·``최대일몫``)이
    바로 이름만으로는 오해되는 열이다.
    """
    if mo.view == "comove":
        return tier_for(HINT_COMOVE_TIERS, width)
    if mo.view == "limits":
        return tier_for(HINT_LIMITS_TIERS, width)
    key = mo.sort_key
    header = _SORT_HELP.get(key) or f"{mo.actor_ko}[억]"
    arrow = "▲" if key == "name" else "▼"
    # 도움말 문장을 그대로 빌려 쓰던 자리다 — 폭 80 은 물론 120 에서도 잘렸다.
    desc = hint_desc(header) or "? 로 설명"
    return hint_line(f" 정렬 {header}{arrow}", desc, width)


def visible_columns(mo: Model, width: int) -> set[str]:
    """**지금 그 폭에서 실제로 그려지는** 표의 열 이름.

    상태줄이 "표에 이미 있는 값" 을 판정하는 데 쓴다. 목록을 손으로 적으면 열을
    하나 떨어뜨렸을 때(폭이 좁아 `_fit` 이 버렸을 때, 아예 표를 안 그릴 때)
    상태줄이 **화면에 없는 값을 숨긴 채로** 남는다 — 그러면 그 값을 볼 데가
    아무 데도 없다. 그래서 그리는 쪽과 같은 함수(`ledger_cols`·`timeline_cols`)에
    같은 `_fit` 을 태워서 묻는다.
    """
    if mo.view == "ledger" and width >= LEDGER_MIN_W:
        return {c[0] for c in _fit(ledger_cols(), width) if c[0]}
    if mo.view == "timeline" and width >= TIMELINE_MIN_W:
        return {c[0] for c in _fit(timeline_cols(), width) if c[0]}
    return set()


#: 상태줄 첫 칸의 들여쓰기. :func:`status_line` 과 :func:`status_title_span` 이
#: **같은 상수**를 봐야 색이 글자와 어긋나지 않는다.
STATUS_INDENT = " "


def status_title_span(mo: Model, width: int) -> tuple[int, int] | None:
    """상태줄에서 **섹터 이름**의 (시작 표시칸, 폭). 세울 이름이 없으면 None.

    이 줄에서 "지금 무엇을 보고 있는가" 를 말하는 건 이름 하나뿐인데, 줄 전체가
    한 색이라 부속 정보에 묻혀 있었다. 좌표를 **뷰가 낸다** — 앱이 문자열을 다시
    뜯어 이름 길이를 추측하면 문구를 고칠 때 색이 조용히 어긋난다. 한글이 두 칸이라
    문자 인덱스가 아니라 **표시 칸**을 낸다(``flow_view.detail_title_span`` 과
    같은 관용구다).
    """
    if not mo.has_cursor:
        return None
    r = mo.selected()
    if not r:
        return None
    start = cell_len(STATUS_INDENT)
    w = min(cell_len(r["sector"]), max(width - start, 0))
    return (start, w) if w > 0 else None


def status_line(mo: Model, width: int) -> str:
    """커서가 놓인 행에서 **표를 봐도 알 수 없는 것** — 터미널에 툴팁이 없다.

    ``최대일몫`` 옆에 **균등 앵커**를 같이 찍는다. 17.9% 가 큰 값인지는 균등
    (=100÷구간일수, 20일이면 5.0%)을 알아야 판단된다 — 그 앵커가 화면 어디에도
    없으면 열 이름만으로는 아무것도 안 잡힌다. 고른 주체도 여기서 이름을 댄다
    (``a`` 를 누르면 그 열만 조용히 바뀌는데 열 헤더에는 주체가 없다).

    ⚠️ **표에 이미 있는 값은 안 적는다.** 예전엔 4주체 금액·잔여·종목수를 늘
    덧붙였는데, 원장 화면에서는 그게 바로 위 행에 열로 그대로 있는 값이다 — 폭만
    먹고, 하필 뒤에 붙던 것이 먼저 잘려 나갔다. 다만 **전개 화면에는 그 열들이
    없으므로** 거기서는 상태줄이 유일한 자리다. 판정은 화면을 그리는 함수에
    묻는다(:func:`visible_columns`) — 목록을 손으로 적으면 폭이 좁아 열이 떨어진
    순간 값이 화면에서 통째로 사라진다.

    커서가 없는 화면(동시성·한계)에서는 **아무 행도 안 적는다**(:attr:`Model.has_cursor`).
    그 자리는 **비운다** — 한때 배너를 넣었는데, 배너는 바로 아래 마지막 줄에 늘
    있으므로 같은 문장이 인접한 두 줄에 그대로 두 번 찍혔다(실측: 동시성·한계).
    빈 줄은 표와 하단 사이를 띄우는 일이라도 한다.
    """
    # ⚠️ 커서 검사가 **먼저**다. 예전엔 `selected()`→`rows()` 를 먼저 부르고
    # 나서 되돌아왔다 — 안 쓸 표를 매 프레임 만들었다.
    if not mo.has_cursor:
        return pad("", width)
    r = mo.selected()
    if not r:
        return pad("", width)
    shown = visible_columns(mo, width)
    parts = [r["sector"]]
    if r["spike"] is not None:
        # 값과 앵커는 **붙어 있어야** 비교가 된다. 값이 표에 있어도 여기 남기는
        # 이유가 그거다 — 앵커만 적으면 눈이 표와 이 줄을 오가야 한다.
        parts.append(f"최대일몫({mo.actor_ko}) {r['spike']:.1f}%"
                     f" · 균등 {mo.uniform_spike():.1f}%")
    # 잔여와 잔여몫은 **따로** 판정한다. 폭이 좁으면 `잔여몫[%]` 열만 먼저
    # 떨어지는데, 한 덩어리로 묶으면 그때 잔여몫이 화면에서 통째로 사라진다.
    resid = []
    if "잔여[억]" not in shown:
        resid.append(f"잔여 {fmt_amt(r['resid'])}")
    if r["resid_pct"] is not None and "잔여몫[%]" not in shown:
        resid.append(f"(일별 {r['resid_pct']:.1f}%)" if resid
                     else f"잔여몫 {r['resid_pct']:.1f}%")
    if resid:
        parts.append(" ".join(resid))
    if "종목[수]" not in shown:
        parts.append(f"종목 {r['n']}")
    parts += [f"{ko} {fmt_amt(r[k])}" for k, ko in ACTORS
              if f"{ko}[억]" not in shown]
    # ⚠️ **우선순위 순서**로 붙이고 안 들어가면 거기서 멈춘다. 예전엔 한 줄로
    # 이어 붙여 `pad` 가 잘랐고, 하필 뒤쪽에 있던 최대1일이 폭 132 에서도
    # 통째로 잘려 나갔다. 앞의 둘은 표를 봐도 뜻을 알 수 없는 값이다(앵커·주체).
    out = ""
    for p in parts:
        cand = f"{out} · {p}" if out else p
        if cell_len(STATUS_INDENT + cand) > width:
            break
        out = cand
    return pad(STATUS_INDENT + out, width)


#: 푸터 — 폭에 맞춰 **단계별로** 줄인다. 예전엔 한 줄 고정이라 배너와 이어 붙인
#: 뒤 문자 슬라이스로 잘랐고, 폭 80(SSH 기본)에서는 배너만으로 76칸이라 푸터가
#: 통째로 사라졌다. 어느 단계에서도 ``?`` 는 남긴다 — 줄어든 안내가 "여기가 전부"
#: 로 읽히면 안 된다(``flow_view.footer_line`` 과 같은 규칙).
#:
#: 그 뒤 반대로 기울었다 — ``v/V w/W m/M a/A s/S`` 처럼 **한 키의 두 방향을 다
#: 적으니** 푸터가 길어져서 정작 무슨 키가 있는지가 안 읽혔다. 이제 소문자 한 벌만
#: 적는다. **키는 하나도 안 건드렸다** — 대문자 역방향은 여전히 듣고 :data:`LEDGER_HELP`
#: 의 키 절에 그대로 적혀 있다. 바뀐 것은 **어느 화면에 적히느냐** 뿐이다
#: (``flow_view.FOOTER_TIERS`` 와 같은 규칙·같은 이유).
FOOTER_TIERS = (
    " v:화면 w:구간 m:시장 a:주체 s:정렬 d:β제거 ↑↓:섹터"
    " Home:처음 ?:도움말 q:종료",
    " v:화면 w:구간 m:시장 a:주체 s:정렬 d:β제거 ↑↓ ?:도움말 q:종료",
    " v화면 w구간 m시장 a주체 s정렬 ?:도움말 q:종료",
    " v w m a s:바꾸기 ?:도움말 q:종료",
    " ?:도움말 q:종료",
    " ?:키 q:종료",
)
#: 예전 이름 — 중간 단계가 기본이다.
FOOTER = FOOTER_TIERS[1]


def footer_line(width: int) -> str:
    """폭에 **온전히** 들어가는 가장 자세한 푸터."""
    return tier_for(FOOTER_TIERS, width)


def banner_footer(width: int) -> str:
    """마지막 줄 — 배너(본문) + 푸터(키). 정확히 ``width`` 칸.

    예전엔 앱이 ``(" " + BANNER + "   " + FOOTER)[:w]`` 로 **문자 슬라이스**해
    붙였다. 두 가지가 틀렸다 — (1) 폭 80(SSH 기본)에서 배너만으로 77칸이라
    푸터가 통째로 사라져 ``?`` 가 화면 어디에도 없었고, (2) 문자 수로 자르니
    한글이 섞인 줄에서 표시 칸과 어긋났다.

    배너가 **먼저** 자리를 잡는다(이 화면이 존재하는 이유가 그 경고다). 그래도
    푸터가 한 단계도 못 들어가면 배너를 한 단계 줄인다 — 배너는 어느 단계에서도
    "미관측" 을 지키므로 줄여도 경고는 남지만, 키는 없으면 아예 못 찾는다.
    """
    short = FOOTER_TIERS[-1]
    ban = banner_for(width)
    room = width - 1 - cell_len(ban) - 2
    if room < cell_len(short):
        for b in BANNER_TIERS:
            r = width - 1 - cell_len(b) - 2
            if r >= cell_len(short):
                ban, room = b, r
                break
        else:
            return pad(" " + ban, width)
    return pad(" " + ban + "  " + tier_for(FOOTER_TIERS, room), width)


#: 표와 하단 세 줄(힌트·상태줄·배너) 사이 **빈 줄**. 표 마지막 행이 힌트바에 딱
#: 붙어 있어서 어디까지가 표인지 눈이 못 끊었다 — 힌트바 첫 글자가 표의 다음 행처럼
#: 읽혔다(``kq-flow`` 가 상세 패널에서 밟은 그 자리다).
BOTTOM_GAP = 1

#: 여백을 넣고도 본문에 남아야 할 최소 줄 수 — 헤더 2줄 + 열 이름 1줄 + 3행.
#: 여백은 **가장 먼저 포기하는** 것이다. 폭에서 :func:`tier_for` 가 문구를 단계적으로
#: 줄이듯 높이에서도 없어도 되는 것부터 뺀다. 빈 줄 하나 때문에 볼 것이 없어지면
#: 읽기 좋아지려던 것이 읽을 것을 없앤 셈이다.
GAP_MIN_BODY = 6


def bottom_gap(height: int) -> int:
    """표와 하단 세 줄 사이 여백 — 자리가 모자라면 **0**.

    그리는 쪽과 :func:`screen` 이 같은 함수를 봐야 어긋나지 않는다.
    """
    return BOTTOM_GAP if height - 3 - BOTTOM_GAP >= GAP_MIN_BODY else 0


def screen(mo: Model, width: int, height: int) -> dict:
    """화면 하나 — 앱은 이걸 그리기만 한다. 길이가 정확히 ``height`` 인 행 목록.

    상태줄과 **한계 배너를 여기서 붙인다.** 앱이 붙이게 두면 "한계가 화면에 있다"를
    순수 함수로 검사할 수 없고, 배너는 정확히 잊히기 쉬운 종류의 것이다.

    dict 로 돌려주는 이유: 화면마다 딸린 것이 다르다(원장/전개는 얇은섹터 표시,
    동시성은 색칠 지시). 튜플로 돌려주면 호출부가 화면 종류를 알고 언패킹해야 해서
    "무엇을 그릴지"가 앱으로 새어 나간다.
    """
    height = max(4, height)          # 본문 1줄 + 힌트·상태줄·배너 3줄
    if mo.help:
        # 도움말은 **전면**이다. 표 위에 반쯤 겹치면 가리는 줄이 하필 지금 읽던
        # 줄이고, 좁은 SSH 창에서는 설명이 두세 줄만 남는다.
        return {"lines": help_screen(mo, width, height), "thin": [], "head": 1,
                "top_head": 1,
                "marks": [], "total": help_total(), "cursor": None,
                "hint_y": None, "status_y": None, "banner_y": None}
    # 힌트 + 상태줄 + 배너 3줄, 그리고 그 위 **여백 한 줄**(자리가 있으면).
    gap = bottom_gap(height)
    body_h = max(1, height - 3 - gap)
    thin: list[bool] = []
    marks: list[tuple] = []
    if mo.view == "limits":
        head_lines: list[str] = []
        body, total = limits_lines(width, mo.hrow, body_h)
        nh = 0
        cursor = None
        lines = body
        top = 0
    else:
        if mo.view == "ledger":
            lines, thin, nh = ledger_lines(mo, width)
        elif mo.view == "timeline":
            lines, thin, nh = timeline_lines(mo, width)
        else:
            lines, marks, nh = comove_lines(mo, width)
        head_lines = header_lines(mo, width)
        # 자리가 모자라면 **맨 위 헤더부터** 버린다(여백 다음 순서다). 예전엔
        # 헤더 두 줄을 끝까지 지켜서 낮은 창에서 표가 **한 행도** 안 남았다 —
        # 무엇을 보는지는 적혀 있는데 볼 것이 없었다. 뒤에서부터 버리므로
        # 날짜 줄이 마지막까지 남는다(``flow_app.layout`` 과 같은 규율).
        while head_lines and body_h - len(head_lines) - nh < 1:
            head_lines = head_lines[:-1]
        avail = body_h - len(head_lines)
        mo.row = max(0, min(mo.row, max(0, len(lines) - nh - 1)))
        view_h = max(1, avail - nh)
        if mo.view == "comove":
            top = max(0, min(mo.row, max(0, len(lines) - nh - view_h)))
        else:
            top = max(0, min(mo.row - view_h + 1, max(0, len(lines) - nh - view_h)))
        sl = slice(nh + top, nh + top + view_h)
        total = len(lines)
        # 커서가 있는 화면인지의 판정은 `Model.has_cursor` 한 곳이다 — 상태줄도
        # 같은 것을 본다(예전엔 여기서만 알고, 상태줄은 없는 선택을 이름까지 댔다).
        cursor = (len(head_lines) + nh + (mo.row - top)) if mo.has_cursor else None
        thin = ([False] * len(head_lines) + thin[:nh] + thin[sl]) if thin else []
        lines = lines[:nh] + lines[sl]

    body = head_lines + lines
    body = body[:body_h] + [pad("", width)] * max(0, body_h - len(body))
    out = (body + [pad("", width)] * gap
           + [pad(hint_text(mo, width), width), status_line(mo, width),
              banner_footer(width)])
    # 소스의 **강조** 는 읽는 사람 눈에 띄라고 쓴 표기지 화면에 나갈 글자가
    # 아니다. 문자열을 하나씩 쫓으면 다음 문구가 또 새어 나온다(동시성 헤더에서
    # 실제로 새어 나왔다) — **출구 한 곳에서** 뗀다. 폭 계산 뒤에 떼면 줄이
    # 짧아지므로, 뗀 만큼 다시 채워 표시 폭을 유지한다.
    out = [pad(ln.replace("**", ""), width) if "**" in ln else ln for ln in out]
    # marks 는 **최종 화면 좌표**다. 예전엔 뷰가 `y - top` 을 내보내고 앱이
    # 다시 `head + my` 를 더했다 — 둘 다 자기가 헤더를 더한다고 믿어서 색이
    # nh 줄만큼 밀렸고(실측 4줄), 부호 글자와 색이 **서로 다른 쌍**을 가리켰다.
    # 이 화면의 존재 이유가 색인데 그렇다. 좌표 계산은 여기 한 곳에서 끝낸다 —
    # 앱은 받은 y 를 그대로 쓴다(더하지 마라).
    hy = len(head_lines)
    # `head` 는 **머리 전체**(맨 위 헤더 + 표의 열 이름)이고, `top_head` 는 그 중
    # 맨 위 헤더 줄 수다. 앱이 두 자리를 다른 색으로 칠하려면 경계를 알아야 하는데,
    # 문자열을 다시 뜯어 맞히면 문구를 고칠 때 조용히 어긋난다.
    return {"lines": out, "thin": thin, "head": hy + nh, "top_head": hy,
            "marks": [(hy + y - top, x, w, lv) for y, x, w, lv in marks
                      if hy + nh <= hy + y - top < body_h],
            "total": total, "cursor": cursor, "hint_y": height - 3,
            "status_y": height - 2, "banner_y": height - 1}


def render_text(mo: Model, width: int = 100) -> str:
    """색 없는 평문 덤프 — 파이프·리다이렉트로 SSH 밖에서도 본다.

    설계 문서는 "인쇄·공유용 산출물"을 터미널이 못 하는 것으로 꼽았다. HTML 대신
    평문을 택한 이유는 3.4MB HTML 이 안 열려서 이 TUI 가 생겼기 때문이다.
    """
    out = []
    keep = mo.vi
    for vi, (v, ko) in enumerate(VIEWS):
        mo.vi = vi
        out.append(f"### {ko}")
        if v == "limits":
            out += limits_body(width)
        else:
            out += header_lines(mo, width)
            if v == "ledger":
                out += ledger_lines(mo, width)[0]
            elif v == "timeline":
                out += timeline_lines(mo, width)[0]
            else:
                out += comove_lines(mo, width)[0]
        out.append("")
    mo.vi = keep
    # 도움말도 **평문 경로에 싣는다.** 키와 열의 뜻은 인쇄·공유용 산출물에서
    # 오히려 더 필요하다(그 사람에게는 ``?`` 를 누를 터미널이 없다).
    out.append("### 키와 열")
    out += help_body(width, 0, 10**6)[0][1:]     # 제목 줄(닫기 안내)은 뺀다
    out.append("")
    out.append(banner_for(width))
    # screen() 과 같은 이유로 여기서도 뗀다 — 평문 경로는 screen() 을 안 거친다.
    return "\n".join(line.replace("**", "").rstrip() for line in out)
