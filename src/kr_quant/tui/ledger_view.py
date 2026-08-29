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

⚠️ 열 자르기(:func:`_fit`)만 ``flow_view._fit`` 을 그대로 베껴 왔다. 지금 그쪽
열 표현이 튜플에서 객체로 바뀌는 중이라 사인이 흔들린다 — 움직이는 사설 API 에
붙는 것보다 6줄을 갖는 편이 싸다. 그쪽이 정착하면 하나로 올릴 것.
"""

from __future__ import annotations

import json
import math
import os
import random

from kr_quant.tui.flow_view import (
    HELP_TITLE_TIERS, MARKET_ORDER, cell_len, cell_width, fmt_amt, help_desc,
    help_lines, hint_line, pad, tier_for)

ACTORS = (("indiv", "개인"), ("forgn", "외국인"), ("inst", "기관"), ("etc", "기타법인"))
ACTOR_KEYS = tuple(k for k, _ in ACTORS)
WINDOWS = (5, 20, 60, 120, 260)
VIEWS = (("ledger", "원장"), ("timeline", "전개"), ("comove", "동시성"), ("limits", "한계"))
SORTS = (("abs", "절대크기"), ("actor", "선택주체"), ("name", "섹터명"),
         ("spike", "최대1일"), ("n", "종목수"))

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

SPARK = "▁▂▃▄▅▆▇█"
#: 반칸 블록. 양수는 오른쪽으로 자라며 꼬리가 `▌`(칸의 왼쪽 절반), 음수는 왼쪽으로
#: 자라며 꼬리가 `▐`(칸의 오른쪽 절반). 두 글자로 양쪽 모두 **반칸 해상도**가 나온다.
#: 한때 `▏▎▍…` 여덟 단계를 썼는데 그건 항상 칸의 *왼쪽*을 채워서 음수 막대의 꼬리가
#: 한 칸 떨어져 보였다.
BLOCK_FULL, BLOCK_L, BLOCK_R = "█", "▌", "▐"

#: 이보다 종목이 적은 (시장,섹터)는 '섹터'로 읽으면 안 된다 — 거래소/부동산은 3종목이다.
THIN_N = 10
_MIN_CORR_N = 20        # 이보다 짧은 구간에서는 상관을 내지 않는다
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


def spark(values: list[float]) -> str:
    """8단계 블록 스파크라인. 전 구간이 같은 값이면 가운데 높이로 눕힌다."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi - lo <= 0:
        return SPARK[len(SPARK) // 2] * len(values)
    span = hi - lo
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / span * len(SPARK)))]
                   for v in values)


def _fit(cols: list[tuple[str, int, bool]], width: int) -> list[tuple[str, int, bool]]:
    """폭에 **온전히** 들어가는 열까지만 남긴다. 첫 열은 잘려도 남긴다
    (섹터 이름은 잘려도 뜻이 남지만, 숫자는 잘리면 다른 값이 된다).

    ``flow_view._fit`` 과 같은 규칙이다(위 모듈 독스트링의 ⚠️ 참조).
    """
    out, used = [], 0
    for c in cols:
        need = c[1] + (1 if out else 0)
        if out and used + need > width:
            break
        out.append(c)
        used += need
    return out


def downsample(values: list[float], n: int) -> list[float]:
    """구간이 화면 칸보다 길 때 **묶어서** 줄인다 — 앞을 잘라내지 않는다.

    ``values[-n:]`` 로 뒤만 남기면 누적 곡선이 시작점을 잃고, 그림이 "이 구간에
    아무 일도 없다가 최근에 움직였다"고 거짓말한다. 각 묶음의 **끝점**을 뽑는다.
    """
    if n <= 0 or len(values) <= n:
        return list(values)
    step = len(values) / n
    return [values[min(len(values) - 1, int((i + 1) * step) - 1)] for i in range(n)]


def signed_bar(value: float, scale: float, half: int) -> str:
    """0 을 가운데 둔 부호 막대. 표시 폭은 **항상** ``2 * half``.

    ``scale`` 은 막대 한쪽 끝(= ``half`` 칸)에 해당하는 값이다. 넘치면 잘리되
    끝 칸을 `█` 로 채워 "넘쳤다"가 보이게 둔다.
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
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)


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

    def net(self, sector: str, key: str) -> float:
        return sum(self.daily(sector, key)[-self.window:])

    def n_stocks(self, sector: str) -> int:
        return sum(int(self.d["n_by_sector"].get(m, {}).get(sector, 0))
                   for m in self._mkts())

    def cap(self, sector: str) -> float:
        return sum(float(self.d["cap"].get(m, {}).get(sector, 0.0)) for m in self._mkts())

    def spike(self, sector: str) -> float | None:
        """구간 중 **최대 하루**가 그 구간 총거래(gross)에서 차지하는 비중(%).

        구간 합만 보면 하루짜리 블록딜이 안 보인다. 실측: 최근 20일 전기/전자는
        2026-07-31 하루가 개인 gross 의 17.9% 였다(GUARDRAILS §10 취약성의 시각화판).
        """
        v = self.daily(sector, self.actor)[-self.window:]
        gross = sum(abs(x) for x in v)
        return (max(abs(x) for x in v) / gross * 100) if gross > 0 else None

    def rows(self) -> list[dict]:
        out = []
        for s in self.sectors:
            r = {"sector": s, "n": self.n_stocks(s), "spike": self.spike(s)}
            for k in ACTOR_KEYS:
                r[k] = self.net(s, k)
            r["resid"] = self.net(s, "resid")
            r["abs"] = sum(abs(r[k]) for k in ACTOR_KEYS)
            out.append(r)
        key = self.sort_key
        if key == "name":
            out.sort(key=lambda r: r["sector"])
        elif key == "actor":
            out.sort(key=lambda r: -r[self.actor])
        else:
            k = key if key in ("abs", "spike", "n") else "abs"
            out.sort(key=lambda r: -(r[k] if r[k] is not None else -1e18))
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
        key = (self.market, self.actor, self.window, self.detrend)
        if key in self._corr_cache:
            return self._corr_cache[key]
        ser = self._norm_series()
        if self.detrend:
            ser = self._detrended(ser)
        keys = [s for s in self.sectors if s in ser]
        m = [[1.0 if a == b else _corr(ser[a], ser[b]) for b in keys] for a in keys]
        self._corr_cache[key] = (keys, m)
        return keys, m

    def null_neg_frac(self) -> float | None:
        """순환이동 널에서 **음의 상관 쌍**이 차지하는 비율.

        순수 노이즈라면 0.5 다. 관측값이 이보다 낮으면 "섹터가 서로 반대로 갔다"는
        서사가 우연보다도 **드물다**는 뜻이다 — 실측이 정확히 그랬다(§3.3).
        """
        key = (self.market, self.actor, self.window, self.detrend)
        if key in self._null_cache:
            return self._null_cache[key]
        ser = self._norm_series()
        if self.detrend:
            ser = self._detrended(ser)
        keys = list(ser)
        n = self.window
        if n < _MIN_CORR_N or len(keys) < 2:
            self._null_cache[key] = None
            return None
        rnd = random.Random(11)
        neg = tot = 0
        for _ in range(_NULL_SHIFTS):
            sh = {}
            for s in keys:
                t = rnd.randrange(n)
                sh[s] = ser[s][t:] + ser[s][:t]
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    tot += 1
                    if _corr(sh[a], sh[b]) < 0:
                        neg += 1
        out = neg / tot if tot else None
        self._null_cache[key] = out
        return out

    def obs_neg_frac(self) -> float | None:
        keys, m = self.corr_matrix()
        if self.window < _MIN_CORR_N or len(keys) < 2:
            return None
        vals = [m[i][j] for i in range(len(keys)) for j in range(i + 1, len(keys))]
        return sum(1 for v in vals if v < 0) / len(vals) if vals else None


# ---------------------------------------------------------------- 렌더

def header_lines(mo: Model, width: int) -> list[str]:
    d = mo.d
    chip = "확정" if d.get("finalized") else "장중·미확정"
    l1 = (f" 자금 원장 · {mo.dates[-1]} {chip} · "
          f"{mo.dates[max(0, len(mo.dates) - mo.window)]}~{mo.dates[-1]} "
          f"({mo.window}거래일)")
    l2 = (f" 화면[{VIEWS[mo.vi][1]}] 구간[{mo.window}일] 시장[{mo.market}]"
          f" 주체[{mo.actor_ko}] 정렬[{SORTS[mo.si][1]}]")
    return [pad(l1, width), pad(l2, width)]


def _col(name: str, want: int, right: bool = True) -> tuple[str, int, bool]:
    """열 하나 — 폭은 **헤더가 잘리지 않는** 최소값 이상으로 잡는다.

    ``기타법인[억]`` 은 표시 폭 12 인데 10 을 주면 ``기타법인[`` 로 잘린다.
    이 저장소는 잘린 헤더를 한 번 겪었고(`test_headers_are_not_truncated_...`),
    폭을 손으로 세는 대신 여기서 계산한다.
    """
    return (name, max(want, cell_len(name)), right)


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
            + [_col("잔여[억]", 9), _col("최대1일[%]", 10), _col("종목[수]", 8)])


def col_span(cols, header: str) -> tuple[int, int] | None:
    cell = 0
    for name, w, _r in cols:
        if name == header:
            return cell, w
        cell += w + 1
    return None


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
                 pad(tier_for(LEDGER_NARROW_TIERS, width), width)],
                [False, False], 2)
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
    for r in rows:
        sp = r["spike"]
        vals = ([r["sector"], "~" if r["n"] < THIN_N else ""]
                + [fmt_amt(r[k]) for k in ACTOR_KEYS] + [fmt_amt(r["resid"])]
                + [f"{sp:.1f}" if sp is not None else "—", str(r["n"])])
        cells = [pad(v, c[1], c[2]) for v, c in zip(vals, cols)]
        line = " ".join(cells)
        if half:
            line += " " + signed_bar(r[mo.actor], scale, half)
        out.append(pad(line, width))
        thin.append(r["n"] < THIN_N)
    return out, thin, 1


#: 스파크라인의 세로 눈금 경고 — 넓은 것부터. :func:`tier_for` 가 폭에 맞춰 고른다.
TIMELINE_NOTE_TIERS = (
    " ↑ 세로 눈금은 행마다 다르다(각 행의 최저~최고를 8단계에 편다)."
    " 행 사이 크기 비교는 진폭 열로 하라.",
    " ↑ 세로 눈금은 행마다 다르다 — 크기 비교는 진폭 열로.",
    " ↑ 눈금은 행마다 다르다",
)


def timeline_lines(mo: Model, width: int) -> tuple[list[str], list[bool], int]:
    """섹터별 **누적** 순매수 스파크라인 — 회전을 보여주되 화살표는 없다.

    순위가 교차하는 게 눈에 보이면 그게 전부다. "A 에서 B 로 옮겨갔다"는 이
    화면에서 **읽을 수 없다** — 돈에 꼬리표가 없기 때문이다.

    ⚠️ 스파크라인은 **행마다 따로 정규화**된다(각 행의 최저~최고를 8단계에 편다).
    3종목짜리 부동산의 그림이 387종목짜리 전기/전자만큼 극적으로 보인다. 그래서
    ``진폭[억]`` 열을 옆에 둔다 — 그림의 세로 눈금이 행마다 다르다는 걸 숫자가 알려준다.
    """
    if width < TIMELINE_MIN_W:
        return ([pad(_too_narrow("전개를", width, TIMELINE_MIN_W), width),
                 pad(tier_for(TIMELINE_NARROW_TIERS, width), width)],
                [False, False], 2)
    cols = _fit([_col("섹터", 13, False), _col("", 1, False),
                 _col("누적[억]", 11), _col("진폭[억]", 11)], width)
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
        amp = (max(cum) - min(cum)) if cum else 0.0
        show = downsample(cum, sw)
        # 진폭은 늘 양수라 부호를 붙이지 않는다 — `+` 를 달면 순매수처럼 읽힌다.
        vals = [r["sector"], "~" if r["n"] < THIN_N else "",
                fmt_amt(acc), f"{amp:,.0f}"]
        line = " ".join(pad(x, c[1], c[2]) for x, c in zip(vals, cols))
        out.append(pad(line + " " + pad(spark(show), sw), width))
        thin.append(r["n"] < THIN_N)
    return out, thin, 2


#: 히트맵 한 칸 = 2글자. 첫 글자가 **부호**, 둘째가 **세기**.
#: 색만으로 부호를 표현하면 8색 터미널·평문 덤프·색맹에서 그림이 통째로 사라진다.
#: 글자에 부호를 박아두면 색은 **보조**가 되고 정보는 색 없이도 남는다.
HEAT_RAMP = " ·░▒▓█"


def heat_level(c: float) -> int:
    """상관을 −5..+5 의 정수 등급으로. 색은 앱이 고른다(여기는 curses 를 모른다)."""
    if c != c:
        return 0
    return max(-5, min(5, int(round(c * 5 / 0.5))))


def heat_cell(c: float) -> str:
    lv = heat_level(c)
    if lv == 0:
        return "  "
    return ("+" if lv > 0 else "-") + HEAT_RAMP[abs(lv)]


def comove_lines(mo: Model, width: int) -> tuple[list[str], list[tuple], int]:
    """27×27 상관 히트맵 + **널 대조**.

    헤더가 스스로 답한다 — "이 무늬가 우연보다 두드러지나". 순수 노이즈면 음수쌍이
    50% 다. 개인·기관·외국인은 17~34% 라 섹터들이 서로 **반대가 아니라 같이**
    움직인다. **기타법인은 45~54% 로 널과 구분되지 않는다** — 주체마다 다르므로
    범위 하나로 단정하지 말고 헤더의 판정줄을 읽어야 한다.

    반환하는 두 번째 값은 색칠 지시 ``(y, x, width, level)`` 이다 — 문자열만으로는
    발산 색상을 표현할 수 없어서, 무엇을 어디에 칠할지만 알려주고 색은 앱이 정한다.
    """
    keys, m = mo.corr_matrix()
    obs, nul = mo.obs_neg_frac(), mo.null_neg_frac()
    if obs is None:
        note = f" 구간이 {mo.window}일이라 상관을 내지 않는다(최소 {_MIN_CORR_N}일)."
        return [pad(note, width), pad(" " + banner_for(width), width)], [], 1

    verdict = ("관측이 널보다 낮다 → 로테이션은 우연보다 드물다"
               if nul is not None and obs < nul - 0.03 else
               "관측이 널과 다르지 않다 → 무늬를 읽지 마라")
    l1 = (f" 음의 상관 쌍  관측 {obs * 100:.0f}%   "
          f"널(순환이동×{_NULL_SHIFTS}) {nul * 100:.0f}%   → {verdict}")
    l2 = " 같은/반대 방향으로 동시에 갔다는 **사실**이다. 옮겨갔다는 뜻이 아니다."
    if mo.detrend:
        l2 += "  ⚠β제거: 전기/전자발 음상관은 구성적 인공물과 구분 안 된다"
    lines = [pad(l1, width), pad(l2, width)]
    marks: list[tuple] = []

    lab = 13
    ncol = max(0, min(len(keys), (width - lab - 1) // 2))
    tens = pad("", lab) + " " + "".join(f"{(i + 1) // 10 or ' ':>2}" for i in range(ncol))
    ones = pad("", lab) + " " + "".join(f"{(i + 1) % 10:>2}" for i in range(ncol))
    lines += [pad(tens, width), pad(ones, width)]
    head = len(lines)

    for i, a in enumerate(keys):
        row = pad(f"{i + 1:>2} {a}", lab) + " "
        x0 = lab + 1
        for j in range(ncol):
            row += heat_cell(m[i][j])
            marks.append((head + i, x0 + 2 * j, 2, heat_level(m[i][j])))
        lines.append(pad(row, width))

    lines.append(pad("", width))
    if ncol < len(keys):
        lines.append(pad(f" 폭이 좁아 {ncol}/{len(keys)} 열만 보인다 — 창을 넓혀라.",
                         width))
    lines.append(pad(" 칸 = 부호 + 세기(" + HEAT_RAMP.strip() + ")."
                     "  +█ = 같이 샀다/팔았다,  -█ = 반대로 갔다.", width))
    # 범례 — 칸을 고정하지 않으면 한글 길이가 제각각이라 열이 안 맞는다.
    ew = 18
    per = max(1, (width - 1) // ew)
    for i in range(0, len(keys), per):
        lines.append(pad(" " + "".join(pad(f"{j + 1:>2} {keys[j]}", ew)
                                       for j in range(i, min(i + per, len(keys)))),
                         width))
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
    "   이건 gross 가중 **평균**이고, 최악의 한 칸은 **37.7%** 다",
    "   (거래소/금속/2026-04-07, 잔여 +237억 / gross 630억).",
    "   전체로 뭉갠 숫자는 '닫힌다'는 안심을 주지만 한 칸에서 잔여가 눈에 보일 만큼",
    "   크다는 사실을 지운다 — 그래서 별도 열로 남긴다.",
    "",
    "6. 섹터쌍 상관은 **동시성**이지 이동이 아니다. 그리고 이 데이터에서는 동시성조차",
    "   널을 못 이긴다 — 개인·기관·외국인은 음의 상관 쌍이 관측 **17~34%** 로",
    "   널(50%)보다 **드물다**. 섹터 자금흐름은 공통요인(위험선호)이 지배한다.",
    "   **기타법인은 예외다** — 45~54% 라 구간을 어떻게 잡아도 널과 구분되지 않는다.",
    "   그 주체에 대해서는 '같이 움직인다'도 '반대로 움직인다'도 말할 수 없다.",
    "   화면 상단 판정줄이 고른 주체·구간에 대해 그때그때 답한다 — 그걸 읽어라.",
    "",
    "7. **생존편향.** 이 페이로드는 require_delisted=False 로 만들어졌고 키움 수급은",
    "   폐지 종목에 빈 응답을 준다. 폐지 종목의 마지막 자금흐름이 빠져 있다 —",
    "   섹터 총량이 과소 측정되는 방향이다. 크기는 실측했다(73종목): 거래대금의",
    "   **0.152%**, 기관 gross 의 0.33%. 그리고 **코스닥이 아니라 거래소에서 크다**",
    "   — 거래소 0.195% vs 코스닥 0.046% 로 4배다. 종목 수는 코스닥이 많지만",
    "   (53 vs 20) 빠진 금액은 거래소가 4배다. 수를 크기로 착각하지 마라.",
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
    ("", "           고른 주체만 막대·최대1일·전개·동시성에 쓰인다. 원장 표의"),
    ("", "           금액 열 넷은 주체 선택과 무관하게 늘 넷 다 나온다."),
    ("s S", "정렬 절대크기·선택주체·섹터명·최대1일·종목수. 대문자는 역방향."),
    ("", "           역순 토글은 없다 — 섹터명만 오름차순, 나머지는 내림차순이다."),
    ("d", "동시성 화면의 β제거 토글. 다른 화면에서는 아무 일도 하지 않는다."),
    ("? F1", "이 도움말. 안에서 ↑↓·PgUp/PgDn·g·G 로 훑고, q·Esc·?·Enter 로"),
    ("", "           닫는다. 도움말은 겹쳐 뜨는 모달이라 뒤의 화면은 그대로 있다."),
    ("q", "종료. 단 도움말 안에서는 **닫기만** 한다(한 번 더 눌러야 종료)."),
    ("Esc", "종료가 **아니다** — 느린 SSH 에서 방향키가 Esc 와 나머지로 쪼개져"),
    ("", "           도착하면(ESCDELAY 기본 1초) ↓ 를 눌렀을 뿐인데 앱이 끝난다."),
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
    ("기관[억]", "같은 계산. 금투·보험·투신·은행·연기금·사모의 합이다."),
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
    ("", "           평균이고 **최악의 한 칸은 37.7%** 다(거래소/금속/2026-04-07)."),
    ("최대1일[%]", "그 구간 총량 중 **하루가 차지한 비중** = max|일별| ÷ Σ|일별| × 100."),
    ("", "           분자·분모 모두 **고른 주체·고른 시장**의 값이고 부호는 절댓값으로"),
    ("", "           죽인다. 매일 고르게 들어왔다면 100 ÷ 구간일수 다 — 20일이면 5%."),
    ("", "           그래서 17.9%(20일 전기/전자 개인, 실측)는 **추세가 아니라 사건**"),
    ("", "           이라는 뜻이다. 하루짜리 블록딜 하나로 구간 합이 만들어진다."),
    ("", "           구간 총량이 0 이면 '—' 다."),
    ("종목[수]", "그 (시장,섹터)의 상장 종목 수. 10 미만이면 ~ 마커가 붙는다."),
    ("절대크기", "정렬 전용 값 — Σ|4주체|. 열로는 안 보인다. 부호를 지우고 더하므로"),
    ("", "           서로 반대로 큰 주체가 있는 섹터가 위로 온다."),
    ("막대", "오른쪽 끝의 '−  주체  +' 열 — 고른 주체의 순매수. 한쪽 끝이 지금"),
    ("", "           화면 안의 최대 절댓값이라 **화면이 바뀌면 눈금도 바뀐다.**"),
    ("", "           반칸(▌▐) 해상도이고 넘치면 끝 칸이 █ 로 찬다. 폭이 좁으면"),
    ("", "           아예 안 그린다."),
    ("", ""),
    ("", "── 전개 열 ──"),
    ("누적[억]", "구간 누적 순매수 [억원] — 고른 주체. 원장의 그 주체 칸과 같다."),
    ("진폭[억]", "누적 경로의 최고−최저 [억원]. 늘 양수라 부호를 안 붙인다."),
    ("", "           스파크라인의 눈금이 행마다 다르다는 걸 이 숫자가 알려준다."),
    ("추이", "구간 동안 순매수가 **누적된 경로**. 왼쪽이 구간 시작, 오른쪽이 끝이다."),
    ("", "           올라가면 들어오는 중 · 내려가면 빠져나가는 중 · 평평하면 멈췄다."),
    ("", "           높이는 **그 행 안에서** 최저~최고를 8단계(▁▂▃▄▅▆▇█)에 편 것이라"),
    ("", "           **행끼리 비교되지 않는다** — 크기는 진폭[억] 열이 말한다."),
    ("", "           구간이 칸보다 길면 묶어서 각 묶음의 **끝점**을 찍는다(앞을 안 버린다)."),
    ("", ""),
    ("", "── 동시성 ──"),
    ("칸", "두 섹터의 상관 = 부호 + 세기(·░▒▓█). +█ 은 같이 샀다/팔았다,"),
    ("", "           -█ 은 반대로 갔다. 세기는 |r| 0.1 마다 한 단계, 0.5 이상이 █ 이다."),
    ("", "           색은 **보조**다 — 부호를 글자에 박아 무색·평문에서도 남는다."),
    ("상관", "섹터 시총으로 나눈 일별 흐름(%p)의 피어슨 상관. 고른 주체·구간."),
    ("", "           구간이 20일 미만이면 상관을 아예 내지 않는다."),
    ("널", "각 계열을 **무작위로 순환이동**해 20번 다시 잰 음의 상관 쌍 비율."),
    ("", "           순수 노이즈면 50% 다. **주체마다 다르다** — 개인·기관·외국인은"),
    ("", "           17~34% 로 널보다 낮다(섹터는 반대가 아니라 **같이** 움직인다)."),
    ("", "           기타법인은 45~54% 로 **널과 구분되지 않는다.** 판정줄이 화면에서"),
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
    ("", "  0.152%(기관 gross 의 0.33%)이고, **코스닥이 아니라 거래소에서 크다**"),
    ("", "  (거래소 0.195% vs 코스닥 0.046%). 종목 수는 코스닥이 많지만(53 vs 20)"),
    ("", "  빠진 **금액**은 거래소가 4배다. 한계 화면 7번은 아직 옛 문장이다."),
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
              "spike": "최대1일[%]", "n": "종목[수]"}

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
    말하지 않는데, 원장에서 설명이 가장 필요한 두 열(``잔여``·``최대1일``)이
    바로 이름만으로는 오해되는 열이다.
    """
    if mo.view == "comove":
        return tier_for(HINT_COMOVE_TIERS, width)
    if mo.view == "limits":
        return tier_for(HINT_LIMITS_TIERS, width)
    key = mo.sort_key
    header = _SORT_HELP.get(key) or f"{mo.actor_ko}[억]"
    arrow = "▲" if key == "name" else "▼"
    desc = help_desc(LEDGER_HELP, header) or "? 로 설명"
    return hint_line(f" 정렬 {header}{arrow}", desc, width)


def status_line(mo: Model, width: int) -> str:
    """커서가 놓인 칸의 전체 수치 — 터미널에 툴팁이 없으니 상태줄이 그 자리다."""
    r = mo.selected()
    if mo.view == "limits" or not r:
        return pad(" " + banner_for(width), width)
    parts = [f"{r['sector']}", f"종목 {r['n']}"]
    parts += [f"{ko} {fmt_amt(r[k])}" for k, ko in ACTORS]
    parts.append(f"잔여 {fmt_amt(r['resid'])}")
    if r["spike"] is not None:
        parts.append(f"최대1일 {r['spike']:.1f}%")
    return pad(" " + " · ".join(parts), width)


#: 푸터 — 폭에 맞춰 **단계별로** 줄인다. 예전엔 한 줄 고정이라 배너와 이어 붙인
#: 뒤 문자 슬라이스로 잘랐고, 폭 80(SSH 기본)에서는 배너만으로 76칸이라 푸터가
#: 통째로 사라졌다. 어느 단계에서도 ``?`` 는 남긴다 — 줄어든 안내가 "여기가 전부"
#: 로 읽히면 안 된다(``flow_view.footer_line`` 과 같은 규칙).
FOOTER_TIERS = (
    " v/V:화면 w/W:구간 m/M:시장 a/A:주체 s/S:정렬 d:β제거 ↑↓:섹터"
    " ?:도움말 q:종료",
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
                "marks": [], "total": help_total(), "cursor": None,
                "hint_y": None, "status_y": None, "banner_y": None}
    body_h = max(1, height - 3)          # 힌트 + 상태줄 + 배너
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
        head_lines = header_lines(mo, width)
        avail = body_h - len(head_lines)
        if mo.view == "ledger":
            lines, thin, nh = ledger_lines(mo, width)
        elif mo.view == "timeline":
            lines, thin, nh = timeline_lines(mo, width)
        else:
            lines, marks, nh = comove_lines(mo, width)
        mo.row = max(0, min(mo.row, max(0, len(lines) - nh - 1)))
        view_h = max(1, avail - nh)
        if mo.view == "comove":
            top = max(0, min(mo.row, max(0, len(lines) - nh - view_h)))
        else:
            top = max(0, min(mo.row - view_h + 1, max(0, len(lines) - nh - view_h)))
        sl = slice(nh + top, nh + top + view_h)
        total = len(lines)
        cursor = None if mo.view == "comove" else len(head_lines) + nh + (mo.row - top)
        thin = ([False] * len(head_lines) + thin[:nh] + thin[sl]) if thin else []
        lines = lines[:nh] + lines[sl]

    body = head_lines + lines
    body = body[:body_h] + [pad("", width)] * max(0, body_h - len(body))
    out = body + [pad(hint_text(mo, width), width), status_line(mo, width),
                  banner_footer(width)]
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
    return {"lines": out, "thin": thin, "head": hy + nh,
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
