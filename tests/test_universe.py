"""PIT small-mid universe — cap-rank band + ADV floor eligibility."""

from __future__ import annotations

import inspect

import pandas as pd

from kr_quant.features.universe import CAP_RANK, ADV_FLOOR, smallmid_universe


def _panels():
    # 6 names, one date. Caps 600..100 → ranks 1..6. ADV: D below a test floor.
    caps = {"A": 600, "B": 500, "C": 400, "D": 300, "E": 200, "F": 100}
    advs = {"A": 5e4, "B": 5e4, "C": 5e4, "D": 1e3, "E": 5e4, "F": 5e4}
    cap = pd.DataFrame([{"code": c, "date": "2020-01-01", "market_cap": v} for c, v in caps.items()])
    adv = pd.DataFrame([{"code": c, "date": "2020-01-01", "adv": v} for c, v in advs.items()])
    return cap, adv


def test_band_excludes_mega_and_tiny_keeps_midtier():
    cap, adv = _panels()
    # Band (2,4] = ranks 3,4 = caps 400(C),300(D). Floor 1e4 drops D (adv 1e3).
    out = smallmid_universe(cap, adv, cap_rank=(2, 4), adv_floor=1e4).set_index("code")["eligible"]
    assert out["C"]                    # rank 3, liquid → eligible
    assert not out["D"]                # rank 4 but ADV below floor → excluded
    assert not out["A"] and not out["B"]   # ranks 1,2 (mega/large) excluded
    assert not out["E"] and not out["F"]   # ranks 5,6 (too small) excluded


def test_defaults_are_frozen_sepa_values():
    sig = inspect.signature(smallmid_universe)
    assert sig.parameters["cap_rank"].default == CAP_RANK == (100, 400)
    assert sig.parameters["adv_floor"].default == ADV_FLOOR == 10000.0


def test_cap_band_immune_to_rank_pitfall():
    # The rank-band pitfall: within a liquidity-pre-filtered (large-cap-only)
    # universe, "rank 2-4 of 6" still lands on genuinely large names. cap_band
    # should instead admit only names whose absolute cap falls in the band,
    # regardless of what else is in the passed-in universe.
    caps = {"MEGA1": 10e12, "MEGA2": 9e12, "LARGE1": 5e12, "MID1": 1e12, "MID2": 0.8e12, "SMALL1": 0.1e12}
    cap = pd.DataFrame([{"code": c, "date": "d1", "market_cap": v} for c, v in caps.items()])
    adv = pd.DataFrame([{"code": c, "date": "d1", "adv": 5e4} for c in caps])
    out = smallmid_universe(cap, adv, cap_band=(3e11, 2e12), adv_floor=1e4).set_index("code")["eligible"]
    assert out["MID1"] and out["MID2"]              # inside the absolute band
    assert not out["MEGA1"] and not out["MEGA2"] and not out["LARGE1"]  # too big
    assert not out["SMALL1"]                        # below the band


def test_cap_band_takes_precedence_over_cap_rank():
    cap, adv = _panels()
    # cap_rank alone would pick rank(2,4]; cap_band should override to its own logic.
    out = smallmid_universe(cap, adv, cap_rank=(2, 4), cap_band=(50, 250), adv_floor=0).set_index("code")["eligible"]
    assert out["E"] and out["F"]                     # caps 200, 100 in [50,250)
    assert not out["C"] and not out["D"]             # caps 400, 300 outside band


def test_ranking_is_per_date():
    cap = pd.DataFrame([
        {"code": "A", "date": "d1", "market_cap": 100}, {"code": "B", "date": "d1", "market_cap": 200},
        {"code": "A", "date": "d2", "market_cap": 300}, {"code": "B", "date": "d2", "market_cap": 50},
    ])
    adv = pd.DataFrame([{"code": c, "date": d, "adv": 9e9} for c in "AB" for d in ("d1", "d2")])
    # Band (1,2] keeps rank 2 (the smaller cap) each date; ranking is per-date.
    out = smallmid_universe(cap, adv, cap_rank=(1, 2), adv_floor=0.0).set_index(["date", "code"])["eligible"]
    assert out[("d1", "A")] and not out[("d1", "B")]   # d1: A(100)<B(200) → A rank2
    assert not out[("d2", "A")] and out[("d2", "B")]   # d2: B(50)<A(300) → B rank2
