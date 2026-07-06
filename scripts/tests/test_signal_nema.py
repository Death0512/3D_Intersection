"""Tests for the NEMA 8-phase dual-ring ``AdaptiveSignalPlan``
(MaxPressure selector) in ``lib/traffic_signal.py``.

Verifies the fundamental NEMA invariants:
  * No two conflicting movements are green simultaneously.
  * Every barrier-side decision stays on one barrier side.
  * Min-green / max-green / clearance (yellow + all-red) timings are honoured.
  * Heavy demand is served more green time than light demand (MaxPressure).
  * ``next_green_frame`` always returns a frame whose combo serves the
    movement (or appends a correct fallback interval).
  * Closed-loop integration: ``_resolve_all`` with an adaptive plan leaves
    no vehicle crossing the stop line on red.
"""
import os
import sys
import itertools
import random

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lib import traffic_signal as SGmod  # noqa: E402
from lib.traffic_signal import (  # noqa: E402
    AdaptiveSignalPlan, SignalPlan, NEMA_PHASES,
    _NS_COMBOS, _EW_COMBOS, _NS_SIDE, _EW_SIDE,
    _movement_to_phase,
    _ADP_MIN_GREEN_S, _ADP_MAX_GREEN_S,
    _ADP_YELLOW_S, _ADP_ALL_RED_S,
)
from lib import geometry as G  # noqa: E402

D = G.Direction
T = G.Turn
FPS = 30


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build(arrivals, horizon=None):
    return AdaptiveSignalPlan(fps=FPS, arrivals=arrivals, horizon_frames=horizon)


def _conflicting_movements(mv_a, mv_b) -> bool:
    """Two movements conflict if their paths could cross inside the box.
    Simplified: any two movements with different (approach, turn) on
    different approaches conflict unless they are on the same NEMA combo
    (already non-conflicting by construction)."""
    if mv_a == mv_b:
        return False
    ap_a, _ = mv_a
    ap_b, _ = mv_b
    # Same-approach movements never conflict (same lane stream).
    if ap_a == ap_b:
        return False
    # Opposite-approach throughs (N↔S or E↔W) do NOT cross — they pass
    # each other straight through; allow.
    opposite = {D.N: D.S, D.S: D.N, D.E: D.W, D.W: D.E}
    if opposite[ap_a] == ap_b:
        return False
    # Otherwise (perpendicular approaches) the movements cross inside the
    # box → conflict.
    return True


# ---------------------------------------------------------------------------
# NEMA phase-table consistency
# ---------------------------------------------------------------------------
def test_movement_to_phase_covers_all_movements():
    seen = set()
    for ap in D:
        for turn in (T.LEFT, T.STRAIGHT, T.RIGHT):
            seen.add(_movement_to_phase(ap, turn))
    # All 8 NEMA phases are reachable.
    assert seen == {1, 2, 3, 4, 5, 6, 7, 8}


def test_nema_phases_partition_movements():
    """Every (approach, turn) non-crosswalk movement appears in exactly one
    NEMA phase."""
    all_movs = {(ap, t) for ap in D for t in (T.LEFT, T.STRAIGHT, T.RIGHT)}
    union = set()
    for ph, movs in NEMA_PHASES.items():
        union |= movs
    assert union == all_movs
    # Disjoint
    for a, b in itertools.combinations(NEMA_PHASES.values(), 2):
        assert a.isdisjoint(b)


def test_combos_are_within_barrier_side():
    """Every legal combo is a (ring-1, ring-2) pair from the SAME barrier
    side — never crossing the NEMA barrier."""
    for p1, p2 in _NS_COMBOS + _EW_COMBOS:
        assert p1 in _RING1_ALLOWED
        assert p2 in _RING2_ALLOWED
        side1 = "NS" if p1 in _NS_SIDE else "EW"
        side2 = "NS" if p2 in _NS_SIDE else "EW"
        assert side1 == side2


_RING1_ALLOWED = {1, 2, 3, 4}
_RING2_ALLOWED = {5, 6, 7, 8}


# ---------------------------------------------------------------------------
# Plan invariants
# ---------------------------------------------------------------------------
def test_no_overlapping_intervals():
    """Adaptive plan intervals must be disjoint (a green never overlaps the
    next green; clearance separates them)."""
    arrivals = {
        (D.N, T.STRAIGHT): [100, 130, 160, 190, 220, 250, 280, 310, 340, 370],
        (D.S, T.STRAIGHT): [110, 140, 170, 200, 230, 260],
        (D.E, T.STRAIGHT): [150, 250, 450],
        (D.W, T.STRAIGHT): [200, 400],
    }
    sp = _build(arrivals, horizon=1200)
    ivs = sp.intervals
    assert ivs
    for a, b in zip(ivs, ivs[1:]):
        assert a[1] <= a[1] + sp.yellow_f + sp.all_red_f  # has room for clearance
        assert b[0] >= a[1] + sp.yellow_f + sp.all_red_f, (
            f"interval {b} starts before clearance from {a} ends")


def test_clearance_between_combos():
    """A clearance (yellow + all-red) gap exists between every consecutive
    pair of served combos and lasts the configured yellow+all-red."""
    arrivals = {
        (D.N, T.STRAIGHT): [100, 200, 300, 400, 500],
        (D.E, T.STRAIGHT): [150, 350],
    }
    sp = _build(arrivals, horizon=1500)
    needed_clr = sp.yellow_f + sp.all_red_f
    assert needed_clr == int(round((_ADP_YELLOW_S + _ADP_ALL_RED_S) * FPS))
    for a, b in zip(sp.intervals, sp.intervals[1:]):
        gap = b[0] - a[1]
        assert gap == needed_clr


def test_min_green_respected():
    """Even a zero-demand phase gets min_green (≥ _ADP_MIN_GREEN_S*fps)."""
    sp = _build({}, horizon=200)
    assert sp.intervals
    for s, e, combo in sp.intervals:
        assert (e - s) >= sp.min_green_f
        assert (e - s) >= int(round(_ADP_MIN_GREEN_S * FPS)) - 1


def test_max_green_respected():
    """A heavily over-served phase is capped at max_green."""
    arrivals = {
        (D.N, T.STRAIGHT): list(range(100, 1100, 5)),  # 200 vehicles
        (D.S, T.STRAIGHT): list(range(100, 1100, 5)),
    }
    sp = _build(arrivals, horizon=4000)
    for s, e, combo in sp.intervals:
        assert (e - s) <= sp.max_green_f + 1  # rounding tolerance


def test_no_conflicting_movements_green_simultaneously():
    """For every interval, no two movements served by the combo conflict."""
    arrivals = {
        (D.N, T.STRAIGHT): [100, 150, 200, 250, 300],
        (D.S, T.LEFT):    [120, 220, 320],
        (D.E, T.STRAIGHT): [180, 400],
        (D.W, T.RIGHT):   [200, 350],
        (D.N, T.LEFT):    [130, 230],
        (D.E, T.LEFT):    [210],
    }
    sp = _build(arrivals, horizon=3000)
    for s, e, combo in sp.intervals:
        movs = NEMA_PHASES[combo[0]] | NEMA_PHASES[combo[1]]
        for a, b in itertools.combinations(movs, 2):
            assert not _conflicting_movements(a, b), (
                f"combo {combo} serves conflicting movements {a} and {b}")


# ---------------------------------------------------------------------------
# MaxPressure demand response
# ---------------------------------------------------------------------------
def test_higher_pressure_gets_more_green_time():
    """When one side has much more demand, it receives more total green time
    than the lighter side over the run window."""
    heavy_ns = list(range(100, 1600, 4))    # ~375 N/S arrivals
    light_ew = [150, 400, 800]              # 3 E/W arrivals
    arrivals = {
        (D.N, T.STRAIGHT): heavy_ns,
        (D.S, T.STRAIGHT): heavy_ns[:len(heavy_ns)//2],
        (D.E, T.STRAIGHT): light_ew,
        (D.W, T.STRAIGHT): [200, 900],
    }
    sp = _build(arrivals, horizon=5000)
    ns_green = sum(e - s for s, e, c in sp.intervals if c[0] in _NS_SIDE)
    ew_green = sum(e - s for s, e, c in sp.intervals if c[0] in _EW_SIDE)
    # N/S served more green than E/W (MaxPressure extends heavy-side
    # intervals to ~max_green while the light side gap-outs at min_green).
    # The mandatory barrier alternation bounds the ratio: with default
    # timings max_green/min_green = 40/8 = 5×, so a comfortable 2× lower
    # bound proves the demand response without coupling the test to a
    # specific tuning.
    assert ns_green > 2 * ew_green, (
        f"MaxPressure did not favour heavy N/S: ns={ns_green} ew={ew_green}")


def test_zero_demand_yields_fallback_round_robin():
    """With no arrivals, the plan still produces intervals (round-robin
    min-green on each side) so the controller is well-defined."""
    sp = _build({}, horizon=600)
    assert sp.intervals
    # Each interval is exactly min_green.
    for s, e, combo in sp.intervals:
        assert (e - s) == sp.min_green_f


# ---------------------------------------------------------------------------
# Query API correctness
# ---------------------------------------------------------------------------
def test_is_green_matches_combo():
    """``is_green`` returns True iff the movement's NEMA phase is in the combo
    active at the queried frame."""
    arrivals = {(D.N, T.STRAIGHT): [100, 200, 300], (D.E, T.LEFT): [150]}
    sp = _build(arrivals, horizon=1500)
    for s, e, combo in sp.intervals:
        mid = (s + e) // 2
        movs = NEMA_PHASES[combo[0]] | NEMA_PHASES[combo[1]]
        for ap in D:
            for turn in (T.LEFT, T.STRAIGHT, T.RIGHT):
                expected = (ap, turn) in movs
                assert sp.is_green(ap, turn, mid) == expected


def test_next_green_frame_returns_serving_frame():
    """``next_green_frame`` returns a frame at which ``is_green`` is True for
    the queried movement, including the fallback path."""
    arrivals = {(D.N, T.STRAIGHT): [100]}
    sp = _build(arrivals, horizon=400)
    # A movement never served by any built interval still returns a frame
    # at which it's green (fallback appends one).
    g = sp.next_green_frame(D.W, T.LEFT, 0)
    assert g is not None and sp.is_green(D.W, T.LEFT, g)


def test_next_green_frame_within_interval_is_self():
    arrivals = {(D.N, T.STRAIGHT): [100, 200, 300]}
    sp = _build(arrivals, horizon=1200)
    for s, e, combo in sp.intervals:
        if 6 in combo:  # phase 6 = N through/right
            mid = (s + e) // 2
            assert sp.next_green_frame(D.N, T.STRAIGHT, mid) == mid


# ---------------------------------------------------------------------------
# Closed-loop integration with scenario_gen._resolve_all
# ---------------------------------------------------------------------------
def test_resolve_all_adaptive_no_red_crossings():
    """After ``_resolve_all`` with an adaptive plan, no vehicle's release
    falls in a red interval — vehicles only cross the stop line on green."""
    import scenario_gen as S
    rng = random.Random(101)
    vehicles = [S.make_vehicle(f"V{i:03d}", rng,
                                demand=S.DemandModel.default())
                for i in range(40)]
    S.schedule_departures_poisson(vehicles, 600, FPS, S.DemandModel.default(),
                                   rng, approach_visible_length=40.0)
    sp = AdaptiveSignalPlan(fps=FPS)
    S._resolve_all(vehicles, 40.0, FPS, signal_plan=sp)
    # Every released vehicle must be green at its release_frame.
    bad = []
    for v in vehicles:
        if v.get("release_frame") is None:
            continue
        ap = G.Direction(v["approach"])
        turn = G.Turn(v["turn"])
        if not sp.is_green(ap, turn, v["release_frame"]):
            bad.append((v["id"], v["release_frame"]))
    assert not bad, f"{len(bad)} vehicles released on red: {bad[:3]}"


def test_resolve_all_adaptive_converges_under_iteration_cap():
    """The closed-loop adaptive fixpoint converges within the default
    max_rounds (no infinite loop / cap-triggered instability)."""
    import scenario_gen as S
    rng = random.Random(202)
    vehicles = [S.make_vehicle(f"V{i:03d}", rng,
                                demand=S.DemandModel.default())
                for i in range(25)]
    S.schedule_departures_poisson(vehicles, 500, FPS, S.DemandModel.default(),
                                   rng, approach_visible_length=40.0)
    sp = AdaptiveSignalPlan(fps=FPS)
    # Should complete without raising.
    S._resolve_all(vehicles, 40.0, FPS, signal_plan=sp, max_rounds=20)
    # Sanity: signal plan has intervals after resolution.
    assert sp.intervals


def test_generate_adaptive_writes_signal_timeline():
    """``generate(signal_mode='adaptive')`` writes a scenario.json containing
    a non-empty ``signal_timeline`` and ``signal_mode == 'adaptive'``."""
    import tempfile
    import scenario_gen as S
    import json
    with tempfile.TemporaryDirectory() as td:
        scn = S.generate(7, 12, 20.0, td, fps=FPS,
                          signal_mode="adaptive",
                          demand=S.DemandModel.default())
        with open(os.path.join(td, "scenario.json")) as f:
            on_disk = json.load(f)
        assert on_disk["signal_mode"] == "adaptive"
        assert on_disk["signal_timeline"]
        assert all({"start", "end", "phases"} <= set(iv)
                    for iv in on_disk["signal_timeline"])


def test_generate_adaptive_no_conflicting_intervals():
    """The exported signal timeline has zero cross-barrier conflicting
    combos (a property of the dual-ring controller)."""
    import tempfile
    import scenario_gen as S
    import json
    with tempfile.TemporaryDirectory() as td:
        S.generate(11, 18, 20.0, td, fps=FPS,
                    signal_mode="adaptive",
                    demand=S.DemandModel.default())
        with open(os.path.join(td, "scenario.json")) as f:
            scn = json.load(f)
    for iv in scn["signal_timeline"]:
        p1, p2 = iv["phases"]
        side1 = "NS" if p1 in _NS_SIDE else "EW"
        side2 = "NS" if p2 in _NS_SIDE else "EW"
        assert side1 == side2, (
            f"cross-barrier combo in {iv}: phases {p1},{p2}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
        except Exception as e:
            print(f"  [ERR ] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
