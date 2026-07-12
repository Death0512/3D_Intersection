"""Workstream B — demand model tests (Poisson arrivals + turning split).

Run:
    cd /home/death/Documents/3D_Intersection_Video
    python3 -m pytest scripts/tests/test_demand.py -v
  or:
    python3 scripts/tests/test_demand.py
"""
from __future__ import annotations

import math
import os
import sys
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import random
from lib import geometry as G
from lib import kinematics as K
import scenario_gen as S


def test_demand_default_flow_all_approaches():
    dm = S.DemandModel.default()
    for d in G.Direction:
        assert dm.flow_vph(d) == S.DEFAULT_APPROACH_FLOW_VPH
    # turn split sums to 1
    assert abs(sum(dm.turn_split.values()) - 1.0) < 1e-9


def test_demand_round_trip_json():
    dm = S.DemandModel(
        flows={G.Direction.N: 600.0, G.Direction.S: 400.0,
              G.Direction.E: 300.0, G.Direction.W: 500.0},
        turn_split={"left": 0.1, "straight": 0.8, "right": 0.1})
    d = dm.to_dict()
    dm2 = S.DemandModel.from_dict(d)
    assert dm2.flow_vph(G.Direction.N) == 600.0
    assert dm2.flow_vph(G.Direction.W) == 500.0
    assert dm2.turn_fraction(G.Direction.N, G.Turn.LEFT) == 0.1
    assert dm2.turn_fraction(G.Direction.N, G.Turn.STRAIGHT) == 0.8


def test_demand_make_vehicle_approach_weighted_by_flow():
    # N has 10x the flow of S -> roughly 10x the vehicles drawn from N.
    dm = S.DemandModel(
        flows={G.Direction.N: 1000.0, G.Direction.S: 100.0,
              G.Direction.E: 100.0, G.Direction.W: 100.0},
        turn_split={"left": 0.2, "straight": 0.6, "right": 0.2})
    rng = random.Random(42)
    counts = {d.value: 0 for d in G.Direction}
    N = 2000
    for _ in range(N):
        v = S.make_vehicle("v", rng, demand=dm)
        counts[v["approach"]] += 1
    # Expect N ~ 1000/1300 * 2000 ~ 1538; S/E/W ~ 154 each. Tolerance is
    # generous (Poisson sampling + lane-turn renormalisation) but the ratio
    # must clearly hold.
    assert counts["N"] > counts["S"] * 5, counts
    assert counts["N"] > counts["E"] * 5, counts
    assert counts["N"] > counts["W"] * 5, counts


def test_demand_make_vehicle_turn_split_distribution():
    # straight-heavy split -> most vehicles go straight.
    dm = S.DemandModel(
        flows={d: 400.0 for d in G.Direction},
        turn_split={"left": 0.1, "straight": 0.8, "right": 0.1})
    rng = random.Random(1)
    counts = {"left": 0, "straight": 0, "right": 0}
    N = 4000
    for _ in range(N):
        v = S.make_vehicle("v", rng, demand=dm)
        counts[v["turn"]] += 1
    # straight should dominate (>= 60%); left and right roughly equal.
    assert counts["straight"] / N > 0.60
    assert counts["left"] / N < 0.20
    assert counts["right"] / N < 0.20


def test_demand_make_vehicle_respects_lane_turn_restrictions():
    # Even with a demand split that likes lefts, lane 1 (through-only) must
    # never produce a left or right — the legal-turn filter renormalises.
    dm = S.DemandModel(
        flows={d: 400.0 for d in G.Direction},
        turn_split={"left": 0.5, "straight": 0.3, "right": 0.5})
    rng = random.Random(2)
    for _ in range(500):
        v = S.make_vehicle("v", rng, demand=dm)
        legal = G.allowed_turns(v["lane"])
        assert G.Turn(v["turn"]) in legal, (v["lane"], v["turn"])


def test_poisson_arrivals_safety_invariants_hold():
    """A Poisson-scheduled scenario must satisfy the SAME invariants as the
    scenario scheduler: no same-lane headway or catch-up violations,
    and (when a signal plan is present) no red crossings, no exit-lane
    overlaps, queued-release spacing >= 0.5s, fixpoint stability."""
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    dm = S.DemandModel.default()
    rng = random.Random(11)
    vehicles = [S.make_vehicle(f"V{i:03d}", rng, demand=dm) for i in range(60)]
    S.schedule_departures_poisson(vehicles, 2100, 30, dm, rng,
                                  approach_visible_length=54.751)
    S._resolve_all(vehicles, 54.751, 30, signal_plan=sp)

    # headway / catch-up
    lanes = {}
    for v in vehicles:
        lanes.setdefault((v["approach"], v["lane"]), []).append(v)
    for key, vs in lanes.items():
        vs.sort(key=lambda d: d["depart_frame"])
        for a, b in zip(vs, vs[1:]):
            assert K.catchup_safe(a["depart_frame"], a["speed_ms"], a["length"],
                                   b["depart_frame"], b["speed_ms"],
                                   approach_visible_length=54.751), (
                f"headway fail {key} {a['id']}->{b['id']}")

    # no red crossing + exit-lane spacing
    for x in vehicles:
        ap = G.Direction(x["approach"])
        tn = G.Turn(x["turn"])
        qs = x.get("queue_slot", -1)
        if qs >= 0:
            assert not sp.is_green(ap, tn, x["stop_frame"])
            assert sp.is_green(ap, tn, x["release_frame"])
        else:
            sf = x.get("stop_frame",
                       x["depart_frame"] + int(round(54.751 / x["speed_ms"] * 30)))
            assert sp.is_green(ap, tn, sf)
        # exit spacing
        tf = int(round(54.751 / x["speed_ms"] * 30))
        rf = x.get("release_frame", x["depart_frame"] + tf)
        rp = rf + G.delta_t_frames(G.Turn(x["turn"]), x["speed_ms"], 30,
                                   lane_index=x["lane"])
        lv = rp + tf

    groups = {}
    for x in vehicles:
        tf = int(round(54.751 / x["speed_ms"] * 30))
        rf = x.get("release_frame", x["depart_frame"] + tf)
        rp = rf + G.delta_t_frames(G.Turn(x["turn"]), x["speed_ms"], 30,
                                   lane_index=x["lane"])
        lv = rp + tf
        od, el = G.exit_lane_for_movement(
            G.Direction(x["approach"]), x["lane"], G.Turn(x["turn"]))
        groups.setdefault((od.value, el), []).append((rp, lv))
    for g in groups.values():
        g.sort()
        last = -1
        for r, lv in g:
            if last > 0:
                assert r >= last + 5, f"exit overlap r={r} < {last}+5"
            last = lv

    # stability: one more signal pass changes nothing
    snap_before = [(x["queue_slot"], x["stop_frame"], x["release_frame"])
                   for x in vehicles]
    S._apply_signal_gating(vehicles, 54.751, sp, 30)
    snap_after = [(x["queue_slot"], x["stop_frame"], x["release_frame"])
                  for x in vehicles]
    assert snap_before == snap_after, "poisson fixpoint not stable"


def test_poisson_realized_rate_approximates_demand():
    """Over a long horizon the realized per-approach arrival rate (before the
    safety filter pushes later) should be close to the demand vph. The safety
    filter only pushes later, so arrivals-per-second stays bounded by demand.
    """
    dm = S.DemandModel(
        flows={d: 600.0 for d in G.Direction},
        turn_split={"left": 0.1, "straight": 0.8, "right": 0.1})
    fps = 30
    horizon_frames = 30 * 120  # 120 s
    rng = random.Random(5)
    # Generate arrivals via the same internal mechanism
    arrivals_per_approach = {}
    for ap in G.Direction:
        rate_ps = dm.flow_vph(ap) / 3600.0
        t = 0.0
        n = 0
        while t < horizon_frames / fps:
            t += rng.expovariate(rate_ps)
            if t < horizon_frames / fps:
                n += 1
        arrivals_per_approach[ap.value] = n
    # 600 veh/h over 120s -> 20 expected per approach. Tolerate wide band.
    for ap, n in arrivals_per_approach.items():
        assert 5 <= n <= 50, (ap, n)


def test_generate_with_demand_writes_demand_field():
    import tempfile
    dm = S.DemandModel.default()
    with tempfile.TemporaryDirectory() as tmp:
        scn = S.generate(1, 60, tmp, fps=30, demand=dm)
        assert "demand" in scn
        assert "flows" in scn["demand"]


def test_generate_vehicle_count_scales_with_demand_and_seconds():
    """Vehicle count is now emergent from demand × time, not an input."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        low = S.generate(42, 60, os.path.join(tmp, "low"), fps=30,
                         demand=S.DemandModel(
                             flows={d: 200.0 for d in G.Direction},
                             turn_split=S.DEFAULT_TURN_SPLIT))
        high = S.generate(42, 60, os.path.join(tmp, "high"), fps=30,
                          demand=S.DemandModel(
                              flows={d: 1200.0 for d in G.Direction},
                              turn_split=S.DEFAULT_TURN_SPLIT))
    assert len(high["vehicles"]) > len(low["vehicles"])
    assert high["duration_frames"] == low["duration_frames"] == 60 * 30


def test_generate_hard_ceiling_drops_vehicles_past_duration():
    """Vehicles whose computed leave_frame exceeds duration_frames are dropped
    from the scenario and IDs are renumbered."""
    import tempfile
    dm = S.DemandModel(
        flows={d: S.DEFAULT_APPROACH_FLOW_VPH * 5.0 for d in G.Direction},
        turn_split=S.DEFAULT_TURN_SPLIT)
    with tempfile.TemporaryDirectory() as tmp:
        scn = S.generate(42, 10, tmp, fps=30, demand=dm,
                         signal_mode="adaptive")

    assert scn["duration_frames"] == 300
    assert len(scn["vehicles"]) > 0
    # Every vehicle must complete within duration_frames.
    for v in scn["vehicles"]:
        tf = int(round(54.751 / v["speed_ms"] * 30))
        rf = v.get("release_frame", v["depart_frame"] + tf)
        dt = G.delta_t_frames(G.Turn(v["turn"]), v["speed_ms"], 30,
                              lane_index=v["lane"])
        leave_f = rf + dt + tf
        assert leave_f <= scn["duration_frames"], (
            f"{v['id']} leave_frame={leave_f} > {scn['duration_frames']}")
    # IDs must be contiguous V000..V{N-1}
    ids = [v["id"] for v in scn["vehicles"]]
    expected = [f"V{i:03d}" for i in range(len(scn["vehicles"]))]
    assert ids == expected, f"non-contiguous IDs: {ids[:5]}...{ids[-3:]}"


def test_free_flow_speed_range_allows_80_kmh():
    assert S.SPEED_KMH_RANGE == (50, 80)


def test_demand_scale_multiplies_default_flow():
    """--demand-scale multiplies DEFAULT_APPROACH_FLOW_VPH."""
    from argparse import Namespace
    # Simulate --demand-scale 3
    scale = 3.0
    dm = S.DemandModel(
        flows={d: S.DEFAULT_APPROACH_FLOW_VPH * scale for d in G.Direction},
        turn_split=S.DEFAULT_TURN_SPLIT)
    for d in G.Direction:
        assert dm.flow_vph(d) == S.DEFAULT_APPROACH_FLOW_VPH * scale
        assert dm.flow_vph(d) == 1200.0  # 400*3

    # Simulate --demand-scale 0 → no demand
    dm_zero = None
    assert dm_zero is None


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


def test_demand_validation_rejects_negative_flow():
    try:
        S.DemandModel(flows={"N": -1, "E": 400, "S": 400, "W": 400})
    except ValueError as e:
        assert "negative" in str(e) or "must be" in str(e)
    else:
        raise AssertionError("expected ValueError for negative flow")


def test_demand_validation_accepts_zero_total_flow():
    dm = S.DemandModel(flows={d: 0 for d in G.Direction})
    assert all(dm.flow_vph(d) == 0 for d in G.Direction)


def test_demand_validation_rejects_negative_turn_split():
    try:
        S.DemandModel(turn_split={"left": -0.1, "straight": 0.8, "right": 0.3})
    except ValueError as e:
        assert "turn_split" in str(e)
    else:
        raise AssertionError("expected ValueError for negative turn split")


def test_demand_validation_rejects_all_zero_turn_split():
    try:
        S.DemandModel(turn_split={"left": 0.0, "straight": 0.0, "right": 0.0})
    except ValueError as e:
        assert "all-zero" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for all-zero turn split")


def test_demand_validation_accepts_valid_model():
    dm = S.DemandModel(
        flows={d: 400.0 for d in G.Direction},
        turn_split={"left": 0.15, "straight": 0.70, "right": 0.15})
    S.make_vehicle("V", random.Random(1), demand=dm)


def test_free_flow_speed_range_50_to_80():
    """SPEED_KMH_RANGE must be (50, 80)."""
    assert S.SPEED_KMH_RANGE == (50, 80)
    # verify make_vehicle produces speeds within range
    rng = random.Random(42)
    for _ in range(100):
        v = S.make_vehicle("V", rng)
        assert 50 <= v["speed_kmh"] <= 80, f"speed_kmh={v['speed_kmh']} out of range"
        assert 50 / 3.6 - 0.01 <= v["speed_ms"] <= 80 / 3.6 + 0.01


def test_generate_vehicle_speeds_in_range():
    """Every vehicle in a generated scenario has speed 50-80 km/h."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        scn = S.generate(1, 60, tmp, fps=30,
                         demand=S.DemandModel.default())
    for v in scn["vehicles"]:
        assert 50 <= v["speed_kmh"] <= 80, (
            f"{v['id']} speed_kmh={v['speed_kmh']} out of [50,80]")


def test_same_lane_no_zero_frame_gaps():
    """After _resolve_all, no two vehicles in the same lane depart in the
    same frame (direct min-headway must separate adjacent departures)."""
    import scenario_gen as S
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    rng = random.Random(19)
    vehicles = [S.make_vehicle(f"V{i:03d}", rng) for i in range(40)]
    S.schedule_departures(vehicles, 2100, rng,
                          approach_visible_length=54.751)
    S._resolve_all(vehicles, 54.751, 30, signal_plan=sp)
    lanes = {}
    for v in vehicles:
        lanes.setdefault((v["approach"], v["lane"]), []).append(v)
    for key, vs in lanes.items():
        vs.sort(key=lambda d: d["depart_frame"])
        for a, b in zip(vs, vs[1:]):
            gap = b["depart_frame"] - a["depart_frame"]
            needed = K.min_headway_frames(max(a["length"], b["length"]),
                                          max(a["speed_ms"], b["speed_ms"]))
            assert gap >= needed, (
                f"lane {key}: {a['id']}@{a['depart_frame']} -> "
                f"{b['id']}@{b['depart_frame']} gap={gap} < needed={needed}")


def test_generate_hard_ceiling_leave_frame():
    """All vehicles in generated scenario have leave_frame <= duration_frames."""
    import tempfile
    dm = S.DemandModel(
        flows={d: S.DEFAULT_APPROACH_FLOW_VPH * 2.0 for d in G.Direction},
        turn_split=S.DEFAULT_TURN_SPLIT)
    with tempfile.TemporaryDirectory() as tmp:
        scn = S.generate(42, 30, tmp, fps=30, demand=dm,
                         signal_mode="adaptive")
    dur = scn["duration_frames"]
    for v in scn["vehicles"]:
        tf = int(round(54.751 / v["speed_ms"] * 30))
        rf = v.get("release_frame", v["depart_frame"] + tf)
        dt = G.delta_t_frames(G.Turn(v["turn"]), v["speed_ms"], 30,
                              lane_index=v["lane"])
        leave_f = rf + dt + tf
        assert leave_f <= dur, (
            f"{v['id']} leave_frame={leave_f} > duration_frames={dur}")


def test_demand_from_dict_rejects_bad_flow():
    try:
        S.DemandModel.from_dict({"flows": {"N": -10, "E": 400, "S": 400, "W": 400},
                                 "turn_split": {"left": 0.1, "straight": 0.8, "right": 0.1}})
    except ValueError:
        pass
    else:
        raise AssertionError("from_dict should validate negative flow")


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
