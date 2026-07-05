"""Phase 2 — Scenario generator (pure Python).

Produces a reproducible, conflict-free list of vehicles for one simulation
run, written to output/<run>/scenario.json.

Each vehicle gets: id, class, color, unique plate, approach, lane, turn,
speed, depart_frame. Departures are scheduled so that no two vehicles in the
same (approach, lane) overlap (min-headway enforced).

The output ``duration_frames`` is auto-sized to fit all scheduled vehicles:
the requested ``--seconds`` is treated as a minimum floor; the actual duration
is extended so every vehicle completes its full appear→box-edge→disappear
arc on screen (no frozen mid-road cars).

Usage:
    python3 scripts/scenario_gen.py --seed 42 --num-vehicles 20 --seconds 12.0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import string
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "lib"))

import geometry as G
import kinematics as K
import envfile as ENV
# Import the traffic signal module unambiguously.  Python's stdlib also
# ships a top-level ``signal`` module (OS signals); plain ``import signal``
# can resolve to stdlib under pytest/CI runs where ``scripts/lib`` isn't on
# the path first.  Force-load our local module and register it under a
# distinct name to guarantee never to import the stdlib one.
import importlib.util as _ilu
_sig_spec = _ilu.spec_from_file_location(
    "traffic_signal_lib", os.path.join(HERE, "lib", "signal.py"))
SG = _ilu.module_from_spec(_sig_spec)
sys.modules["traffic_signal_lib"] = SG
_sig_spec.loader.exec_module(SG)
from gen_plate import random_plate

ROAD_JSON = os.path.join(HERE, "..", "assets", "road.json")


# ---- demand model (real-world traffic volume) -------------------------------
# A per-approach flow rate (veh/h) plus a turning-movement split. Drives
# Poisson arrivals so platoons and gaps emerge naturally, matching real
# urban intersection demand. Defaults reflect a moderate urban arterial
# (~400 veh/h per approach, straight-dominant with realistic left/right
# shares). Lane is chosen uniformly within the approach (lane choice does
# not affect arrival rate; lanes share the approach's demand and the
# headway filter keeps them conflict-free).
DEFAULT_APPROACH_FLOW_VPH = 400.0
# default turning split {turn: fraction}. Lane-turn restrictions still apply.
DEFAULT_TURN_SPLIT = {"left": 0.15, "straight": 0.70, "right": 0.15}


class DemandModel:
    """Per-approach Poisson demand with a turning-movement split.

    ``flows`` maps G.Direction -> veh/h. ``turn_split`` is shared across all
    approaches (typical for a symmetric 4-way); per-approach splits can be
    supplied via ``turn_splits``.  Falls back to uniform VPH for any approach
    not listed in ``flows``.

    Used by ``schedule_departures_poisson`` to draw exponential inter-arrival
    times per approach (rate = flow_vph/3600 veh/s), and by ``make_vehicle`` to
    draw the approach+turn from the demand distribution.
    """

    def __init__(self,
                 flows: Optional[dict] = None,
                 turn_split: Optional[dict] = None,
                 turn_splits: Optional[dict] = None):
        self.flows = dict(flows) if flows else {}
        # per-approach turn split overrides turn_split when provided
        self.turn_splits = turn_splits or {}
        self.turn_split = dict(turn_split) if turn_split else dict(DEFAULT_TURN_SPLIT)

    @classmethod
    def default(cls) -> "DemandModel":
        flows = {d: DEFAULT_APPROACH_FLOW_VPH for d in G.Direction}
        return cls(flows=flows, turn_split=DEFAULT_TURN_SPLIT)

    def flow_vph(self, approach: G.Direction) -> float:
        return self.flows.get(approach, DEFAULT_APPROACH_FLOW_VPH)

    def turn_fraction(self, approach: G.Direction, turn: G.Turn) -> float:
        split = self.turn_splits.get(approach, self.turn_split)
        return split.get(turn.value, 0.0)

    def to_dict(self) -> dict:
        return {
            "flows": {d.value: v for d, v in self.flows.items()},
            "turn_split": dict(self.turn_split),
            "turn_splits": {d.value: s for d, s in self.turn_splits.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DemandModel":
        flows = {G.Direction(k): float(v) for k, v in (d.get("flows") or {}).items()}
        ts = d.get("turn_split")
        tss = d.get("turn_splits")
        turn_splits = None
        if tss:
            turn_splits = {G.Direction(k): {kk: float(vv) for kk, vv in s.items()}
                           for k, s in tss.items()}
        return cls(flows=flows, turn_split=ts, turn_splits=turn_splits)

    @classmethod
    def from_file(cls, path: str) -> "DemandModel":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _approach_turn_from_demand(demand: DemandModel, rng: random.Random,
                               lane: int) -> tuple:
    """Pick an approach weighted by flow and a legal turn from the demand split.

    The approach is drawn proportional to flow so higher-demand approaches
    generate more vehicles.  The turn is drawn from the demand split filtered
    by the lane's legal turns (restricted lanes still only see legal turns;
    fractions renormalise over the legal subset).
    """
    approaches = list(G.Direction)
    weights = [demand.flow_vph(a) for a in approaches]
    approach = rng.choices(approaches, weights=weights, k=1)[0]
    legal = [t for t in TURNS if t in G.allowed_turns(lane)]
    fracs = [demand.turn_fraction(approach, t) for t in legal]
    total = sum(fracs)
    if total <= 0:
        # all-zero split for this lane's legal turns — fall back to equal
        weights_t = [1.0] * len(legal)
    else:
        weights_t = [f / total for f in fracs]
    turn = rng.choices(legal, weights=weights_t, k=1)[0]
    return approach, turn


# ---- defaults ---------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_NUM_VEHICLES = 20
DEFAULT_SECONDS = 12.0
VEHICLE_CLASSES = ["car"]
# representative lengths for headway (m)
VEHICLE_LENGTH = {"car": 4.47}
# common CCTV-ish speeds (km/h) in free flow
SPEED_KMH_RANGE = (30, 60)
COLOR_LIST = [
    ((0.8, 0.1, 0.1, 1.0), "red"),
    ((0.1, 0.2, 0.8, 1.0), "blue"),
    ((0.9, 0.9, 0.9, 1.0), "white"),
    ((0.1, 0.1, 0.1, 1.0), "black"),
    ((0.8, 0.8, 0.1, 1.0), "yellow"),
    ((0.5, 0.5, 0.5, 1.0), "grey"),
    ((0.1, 0.6, 0.3, 1.0), "green"),
]
TURNS = [G.Turn.LEFT, G.Turn.STRAIGHT, G.Turn.RIGHT]
TURN_WEIGHTS = [1, 3, 2]  # straight most common
TURN_WEIGHT_BY_TURN = dict(zip(TURNS, TURN_WEIGHTS))


def choose_legal_turn(lane: int, rng: random.Random) -> G.Turn:
    legal = [t for t in TURNS if t in G.allowed_turns(lane)]
    weights = [TURN_WEIGHT_BY_TURN[t] for t in legal]
    return rng.choices(legal, weights=weights)[0]


def make_vehicle(vid: str, rng: random.Random,
                 demand: Optional[DemandModel] = None) -> dict:
    cls = rng.choices(VEHICLE_CLASSES, weights=[1])[0]
    rgba, color_name = rng.choice(COLOR_LIST)
    plate = random_plate(rng)
    lane = rng.randint(0, G.NUM_LANES - 1)
    if demand is not None:
        approach, turn = _approach_turn_from_demand(demand, rng, lane)
    else:
        approach = rng.choice(list(G.Direction))
        turn = choose_legal_turn(lane, rng)
    speed_kmh = rng.uniform(*SPEED_KMH_RANGE)
    speed_ms = K.speed_kmh_to_ms(speed_kmh)
    return {
        "id": vid,
        "class": cls,
        "color": list(rgba),
        "color_name": color_name,
        "plate": plate,
        "approach": approach.value,
        "lane": lane,
        "turn": turn.value,
        "speed_kmh": round(speed_kmh, 2),
        "speed_ms": round(speed_ms, 3),
        "length": VEHICLE_LENGTH[cls],
    }


def schedule_departures(vehicles: list, duration_frames: int, rng: random.Random,
                        safety_gap: float = 2.0,
                        approach_visible_length: float = 40.0) -> list:
    """Assign a depart_frame to each vehicle so no two in the same
    (approach, lane) overlap, either at departure OR by catch-up on the
    approach segment. Uniform-random target per vehicle (legacy behaviour
    — used when no ``DemandModel`` is provided).

    Two checks per candidate frame against every existing vehicle in the lane:
      * start-gap headway (min_headway_frames) — no overlap at the depart instant
      * catch-up safety (catchup_safe) — a faster follower must not close to
        within safety_gap of a slower leader before the leader enters the
        Black Box. When a faster vehicle would catch a slower one, its
        depart_frame is pushed later (speed variety preserved; only timing
        shifts) until both checks pass.
    Vehicles are placed in a random order; each is given the earliest feasible
    frame >= a random target.
    """
    # group existing departures by (approach, lane)
    lanes: dict = {}
    order = list(range(len(vehicles)))
    rng.shuffle(order)
    for i in order:
        v = vehicles[i]
        key = (v["approach"], v["lane"])
        target = rng.randint(0, max(1, duration_frames // 2))
        # find earliest frame >= target with no conflict in this lane
        existing = lanes.setdefault(key, [])
        frame = target
        step = 1
        # bounded search
        for _ in range(2000):
            ok = True
            for (ef, el, es) in existing:
                # 1. start-gap headway (no overlap at the depart instant)
                needed = K.min_headway_frames(max(v["length"], el),
                                              max(v["speed_ms"], es), safety_gap)
                if abs(frame - ef) < needed:
                    ok = False
                    break
                # 2. catch-up on the approach segment — check BOTH orderings:
                #    the LATER-departing vehicle is the potential follower that
                #    could close the gap. A slower new vehicle placed BEFORE an
                #    existing faster one would be rear-ended by it, so the check
                #    must run in both directions (not only new-as-follower).
                if frame >= ef:
                    # new vehicle is the follower
                    if not K.catchup_safe(ef, es, el, frame, v["speed_ms"],
                                          approach_visible_length, safety_gap):
                        ok = False
                        break
                else:
                    # new vehicle is the leader; existing departs later
                    if not K.catchup_safe(frame, v["speed_ms"], v["length"],
                                          ef, es, approach_visible_length, safety_gap):
                        ok = False
                        break
            if ok:
                break
            frame += step
        else:
            # C8: loop exhausted without finding a safe frame — the vehicle
            # is placed at the last tried frame, which may overlap a prior.
            # Warn so the user knows the scenario has a headway violation
            # (dense demand on a short approach is the usual cause).
            import sys as _sys
            print(f"[WARN] schedule_departures: cap hit for V{i} "
                  f"({v['approach']}/lane {v['lane']}, frame {frame}) — "
                  f"vehicle may overlap prior in same lane",
                  file=_sys.stderr, flush=True)
        v["depart_frame"] = int(frame)
        existing.append((frame, v["length"], v["speed_ms"]))
    return vehicles


def _enforce_lane_safety(vehicles: list, fps: int, safety_gap: float,
                         approach_visible_length: float, rng: random.Random):
    """Push-later-only feasibility pass ensuring no two vehicles in the same
    (approach, lane) overlap at depart or by catch-up. Used by the Poisson
    scheduler after raw arrivals are assigned so the headway / catch-up
    invariants hold (same guarantees as ``schedule_departures``).

    Vehicles are processed in depart_frame order within each lane; for each
    vehicle the earliest safe frame >= its current frame is found by pushing it
    later only.
    """
    lanes: dict = {}
    for v in vehicles:
        lanes.setdefault((v["approach"], v["lane"]), []).append(v)
    for key, vs in lanes.items():
        vs.sort(key=lambda x: x["depart_frame"])
        placed: list = []
        for v in vs:
            frame = v["depart_frame"]
            for _ in range(4000):
                ok = True
                for (ef, el, es) in placed:
                    needed = K.min_headway_frames(max(v["length"], el),
                                                  max(v["speed_ms"], es), safety_gap)
                    if abs(frame - ef) < needed:
                        ok = False
                        break
                    if frame >= ef:
                        if not K.catchup_safe(ef, es, el, frame, v["speed_ms"],
                                              approach_visible_length, safety_gap):
                            ok = False
                            break
                    else:
                        if not K.catchup_safe(frame, v["speed_ms"], v["length"],
                                              ef, es, approach_visible_length, safety_gap):
                            ok = False
                            break
                if ok:
                    break
                frame += 1
            else:
                # C8: 4000-iter cap exhausted — vehicle placed at last frame,
                # may overlap. Warn per-vehicle so dense-demand scenarios
                # surface the constraint violation.
                import sys as _sys
                print(f"[WARN] _enforce_lane_safety: cap hit for "
                      f"{v['id']} ({v['approach']}/lane {v['lane']}, "
                      f"frame {frame}) — vehicle may overlap prior",
                      file=_sys.stderr, flush=True)
            v["depart_frame"] = int(frame)
            placed.append((frame, v["length"], v["speed_ms"]))


def schedule_departures_poisson(vehicles: list, duration_frames: int,
                                 fps: int, demand: DemandModel,
                                 rng: random.Random,
                                 safety_gap: float = 2.0,
                                 approach_visible_length: float = 40.0) -> list:
    """Assign depart_frame via a Poisson arrival process per approach.

    For each approach, inter-arrival times are drawn exponential with rate
    ``flow_vph / 3600`` veh/s. Vehicles are grouped by approach and assigned
    arrival frames in the order the Poisson clock produces them; lane is
    kept from ``make_vehicle`` (which already chose it) so per-lane headway
    filtering applies.

    After raw arrivals are assigned, the same headway / catch-up safety filter
    as ``schedule_departures`` runs as a push-later-only feasibility pass: any
    arrival that would violate same-lane headway or catch-up is pushed to the
    earliest safe frame >= its Poisson arrival. This preserves all existing
    safety invariants while letting platoons / gaps emerge from the demand.

    Vehicles whose pushed-later frame exceeds ``duration_frames`` are kept
    (the scenario duration auto-extends in ``generate`` to fit every vehicle).
    """
    fps_f = float(fps)
    horizon_s = duration_frames / fps_f

    # Generate per-approach arrival sequences from independent Poisson clocks.
    arrivals_by_approach: dict = {}
    for approach in G.Direction:
        rate_ps = demand.flow_vph(approach) / 3600.0
        if rate_ps <= 0:
            arrivals_by_approach[approach] = []
            continue
        arrivals = []
        t = 0.0
        while t < horizon_s:
            # exponential inter-arrival (Poisson process)
            t += rng.expovariate(rate_ps)
            if t < horizon_s:
                arrivals.append(int(round(t * fps_f)))
        arrivals_by_approach[approach] = arrivals

    # Group vehicles by approach, assign arrival frames in Poisson order.
    by_approach: dict = {}
    for v in vehicles:
        by_approach.setdefault(G.Direction(v["approach"]), []).append(v)
    for ap, vs in by_approach.items():
        vs.sort(key=lambda x: x.get("depart_frame", 0))
        arrivals = arrivals_by_approach.get(ap, [])
        n_to_assign = min(len(vs), len(arrivals))
        for i in range(n_to_assign):
            vs[i]["depart_frame"] = int(arrivals[i])
        # leftover vehicles (more vehicles than arrivals in window) get frames
        # just past the last arrival so they still appear in the run.
        if n_to_assign < len(vs):
            tail = (arrivals[-1] + 1) if arrivals else 0
            for i in range(n_to_assign, len(vs)):
                vs[i]["depart_frame"] = int(tail)
                tail += 1

    # Push-later-only feasibility pass (preserves all safety invariants).
    _enforce_lane_safety(vehicles, fps, safety_gap,
                         approach_visible_length, rng)
    return vehicles


def _compute_max_leave_frame(vehicles: list, road_meta: dict, fps: int) -> int:
    """Compute the maximum ``leave_frame`` across all vehicles using the
    identical env-anchor + plan_motion path as ``render.compute_metadata``.
    Returns 0 if the list is empty."""
    envs = {tag: ENV.load_env(tag, ROOT) for tag in G.camera_names()}
    max_lf = 0
    for veh in vehicles:
        approach = G.Direction(veh["approach"])
        turn = G.Turn(veh["turn"])
        ex_dir, ex_lane = G.exit_lane_for_movement(approach, veh["lane"], turn)
        in_anchor, _ = ENV.lane_default_anchor(envs[f"in_{approach.value}"], veh["lane"])
        out_anchor, _ = ENV.lane_default_anchor(envs[f"out_{ex_dir.value}"], ex_lane)
        motion = K.plan_motion(
            veh["id"], approach, veh["lane"], turn,
            veh["speed_ms"], veh["depart_frame"], fps=fps,
            appear_anchor=in_anchor[:2],
            reappear_anchor=out_anchor[:2],
            road_meta=road_meta,
            stop_frame=veh.get("stop_frame"),
            release_frame=veh.get("release_frame"))
        if motion.leave_frame > max_lf:
            max_lf = motion.leave_frame
    return max_lf


def generate(seed: int, num_vehicles: int, seconds: float,
             out_dir: str, fps: int = G.FPS,
             signal_plan: Optional[SG.SignalPlan] = None,
             demand: Optional[DemandModel] = None,
             signal_mode: str = "fixed") -> dict:
    """Generate a scenario and write ``scenario.json``.

    Args:
        signal_plan: pre-built signal plan (overrides ``signal_mode``).
            If None and ``signal_mode`` is ``"adaptive"`` / ``"fixed"``,
            the appropriate plan is constructed here.
        signal_mode: ``"fixed"`` (default fixed-cycle permissive-left
            ``SignalPlan``) or ``"adaptive"`` (NEMA 8-phase MaxPressure
            ``AdaptiveSignalPlan`` — closed-loop, rebuilds inside the
            ``_resolve_all`` fixpoint from realised arrivals).  Ignored when
            ``signal_plan`` is given explicitly.
        demand: Poisson per-approach flow + turning-movement split.  None
            falls back to the legacy uniform scheduler.
    """
    min_duration_frames = int(round(seconds * fps))

    rng = random.Random(seed)
    vehicles = [make_vehicle(f"V{i:03d}", rng, demand=demand)
                for i in range(num_vehicles)]

    road_meta = {}
    approach_len = 40.0
    if os.path.exists(ROAD_JSON):
        with open(ROAD_JSON) as f:
            road_meta = json.load(f)
            approach_len = road_meta.get("approach_length", 40.0)

    if demand is not None:
        vehicles = schedule_departures_poisson(
            vehicles, min_duration_frames, fps, demand, rng,
            approach_visible_length=approach_len)
    else:
        vehicles = schedule_departures(
            vehicles, min_duration_frames, rng,
            approach_visible_length=approach_len)

    # Construct signal plan from mode if not supplied explicitly.
    if signal_plan is None and signal_mode == "adaptive":
        signal_plan = SG.AdaptiveSignalPlan(fps=fps)
    elif signal_plan is None and signal_mode == "fixed":
        signal_plan = None  # fixed mode is only active when explicitly built
    # Preserve a requested fixed plan that was passed in.
    if signal_mode == "adaptive" and signal_plan is not None and not isinstance(
            signal_plan, SG.AdaptiveSignalPlan):
        # User mixed the two: prefer the explicit plan, switch the mode label.
        signal_mode = "fixed"

    _resolve_all(vehicles, approach_len, fps, signal_plan=signal_plan)

    required = _compute_max_leave_frame(vehicles, road_meta, fps)
    tail = fps  # 1 s of empty road after the last car leaves
    duration_frames = max(min_duration_frames, required + tail)

    # Surface a large mismatch between the requested --seconds floor and the
    # actual rendered duration. The video auto-extends to fit every vehicle
    # (it can be longer than the minimum, never shorter — that's by design),
    # but a 12s request that becomes a ~490s render is almost always a user
    # mistake (too many vehicles for the requested window). Warn loudly so the
    # user can Ctrl-C BEFORE the render step spends an hour producing a much
    # longer clip than intended. Print to both stdout (visible in terminal)
    # and the scenario line below echoes it in the log.
    actual_seconds = duration_frames / fps
    if actual_seconds > seconds * 1.5 and duration_frames > min_duration_frames * 2:
        import sys as _sys
        print(f"\n[WARN] --seconds={seconds:g} is a MINIMUM. The actual video "
              f"will be {actual_seconds:.1f}s ({duration_frames} frames) because "
              f"{num_vehicles} vehicles cannot fit in {seconds:g}s and auto-extend to "
              f"{required} frames (required) + {tail} (tail). Rendering "
              f"{duration_frames} frames per camera.", file=_sys.stderr, flush=True)
        print(f"[WARN] For a ~{seconds:g}s clip, use ~{int(seconds * 1600 / 3600 * 4)} "
              f"vehicles at default demand; for {num_vehicles} vehicles, expect "
              f"a ~{actual_seconds:.0f}s video.", file=_sys.stderr, flush=True)

    scenario = {
        "seed": seed,
        "fps": fps,
        "duration_frames": duration_frames,
        "box_size": G.BOX_SIZE,
        "num_lanes": G.NUM_LANES,
        "lane_centerlines_x": G.LANE_CENTERLINES,
        "cameras": G.camera_names(),
        "vehicles": vehicles,
    }
    if signal_plan is not None:
        scenario["signal_mode"] = signal_mode
        scenario["signal_cycle_frames"] = signal_plan.cycle_frames
        # Emit a per-frame phase timeline for downstream vision/behaviour
        # labels and for replay/debug.  For adaptive plans this is the
        # interval list; for fixed plans it is the phase-name boundaries.
        if isinstance(signal_plan, SG.AdaptiveSignalPlan):
            scenario["signal_timeline"] = [
                {"start": s, "end": e, "phases": list(combo)}
                for (s, e, combo) in signal_plan.intervals
            ]
            scenario["signal_clearances"] = [
                {"start": s, "end": e}
                for (s, e) in signal_plan._clearances
            ]
        else:
            scenario["signal_timeline"] = [
                {"start": s, "phase": name}
                for (s, name) in signal_plan._boundaries
            ]
    if demand is not None:
        scenario["demand"] = demand.to_dict()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scenario.json")
    with open(out_path, "w") as f:
        json.dump(scenario, f, indent=2)
    print(f"Wrote scenario: {out_path}  ({len(vehicles)} vehicles, {fps} fps, "
          f"duration_frames={duration_frames}f"
          f" (floor={min_duration_frames}f, required={required}f, tail=+{tail}f))")
    return scenario


def _apply_signal_gating(vehicles, approach_visible_length, signal_plan, fps):
    """Set stop_frame / release_frame so queued vehicles stop at the
    stop line during red and enter the box at green.

    ``stop_frame`` = frame the vehicle would reach the stop line at free-flow
    (always = depart_frame + travel_time). ``release_frame`` = frame the
    vehicle enters the box (= stop_frame for free-flow, or next-green + queue
    slot + extra_stagger_frames for queued).

    Does NOT modify ``depart_frame`` — vehicles appear at their scheduled time
    and queue visually if they would arrive on red.

    Reads ``extra_stagger_frames`` (set by exit-conflict resolution for queued
    vehicles).  Initialises the key if absent; resets to 0 when a vehicle
    becomes free-flow (green arrival).  Exit-conflict resolution is the only
    writer of non-zero values.

    Queue slots are ordered by ``stop_arrival`` (actual stop-line arrival)
    within each contiguous red batch so the ``slot * reaction_per_slot``
    stagger reflects true physical arrival order.
    """
    reaction_per_slot = int(round(0.5 * fps))
    groups: dict = {}
    for v in vehicles:
        key = (v["approach"], v["lane"])
        groups.setdefault(key, []).append(v)
        v["extra_stagger_frames"] = v.get("extra_stagger_frames", 0)
    for key, group in groups.items():
        group.sort(key=lambda x: x["depart_frame"])
        approach = G.Direction(key[0])

        # Pre-compute stop_arrival and red/green status for each vehicle
        tagged = []
        for v in group:
            turn = G.Turn(v["turn"])
            travel_time_f = int(round(approach_visible_length / v["speed_ms"] * fps))
            stop_arrival = v["depart_frame"] + travel_time_f
            is_green = signal_plan.is_green(approach, turn, stop_arrival)
            tagged.append((v, stop_arrival, is_green))

        # Walk tagged list, grouping red vehicles into batches.
        i = 0
        n = len(tagged)
        while i < n:
            # Green vehicle — free-flow (reset queue).
            if tagged[i][2]:  # is_green
                v = tagged[i][0]
                v["extra_stagger_frames"] = 0
                v["stop_frame"] = tagged[i][1]
                v["release_frame"] = tagged[i][1]
                v["queue_slot"] = -1
                v["wait_frames"] = 0
                i += 1
                continue

            # Red batch: collect all consecutive red vehicles.
            batch_start = i
            while i < n and not tagged[i][2]:
                i += 1
            red_vehicles = tagged[batch_start:i]

            # Sort red batch by stop_arrival → true arrival order at stop line.
            red_vehicles.sort(key=lambda tup: (tup[1], tup[0]["depart_frame"], tup[0]["id"]))

            for slot, (v, stop_arrival, _) in enumerate(red_vehicles):
                turn = G.Turn(v["turn"])
                next_green = signal_plan.next_green_frame(approach, turn, stop_arrival)
                release = next_green + slot * reaction_per_slot + v["extra_stagger_frames"]
                # Rebase if extra_stagger pushes the release past the green window.
                if not signal_plan.is_green(approach, turn, release):
                    rebased = signal_plan.next_green_frame(approach, turn, release)
                    rb_delta = rebased - release
                    v["extra_stagger_frames"] = v.get("extra_stagger_frames", 0) + rb_delta
                    release = rebased
                v["stop_frame"] = stop_arrival
                v["release_frame"] = release
                v["queue_slot"] = slot
                v["wait_frames"] = release - stop_arrival

        # --- Post-hoc release-gap guard on queued vehicles in this group ---
        # ``extra_stagger_frames`` from exit-conflict resolution can reorder or
        # collapse release times relative to the pure ``slot * reaction``
        # pattern.  Ensure every queued pair releases ≥ reaction_per_slot apart.
        qs = [v for v in group if v.get("queue_slot", -1) >= 0]
        qs.sort(key=lambda v: v["release_frame"])
        for a, b in zip(qs, qs[1:]):
            need = a["release_frame"] + reaction_per_slot
            if b["release_frame"] < need:
                delta = need - b["release_frame"]
                old_rel = b["release_frame"]
                b["release_frame"] = need
                b["wait_frames"] += delta
                # Accumulate into extra_stagger so the adjustment survives
                # exit-conflict resolution in later fixpoint rounds.
                b["extra_stagger_frames"] = b.get("extra_stagger_frames", 0) + delta
                # If the push leaves the green window, rebase to next green.
                if signal_plan is not None:
                    approach = G.Direction(b["approach"])
                    turn = G.Turn(b["turn"])
                    if not signal_plan.is_green(approach, turn, b["release_frame"]):
                        rebased = signal_plan.next_green_frame(
                            approach, turn, b["release_frame"])
                        rebase_delta = rebased - b["release_frame"]
                        b["release_frame"] = rebased
                        b["wait_frames"] += rebase_delta
                        b["extra_stagger_frames"] += rebase_delta


def _resolve_exit_conflicts(vehicles, approach_visible_length, fps,
                            signal_plan=None):
    """Resolve exit-lane interval overlaps across different approaches.

    For **free-flow** vehicles (queue_slot < 0): push depart_frame later
    (moves the entire timeline forward).

    For **queued** vehicles (queue_slot >= 0): add stagger to
    ``extra_stagger_frames`` so the release frame shifts within the same
    green cycle, without moving depart_frame (which can't change the
    box-entry time, pinned to green).  If the stagger would push the
    release past the green window, rebase to the next green cycle.

    Returns True if any depart_frame or extra_stagger_frames was changed."""
    any_changed = False
    for _ in range(5):
        changed = False
        groups: dict = {}
        for v in vehicles:
            travel_f = int(round(approach_visible_length / v["speed_ms"] * fps))
            release_f = v.get("release_frame", v["depart_frame"] + travel_f)
            reappear_f = release_f + G.delta_t_frames(
                G.Turn(v["turn"]), v["speed_ms"], fps, lane_index=v["lane"])
            leave_f = reappear_f + travel_f
            out_dir, ex_lane = G.exit_lane_for_movement(
                G.Direction(v["approach"]), v["lane"], G.Turn(v["turn"]))
            key = (out_dir.value, ex_lane)
            groups.setdefault(key, []).append((v, reappear_f, leave_f))
        for key, grp in groups.items():
            grp.sort(key=lambda x: x[1])
            last_leave = -1
            for v, r, l in grp:
                if last_leave > 0 and r < last_leave + 5:
                    delay = last_leave + 5 - r
                    if v.get("queue_slot", -1) >= 0:
                        # Queued: push stagger so release moves within green
                        old_extra = v.get("extra_stagger_frames", 0)
                        v["extra_stagger_frames"] = old_extra + delay
                        # Recompute release and keep within green window
                        old_release = v.get("release_frame", 0)
                        new_release = old_release + delay
                        if signal_plan is not None:
                            approach = G.Direction(v["approach"])
                            turn = G.Turn(v["turn"])
                            if not signal_plan.is_green(approach, turn, new_release):
                                # Rebases to next green — the extra frames beyond
                                # the green window are added to the stagger so
                                # the delay still counts toward exit ordering.
                                rebased = signal_plan.next_green_frame(
                                    approach, turn, new_release)
                                rebase_extra = rebased - new_release
                                v["extra_stagger_frames"] += rebase_extra
                                new_release = rebased
                        v["release_frame"] = new_release
                        last_leave = new_release + (l - r)
                        changed = True
                        any_changed = True
                    else:
                        # Free-flow: push depart_frame later
                        v["depart_frame"] += delay
                        if "stop_frame" in v:
                            v["stop_frame"] += delay
                            v["release_frame"] += delay
                        last_leave = l + delay
                        changed = True
                        any_changed = True
                else:
                    last_leave = l
        if not changed:
            break
    return any_changed


def _check_headway_fixpoint(vehicles, approach_visible_length, fps):
    """Check same-lane headway and catch-up safety. Push depart_frame later
    for any violating pair. Returns True if any frame was changed."""
    changed = False
    lanes: dict = {}
    for v in vehicles:
        key = (v["approach"], v["lane"])
        lanes.setdefault(key, []).append(v)
    for key, vs in lanes.items():
        vs.sort(key=lambda x: x["depart_frame"])
        for i in range(len(vs) - 1):
            lead = vs[i]
            follow = vs[i + 1]
            if not K.catchup_safe(lead["depart_frame"], lead["speed_ms"],
                                  lead["length"],
                                  follow["depart_frame"], follow["speed_ms"],
                                  approach_visible_length):
                # Push follower later to make it safe
                needed = K.min_follow_depart_frame(
                    lead["depart_frame"], lead["speed_ms"], lead["length"],
                    follow["speed_ms"], approach_visible_length)
                if needed > follow["depart_frame"]:
                    delta = needed - follow["depart_frame"]
                    follow["depart_frame"] = needed
                    if "stop_frame" in follow:
                        follow["stop_frame"] += delta
                        follow["release_frame"] += delta
                    changed = True
    return changed


def _collect_stop_arrivals(vehicles: list, approach_visible_length: float,
                           fps: int) -> Dict[Tuple[G.Direction, G.Turn], List[int]]:
    """Compute the natural (un-gated) stop-line arrival frame for every
    vehicle and group by movement.  This is the *demand* signal an adaptive
    controller reacts to: arrival = depart_frame + travel_time at free-flow
    speed, independent of any signal-induced wait.

    Used to rebuild an ``AdaptiveSignalPlan`` from current arrivals inside the
    ``_resolve_all`` fixpoint, closing the loop (signal reacts to arrivals,
    release times shift, downstream arrivals change).  Feeding *natural*
    arrivals (not gated ones) stabilises the fixpoint: the signal plan is a
    function of demand only, not of itself.
    """
    arrivals: Dict[Tuple[G.Direction, G.Turn], List[int]] = {}
    for v in vehicles:
        ap = G.Direction(v["approach"])
        turn = G.Turn(v["turn"])
        travel_f = int(round(approach_visible_length / v["speed_ms"] * fps))
        t = v["depart_frame"] + travel_f
        arrivals.setdefault((ap, turn), []).append(t)
    for key in arrivals:
        arrivals[key].sort()
    return arrivals


def _resolve_all(vehicles, approach_visible_length, fps,
                 signal_plan=None, max_rounds=20):
    """Unified fixpoint: iterate headway → signal → exit until all
    constraints are satisfied and the full vehicle state is stable.

    Convergence is detected by comparing a snapshot of
    ``(depart_frame, stop_frame, release_frame, queue_slot, extra_stagger)``
    before and after each round.  Exit resolution and headway only push
    later (monotonic), so the loop converges.

    If ``signal_plan`` is an ``AdaptiveSignalPlan`` the plan is *rebuilt*
    from the current natural stop-line arrivals at the top of each round
    (closed loop: signal reacts to demand).  The arrivals snapshot is taken
    once on the first round and reused on subsequent rounds so the signal
    plan is a function of *demand* only — feeding gated arrivals back in
    would cause the plan to oscillate.  The plan stabilises when the arrival
    snapshot is unchanged across rounds (a direct consequence of the
    snapshot comparison already used for fixpoint detection).
    """
    def _snapshot(vs):
        return [
            (v["depart_frame"], v.get("stop_frame"), v.get("release_frame"),
             v.get("queue_slot"), v.get("extra_stagger_frames"))
            for v in vs
        ]

    is_adaptive = SG.AdaptiveSignalPlan is not None and isinstance(
        signal_plan, SG.AdaptiveSignalPlan)
    arrivals_snapshot: Optional[Dict] = None
    if is_adaptive:
        arrivals_snapshot = _collect_stop_arrivals(
            vehicles, approach_visible_length, fps)

    for _round in range(max_rounds):
        before = _snapshot(vehicles)

        # 0. Closed-loop adaptive signal: rebuild plan from current arrivals.
        if is_adaptive:
            horizon = (
                max((t for ts in arrivals_snapshot.values() for t in ts),
                    default=0) + 20 * signal_plan.max_green_f)
            signal_plan.rebuild(arrivals_snapshot, horizon_frames=horizon)

        # 1. Same-lane headway (re-check after exit shifts)
        _check_headway_fixpoint(vehicles, approach_visible_length, fps)

        # 2. Signal gating (compute stop/release from current depart_frame)
        if signal_plan is not None:
            _apply_signal_gating(vehicles, approach_visible_length,
                                 signal_plan, fps)

        # 3. Exit-lane conflicts (may push depart_frame or stagger)
        _resolve_exit_conflicts(vehicles, approach_visible_length, fps,
                                signal_plan=signal_plan)

        if _snapshot(vehicles) == before:
            break
    else:
        # C7: loop exhausted without convergence — vehicles may still be
        # shifting stop/release frames across rounds. Warn so the scenario
        # isn't silently accepted with an unstable state. Compare the final
        # snapshot to the last `before` to report what's still drifting.
        import sys as _sys
        print(f"[WARN] _resolve_all did not converge after {max_rounds} "
              f"rounds — vehicle stop/release state may still be shifting. "
              f"This usually means a dense scenario + tight headway + signal "
              f"gating can't all stabilise; consider lowering --num-vehicles "
              f"or increasing --seconds.",
              file=_sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--num-vehicles", type=int, default=DEFAULT_NUM_VEHICLES)
    ap.add_argument("--fps", type=int, default=G.FPS,
                    help="frames per second (default: %(default)s)")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                    help="minimum video length in seconds (default: %(default)s); "
                         "the actual duration auto-extends to fit all vehicles")
    ap.add_argument("--signal", action="store_true",
                    help="enable traffic signal SPaT gating + queue")
    ap.add_argument("--signal-mode", type=str, default="fixed",
                    choices=["fixed", "adaptive"],
                    help="signal controller type when --signal is set: "
                         "'fixed' (default, 70s cycle permissive-left) or "
                         "'adaptive' (NEMA 8-phase MaxPressure, closed-loop "
                         "on realised arrivals)")
    ap.add_argument("--demand", type=str, default=None,
                    help="path to a demand JSON (per-approach flow veh/h + turning "
                         "split). When omitted, the default demand model "
                         f"({DEFAULT_APPROACH_FLOW_VPH:g} veh/h/approach) is used. "
                         "Pass 'none' to disable demand and use the legacy "
                         "uniform-random scheduler.")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "..", "output", "run1"))
    args = ap.parse_args()
    if args.signal:
        if args.signal_mode == "adaptive":
            signal_plan = SG.AdaptiveSignalPlan(fps=args.fps)
        else:
            signal_plan = SG.SignalPlan(fps=args.fps)
    else:
        signal_plan = None
    if args.demand and args.demand.lower() == "none":
        demand = None
    elif args.demand:
        demand = DemandModel.from_file(args.demand)
    else:
        demand = DemandModel.default()
    generate(args.seed, args.num_vehicles, args.seconds, args.out, fps=args.fps,
             signal_plan=signal_plan, demand=demand,
             signal_mode=args.signal_mode)


if __name__ == "__main__":
    main()
