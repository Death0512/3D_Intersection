"""Phase 2 — Scenario generator (pure Python).

Produces a reproducible, conflict-free list of vehicles for one simulation
run, written to output/<run>/scenario.json.

Each vehicle gets: id, class, color, unique plate, approach, lane, turn,
speed, depart_frame. Departures are scheduled so that no two vehicles in the
same (approach, lane) overlap (min-headway enforced).

Vehicles are generated from a steady-state Poisson stream with a warm-up period
before frame 0. ``duration_frames`` is exactly ``int(round(seconds*fps))``;
only vehicles that are visible during the rendered clip are emitted. This
avoids an artificially empty opening and avoids discarding dense but valid
traffic that is still on screen when the fixed-length clip ends.

Usage:
    python3 scripts/scenario_gen.py --seed 42 --seconds 12.0
"""
from __future__ import annotations

import argparse
import json
import math
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
import traffic_signal as SG
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

# Margin applied when scaling the Poisson arrival horizon to the vehicle count
# (see schedule_departures_poisson). The natural horizon is
#   n_vehicles / total_rate_per_s
# and the ×1.5 covers Poisson variance so the drawn arrivals >= vehicle count
# with very high probability — making the consecutive-frame "leftover cramming"
# path a rare safety net instead of the norm.
POISSON_HORIZON_MARGIN = 1.5

# Generate traffic before frame 0 so short clips begin in steady state instead
# of an empty-road cold start.  One clip-length of warm-up is enough to populate
# entry roads, stop-line queues, and early out-camera exits for 30s/60s clips;
# the lower bound keeps very short test clips from looking empty.
MIN_WARMUP_SECONDS = 30.0


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
        # Normalise direction keys to G.Direction so all downstream code
        # (to_dict, turn_fraction, flow_vph) works with a single key type.
        self.flows = {}
        if flows:
            for k, v in flows.items():
                d = G.Direction(k) if isinstance(k, str) else k
                self.flows[d] = v
        self.turn_splits = {}
        if turn_splits:
            for k, s in turn_splits.items():
                d = G.Direction(k) if isinstance(k, str) else k
                self.turn_splits[d] = dict(s)
        self.turn_split = dict(turn_split) if turn_split else dict(DEFAULT_TURN_SPLIT)
        self._validate()

    @classmethod
    def default(cls) -> "DemandModel":
        flows = {d: DEFAULT_APPROACH_FLOW_VPH for d in G.Direction}
        return cls(flows=flows, turn_split=DEFAULT_TURN_SPLIT)

    def flow_vph(self, approach: G.Direction) -> float:
        return self.flows.get(approach, DEFAULT_APPROACH_FLOW_VPH)

    def turn_fraction(self, approach: G.Direction, turn: G.Turn) -> float:
        split = self.turn_splits.get(approach, self.turn_split)
        return split.get(turn.value, 0.0)

    def _validate(self):
        def _raise(msg):
            raise ValueError(f"DemandModel: {msg}")

        def _dir_name(d):
            return d.value if isinstance(d, G.Direction) else str(d)

        # flows: every supplied flow must be a finite, non-negative number;
        # total across all supplied approaches must be > 0.
        for d, v in self.flows.items():
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                _raise(f"flow {_dir_name(d)}={v!r} must be a finite non-negative float")
        # All-zero demand is valid: it intentionally produces an empty scenario.

        # turn_split: every value must be finite non-negative; at least one
        # value in the split must be > 0 (otherwise vehicle generation is
        # impossible on any lane).
        for k, v in self.turn_split.items():
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                _raise(f"turn_split[{k!r}]={v!r} must be a finite non-negative float")
        if self.turn_split and all(v == 0 for v in self.turn_split.values()):
            _raise("turn_split all-zero — no vehicle generation possible")

        # turn_splits: same rules per-approach.
        for d, split in self.turn_splits.items():
            for k, v in split.items():
                if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                    _raise(
                        f"turn_splits[{_dir_name(d)}][{k!r}]={v!r} "
                        f"must be a finite non-negative float")
            if split and all(v == 0 for v in split.values()):
                _raise(f"turn_splits[{_dir_name(d)}] all-zero — "
                       f"no vehicles produced for approach {_dir_name(d)}")

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
        raise SystemExit(
            f"DemandModel: all-zero turn fraction for lane {lane} "
            f"on approach {approach.value} "
            f"(legal turns: {[t.value for t in legal]})")
    weights_t = [f / total for f in fracs]
    turn = rng.choices(legal, weights=weights_t, k=1)[0]
    return approach, turn


# ---- defaults ---------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_SECONDS = 12.0
VEHICLE_CLASSES = ["car"]
# representative lengths for headway (m)
VEHICLE_LENGTH = {"car": 4.47}
# common CCTV-ish urban/arterial speeds (km/h) in free flow
SPEED_KMH_RANGE = (50, 80)
COLOR_LIST = [
    ((0.8, 0.1, 0.1, 1.0), "red"),
    ((0.1, 0.2, 0.8, 1.0), "blue"),
    ((0.9, 0.9, 0.9, 1.0), "white"),
    ((0.1, 0.1, 0.1, 1.0), "black"),
    ((0.8, 0.8, 0.1, 1.0), "yellow"),
    ((0.5, 0.5, 0.5, 1.0), "grey"),
    ((0.1, 0.6, 0.3, 1.0), "green"),
]


def _vehicle_leave_frame(v: dict, approach_visible_length: float, fps: int) -> int:
    """Frame where a vehicle completes its out-camera segment."""
    travel_f = int(round(approach_visible_length / v["speed_ms"] * fps))
    release_f = v.get("release_frame", v["depart_frame"] + travel_f)
    dt = G.delta_t_frames(G.Turn(v["turn"]), v["speed_ms"], fps,
                          lane_index=v["lane"])
    return release_f + dt + travel_f


def _filter_to_duration(vehicles: list, duration_frames: int,
                        approach_visible_length: float, fps: int) -> int:
    """Remove vehicles outside the rendered clip; return count.

    A vehicle is admitted when its full motion intersects the clip.  This
    supports steady-state warm-up traffic that may have a negative
    ``depart_frame`` (already on the approach at frame 0) and dense traffic that
    is still visible when the fixed-length clip ends.  Rendering/metadata clamp
    per-frame outputs to [0, duration_frames), so no out-of-clip frames leak.
    """
    keep = [v for v in vehicles
            if _vehicle_leave_frame(v, approach_visible_length, fps) >= 0
            and v["depart_frame"] < duration_frames]
    dropped = len(vehicles) - len(keep)
    if dropped:
        vehicles[:] = keep
    return dropped
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

    This low-level scheduler assigns all provided vehicles and may push some
    past ``duration_frames``. The high-level ``generate`` entry point applies
    the project contract: ``--seconds`` is the fixed rendered clip, so vehicles
    that never intersect that clip are removed before writing the scenario.
    """
    fps_f = float(fps)
    floor_horizon_s = duration_frames / fps_f

    # Scale the arrival horizon to the vehicle count so every vehicle gets a
    # real Poisson arrival at the demand rate, spread over the natural window.
    # The previous behaviour pinned the horizon to `duration_frames/fps` (the
    # --seconds floor): at default demand (400 veh/h/approach ~= 0.444 veh/s
    # total) a 12s window only emits ~5 Poisson arrivals, so the other ~85 of
    # 90 vehicles fell into the "leftover cramming" path (consecutive frames)
    # -- producing a massive queue, signal saturation, and dense scenes that
    # render pathologically slowly. See memory 39 / the n=90 Kaggle hang.
    #
    # natural_horizon = n_vehicles / total_rate_per_s  (the time demand would
    # take to emit that many arrivals); ×POISSON_HORIZON_MARGIN covers Poisson
    # variance. The passed `duration_frames` becomes a true floor only.
    total_rate_ps = sum(demand.flow_vph(a) for a in G.Direction) / 3600.0
    n_veh = len(vehicles)
    if total_rate_ps > 0 and n_veh > 0:
        natural_horizon_s = (n_veh / total_rate_ps) * POISSON_HORIZON_MARGIN
        horizon_s = max(floor_horizon_s, natural_horizon_s)
    else:
        horizon_s = floor_horizon_s

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


def generate(seed: int, seconds: float,
             out_dir: str, fps: int = G.FPS,
             signal_plan: Optional[SG.SignalPlan] = None,
             demand: Optional[DemandModel] = None,
             signal_mode: str = "fixed") -> dict:
    """Generate a scenario and write ``scenario.json``.

    Vehicles are generated from a steady-state Poisson stream that starts
    before frame 0, then clipped to the rendered window. ``duration_frames`` is
    exactly ``int(round(seconds*fps))``. All vehicles in the output intersect
    the clip; stale pre-clip vehicles and never-visible post-clip vehicles are
    dropped.

    Args:
        signal_plan: pre-built signal plan (overrides ``signal_mode``).
            If None and ``signal_mode`` is ``"adaptive"`` / ``"fixed"``,
            the appropriate plan is constructed here.
        signal_mode: ``"fixed"`` (default fixed-cycle permissive-left
            ``SignalPlan``) or ``"adaptive"`` (NEMA 8-phase MaxPressure
            ``AdaptiveSignalPlan`` — closed-loop, rebuilds inside the
            ``_resolve_all`` fixpoint from realised arrivals).  Ignored when
            ``signal_plan`` is given explicitly.
        demand: Poisson per-approach flow + turning-movement split. If
            None, the default ``DemandModel`` is used.
    """
    duration_frames = int(round(seconds * fps))

    rng = random.Random(seed)
    demand = demand if demand is not None else DemandModel.default()

    road_meta = {}
    approach_len = 40.0
    if os.path.exists(ROAD_JSON):
        with open(ROAD_JSON) as f:
            road_meta = json.load(f)
            approach_len = road_meta.get("approach_length", 40.0)

    # Generate per-approach Poisson arrivals from a warm-up period before frame
    # 0 through the end of the clip.  Starting empty at frame 0 made short,
    # high-demand clips look artificially thin because there were no vehicles
    # already on screen or queued at the opening.  Warm-up creates the normal
    # steady-state condition: vehicles/queues may already be visible at frame 0.
    fps_f = float(fps)
    warmup_seconds = max(MIN_WARMUP_SECONDS, seconds)
    # Stable base speeds per (approach, lane) so same-lane vehicles share a
    # platoon speed and avoid arbitrary 50-behind-80 gaps. Base is sampled
    # uniformly in [50,80]; each vehicle jitters ±5 km/h, clamped.
    base_speed: dict = {}  # (approach.value, lane) -> km/h base
    arrivals_by_approach: dict = {}
    vehicle_idx = 0
    vehicles: list = []
    for approach in G.Direction:
        rate_ps = demand.flow_vph(approach) / 3600.0
        arrivals = []
        t = -warmup_seconds
        while t < seconds:
            t += rng.expovariate(rate_ps) if rate_ps > 0 else seconds
            if t < seconds:
                frame = int(round(t * fps_f))
                arrivals.append(frame)
        arrivals_by_approach[approach] = arrivals
        # Create vehicles for each arrival on this approach.
        for af in arrivals:
            v = make_vehicle(f"V{vehicle_idx:03d}", rng, demand=demand)
            vehicle_idx += 1
            # Force approach to match the Poisson clock.
            v["approach"] = approach.value
            v["depart_frame"] = af
            # ponytail: stable base speed + jitter per (approach,lane)
            lane_key = (approach.value, v["lane"])
            if lane_key not in base_speed:
                base_speed[lane_key] = rng.uniform(50, 80)
            jitter = rng.uniform(-5, 5)
            speed_kmh = max(50, min(80, base_speed[lane_key] + jitter))
            v["speed_kmh"] = round(speed_kmh, 2)
            v["speed_ms"] = round(K.speed_kmh_to_ms(speed_kmh), 3)
            vehicles.append(v)

    # Sort by depart_frame, approach, id for reproducibility.
    vehicles.sort(key=lambda v: (v["depart_frame"], v["approach"], v["id"]))

    # Enforce lane safety (push-later-only).
    _enforce_lane_safety(vehicles, fps, safety_gap=2.0,
                         approach_visible_length=approach_len, rng=rng)

    # Construct signal plan from mode if not supplied explicitly.
    if signal_plan is None and signal_mode == "adaptive":
        signal_plan = SG.AdaptiveSignalPlan(fps=fps)
    elif signal_plan is None and signal_mode == "fixed":
        signal_plan = SG.SignalPlan(fps=fps)
    # Preserve a requested fixed plan that was passed in.
    if signal_mode == "adaptive" and signal_plan is not None and not isinstance(
            signal_plan, SG.AdaptiveSignalPlan):
        # User mixed the two: prefer the explicit plan, switch the mode label.
        signal_mode = "fixed"

    _resolve_all(vehicles, approach_len, fps, signal_plan=signal_plan)

    # Clip-window admission: remove vehicles that never appear in the rendered
    # interval and re-resolve because taking invisible vehicles out can change
    # queue slots/exit conflicts for the remaining visible set.  Iterate to a
    # fixed point so scenario.json only contains vehicles that actually
    # contribute to the clip.
    dropped = 0
    while True:
        n = _filter_to_duration(vehicles, duration_frames, approach_len, fps)
        dropped += n
        if n == 0:
            break
        _resolve_all(vehicles, approach_len, fps, signal_plan=signal_plan)
    if dropped:
        import sys as _sys
        print(f"[INFO] clip window: admitted {len(vehicles)} visible "
              f"vehicles; skipped {dropped} invisible candidates "
              f"(outside 0..{duration_frames - 1})",
              file=_sys.stderr, flush=True)
    # Renumber IDs so the output is contiguous 0..N-1.
    for i, v in enumerate(vehicles):
        v["id"] = f"V{i:03d}"

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
          f"duration_frames={duration_frames}f)")
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

    Queueing is lane-physical, not just signal-color based: a vehicle also
    joins the queue if it reaches the approach while an earlier same-lane
    vehicle is still stopped/releasing ahead.  This prevents a later vehicle
    from free-flowing over an upstream stopped queue just because its own
    natural stop-line arrival happens during green.
    """
    assert signal_plan is not None
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

        # Walk vehicles in physical arrival order.  The old implementation
        # grouped only consecutive red arrivals; a green-arriving vehicle right
        # behind an unreleased red queue was marked free-flow and could pass
        # through stopped cars.  Keep an active same-lane queue instead.
        active_queue: list = []
        last_queued_release = None
        for v, stop_arrival, is_green in sorted(
                tagged, key=lambda tup: (tup[1], tup[0]["depart_frame"], tup[0]["id"])):
            turn = G.Turn(v["turn"])

            # Vehicles whose release time has passed no longer occupy a queue
            # slot on the approach.  Remaining entries are still stopped or
            # launching ahead of the current vehicle.
            active_queue = [q for q in active_queue
                            if q["release_frame"] > stop_arrival]
            if not active_queue:
                last_queued_release = None
            blocked_by_queue = bool(active_queue)

            if is_green and not blocked_by_queue:
                v["extra_stagger_frames"] = 0
                v["stop_frame"] = stop_arrival
                v["release_frame"] = stop_arrival
                v["queue_slot"] = -1
                v["wait_frames"] = 0
                continue

            # Red arrival OR green arrival blocked by a queue ahead: stop at
            # the next physical queue slot and release FIFO with reaction gap.
            slot = len(active_queue)
            base_release = stop_arrival if is_green else signal_plan.next_green_frame(
                approach, turn, stop_arrival)
            release = base_release + v["extra_stagger_frames"]
            if last_queued_release is not None:
                release = max(release, last_queued_release + reaction_per_slot)

            # Preserve later release pushes made by exit-conflict resolution in
            # the previous fixpoint step.  Those pushes are safety constraints;
            # recomputing from signal state alone must not pull the car earlier
            # and re-create an exit overlap.
            old_release = v.get("release_frame")
            if old_release is not None and old_release > release:
                release = old_release

            # Rebase if the queue/reaction/extra stagger pushes release outside
            # a legal green window for this movement.
            if not signal_plan.is_green(approach, turn, release):
                rebased = signal_plan.next_green_frame(approach, turn, release)
                rb_delta = rebased - release
                v["extra_stagger_frames"] = v.get("extra_stagger_frames", 0) + rb_delta
                release = rebased

            v["stop_frame"] = stop_arrival
            v["release_frame"] = release
            v["queue_slot"] = slot
            v["wait_frames"] = release - stop_arrival
            active_queue.append(v)
            last_queued_release = release

        # --- Post-hoc release-gap guard on queued vehicles in this group ---
        # ``extra_stagger_frames`` from exit-conflict resolution can reorder or
        # collapse release times relative to the pure ``slot * reaction``
        # pattern.  Ensure every queued pair releases ≥ reaction_per_slot apart.
        qs = [v for v in group if v.get("queue_slot", -1) >= 0]
        # Preserve physical arrival order.  Sorting by release_frame lets a rear
        # vehicle with a large exit-conflict stagger invert ahead of the front
        # vehicle; sorting by queue_slot alone can also mix separate platoons
        # because queue_slot resets after a queue has cleared.
        qs.sort(key=lambda v: (v["stop_frame"], v.get("queue_slot", 0), v["id"]))
        last_release = None
        last_leave_by_exit: dict = {}
        for v in qs:
            need = v["release_frame"]
            if last_release is not None:
                need = max(need, last_release + reaction_per_slot)

            # A later queued vehicle may have a shorter black-box turn time
            # than ANY earlier queue member that feeds the same exit lane. A
            # plain adjacent release-frame gap can therefore still let it
            # reappear before an earlier vehicle has left the out-camera. Track
            # last leave per exit lane across the whole physical queue order.
            out_dir, ex_lane = G.exit_lane_for_movement(
                G.Direction(v["approach"]), v["lane"], G.Turn(v["turn"]))
            exit_key = (out_dir.value, ex_lane)
            v_dt = G.delta_t_frames(G.Turn(v["turn"]), v["speed_ms"], fps,
                                    lane_index=v["lane"])
            if exit_key in last_leave_by_exit:
                need = max(need, last_leave_by_exit[exit_key] + 5 - v_dt)

            if v["release_frame"] < need:
                delta = need - v["release_frame"]
                v["release_frame"] = need
                v["wait_frames"] += delta
                # Accumulate into extra_stagger so the adjustment survives
                # exit-conflict resolution in later fixpoint rounds.
                v["extra_stagger_frames"] = v.get("extra_stagger_frames", 0) + delta
                # If the push leaves the green window, rebase to next green.
                if signal_plan is not None:
                    approach = G.Direction(v["approach"])
                    turn = G.Turn(v["turn"])
                    if not signal_plan.is_green(approach, turn, v["release_frame"]):
                        rebased = signal_plan.next_green_frame(
                            approach, turn, v["release_frame"])
                        rebase_delta = rebased - v["release_frame"]
                        v["release_frame"] = rebased
                        v["wait_frames"] += rebase_delta
                        v["extra_stagger_frames"] += rebase_delta

            v_tf = int(round(approach_visible_length / v["speed_ms"] * fps))
            last_release = v["release_frame"]
            last_leave_by_exit[exit_key] = v["release_frame"] + v_dt + v_tf


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
    # Dense queues can require many cascading pushes through the same exit lane:
    # delaying one vehicle may push it into the next platoon/green window, which
    # then forces later vehicles to move too. Iterate to stability instead of a
    # small fixed cap so the 5-frame exit buffer is actually guaranteed.
    for _ in range(100):
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
    """Check same-lane headway and catch-up safety. Enforces BOTH direct
    min-headway (no zero-frame same-lane gaps) AND catch-up safety for faster
    followers. Push depart_frame later for any violating pair. Returns True
    if any frame was changed."""
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
            # 1. Direct min-headway: no zero-frame/instant same-lane gap
            needed_min = K.min_headway_frames(max(lead["length"], follow["length"]),
                                              max(lead["speed_ms"], follow["speed_ms"]))
            if follow["depart_frame"] - lead["depart_frame"] < needed_min:
                new_f = lead["depart_frame"] + needed_min
                if new_f > follow["depart_frame"]:
                    delta = new_f - follow["depart_frame"]
                    follow["depart_frame"] = new_f
                    if "stop_frame" in follow:
                        follow["stop_frame"] += delta
                        follow["release_frame"] += delta
                    changed = True
            # 2. Catch-up: faster follower must not close gap
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
                 signal_plan=None, max_rounds=120):
    """Unified fixpoint: iterate headway → signal → exit until all
    constraints are satisfied and the full vehicle state is stable.

    Convergence is detected by comparing a snapshot of
    ``(depart_frame, stop_frame, release_frame, queue_slot, extra_stagger)``
    before and after each round.  Exit resolution and headway only push
    later (monotonic), so the loop converges.

    If ``signal_plan`` is an ``AdaptiveSignalPlan`` the plan is built once
    from the natural stop-line demand snapshot. Rebuilding the same timeline
    every fixpoint round was pure cost and erased late fallback intervals;
    feeding gated arrivals back in would make the controller chase itself.
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
        horizon = (
            max((t for ts in arrivals_snapshot.values() for t in ts),
                default=0) + 20 * signal_plan.max_green_f)
        signal_plan.rebuild(arrivals_snapshot, horizon_frames=horizon)

    for _round in range(max_rounds):
        before = _snapshot(vehicles)

        # 1. Same-lane headway (re-check after exit shifts)
        _check_headway_fixpoint(vehicles, approach_visible_length, fps)

        # 2. Signal gating (compute stop/release from current depart_frame)
        if signal_plan is not None:
            _apply_signal_gating(vehicles, approach_visible_length,
                                 signal_plan, fps)
            _resolve_exit_conflicts(vehicles, approach_visible_length, fps,
                                    signal_plan=signal_plan)
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
              f"gating can't all stabilise; consider increasing --seconds "
              f"or lowering --demand-scale.",
              file=_sys.stderr, flush=True)
        # Do not run another standalone signal pass here: the final fixpoint
        # round ends with exit-conflict resolution, and a signal-only pass can
        # undo those last release-frame shifts in saturated scenarios.  Keep the
        # safer post-exit state and surface the warning instead.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--fps", type=int, default=G.FPS,
                    help="frames per second (default: %(default)s)")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                    help="video length in seconds (default: %(default)s); "
                         "duration_frames = int(round(seconds*fps)); "
                         "steady-state traffic is clipped to this rendered "
                         "window")
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
                         f"({DEFAULT_APPROACH_FLOW_VPH:g} veh/h/approach) is used, "
                         "scaled by --demand-scale.")
    ap.add_argument("--demand-scale", type=float, default=1.0,
                    help="convenience density multiplier applied to the default "
                         "demand model when --demand is not given (default: "
                         "%(default)s). E.g. --demand-scale 3 makes ~1200 "
                         "veh/h/approach — denser on-screen traffic without "
                         "authoring a JSON. Ignored when --demand is a path. "
                         "Scale <= 0 produces zero vehicles.")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "..", "output", "run1"))
    args = ap.parse_args()
    if args.signal:
        if args.signal_mode == "adaptive":
            signal_plan = SG.AdaptiveSignalPlan(fps=args.fps)
        else:
            signal_plan = SG.SignalPlan(fps=args.fps)
    else:
        signal_plan = None
    if args.demand:
        demand = DemandModel.from_file(args.demand)
    else:
        scale = max(0.0, args.demand_scale)
        if scale <= 0.0:
            # ponytail: zero demand → zero vehicles, still write scenario.json
            demand = DemandModel(
                flows={d: 0.0 for d in G.Direction},
                turn_split=DEFAULT_TURN_SPLIT)
        else:
            demand = DemandModel(
                flows={d: DEFAULT_APPROACH_FLOW_VPH * scale for d in G.Direction},
                turn_split=DEFAULT_TURN_SPLIT)
    generate(args.seed, args.seconds, args.out, fps=args.fps,
             signal_plan=signal_plan, demand=demand,
             signal_mode=args.signal_mode)


if __name__ == "__main__":
    main()
