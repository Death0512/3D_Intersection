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
import envfile as ENV
import traffic_signal as SG
from gen_plate import random_plate
import micro_sim as MS
import research_sim as RS
from sim.exporter import write_simulation_artifacts
from sim.trajectory import enrich_trajectory_samples

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


def _remap_trajectory_samples(sim_meta: dict, id_map: Dict[str, str]) -> None:
    """Keep trajectory samples for final vehicles and remap ids in place."""
    if "trajectory_samples" not in sim_meta:
        return
    kept = []
    for sample in sim_meta.get("trajectory_samples", []):
        vid = sample.get("vehicle_id")
        if vid in id_map:
            sample["vehicle_id"] = id_map[vid]
            kept.append(sample)
    sim_meta["trajectory_samples"] = kept


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
    speed_ms = speed_kmh / 3.6
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


def _enforce_lane_safety(vehicles: list, fps: int, safety_gap: float,
                         approach_visible_length: float, rng: random.Random):
    """Push-later-only feasibility pass ensuring no two vehicles in the same
    (approach, lane) overlap at depart. Vehicles are processed in depart_frame
    order within each lane; each vehicle is pushed to the earliest frame that
    satisfies a minimum headway >= (max_length + safety_gap) / max_speed * fps.
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
                    max_len = max(v["length"], el)
                    max_spd = max(v["speed_ms"], es)
                    needed = int(math.ceil((max_len + safety_gap) / max_spd * fps)) if max_spd > 0 else fps
                    if abs(frame - ef) < needed:
                        ok = False
                        break
                if ok:
                    break
                frame += 1
            else:
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
             signal_mode: str = "fixed",
             simulator: str = "micro") -> dict:
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
            ``AdaptiveSignalPlan`` — event-driven v2 microsim builds the
            realised signal timeline).  Ignored when
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
        if rate_ps <= 0.0:
            arrivals_by_approach[approach] = arrivals
            continue
        t = -warmup_seconds
        while t < seconds:
            t += rng.expovariate(rate_ps)
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
            v["speed_ms"] = round(speed_kmh / 3.6, 3)
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

    # ---- microsimulation ---------------------------------------------------
    def _run_sim(veh_list):
        if simulator == "research":
            return RS.simulate(veh_list, approach_len, fps,
                               signal_plan=signal_plan, seed=seed,
                               record_trajectories=True)
        if simulator == "micro":
            return MS.simulate(veh_list, approach_len, fps,
                               signal_plan=signal_plan, seed=seed)
        raise SystemExit(
            f"ERROR: simulator '{simulator}' is not supported. "
            f"Use 'micro', 'research', or 'sumo' (via run_sumo_unified.py)."
        )

    vehicles, sim_meta = _run_sim(vehicles)
    sim_arrivals = sim_meta.get("arrival_events", {})

    # Build adaptive signal timeline from realised stop-line arrivals.
    # For fixed plans the pre-built SignalPlan is_green is used directly.
    adaptive_intervals = None
    adaptive_clearances = None
    if isinstance(signal_plan, SG.AdaptiveSignalPlan):
        adaptive_intervals = sim_meta.get("adaptive_intervals")
        adaptive_clearances = sim_meta.get("adaptive_clearances")

    # Clip-window: remove vehicles with no visible footprint. Dropping vehicles
    # can shift queues earlier, making more warm-up vehicles stale; iterate to a
    # fixed point (normally one pass, hard-capped as a safety net).
    total_dropped = 0
    for _ in range(5):
        dropped = _filter_to_duration(vehicles, duration_frames, approach_len, fps)
        if not dropped:
            break
        total_dropped += dropped
        vehicles, sim_meta = _run_sim(vehicles)
        sim_arrivals = sim_meta.get("arrival_events", {})
    else:
        # Loop exhausted without reaching a fixed point (re-sim after the last
        # drop can still shift a few vehicles out of window). Guarantee no
        # out-of-window vehicle ever ships with one final drop-only pass.
        dropped = _filter_to_duration(vehicles, duration_frames, approach_len, fps)
        total_dropped += dropped

    # Rebuild adaptive timeline if needed after final clipping/re-sim.
    if isinstance(signal_plan, SG.AdaptiveSignalPlan):
        adaptive_intervals = sim_meta.get("adaptive_intervals")
        adaptive_clearances = sim_meta.get("adaptive_clearances")
    if total_dropped:
        print(f"[INFO] clip window: admitted {len(vehicles)} visible "
              f"vehicles; skipped {total_dropped} invisible candidates "
              f"(outside 0..{duration_frames - 1})",
              file=sys.stderr, flush=True)

    # Soft validation warn
    if signal_plan is not None:
        waits = [v.get("wait_frames", 0) for v in vehicles]
        if waits:
            mean_w = sum(waits) / len(waits)
            if mean_w > 2 * duration_frames:
                print(f"[WARN] mean wait {mean_w:.0f}f > 2*duration "
                      f"({2*duration_frames}f) — consider lower demand-scale",
                      file=sys.stderr, flush=True)

    # Renumber IDs
    id_map = {}
    for i, v in enumerate(vehicles):
        old_id = v["id"]
        v["id"] = f"V{i:03d}"
        id_map[old_id] = v["id"]
    _remap_trajectory_samples(sim_meta, id_map)

    scenario = {
        "seed": seed,
        "fps": fps,
        "duration_frames": duration_frames,
        "box_size": G.BOX_SIZE,
        "num_lanes": G.NUM_LANES,
        "lane_centerlines_x": G.LANE_CENTERLINES,
        "cameras": G.camera_names(),
        "vehicles": vehicles,
        "generator": "v2",
        "simulator": simulator,
    }
    if signal_plan is not None:
        scenario["signal_mode"] = signal_mode
        scenario["signal_cycle_frames"] = signal_plan.cycle_frames
        if adaptive_intervals is not None:
            # Adaptive: timeline is built by v2 sim
            scenario["signal_timeline"] = [
                {"start": s, "end": e, "phases": list(combo)}
                for (s, e, combo) in adaptive_intervals
            ]
            scenario["signal_clearances"] = [
                {"start": s, "end": e}
                for (s, e) in adaptive_clearances
            ]
        else:
            # Fixed plan: use pre-built boundaries
            scenario["signal_timeline"] = [
                {"start": s, "phase": name}
                for (s, name) in signal_plan._boundaries
            ]
    if demand is not None:
        scenario["demand"] = demand.to_dict()
    os.makedirs(out_dir, exist_ok=True)
    if simulator == "research":
        enrich_trajectory_samples(scenario, sim_meta, ROOT, road_meta=road_meta)
        scenario["simulation_artifacts"] = write_simulation_artifacts(
            out_dir, scenario, sim_meta)
    out_path = os.path.join(out_dir, "scenario.json")
    with open(out_path, "w") as f:
        json.dump(scenario, f, indent=2)
    print(f"Wrote scenario: {out_path}  ({len(vehicles)} vehicles, {fps} fps, "
          f"duration_frames={duration_frames}f)")
    return scenario


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
    ap.add_argument("--generator", type=str, default="v2",
                    choices=["v2"],
                    help="scheduler version: only v2 (event-driven) "
                         "supported (default: %(default)s)")
    ap.add_argument("--simulator", type=str, default="micro",
                    choices=["legacy", "micro", "research"],
                    help="simulation engine: 'legacy' (event-driven queue, "
                         "default), 'micro' (IDM prototype), or 'research' "
                         "(formal state-based simulation kernel)")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "..", "output", "run1"))
    args = ap.parse_args()
    if args.generator != "v2":
        sys.exit(f"ERROR: --generator '{args.generator}' not supported. Only 'v2'.")
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
             signal_mode=args.signal_mode,
             simulator=args.simulator)


if __name__ == "__main__":
    main()
