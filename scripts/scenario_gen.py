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
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "lib"))

import geometry as G
import kinematics as K
import envfile as ENV
from gen_plate import random_plate

try:
    from lib import signal as SG
except ImportError:
    import signal as SG

ROAD_JSON = os.path.join(HERE, "..", "assets", "road.json")


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


def make_vehicle(vid: str, rng: random.Random) -> dict:
    cls = rng.choices(VEHICLE_CLASSES, weights=[1])[0]
    rgba, color_name = rng.choice(COLOR_LIST)
    plate = random_plate(rng)
    approach = rng.choice(list(G.Direction))
    lane = rng.randint(0, G.NUM_LANES - 1)
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
    approach segment.

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
        v["depart_frame"] = int(frame)
        existing.append((frame, v["length"], v["speed_ms"]))
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
             signal_plan: Optional[SG.SignalPlan] = None) -> dict:
    min_duration_frames = int(round(seconds * fps))

    rng = random.Random(seed)
    vehicles = [make_vehicle(f"V{i:03d}", rng) for i in range(num_vehicles)]

    road_meta = {}
    approach_len = 40.0
    if os.path.exists(ROAD_JSON):
        with open(ROAD_JSON) as f:
            road_meta = json.load(f)
            approach_len = road_meta.get("approach_length", 40.0)

    vehicles = schedule_departures(
        vehicles, min_duration_frames, rng,
        approach_visible_length=approach_len)

    _resolve_all(vehicles, approach_len, fps, signal_plan=signal_plan)

    required = _compute_max_leave_frame(vehicles, road_meta, fps)
    tail = fps  # 1 s of empty road after the last car leaves
    duration_frames = max(min_duration_frames, required + tail)

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
        scenario["signal_cycle_frames"] = signal_plan.cycle_frames
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


def _resolve_all(vehicles, approach_visible_length, fps,
                 signal_plan=None, max_rounds=20):
    """Unified fixpoint: iterate headway → signal → exit until all
    constraints are satisfied and the full vehicle state is stable.

    Convergence is detected by comparing a snapshot of
    ``(depart_frame, stop_frame, release_frame, queue_slot, extra_stagger)``
    before and after each round.  Exit resolution and headway only push
    later (monotonic), so the loop converges.
    """
    def _snapshot(vs):
        return [
            (v["depart_frame"], v.get("stop_frame"), v.get("release_frame"),
             v.get("queue_slot"), v.get("extra_stagger_frames"))
            for v in vs
        ]

    for _ in range(max_rounds):
        before = _snapshot(vehicles)

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
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "..", "output", "run1"))
    args = ap.parse_args()
    signal_plan = SG.SignalPlan(fps=args.fps) if args.signal else None
    generate(args.seed, args.num_vehicles, args.seconds, args.out, fps=args.fps,
             signal_plan=signal_plan)


if __name__ == "__main__":
    main()
