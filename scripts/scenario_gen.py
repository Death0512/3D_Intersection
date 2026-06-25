"""Phase 2 — Scenario generator (pure Python).

Produces a reproducible, conflict-free list of vehicles for one simulation
run, written to output/<run>/scenario.json.

Each vehicle gets: id, class, color, unique plate, approach, lane, turn,
speed, depart_frame. Departures are scheduled so that no two vehicles in the
same (approach, lane) overlap (min-headway enforced).

Usage:
    python3 scripts/scenario_gen.py --seed 42 --num-vehicles 20 --duration 300
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import string

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

import geometry as G
import kinematics as K
from gen_plate import random_plate


# ---- defaults ---------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_NUM_VEHICLES = 20
DEFAULT_DURATION_FRAMES = 300  # 10 s @ 30 fps
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


def make_vehicle(vid: str, rng: random.Random) -> dict:
    cls = rng.choices(VEHICLE_CLASSES, weights=[1])[0]
    rgba, color_name = rng.choice(COLOR_LIST)
    plate = random_plate(rng)
    approach = rng.choice(list(G.Direction))
    lane = rng.randint(0, G.NUM_LANES - 1)
    turn = rng.choices(TURNS, weights=TURN_WEIGHTS)[0]
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


def generate(seed: int, num_vehicles: int, duration_frames: int,
             out_dir: str, fps: int = G.FPS) -> dict:
    rng = random.Random(seed)
    vehicles = [make_vehicle(f"V{i:03d}", rng) for i in range(num_vehicles)]
    vehicles = schedule_departures(vehicles, duration_frames, rng)

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
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scenario.json")
    with open(out_path, "w") as f:
        json.dump(scenario, f, indent=2)
    print(f"Wrote scenario: {out_path}  ({len(vehicles)} vehicles, {fps} fps, {duration_frames} frames)")
    return scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--num-vehicles", type=int, default=DEFAULT_NUM_VEHICLES)
    ap.add_argument("--fps", type=int, default=G.FPS,
                    help="frames per second (default: %(default)s)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="video length in seconds; overrides --duration")
    ap.add_argument("--duration", type=int, default=DEFAULT_DURATION_FRAMES,
                    help="duration in frames (ignored if --seconds given)")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "..", "output", "run1"))
    args = ap.parse_args()
    if args.seconds is not None:
        duration = int(round(args.seconds * args.fps))
    else:
        duration = args.duration
    generate(args.seed, args.num_vehicles, duration, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
