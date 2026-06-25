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
VEHICLE_CLASSES = ["car", "van", "truck", "bus"]
# representative lengths for headway (m)
VEHICLE_LENGTH = {"car": 4.47, "van": 5.0, "truck": 7.0, "bus": 8.33}
# common CCTV-ish speeds (km/h) in free flow
SPEED_KMH_RANGE = (30, 60)
COLORS = [
    (0.8, 0.1, 0.1, 1.0), "red",
    (0.1, 0.2, 0.8, 1.0), "blue",
    (0.9, 0.9, 0.9, 1.0), "white",
    (0.1, 0.1, 0.1, 1.0), "black",
    (0.8, 0.8, 0.1, 1.0), "yellow",
    (0.5, 0.5, 0.5, 1.0), "grey",
    (0.1, 0.6, 0.3, 1.0), "green",
]
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
    cls = rng.choices(VEHICLE_CLASSES, weights=[5, 2, 2, 1])[0]
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
                        safety_gap: float = 2.0) -> list:
    """Assign a depart_frame to each vehicle so no two in the same
    (approach, lane) violate min-headway. Vehicles are placed in a random
    order; each is given the earliest feasible frame >= a random target.
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
                needed = K.min_headway_frames(max(v["length"], el),
                                              max(v["speed_ms"], es), safety_gap)
                if abs(frame - ef) < needed:
                    ok = False
                    break
            if ok:
                break
            frame += step
        v["depart_frame"] = int(frame)
        existing.append((frame, v["length"], v["speed_ms"]))
    return vehicles


def generate(seed: int, num_vehicles: int, duration_frames: int,
             out_dir: str) -> dict:
    rng = random.Random(seed)
    vehicles = [make_vehicle(f"V{i:03d}", rng) for i in range(num_vehicles)]
    vehicles = schedule_departures(vehicles, duration_frames, rng)

    scenario = {
        "seed": seed,
        "fps": G.FPS,
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
    print(f"Wrote scenario: {out_path}  ({len(vehicles)} vehicles)")
    return scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--num-vehicles", type=int, default=DEFAULT_NUM_VEHICLES)
    ap.add_argument("--duration", type=int, default=DEFAULT_DURATION_FRAMES,
                    help="duration in frames")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "..", "output", "run1"))
    args = ap.parse_args()
    generate(args.seed, args.num_vehicles, args.duration, args.out)


if __name__ == "__main__":
    main()
