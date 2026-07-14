"""Compare legacy, micro (IDM), and research simulators side-by-side.

Runs all three simulators on the same seed/demand/signal and prints
thesis-ready comparison metrics: delay, throughput, queue statistics,
and per-vehicle timing differences.

Usage:
    python3 scripts/compare_simulators.py --seed 42 --seconds 12 --signal
    python3 scripts/compare_simulators.py --seed 42 --seconds 30 --signal --signal-mode adaptive
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import List, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

import geometry as G
import traffic_signal as SG
import scenario_gen as SGEN


def _stats(values: List[float]) -> Dict[str, float]:
    """Basic descriptive stats."""
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = len(values)
    mu = sum(values) / n
    var = sum((x - mu) ** 2 for x in values) / max(n, 1)
    return {
        "count": n,
        "mean": round(mu, 2),
        "std": round(var ** 0.5, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def compare(seed: int, seconds: float, fps: int,
            signal_plan, signal_mode: str,
            demand: SGEN.DemandModel):
    """Run both simulators and return comparison dict."""
    results = {}
    for sim_name in ("legacy", "micro", "research"):
        with tempfile.TemporaryDirectory() as td:
            scn = SGEN.generate(
                seed, seconds, td, fps=fps,
                signal_plan=signal_plan, demand=demand,
                signal_mode=signal_mode, simulator=sim_name,
            )
            vehicles = scn["vehicles"]
            dur = scn["duration_frames"]

            waits = [v.get("wait_frames", 0) for v in vehicles]
            delays = [v.get("release_frame", 0) - v.get("stop_frame", 0)
                      for v in vehicles
                      if v.get("stop_frame") is not None and
                      v.get("release_frame") is not None]
            queued = [v for v in vehicles if v.get("queue_slot", -1) >= 0]
            slots = [v["queue_slot"] for v in queued]

            results[sim_name] = {
                "num_vehicles": len(vehicles),
                "duration_frames": dur,
                "wait_frames": _stats([float(w) for w in waits]),
                "delay_frames": _stats([float(d) for d in delays]),
                "queued_count": len(queued),
                "queue_slot": _stats([float(s) for s in slots]),
                "throughput_vph": round(
                    len(vehicles) / (dur / fps / 3600), 1) if dur > 0 else 0,
            }

    return results


def print_report(results: Dict):
    """Pretty-print the comparison report."""
    print()
    W = 85
    print("=" * W)
    print("  SIMULATOR COMPARISON REPORT")
    print("=" * W)

    header = f"{'Metric':<30} {'Legacy':>15} {'Micro (IDM)':>15} {'Research':>15}"
    print(header)
    print("-" * W)

    leg = results["legacy"]
    mic = results["micro"]
    res = results["research"]

    rows = [
        ("Vehicles", leg["num_vehicles"], mic["num_vehicles"], res["num_vehicles"]),
        ("Duration (frames)", leg["duration_frames"], mic["duration_frames"], res["duration_frames"]),
        ("Throughput (veh/h)", leg["throughput_vph"], mic["throughput_vph"], res["throughput_vph"]),
        ("Queued count", leg["queued_count"], mic["queued_count"], res["queued_count"]),
        ("Mean wait (frames)", leg["wait_frames"]["mean"], mic["wait_frames"]["mean"], res["wait_frames"]["mean"]),
        ("Max wait (frames)", leg["wait_frames"]["max"], mic["wait_frames"]["max"], res["wait_frames"]["max"]),
        ("Std wait (frames)", leg["wait_frames"]["std"], mic["wait_frames"]["std"], res["wait_frames"]["std"]),
        ("Mean delay (frames)", leg["delay_frames"]["mean"], mic["delay_frames"]["mean"], res["delay_frames"]["mean"]),
        ("Max delay (frames)", leg["delay_frames"]["max"], mic["delay_frames"]["max"], res["delay_frames"]["max"]),
        ("Mean queue slot", leg["queue_slot"]["mean"], mic["queue_slot"]["mean"], res["queue_slot"]["mean"]),
        ("Max queue slot", leg["queue_slot"]["max"], mic["queue_slot"]["max"], res["queue_slot"]["max"]),
    ]

    for label, lv, mv, rv in rows:
        print(f"{label:<30} {str(lv):>15} {str(mv):>15} {str(rv):>15}")

    print("=" * W)
    print()
    print("Key thesis observation:")
    print("  In MICRO and RESEARCH simulators, vehicle speed is an OUTPUT of IDM")
    print("  dynamics, not a fixed input. RESEARCH adds formal driver profiles,")
    print("  conflict-resource intersection, and closed-loop adaptive signal FSM.")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Compare legacy, micro (IDM), and research simulators")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=G.FPS)
    ap.add_argument("--signal", action="store_true")
    ap.add_argument("--signal-mode", type=str, default="fixed",
                    choices=["fixed", "adaptive"])
    ap.add_argument("--demand-scale", type=float, default=1.0)
    ap.add_argument("--json", action="store_true",
                    help="output raw JSON instead of formatted report")
    args = ap.parse_args()

    if args.signal:
        if args.signal_mode == "adaptive":
            signal_plan = SG.AdaptiveSignalPlan(fps=args.fps)
        else:
            signal_plan = SG.SignalPlan(fps=args.fps)
    else:
        signal_plan = None

    demand = SGEN.DemandModel(
        flows={d: SGEN.DEFAULT_APPROACH_FLOW_VPH * args.demand_scale
               for d in G.Direction},
        turn_split=SGEN.DEFAULT_TURN_SPLIT,
    )

    results = compare(args.seed, args.seconds, args.fps,
                      signal_plan, args.signal_mode, demand)

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        print_report(results)


if __name__ == "__main__":
    main()
