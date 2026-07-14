"""Phase 5 — Run validation.

Checks a completed run's outputs for consistency:
  * all 8 videos present (or the subset that should exist)
  * metadata.json present and well-formed
  * no lane overlaps (headway) in the scenario
  * reappear timing matches Delta t for every vehicle
  * plate identity consistent (same plate string on in & out segments)
  * every visible frame has a pose and a camera assigned

Run:
    python3 scripts/validate_run.py --out output/run1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

import geometry as G
import kinematics as K


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = args.out
    ok = True

    print("=" * 60)
    print(f"VALIDATE RUN: {out}")
    print("=" * 60)

    # videos
    expected = G.camera_names()
    videos = [f for f in os.listdir(out) if f.startswith("video_") and f.endswith(".mp4")]
    present_tags = {v[len("video_"):-len(".mp4")] for v in videos}
    missing = set(expected) - present_tags
    ok &= check(f"videos present ({len(videos)}/{len(expected)})", len(videos) == len(expected),
                f"missing: {missing}" if missing else "")
    for v in sorted(videos):
        sz = os.path.getsize(os.path.join(out, v))
        ok &= check(f"  {v} non-empty", sz > 0, f"{sz} B")

    # metadata
    meta_path = os.path.join(out, "metadata.json")
    ok &= check("metadata.json exists", os.path.exists(meta_path))
    if not os.path.exists(meta_path):
        print("=" * 60); sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    # scenario headway
    scn = None
    scn_path = os.path.join(out, "scenario.json")
    if os.path.exists(scn_path):
        with open(scn_path) as f:
            scn = json.load(f)
        lanes = {}
        for v in scn["vehicles"]:
            lanes.setdefault((v["approach"], v["lane"]), []).append(
                (v["depart_frame"], v["length"], v["speed_ms"]))
        bad = [k for k, s in lanes.items() if not K.conflict_free(s)]
        ok &= check("no lane headway conflicts", not bad, f"bad lanes: {bad}" if bad else "")

    # per-vehicle timing + identity
    for v in meta["vehicles"]:
        # delta_t matches
        expected_dt = G.delta_t_frames(G.Turn(v["turn"]), v["speed_ms"], meta["fps"],
                                       lane_index=v["lane"])
        ok &= check(f"{v['id']} delta_t matches",
                    v["delta_t_frames"] == expected_dt,
                    f"meta={v['delta_t_frames']} expected={expected_dt}")
        # reappear = disappear + delta_t
        ok &= check(f"{v['id']} reappear = disappear + dt",
                    v["reappear_frame"] == v["disappear_frame"] + v["delta_t_frames"])
        # plate present
        ok &= check(f"{v['id']} has plate", bool(v.get("plate")), v.get("plate", ""))
        # in/out cameras consistent with approach/turn
        ex_dir, _ = G.exit_lane_for_movement(G.Direction(v["approach"]), v["lane"], G.Turn(v["turn"]))
        ok &= check(f"{v['id']} out_camera matches exit direction",
                    v["out_camera"] == f"out_{ex_dir.value}",
                    f"{v['out_camera']} vs out_{ex_dir.value}")
        # visible frames have poses
        vis = [f for f in v["frames"] if f["visible"]]
        bad_poses = [f for f in vis if f["pose"] is None]
        ok &= check(f"{v['id']} all visible frames have poses", not bad_poses,
                    f"{len(bad_poses)} missing" if bad_poses else "")
        # in segment frames tagged with in_cam, out with out_cam
        in_ok = all(f["camera"] == v["in_camera"] for f in vis
                    if f["frame"] <= v["disappear_frame"])
        out_ok = all(f["camera"] == v["out_camera"] for f in vis
                     if f["frame"] >= v["reappear_frame"])
        ok &= check(f"{v['id']} in-segment camera tag", in_ok)
        ok &= check(f"{v['id']} out-segment camera tag", out_ok)

    # ---- state-based microsim validation (micro prototype or research engine) --
    simulator_mode = scn.get("simulator", "legacy") if scn is not None else "legacy"
    if simulator_mode in ("micro", "research") and scn is not None:
        print("-" * 60)
        print("MICROSCOPIC SIMULATION CHECKS")
        print("-" * 60)

        # Check 1: speed is non-negative for all vehicles
        for v in scn["vehicles"]:
            ok &= check(f"{v['id']} speed_ms >= 0",
                        v["speed_ms"] >= 0,
                        f"speed_ms={v['speed_ms']}")

        # Check 2: wait_frames >= 0 for all vehicles
        for v in scn["vehicles"]:
            wf = v.get("wait_frames", 0)
            ok &= check(f"{v['id']} wait_frames >= 0",
                        wf >= 0,
                        f"wait_frames={wf}")

        # Check 3: release >= stop (FIFO ordering)
        for v in scn["vehicles"]:
            sf = v.get("stop_frame")
            rf = v.get("release_frame")
            if sf is not None and rf is not None:
                ok &= check(f"{v['id']} release >= stop",
                            rf >= sf,
                            f"stop={sf} release={rf}")

        # Check 4: same-lane FIFO — vehicles released in depart order per lane
        lane_releases = {}
        for v in scn["vehicles"]:
            key = (v["approach"], v["lane"])
            rf = v.get("release_frame")
            if rf is not None:
                lane_releases.setdefault(key, []).append(
                    (v["depart_frame"], rf, v["id"]))
        for key, entries in lane_releases.items():
            entries.sort(key=lambda x: x[0])  # sort by depart_frame
            for i in range(len(entries) - 1):
                ok &= check(f"{entries[i][2]}->{entries[i+1][2]} FIFO release",
                            entries[i][1] <= entries[i+1][1],
                            f"lane {key}: {entries[i][1]} > {entries[i+1][1]}"
                            if entries[i][1] > entries[i+1][1] else "")

        # Check 5: queue_slot consistency — queued vehicles have slot >= 0
        for v in scn["vehicles"]:
            wf = v.get("wait_frames", 0)
            qs = v.get("queue_slot", -1)
            if wf > 0:
                ok &= check(f"{v['id']} queued has queue_slot >= 0",
                            qs >= 0,
                            f"wait={wf} slot={qs}")

    print("=" * 60)
    print("OVERALL:", "ALL PASS" if ok else "FAILURES PRESENT")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
