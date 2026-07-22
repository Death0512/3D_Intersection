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
import envfile as ENV

TRACE_POS_TOL_M = 0.01


def _load_json(path):
    with open(path) as f:
        return json.load(f)


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

    # Metadata records the expected video subset for partial-camera runs.
    meta_path = os.path.join(out, "metadata.json")
    meta = _load_json(meta_path) if os.path.exists(meta_path) else None

    # videos
    if meta and meta.get("videos"):
        expected = [v[len("video_"):-len(".mp4")] for v in meta.get("videos", [])]
    else:
        expected = G.camera_names()
    videos = [f for f in os.listdir(out) if f.startswith("video_") and f.endswith(".mp4")]
    present_tags = {v[len("video_"):-len(".mp4")] for v in videos}
    missing = set(expected) - present_tags
    expected_present = [t for t in expected if t in present_tags]
    ok &= check(f"videos present ({len(expected_present)}/{len(expected)})", not missing,
                f"missing: {missing}" if missing else "")
    for v in sorted(videos):
        sz = os.path.getsize(os.path.join(out, v))
        ok &= check(f"  {v} non-empty", sz > 0, f"{sz} B")

    # metadata
    ok &= check("metadata.json exists", os.path.exists(meta_path))
    if not os.path.exists(meta_path):
        print("=" * 60); sys.exit(1)
    if meta is None:
        meta = _load_json(meta_path)

    # scenario headway
    scn = None
    scn_path = os.path.join(out, "scenario.json")
    if os.path.exists(scn_path):
        scn = _load_json(scn_path)

    simulator_mode = scn.get("simulator", "legacy") if scn is not None else "legacy"
    if simulator_mode == "sumo":
        ok &= check("scenario simulator is SUMO", True)
        ok &= check("metadata simulator is SUMO", meta.get("simulator") == "sumo",
                    str(meta.get("simulator")))
        scenario_vehicles = scn.get("vehicles", []) if scn else []
        meta_vehicles = meta.get("vehicles", [])
        ok &= check("metadata vehicle count matches scenario",
                    len(meta_vehicles) == len(scenario_vehicles),
                    f"metadata={len(meta_vehicles)} scenario={len(scenario_vehicles)}")
        scn_by_id = {v.get("id"): v for v in scenario_vehicles}
        for mv in meta_vehicles:
            vid = mv.get("id")
            sv = scn_by_id.get(vid)
            ok &= check(f"{vid} exists in scenario", sv is not None)
            frames = mv.get("frames", [])
            ok &= check(f"{vid} has metadata frames", bool(frames))
            bad_pose = [fr for fr in frames if not fr.get("pose")]
            ok &= check(f"{vid} all frames have poses", not bad_pose,
                        f"{len(bad_pose)} missing" if bad_pose else "")
            if sv is not None:
                traj = sv.get("trajectory", [])
                ok &= check(f"{vid} has SUMO trajectory", bool(traj))
                ok &= check(f"{vid} metadata frames match trajectory",
                            len(frames) == len(traj),
                            f"metadata={len(frames)} trajectory={len(traj)}")
                if traj and frames:
                    ok &= check(f"{vid} first/last frame match trajectory",
                                frames[0].get("frame") == traj[0].get("frame") and
                                frames[-1].get("frame") == traj[-1].get("frame"))
        print("=" * 60)
        print("OVERALL:", "ALL PASS" if ok else "FAILURES PRESENT")
        print("=" * 60)
        sys.exit(0 if ok else 1)

    if scn is not None:
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

    # trajectory artifact integrity
    traj_path = os.path.join(out, "trajectory.json")
    if os.path.exists(traj_path):
        print("-" * 60)
        print("TRAJECTORY INTEGRITY CHECKS")
        print("-" * 60)
        traj = _load_json(traj_path)
        samples = traj.get("samples", [])
        by_vid = {}
        for sample in samples:
            vid = sample.get("vehicle_id")
            if vid:
                by_vid.setdefault(vid, []).append(sample)

        meta_by_vid = {v["id"]: v for v in meta["vehicles"]}
        scenario_by_vid = {v["id"]: v for v in scn["vehicles"]} if scn else {}

        ok &= check("trajectory schema is v2/v3", traj.get("schema") in {"trajectory.v2", "trajectory.v3"},
                    traj.get("schema", ""))

        for vid, veh_samples in by_vid.items():
            veh_meta = meta_by_vid.get(vid)
            scn_veh = scenario_by_vid.get(vid)
            if veh_meta is None or scn_veh is None:
                ok &= check(f"{vid} trajectory has matching metadata/scenario",
                            False, "orphan trajectory vehicle")
                continue

            veh_samples.sort(key=lambda s: int(s.get("frame", 0)))
            # world coordinates must be present and finite for every sample
            bad_world = [s for s in veh_samples
                         if any(k not in s for k in ("world_x", "world_y", "world_z"))]
            ok &= check(f"{vid} trajectory has world coords", not bad_world,
                        f"{len(bad_world)} missing" if bad_world else "")

            # samples must stay on one route / lane identity
            route_ok = all(
                s.get("approach") == veh_meta["approach"] and
                s.get("lane") == veh_meta["lane"] and
                s.get("turn") == veh_meta["turn"]
                for s in veh_samples
            )
            ok &= check(f"{vid} trajectory route identity consistent", route_ok)

            trace_vehicle = traj.get("vehicles", {}).get(vid, {})
            spawn = trace_vehicle.get("spawn_position", {})
            ok &= check(f"{vid} has spawn position",
                        all(k in spawn for k in ("x", "y", "z")))
            env = ENV.load_env(f"in_{scn_veh['approach']}", os.path.abspath(os.path.join(HERE, "..")))
            expected_anchor, _ = ENV.lane_default_anchor(env, scn_veh["lane"])
            if all(k in spawn for k in ("x", "y")):
                spawn_err = ((spawn["x"] - expected_anchor[0]) ** 2 +
                             (spawn["y"] - expected_anchor[1]) ** 2) ** 0.5
                ok &= check(f"{vid} spawn position matches lane anchor",
                            spawn_err < TRACE_POS_TOL_M,
                            f"err={spawn_err:.3f}m")

            # disappearance timing must not regress before release
            release = scn_veh.get("release_frame")
            disappear = veh_meta["disappear_frame"]
            if release is not None:
                ok &= check(f"{vid} disappear >= release",
                            disappear >= release,
                            f"release={release} disappear={disappear}")

            # trajectory samples should be monotone in frame and s for approach-side samples
            frames = [int(s.get("frame", 0)) for s in veh_samples]
            ok &= check(f"{vid} trajectory frames sorted", frames == sorted(frames))
            approach_samples = [s for s in veh_samples
                                if s.get("stage") in ("APPROACH", "QUEUED")]
            if len(approach_samples) >= 2:
                s_vals = [float(s.get("s", 0.0)) for s in approach_samples]
                ok &= check(f"{vid} approach s monotone", all(a <= b for a, b in zip(s_vals, s_vals[1:])),
                            f"s={s_vals[:5]}..." if len(s_vals) > 5 else f"s={s_vals}")

            if traj.get("schema") == "trajectory.v3":
                exit_samples = [s for s in veh_samples if s.get("stage") == "EXIT"]
                out_meta_frames = [f for f in veh_meta["frames"]
                                   if f.get("camera") == veh_meta.get("out_camera")]
                if out_meta_frames:
                    exit_frames = [int(s.get("frame", -1)) for s in exit_samples]
                    out_frames = [int(f.get("frame", -2)) for f in out_meta_frames]
                    ok &= check(f"{vid} v3 has EXIT samples for out-camera frames",
                                exit_frames == out_frames,
                                f"exit={exit_frames[:5]} out={out_frames[:5]}")
                    last_exit = exit_samples[-1] if exit_samples else None
                    leave = trace_vehicle.get("leave_position") or trace_vehicle.get("exit_position") or {}
                    if last_exit and all(k in leave for k in ("x", "y")):
                        leave_err = ((float(last_exit["world_x"]) - float(leave["x"])) ** 2 +
                                     (float(last_exit["world_y"]) - float(leave["y"])) ** 2) ** 0.5
                        ok &= check(f"{vid} v3 final EXIT sample matches leave position",
                                    leave_err < TRACE_POS_TOL_M,
                                    f"err={leave_err:.3f}m")

            # geometry consistency: trajectory world pose should be near metadata pose at matching frames
            meta_frames = {f["frame"]: f for f in veh_meta["frames"]}
            matched = 0
            for sample in veh_samples:
                fr = int(sample.get("frame", -1))
                meta_fr = meta_frames.get(fr)
                if not meta_fr:
                    continue
                matched += 1
                pose = meta_fr.get("pose") or {}
                dx = abs(float(sample.get("world_x", 0.0)) - float(pose.get("x", 0.0)))
                dy = abs(float(sample.get("world_y", 0.0)) - float(pose.get("y", 0.0)))
                if dx > TRACE_POS_TOL_M or dy > TRACE_POS_TOL_M:
                    ok &= check(f"{vid} metadata/trajectory pose match",
                                False, f"frame={fr} dx={dx:.3f} dy={dy:.3f}")
                    break
            else:
                ok &= check(f"{vid} metadata/trajectory pose match", matched > 0,
                            f"matched={matched}")

    # ---- state-based microsim validation (micro prototype or research engine) --
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
