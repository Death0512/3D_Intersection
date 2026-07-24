"""Phase 6 — convert simulation trajectory samples to render tracks.

The research simulator exports lane-longitudinal state samples (``s`` metres
from the camera-side spawn anchor toward the stop line).  This module maps those
samples into ``geometry.TrackPoint`` objects so Blender can consume simulation
state directly instead of recomputing in-camera traffic motion.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import geometry as G

try:
    import envfile as ENV
except ImportError:  # imported as lib.sim.trajectory
    from .. import envfile as ENV


def load_trajectory_index(scenario: dict, base_dir: str) -> Dict[str, List[dict]]:
    """Load ``trajectory.json`` referenced by scenario and index by vehicle id."""
    rel = scenario.get("simulation_artifacts", {}).get("trajectory")
    if not rel:
        return {}
    path = os.path.join(base_dir, rel)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        payload = json.load(f)
    index: Dict[str, List[dict]] = {}
    for sample in payload.get("samples", []):
        vid = sample.get("vehicle_id")
        if not vid:
            continue
        index.setdefault(vid, []).append(sample)
    for samples in index.values():
        samples.sort(key=lambda s: int(s.get("frame", 0)))
    return index


def _interpolate_world_pos(s: float,
                           appear_anchor: Tuple[float, float],
                           stop_pos: Tuple[float, float],
                           approach_len: float) -> Tuple[float, float]:
    """Map lane-longitudinal ``s`` to the approach centerline world point."""
    ax, ay = appear_anchor
    sx, sy = stop_pos
    u = max(0.0, min(float(s), float(approach_len))) / float(approach_len)
    return ax + (sx - ax) * u, ay + (sy - ay) * u


def enrich_trajectory_samples(scenario: dict, sim_meta: dict, root: str,
                              road_meta: Optional[dict] = None) -> None:
    """Enrich logical trajectory samples with geometric trace data in place.

    The recorder remains geometry-free. This bridge loads env/road data and
    derives world-space sample fields plus a per-vehicle side table for
    constants that do not vary by frame.
    """
    samples = sim_meta.get("trajectory_samples", [])
    sim_meta["trajectory_schema"] = "trajectory.v2"
    if not samples:
        sim_meta["trajectory_vehicles"] = {}
        return

    if road_meta is None:
        road_path = os.path.join(root, "assets", "road.json")
        road_meta = {}
        if os.path.exists(road_path):
            with open(road_path) as f:
                road_meta = json.load(f)

    envs = {tag: ENV.load_env(tag, root) for tag in G.camera_names()}
    vehicles_by_id = {v["id"]: v for v in scenario.get("vehicles", [])}
    vehicle_meta: Dict[str, dict] = {}

    for sample in samples:
        vid = sample.get("vehicle_id")
        veh = vehicles_by_id.get(vid)
        if not veh:
            continue
        approach = G.Direction(veh["approach"])
        lane = int(veh["lane"])
        turn = G.Turn(veh["turn"])
        exit_dir, exit_lane = G.exit_lane_for_movement(approach, lane, turn)
        in_cam = f"in_{approach.value}"
        out_cam = f"out_{exit_dir.value}"
        in_anchor, in_rot_z = ENV.lane_default_anchor(envs[in_cam], lane)
        out_anchor, out_rot_z = ENV.lane_default_anchor(envs[out_cam], exit_lane)
        stop_pos = G.lane_entry_box_edge(approach, lane)
        s = float(sample.get("s", 0.0))
        approach_len = float((road_meta or {}).get("approach_length") or
                             max(1e-6, ((stop_pos[0] - in_anchor[0]) ** 2 +
                                        (stop_pos[1] - in_anchor[1]) ** 2) ** 0.5))
        wx, wy = _interpolate_world_pos(
            s, in_anchor[:2], stop_pos, approach_len)
        sample["world_x"] = round(wx, 3)
        sample["world_y"] = round(wy, 3)
        sample["world_z"] = 0.0
        sample["velocity_x"] = round(sample.get("speed", 0.0) * G.approach_forward(approach)[0], 6)
        sample["velocity_y"] = round(sample.get("speed", 0.0) * G.approach_forward(approach)[1], 6)

        if vid not in vehicle_meta:
            vehicle_meta[vid] = {
                "vehicle_id": vid,
                "approach": approach.value,
                "lane": lane,
                "turn": turn.value,
                "camera_id": in_cam,
                "spawn_position": {
                    "x": round(in_anchor[0], 3),
                    "y": round(in_anchor[1], 3),
                    "z": round(in_anchor[2], 3),
                },
                "exit_position": {
                    "x": round(out_anchor[0], 3),
                    "y": round(out_anchor[1], 3),
                    "z": round(out_anchor[2], 3),
                },
                "heading": round(in_rot_z, 6),
                "route_polyline": [
                    [round(in_anchor[0], 3), round(in_anchor[1], 3), round(in_anchor[2], 3)],
                    [round(stop_pos[0], 3), round(stop_pos[1], 3), 0.0],
                    [round(out_anchor[0], 3), round(out_anchor[1], 3), round(out_anchor[2], 3)],
                ],
            }

    sim_meta["trajectory_vehicles"] = vehicle_meta


def samples_to_track(samples: Iterable[dict],
                     approach: G.Direction,
                     lane: int,
                     appear_anchor: Tuple[float, float],
                     road_meta: Optional[dict] = None) -> List[G.TrackPoint]:
    """Convert lane-longitudinal samples into world-space TrackPoints.

    ``s=0`` is the env-file spawn anchor and ``s=approach_len`` is the entry
    box edge for the lane.  The approach track consumes APPROACH/QUEUED samples
    plus the single release-tick IN_BOX sample, so trajectory-backed rendering
    reaches the box edge.  EXIT samples are consumed separately for the out
    camera track.
    """
    stop_pos = G.lane_entry_box_edge(approach, lane)
    approach_len = (road_meta or {}).get("approach_length")
    if not approach_len:
        ax, ay = appear_anchor
        sx, sy = stop_pos
        approach_len = max(1e-6, ((sx - ax) ** 2 + (sy - ay) ** 2) ** 0.5)

    out: List[G.TrackPoint] = []
    seen_frames = set()
    for sample in samples:
        frame = int(sample.get("frame", 0))
        release_frame = sample.get("release_frame")
        stage = str(sample.get("stage", ""))
        is_release_sample = (
            stage == "IN_BOX" and release_frame is not None
            and int(release_frame) == frame
        )
        if stage not in {"APPROACH", "QUEUED"} and not is_release_sample:
            continue
        if "world_x" in sample and "world_y" in sample:
            # v2 samples already contain the bridge-computed world point. Keep
            # this path in sync with enrich_trajectory_samples(): both represent
            # the same env-anchor + stop-line interpolation.
            wx = float(sample["world_x"])
            wy = float(sample["world_y"])
        else:
            # v1 fallback computes the same world point lazily from s.
            wx, wy = _interpolate_world_pos(
                float(sample.get("s", 0.0)), appear_anchor, stop_pos,
                float(approach_len))
        if frame in seen_frames:
            continue
        seen_frames.add(frame)
        out.append(G.TrackPoint(
            frame=frame,
            x=wx,
            y=wy,
            visible=True,
            interp="LINEAR",
        ))
    out.sort(key=lambda p: p.frame)
    return out


def _has_release_track_sample(samples: Iterable[dict]) -> bool:
    """Whether samples include the true release-tick IN_BOX stop-line point."""
    for sample in samples:
        frame = int(sample.get("frame", 0))
        release_frame = sample.get("release_frame")
        if (sample.get("stage") == "IN_BOX" and release_frame is not None
                and int(release_frame) == frame):
            return True
    return False


def exit_samples_to_track(samples: Iterable[dict]) -> List[G.TrackPoint]:
    """Convert v3 EXIT samples with world coordinates into TrackPoints."""
    out: List[G.TrackPoint] = []
    seen_frames = set()
    for sample in samples:
        if sample.get("stage") != "EXIT":
            continue
        if "world_x" not in sample or "world_y" not in sample:
            continue
        frame = int(sample.get("frame", 0))
        if frame in seen_frames:
            continue
        seen_frames.add(frame)
        out.append(G.TrackPoint(
            frame=frame,
            x=float(sample["world_x"]),
            y=float(sample["world_y"]),
            visible=True,
            interp="LINEAR",
        ))
    out.sort(key=lambda p: p.frame)
    return out


def apply_samples_to_motion(motion: G.VehicleMotion,
                            samples: Iterable[dict],
                            appear_anchor: Tuple[float, float],
                            road_meta: Optional[dict] = None) -> G.VehicleMotion:
    """Replace visible motion tracks with recorded simulation samples if usable."""
    sample_list = list(samples)
    track = samples_to_track(
        sample_list,
        motion.approach,
        motion.lane,
        appear_anchor,
        road_meta=road_meta,
    )
    if len(track) < 2:
        # Trajectory data too sparse to define an approach path; return motion unchanged.
        return motion
    motion.track_in = track
    if _has_release_track_sample(sample_list):
        # Only update timing when the recorded trace contains the true release
        # sample. Warm-up/clipped runs may only have a partial approach trace;
        # using its last sample as disappear_frame corrupts delta_t.
        motion.appear_frame = track[0].frame
        motion.appear_pos = (track[0].x, track[0].y)
        motion.disappear_frame = track[-1].frame
        motion.disappear_pos = (track[-1].x, track[-1].y)
    exit_track = exit_samples_to_track(sample_list)
    if len(exit_track) >= 2:
        motion.track_out = exit_track
        motion.reappear_frame = exit_track[0].frame
        motion.leave_frame = exit_track[-1].frame
        motion.reappear_pos = (exit_track[0].x, exit_track[0].y)
        motion.leave_pos = (exit_track[-1].x, exit_track[-1].y)
    return motion


def _exit_sample_from_metadata(veh_meta: dict, frame: dict,
                               s_exit: float) -> dict:
    """Build one trajectory.v3 EXIT sample from metadata's canonical pose."""
    pose = frame.get("pose") or {}
    speed = float(veh_meta.get("speed_ms", 0.0))
    exit_dir = G.Direction(veh_meta["exit_direction"])
    vx, vy = exit_dir.vec
    return {
        "frame": int(frame["frame"]),
        "vehicle_id": veh_meta["id"],
        "approach": veh_meta["approach"],
        "lane": int(veh_meta["lane"]),
        "turn": veh_meta["turn"],
        "stage": "EXIT",
        "s": round(s_exit, 3),
        "speed": speed,
        "accel": 0.0,
        "leader_id": None,
        "gap": 0.0,
        "release_frame": veh_meta.get("release_frame"),
        "world_x": round(float(pose["x"]), 3),
        "world_y": round(float(pose["y"]), 3),
        "world_z": round(float(pose.get("z", 0.0)), 3),
        "velocity_x": round(speed * vx, 6),
        "velocity_y": round(speed * vy, 6),
    }


def complete_trajectory_with_metadata(scenario: dict, base_dir: str,
                                      metadata: dict) -> None:
    """Finalize trajectory.json with v3 EXIT samples from metadata poses.

    ``scenario_gen`` writes the geometry-enriched approach trace before any
    rendering/metadata pass exists.  The metadata pass already computes the
    deterministic out-camera motion for every vehicle, so this helper upgrades
    the trajectory artifact in place without duplicating env/road loading in the
    generator or physics recorder.
    """
    rel = scenario.get("simulation_artifacts", {}).get("trajectory")
    if not rel:
        return
    path = os.path.join(base_dir, rel)
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            payload = json.load(f)
        changed = _complete_trajectory_payload(payload, metadata)
        if changed:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
    except Exception as exc:
        print(f"[WARN] trajectory v3 completion skipped: {exc}")


def _complete_trajectory_payload(payload: dict, metadata: dict) -> bool:
    """Mutate a trajectory payload with EXIT samples. Return True if upgraded."""
    samples = payload.get("samples", [])
    by_vid: Dict[str, List[dict]] = {}
    for sample in samples:
        vid = sample.get("vehicle_id")
        if vid:
            by_vid.setdefault(vid, []).append(sample)

    vehicles_table = payload.setdefault("vehicles", {})
    meta_vids = {v.get("id") for v in metadata.get("vehicles", [])}
    completed_samples: List[dict] = [
        s for vid, lst in by_vid.items() if vid not in meta_vids for s in lst
    ]
    added_exit = False
    for veh_meta in metadata.get("vehicles", []):
        vid = veh_meta["id"]
        existing = [s for s in by_vid.get(vid, []) if s.get("stage") != "EXIT"]
        out_frames = [f for f in veh_meta.get("frames", [])
                      if f.get("camera") == veh_meta.get("out_camera")]
        out_frames.sort(key=lambda f: int(f.get("frame", 0)))
        exit_samples: List[dict] = []
        if out_frames:
            first_pose = out_frames[0].get("pose") or {}
            fx = float(first_pose.get("x", 0.0))
            fy = float(first_pose.get("y", 0.0))
            for frame in out_frames:
                pose = frame.get("pose") or {}
                dx = float(pose.get("x", 0.0)) - fx
                dy = float(pose.get("y", 0.0)) - fy
                # ponytail: EXIT s is chord distance from reappear; switch to
                # arc length only if a downstream consumer needs it.
                exit_samples.append(_exit_sample_from_metadata(
                    veh_meta, frame, (dx * dx + dy * dy) ** 0.5))
            added_exit = bool(exit_samples) or added_exit

            trace_vehicle = vehicles_table.setdefault(vid, {"vehicle_id": vid})
            trace_vehicle.update({
                "depart_frame": veh_meta.get("depart_frame"),
                "stop_frame": veh_meta.get("stop_frame"),
                "release_frame": veh_meta.get("release_frame"),
                "appear_frame": veh_meta.get("appear_frame"),
                "disappear_frame": veh_meta.get("disappear_frame"),
                "reappear_frame": veh_meta.get("reappear_frame"),
                "leave_frame": veh_meta.get("leave_frame"),
                "exit_direction": veh_meta.get("exit_direction"),
                "exit_lane": veh_meta.get("exit_lane"),
            })
            reappear_pose = out_frames[0].get("pose") or {}
            leave_pose = out_frames[-1].get("pose") or {}
            trace_vehicle["reappear_position"] = {
                "x": round(float(reappear_pose.get("x", 0.0)), 3),
                "y": round(float(reappear_pose.get("y", 0.0)), 3),
                "z": round(float(reappear_pose.get("z", 0.0)), 3),
            }
            trace_vehicle["leave_position"] = {
                "x": round(float(leave_pose.get("x", 0.0)), 3),
                "y": round(float(leave_pose.get("y", 0.0)), 3),
                "z": round(float(leave_pose.get("z", 0.0)), 3),
            }
            trace_vehicle["exit_position"] = dict(trace_vehicle["leave_position"])
            route = trace_vehicle.get("route_polyline") or []
            if len(route) >= 3:
                route[-1] = [trace_vehicle["leave_position"]["x"],
                             trace_vehicle["leave_position"]["y"],
                             trace_vehicle["leave_position"]["z"]]
            trace_vehicle["route_polyline"] = route

        completed_samples.extend(existing + exit_samples)

    if not added_exit:
        return False
    payload["schema"] = "trajectory.v3"
    payload["description"] = (
        "Per-frame ground-truth trace before Blender visualization. v3 keeps "
        "legacy lane-longitudinal approach samples, world-space approach poses, "
        "the release-frame IN_BOX sample, and full visible EXIT samples. The "
        "intersection box itself remains a black-box delay between release and "
        "reappear frames. In v3, route_polyline[-1] and exit_position represent "
        "the actual leave pose; reappear_position preserves the out-camera anchor."
    )
    payload["samples"] = sorted(
        completed_samples,
        key=lambda s: (str(s.get("vehicle_id", "")), int(s.get("frame", 0))),
    )
    return True


__all__ = [
    "load_trajectory_index",
    "complete_trajectory_with_metadata",
    "exit_samples_to_track",
    "enrich_trajectory_samples",
    "_complete_trajectory_payload",
    "_exit_sample_from_metadata",
    "samples_to_track",
    "apply_samples_to_motion",
]
