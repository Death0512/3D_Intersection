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
    reaches the box edge.  Later in-box/exit motion still uses the legacy
    black-box exit track.
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


def apply_samples_to_motion(motion: G.VehicleMotion,
                            samples: Iterable[dict],
                            appear_anchor: Tuple[float, float],
                            road_meta: Optional[dict] = None) -> G.VehicleMotion:
    """Replace ``motion.track_in`` with recorded simulation samples if usable."""
    track = samples_to_track(
        samples,
        motion.approach,
        motion.lane,
        appear_anchor,
        road_meta=road_meta,
    )
    if len(track) < 2:
        # Fallback to legacy kinematics if trajectory data is too sparse to
        # define an approach path. Callers still clamp emitted/rendered frames.
        return motion
    motion.track_in = track
    motion.appear_frame = track[0].frame
    motion.disappear_frame = track[-1].frame
    motion.appear_pos = (track[0].x, track[0].y)
    motion.disappear_pos = (track[-1].x, track[-1].y)
    return motion


__all__ = [
    "load_trajectory_index",
    "enrich_trajectory_samples",
    "samples_to_track",
    "apply_samples_to_motion",
]
