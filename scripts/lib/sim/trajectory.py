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


def samples_to_track(samples: Iterable[dict],
                     approach: G.Direction,
                     lane: int,
                     appear_anchor: Tuple[float, float],
                     road_meta: Optional[dict] = None,
                     frame_end: Optional[int] = None) -> List[G.TrackPoint]:
    """Convert lane-longitudinal samples into world-space TrackPoints.

    ``s=0`` is the env-file spawn anchor and ``s=approach_len`` is the entry
    box edge for the lane.  The approach track consumes APPROACH/QUEUED samples
    plus the single release-tick IN_BOX sample, so trajectory-backed rendering
    reaches the box edge.  Later in-box/exit motion still uses the legacy
    black-box exit track.
    """
    stop_pos = G.lane_entry_box_edge(approach, lane)
    ax, ay = appear_anchor
    sx, sy = stop_pos
    dx, dy = sx - ax, sy - ay
    approach_len = (road_meta or {}).get("approach_length")
    if not approach_len:
        approach_len = max(1e-6, (dx * dx + dy * dy) ** 0.5)

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
        if frame_end is not None and frame > frame_end + 1:
            continue
        if frame in seen_frames:
            continue
        seen_frames.add(frame)
        s = max(0.0, min(float(sample.get("s", 0.0)), float(approach_len)))
        u = s / float(approach_len)
        out.append(G.TrackPoint(
            frame=frame,
            x=ax + dx * u,
            y=ay + dy * u,
            visible=True,
            interp="LINEAR",
        ))
    out.sort(key=lambda p: p.frame)
    return out


def apply_samples_to_motion(motion: G.VehicleMotion,
                            samples: Iterable[dict],
                            appear_anchor: Tuple[float, float],
                            road_meta: Optional[dict] = None,
                            frame_end: Optional[int] = None) -> G.VehicleMotion:
    """Replace ``motion.track_in`` with recorded simulation samples if usable."""
    track = samples_to_track(
        samples,
        motion.approach,
        motion.lane,
        appear_anchor,
        road_meta=road_meta,
        frame_end=frame_end,
    )
    if len(track) < 2:
        return motion
    motion.track_in = track
    motion.appear_frame = track[0].frame
    motion.disappear_frame = track[-1].frame
    motion.appear_pos = (track[0].x, track[0].y)
    motion.disappear_pos = (track[-1].x, track[-1].y)
    return motion


__all__ = [
    "load_trajectory_index",
    "samples_to_track",
    "apply_samples_to_motion",
]
