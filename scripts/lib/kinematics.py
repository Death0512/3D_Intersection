"""Phase 1 — Kinematics helpers (pure Python).

Wraps geometry.compute_motion and adds headway / scheduling utilities used by
the scenario generator (Phase 2) and scene builder (Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from . import geometry as G
except ImportError:  # imported as a top-level module (scripts/lib on sys.path)
    import geometry as G


def frames_to_travel(distance_m: float, speed_ms: float, fps: int = G.FPS) -> int:
    if speed_ms <= 0:
        raise ValueError("speed must be > 0")
    return int(round((distance_m / speed_ms) * fps))


def min_headway_frames(vehicle_length: float, speed_ms: float,
                       safety_gap: float = 2.0, fps: int = G.FPS) -> int:
    """Minimum frame gap between two vehicles in the SAME lane so they never
    overlap: gap = (length + safety_gap) / speed, in frames."""
    return frames_to_travel(vehicle_length + safety_gap, speed_ms, fps)


def plan_motion(vehicle_id: str, approach, lane, turn, speed_ms, depart_frame,
                approach_visible_length=40.0, exit_visible_length=40.0,
                fps=G.FPS) -> G.VehicleMotion:
    return G.compute_motion(vehicle_id, approach, lane, turn, speed_ms,
                            depart_frame, approach_visible_length,
                            exit_visible_length, fps)


def conflict_free(departures: List[Tuple[int, float, float]],
                  fps: int = G.FPS, safety_gap: float = 2.0) -> bool:
    """Given a list of (depart_frame, vehicle_length, speed) for one lane,
    check that no two vehicles overlap (sorted by depart_frame)."""
    ds = sorted(departures, key=lambda x: x[0])
    for (f0, l0, s0), (f1, l1, s1) in zip(ds, ds[1:]):
        gap_frames = f1 - f0
        needed = min_headway_frames(max(l0, l1), max(s0, s1), safety_gap, fps)
        if gap_frames < needed:
            return False
    return True


def speed_kmh_to_ms(kmh: float) -> float:
    return kmh / 3.6
