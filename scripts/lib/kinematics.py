"""Phase 1 — Kinematics helpers (pure Python).

Wraps geometry.compute_motion and adds headway / scheduling utilities used by
the scenario generator (Phase 2) and scene builder (Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

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
                fps=G.FPS,
                appear_anchor=None, reappear_anchor=None,
                road_meta=None) -> G.VehicleMotion:
    return G.compute_motion(vehicle_id, approach, lane, turn, speed_ms,
                            depart_frame, approach_visible_length,
                            exit_visible_length, fps,
                            appear_anchor=appear_anchor,
                            reappear_anchor=reappear_anchor,
                            road_meta=road_meta)


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


def catchup_safe(lead_depart_frame: int, lead_speed: float, lead_length: float,
                 follow_depart_frame: int, follow_speed: float,
                 approach_visible_length: float = 40.0,
                 safety_gap: float = 2.0, fps: int = G.FPS) -> bool:
    """Check that a faster follower cannot close to within `safety_gap` of the
    leader before the leader reaches the Black Box (disappears).

    Both vehicles share the same (approach, lane) visible segment of length
    `approach_visible_length`. The leader enters at `lead_depart_frame` and
    reaches the box after `lead_time_to_box = L / lead_speed` seconds. The
    follower enters at `follow_depart_frame` (>= lead_depart_frame).

    If follow_speed <= lead_speed the gap can only grow -> always safe.
    Otherwise the follower closes at relative speed (follow_speed - lead_speed).
    Starting gap at the follower's depart = lead_speed * head_start_seconds
    (distance the leader covered before the follower appears). We require that
    the gap never drops below (lead_length + safety_gap) during the interval the
    leader is still on the approach. If it would, the follower must be delayed.
    """
    if follow_speed <= lead_speed:
        return True
    head_start_s = max(0.0, (follow_depart_frame - lead_depart_frame) / fps)
    # distance the leader has already covered when the follower appears
    lead_ahead = lead_speed * head_start_s
    # remaining time before the leader reaches the box from the follower's POV
    # (leader still has L - lead_ahead metres to travel)
    lead_remaining_s = max(0.0, (approach_visible_length - lead_ahead) / lead_speed)
    # gap shrinks at (follow_speed - lead_speed); smallest gap at the moment the
    # leader disappears (or the follower catches up, whichever first)
    rel_speed = follow_speed - lead_speed
    min_gap = lead_ahead - rel_speed * lead_remaining_s
    return min_gap >= (lead_length + safety_gap)


def min_follow_depart_frame(lead_depart_frame: int, lead_speed: float,
                            lead_length: float, follow_speed: float,
                            approach_visible_length: float = 40.0,
                            safety_gap: float = 2.0, fps: int = G.FPS) -> int:
    """Earliest depart_frame for a faster follower so it never catches the
    leader before the leader enters the Black Box.

    Solve for the head_start that leaves exactly (lead_length + safety_gap) at
    the leader's box-entry moment, then convert to frames and add to the lead
    depart frame. For follow_speed <= lead_speed this is just the lead depart
    frame (no extra delay needed)."""
    if follow_speed <= lead_speed:
        return lead_depart_frame
    # Let h = head_start (s). gap_at_box = lead_speed*h - (follow-lead)*(L/lead - h)
    #   = lead_speed*h - (follow-lead)*L/lead + (follow-lead)*h
    #   = follow_speed*h - (follow-lead)*L/lead
    # Require >= lead_length + safety_gap:
    #   follow_speed*h >= (lead_length+safety_gap) + (follow-lead)*L/lead
    #   h >= [ (lead_length+safety_gap) + (follow-lead)*L/lead ] / follow_speed
    L = approach_visible_length
    needed_h = ((lead_length + safety_gap)
                + (follow_speed - lead_speed) * L / lead_speed) / follow_speed
    return lead_depart_frame + int(math.ceil(needed_h * fps))


def speed_kmh_to_ms(kmh: float) -> float:
    return kmh / 3.6
