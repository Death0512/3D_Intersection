"""Phase 1 — Intersection geometry (pure Python, no bpy).

Defines the world-space layout of the 4-way intersection consistent with
KNOWLEDGE_EN.md and the locked parameters:

  * Intersection box: 30 x 30 m  (centred at world origin)
  * Each axis: 4 lanes @ 3.5 m (14 m) per direction + 2 m median
  * Rendered road arm width: 14 m (one travel direction)
  * Lane centre lines (arm-local, entry direction +X = right):
        [-5.25, -1.75, 1.75, 5.25]
  * 4 approaches: N, E, S, W
  * 12 movements: approach x {left, straight, right}

Coordinate conventions:
  * World Z = up, ground at Z = 0.
  * World Y axis = the N-S axis (N = +Y), World X axis = the E-W axis (E = +X).
  * Approach direction = the direction vehicles travel TOWARD the intersection.
    e.g. "N approach" = vehicles moving northbound (+Y) toward the box.
  * Entry lanes (vehicles approaching) and exit lanes (vehicles leaving) are
    mirror images across the median for a given axis.

This module is unit-testable without Blender.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


# ---- locked constants -------------------------------------------------------
BOX_SIZE = 30.0          # intersection box side (m)
LANE_WIDTH = 3.5
NUM_LANES = 4            # per direction
MEDIAN = 2.0
ARM_WIDTH = NUM_LANES * LANE_WIDTH          # 14 m (one direction)
AXIS_WIDTH = 2 * ARM_WIDTH + MEDIAN         # 30 m (full carriageway)

# lane centre lines relative to the centre of one 4-lane direction (arm-local)
LANE_CENTERLINES = [-5.25, -1.75, 1.75, 5.25]

# Carriageway lateral offset from the axis centerline to a carriageway centre.
# Each axis has TWO carriageways (entry + exit) separated by the 2 m median.
# Carriageway centre = axis_centre ± CARRIAGEWAY_OFFSET.
CARRIAGEWAY_OFFSET = ARM_WIDTH / 2 + MEDIAN / 2   # = 8.0 m

FPS = 30


class Direction(str, Enum):
    N = "N"; E = "E"; S = "S"; W = "W"

    @property
    def opposite(self) -> "Direction":
        return {Direction.N: Direction.S, Direction.S: Direction.N,
                Direction.E: Direction.W, Direction.W: Direction.E}[self]

    @property
    def vec(self) -> Tuple[float, float]:
        return {Direction.N: (0, 1), Direction.S: (0, -1),
                Direction.E: (1, 0), Direction.W: (-1, 0)}[self]

    @property
    def angle_rad(self) -> float:
        # heading angle measured from +Y axis, clockwise (so N=0, E=pi/2, S=pi, W=3pi/2)
        return {Direction.N: 0.0, Direction.E: math.pi / 2,
                Direction.S: math.pi, Direction.W: 3 * math.pi / 2}[self]


class Turn(str, Enum):
    LEFT = "left"; STRAIGHT = "straight"; RIGHT = "right"


# ---------------------------------------------------------------------------
# Lane geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lane:
    approach: Direction       # which approach this lane belongs to
    index: int                # 0..3 (lane number within the approach)
    center_offset: float      # signed offset perpendicular to travel dir (m)
    role: str = "entry"       # "entry" (approaching) or "exit" (leaving)


def approach_forward(approach: Direction) -> Tuple[float, float]:
    """Unit forward vector of vehicles on the given approach (toward box)."""
    return approach.vec


def approach_right(approach: Direction) -> Tuple[float, float]:
    """Unit vector pointing to the vehicle's right side (perpendicular to
    forward, in ground plane). For right-hand traffic, lanes sit on +right."""
    fx, fy = approach_forward(approach)
    # right = forward rotated -90deg in Z-up
    return (fy, -fx)


def lane_entry_box_edge(approach: Direction, lane_index: int) -> Tuple[float, float]:
    """World (x, y) of the centre of an ENTRY lane at the box boundary where
    the vehicle disappears (the near edge of the box relative to the approach).

    Each approach has its own self-contained 14 m road arm centred on the
    branch axis.  Lane index selects a lane from LANE_CENTERLINES (no median
    offset — the arm covers the full width on its own).
    """
    fx, fy = approach_forward(approach)
    rx, ry = approach_right(approach)
    off = LANE_CENTERLINES[lane_index]
    cx = rx * off - fx * (BOX_SIZE / 2)
    cy = ry * off - fy * (BOX_SIZE / 2)
    return (cx, cy)


def exit_lane_for_movement(approach: Direction, lane_index: int, turn: Turn) -> Tuple[Direction, int]:
    """Given an entry (approach, lane) and a turn, return
    (outbound_direction, exit_lane_index).

    `outbound_direction` is the HEADING the vehicle travels AFTER exiting the
    box (i.e. the direction it moves on its exit road, away from the box):
      * straight: outbound = approach (keeps going the same way)
      * right:    outbound = approach rotated right (clockwise from above)
      * left:     outbound = approach rotated left  (counter-clockwise)
    """
    right_map = {Direction.N: Direction.E, Direction.E: Direction.S,
                 Direction.S: Direction.W, Direction.W: Direction.N}
    left_map = {Direction.N: Direction.W, Direction.W: Direction.S,
                Direction.S: Direction.E, Direction.E: Direction.N}
    if turn == Turn.STRAIGHT:
        return (approach, lane_index)
    if turn == Turn.RIGHT:
        return (right_map[approach], 0)
    if turn == Turn.LEFT:
        return (left_map[approach], NUM_LANES - 1)


def lane_exit_box_edge(outbound: Direction, lane_index: int) -> Tuple[float, float]:
    """World (x, y) of the centre of an EXIT lane at the box boundary where the
    vehicle reappears (the far edge of the box relative to `outbound`, i.e. the
    edge the vehicle emerges from).

    Each exit road is its own 14 m arm centred on the outbound branch axis.
    Lane index selects from LANE_CENTERLINES (right-hand traffic: lane 0 is
    the leftmost/innermost lane of the outbound arm).
    """
    fx, fy = outbound.vec
    rx, ry = approach_right(outbound)
    off = LANE_CENTERLINES[lane_index]
    cx = rx * off + fx * (BOX_SIZE / 2)
    cy = ry * off + fy * (BOX_SIZE / 2)
    return (cx, cy)


def approach_rotation(approach: Direction) -> float:
    """Rotation (radians, about Z) to apply to a +Y-forward road arm asset so
    that its forward (+Y) aligns with the given approach's forward direction.
    """
    fx, fy = approach_forward(approach)
    # arm forward is +Y = (0,1); we need to rotate (0,1) -> (fx,fy)
    return math.atan2(fx, fy)


# ---------------------------------------------------------------------------
# Stop lines, crosswalks, camera reference Y (in arm-local coords)
# ---------------------------------------------------------------------------

# In the prepped road asset (assets/road.json), the road arm is centred on its
# origin with forward = +Y and the crosswalk/stop-line at +Y = crosswalk_y.
# When the arm is placed in world for a given approach, the stop line sits at
# the box boundary on the approach side.
STOP_LINE_LOCAL_Y = None  # filled from road.json at runtime
CROSSWALK_LOCAL_Y = None


# ---------------------------------------------------------------------------
# Routing table (12 movements)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Movement:
    approach: Direction
    lane: int                # entry lane index 0..3
    turn: Turn
    exit_direction: Direction
    exit_lane: int
    in_camera: str           # e.g. "in_N"
    out_camera: str          # e.g. "out_S"


def build_routing_table() -> List[Movement]:
    table = []
    for approach in Direction:
        for lane in range(NUM_LANES):
            for turn in Turn:
                ex_dir, ex_lane = exit_lane_for_movement(approach, lane, turn)
                table.append(Movement(
                    approach=approach, lane=lane, turn=turn,
                    exit_direction=ex_dir, exit_lane=ex_lane,
                    in_camera=f"in_{approach.value}",
                    out_camera=f"out_{ex_dir.value}",
                ))
    return table


ROUTING_TABLE: List[Movement] = build_routing_table()


def camera_names() -> List[str]:
    """The 8 camera names."""
    names = []
    for d in Direction:
        names.append(f"in_{d.value}")
        names.append(f"out_{d.value}")
    return names


# ---------------------------------------------------------------------------
# Path length through the intersection (for Delta t)
# ---------------------------------------------------------------------------

def intersection_path_length(turn: Turn, lane_index: int) -> float:
    """Distance travelled inside the 30x30 box for a given turn.

      straight: BOX_SIZE (30 m)
      right:    quarter-circle of radius = lane offset from corner
      left:     quarter-circle of larger radius

    For simplicity and deterministic timing we use a fixed geometric model:
      straight = 30
      right    = (pi/2) * R_right    with R_right = 6.0  (curb radius)
      left     = (pi/2) * R_left     with R_left  = 12.0 (wider arc)
    """
    if turn == Turn.STRAIGHT:
        return BOX_SIZE
    if turn == Turn.RIGHT:
        return (math.pi / 2) * 6.0
    if turn == Turn.LEFT:
        return (math.pi / 2) * 12.0


def delta_t_seconds(turn: Turn, speed_ms: float) -> float:
    """Blind-zone delay time (s) = path length / speed."""
    if speed_ms <= 0:
        raise ValueError("speed must be > 0")
    return intersection_path_length(turn, 0) / speed_ms


def delta_t_frames(turn: Turn, speed_ms: float, fps: int = FPS) -> int:
    """Integer frame count of invisibility (rounded)."""
    return int(round(delta_t_seconds(turn, speed_ms) * fps))


# ---------------------------------------------------------------------------
# Helpers for scenario / motion
# ---------------------------------------------------------------------------

@dataclass
class VehicleMotion:
    """Computed motion plan for one vehicle on one approach camera view."""
    vehicle_id: str
    approach: Direction
    lane: int
    turn: Turn
    speed_ms: float
    depart_frame: int
    fps: int
    # filled by compute()
    appear_frame: int = 0
    disappear_frame: int = 0
    reappear_frame: int = 0
    leave_frame: int = 0
    appear_pos: Tuple[float, float] = (0.0, 0.0)
    disappear_pos: Tuple[float, float] = (0.0, 0.0)
    reappear_pos: Tuple[float, float] = (0.0, 0.0)
    leave_pos: Tuple[float, float] = (0.0, 0.0)
    exit_direction: Direction = Direction.N
    exit_lane: int = 0


def compute_motion(vehicle_id: str, approach: Direction, lane: int, turn: Turn,
                   speed_ms: float, depart_frame: int,
                   approach_visible_length: float = 40.0,
                   exit_visible_length: float = 40.0,
                   fps: int = FPS) -> VehicleMotion:
    """Compute the frame numbers and world positions for a vehicle's full
    In -> Black-Box -> Out trajectory.

    Conventions:
      * `approach` is the vehicle's heading TOWARD the box on entry.
      * The vehicle disappears at the box's near edge (entry side) along
        `approach` forward, at the entry lane centre.
      * After the Black-Box delay it reappears at the box's far edge along the
        OUTBOUND heading (the heading it has after the turn), at the exit lane
        centre, and continues OUTWARD along that outbound heading.

    approach_visible_length: how far back along the approach the In-camera can
        see (vehicle appears this far behind the stop line).
    exit_visible_length: how far along the exit the Out-camera can see (vehicle
        leaves when it passes this far beyond the crosswalk).
    """
    dt_approach = approach_visible_length / speed_ms
    dt_exit = exit_visible_length / speed_ms
    dt_box_frames = delta_t_frames(turn, speed_ms, fps)

    appear_frame = depart_frame
    disappear_frame = depart_frame + int(round(dt_approach * fps))
    reappear_frame = disappear_frame + dt_box_frames
    leave_frame = reappear_frame + int(round(dt_exit * fps))

    # ENTRY segment: along `approach` forward, on the entry lane (right-hand).
    fx, fy = approach_forward(approach)
    disappear_pos = lane_entry_box_edge(approach, lane)
    appear_pos = (disappear_pos[0] - fx * approach_visible_length,
                  disappear_pos[1] - fy * approach_visible_length)

    # EXIT segment: along OUTBOUND heading, on the exit lane (right-hand).
    outbound, ex_lane = exit_lane_for_movement(approach, lane, turn)
    ofx, ofy = outbound.vec
    reappear_pos = lane_exit_box_edge(outbound, ex_lane)
    leave_pos = (reappear_pos[0] + ofx * exit_visible_length,
                 reappear_pos[1] + ofy * exit_visible_length)

    return VehicleMotion(
        vehicle_id=vehicle_id, approach=approach, lane=lane, turn=turn,
        speed_ms=speed_ms, depart_frame=depart_frame, fps=fps,
        appear_frame=appear_frame, disappear_frame=disappear_frame,
        reappear_frame=reappear_frame, leave_frame=leave_frame,
        appear_pos=appear_pos, disappear_pos=disappear_pos,
        reappear_pos=reappear_pos, leave_pos=leave_pos,
        exit_direction=outbound, exit_lane=ex_lane,
    )


# ---------------------------------------------------------------------------
# Summary / sanity
# ---------------------------------------------------------------------------

def summary() -> Dict:
    return {
        "box_size": BOX_SIZE,
        "arm_width": ARM_WIDTH,
        "axis_width": AXIS_WIDTH,
        "lane_width": LANE_WIDTH,
        "num_lanes": NUM_LANES,
        "median": MEDIAN,
        "lane_centerlines": LANE_CENTERLINES,
        "fps": FPS,
        "num_movements": len(ROUTING_TABLE),
        "cameras": camera_names(),
    }
