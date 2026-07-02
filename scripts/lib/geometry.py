"""Phase 1 — Intersection geometry (pure Python, no bpy).

Defines the world-space layout for a single-camera shot of the 4-way
intersection.  Each of the 8 videos is an independent .blend; lanes are
centred on the road axis (x = 0) with no shared-world carriageway offset.

Locked parameters:
  * Intersection box: 30 x 30 m  (centred at world origin)
  * Each axis: 4 lanes @ 3.5 m (14 m) per direction
  * Lane centre lines (relative to road axis, x = 0):
        [-5.25, -1.75, +1.75, +5.25]
  * 4 approaches: N, E, S, W
  * 12 movements: approach x {left, straight, right}

Per-shot coordinate frame:
  * Each .blend is built around ONE road arm centred at (0,0,0).
  * Road forward = +Y (car drives +Y toward the box).
  * Lanes at x = LANE_CENTERLINES[k] (no carriageway offset).
  * Camera at (0, ±road_length/2, CAM_HEIGHT) looking along the road.

Lane-index convention (LOCKED):
  * index 0 = MEDIAN-side (innermost); index 3 = CURB-side (outermost).
  * Turning rules (right-hand driving):
        LEFT  -> exit lane 0          (median-side)
        RIGHT -> exit lane NUM_LANES-1 (curb-side)
        STRAIGHT -> keep entry lane index

This module is unit-testable without Blender.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---- locked constants -------------------------------------------------------
BOX_SIZE = 30.0          # intersection box side (m)
LANE_WIDTH = 3.5
NUM_LANES = 4            # per direction
MEDIAN = 2.0
ARM_WIDTH = NUM_LANES * LANE_WIDTH          # 14 m (one direction)
AXIS_WIDTH = 2 * ARM_WIDTH + MEDIAN         # 30 m (full carriageway)

# lane centre lines relative to the centre of one 4-lane direction (arm-local)
LANE_CENTERLINES = [-5.25, -1.75, 1.75, 5.25]

FPS = 30


# Camera parameters (single source of truth — shared by build_scene.place_camera
# and render.compute_metadata). Keeping them here means the Blender camera and
# the per-frame pose ground truth in metadata.json always agree.
CAM_HEIGHT = 7.0       # camera elevation (m)
CAM_BACK_DIST = 100.0   # how far past the box edge the out-camera looks (m)
LENS_MM = 60.0         # telephoto focal length
SENSOR_MM = 36.0       # full-frame sensor width (set on the Blender camera too)
RES_X = 1920
RES_Y = 1080


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


def lane_lateral_offset(direction: Direction, lane_index: int) -> Tuple[float, float]:
    """Lateral (x, y) offset from the road axis to the centre of a specific
    lane.  In the per-shot frame the road is centred on the axis (x = 0), so
    the offset is purely the arm-local lane centerline rotated into the
    approach's right-hand perpendicular direction."""
    rx, ry = approach_right(direction)
    off = LANE_CENTERLINES[lane_index]
    return (rx * off, ry * off)


def lane_entry_box_edge(approach: Direction, lane_index: int) -> Tuple[float, float]:
    """World (x, y) of the centre of an ENTRY lane at the box boundary where
    the vehicle disappears (the near edge of the box relative to the approach).
    Lanes are centred on the road axis (no carriageway offset).
    """
    fx, fy = approach_forward(approach)
    ox, oy = lane_lateral_offset(approach, lane_index)
    cx = ox - fx * (BOX_SIZE / 2)
    cy = oy - fy * (BOX_SIZE / 2)
    return (cx, cy)


def exit_lane_for_movement(approach: Direction, lane_index: int, turn: Turn) -> Tuple[Direction, int]:
    """Given an entry (approach, lane) and a turn, return
    (outbound_direction, exit_lane_index).

    `outbound_direction` is the HEADING the vehicle travels AFTER exiting the
    box (i.e. the direction it moves on its exit road, away from the box):
      * straight: outbound = approach (keeps going the same way)
      * right:    outbound = approach rotated right (clockwise from above)
      * left:     outbound = approach rotated left  (counter-clockwise)

    Exit lane index follows right-hand driving with the locked convention
    index 0 = median-side, NUM_LANES-1 = curb-side:
      * straight: keep the entry lane index
      * left:     exit lane 0          (median-side)
      * right:    exit lane NUM_LANES-1 (curb-side)
    """
    right_map = {Direction.N: Direction.E, Direction.E: Direction.S,
                 Direction.S: Direction.W, Direction.W: Direction.N}
    left_map = {Direction.N: Direction.W, Direction.W: Direction.S,
                Direction.S: Direction.E, Direction.E: Direction.N}
    if turn == Turn.STRAIGHT:
        return (approach, lane_index)
    if turn == Turn.RIGHT:
        return (right_map[approach], NUM_LANES - 1)
    if turn == Turn.LEFT:
        return (left_map[approach], 0)


def lane_exit_box_edge(outbound: Direction, lane_index: int) -> Tuple[float, float]:
    """World (x, y) of the centre of an EXIT lane at the box boundary where the
    vehicle reappears (the far edge of the box relative to `outbound`).
    Lanes are centred on the road axis (no carriageway offset).
    """
    fx, fy = outbound.vec
    ox, oy = lane_lateral_offset(outbound, lane_index)
    cx = ox + fx * (BOX_SIZE / 2)
    cy = oy + fy * (BOX_SIZE / 2)
    return (cx, cy)


def approach_rotation(approach: Direction) -> float:
    """Rotation (radians, about Z) to apply to a +Y-forward road arm asset so
    that its forward (+Y) aligns with the given approach's forward direction.
    """
    fx, fy = approach_forward(approach)
    # arm forward is +Y = (0,1); we need to rotate (0,1) -> (fx,fy)
    return math.atan2(fx, fy)


def road_arm_transform(approach: Direction, road_meta: dict,
                       is_entry: bool) -> Tuple[Tuple[float, float, float], float]:
    """World (location xyz, rotation_z) for the road arm empty in the per-shot
    frame.  Mirrors build_scene.place_road exactly — kept here as the single
    source of truth so the env-file generator and the Blender placement agree.

    Entry (is_entry=True): arm +Y (crosswalk) at box near-edge, body outward.
    Exit (is_entry=False): arm −Y (back) at box far edge, body outward.
    """
    fx, fy = approach_forward(approach)
    crosswalk_y = road_meta.get("crosswalk_y", 0.0)
    approach_length = road_meta.get("approach_length", crosswalk_y)
    arm_back = approach_length - crosswalk_y
    half = BOX_SIZE / 2
    rot = approach_rotation(approach)
    if is_entry:
        edge = (-fx * half, -fy * half)
        ox, oy = edge[0] - fx * crosswalk_y, edge[1] - fy * crosswalk_y
    else:
        edge = (fx * half, fy * half)
        ox, oy = edge[0] + fx * arm_back, edge[1] + fy * arm_back
    return ((ox, oy, 0.0), rot)


def visible_heading(motion: "VehicleMotion", is_in_camera: bool,
                    forward_offset_deg: float = 0.0) -> float:
    """World Z rotation (radians) a vehicle must face on the visible segment of
    the given camera view, INCLUDING the per-class forward_offset_deg correction
    for models whose nose does not point along +Y.

    In-camera  -> the vehicle faces its approach heading (drives toward box).
    Out-camera -> the vehicle faces its exit/outbound heading (drives away).
    The forward_offset_deg is added so a sideways model is corrected on BOTH
    entry and exit shots (the keyframed rotation would otherwise override the
    static root rotation set in make_vehicle_instance and drop the correction).
    """
    base = (approach_rotation(motion.approach) if is_in_camera
            else approach_rotation(motion.exit_direction))
    return base + math.radians(forward_offset_deg)


def camera_pose(approach: Direction, is_in: bool, road_meta: dict,
                cam_height: float = CAM_HEIGHT,
                cam_back_dist: float = CAM_BACK_DIST):
    """Return (cam_loc (x,y,z), look_at (x,y,z)) for the telephoto CCTV.

    SINGLE SOURCE OF TRUTH — shared by build_scene.place_camera (Blender) and
    render.compute_metadata (pure-python per-frame pose).

    Per-shot frame: each .blend is independent; road axis centred at (0,0,0).
    Camera sits on the road axis centre line (lateral x = 0 in the approach's
    rotated frame) at road_length/2 from the road centre, elevated CAM_HEIGHT.

    Entry (in_<D>): camera at the back/outer end of the entry arm (farthest
    from box), at -approach × (half + approach_length), looking toward the box.
    Cars appear near the camera and drive toward the stop line.

    Exit (out_<D>): camera at the box's far (outbound) edge, at
    +approach × half, looking outward. Cars emerge just ahead and drive away;
    the box is behind the camera.
    """
    fx, fy = approach_forward(approach)
    half = BOX_SIZE / 2
    approach_length = road_meta.get("approach_length", 0.0)
    crosswalk_y = road_meta.get("crosswalk_y", 0.0)
    if is_in:
        # Camera at the outer/back end of the entry arm.
        # arm extends approach_length from the stop line outward;
        # stop line is at -approach * half from world origin.
        cam_ground = (-fx * (half + approach_length),
                      -fy * (half + approach_length))
        look_ground = (-fx * (half - 2.0),
                       -fy * (half - 2.0), 0.0)
    else:
        # Camera at the box far edge, looking outward.
        cam_ground = (fx * half, fy * half)
        look_ground = (fx * (half + cam_back_dist),
                       fy * (half + cam_back_dist), 0.0)
    cam_loc = (cam_ground[0], cam_ground[1], cam_height)
    return cam_loc, look_ground


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

def intersection_path_length(turn: Turn) -> float:
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
    return intersection_path_length(turn) / speed_ms


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
                   fps: int = FPS,
                   appear_anchor: Optional[Tuple[float, float]] = None,
                   reappear_anchor: Optional[Tuple[float, float]] = None,
                   road_meta: Optional[dict] = None) -> VehicleMotion:
    """Compute the frame numbers and world positions for a vehicle's full
    In -> Black-Box -> Out trajectory.

    Conventions:
      * `approach` is the vehicle's heading TOWARD the box on entry.
      * The vehicle disappears at the box's near edge (entry side) along
        `approach` forward, at the entry lane centre.
      * After the Black-Box delay it reappears at the box's far edge along the
        OUTBOUND heading (the heading it has after the turn), at the exit lane
        centre, and continues OUTWARD along that outbound heading.
      * With ``road_meta`` (from ``road.json``) the IN segment ends at the box
        edge and the OUT segment ends at the road far end, so the vehicle
        drives the **full visible length** of the road arm.  Without it the
        fixed ``approach_visible_length`` / ``exit_visible_length`` defaults
        are used (legacy behaviour).

    appear_anchor / reappear_anchor: optional explicit START positions for the
        in-segment (appear) and out-segment (reappear). When given and
        ``road_meta`` is also provided the vehicle traverses the full road;
        when ``road_meta`` is absent the fixed-length defaults are used.
    """
    # ---- effective visible lengths -------------------------------------------
    _avl = approach_visible_length
    _evl = exit_visible_length
    if appear_anchor is not None and road_meta is not None:
        # IN segment ends at intersection box near edge.
        disappear_pos = lane_entry_box_edge(approach, lane)
        dx = disappear_pos[0] - appear_anchor[0]
        dy = disappear_pos[1] - appear_anchor[1]
        _avl = math.sqrt(dx * dx + dy * dy)
    if reappear_anchor is not None and road_meta is not None:
        _evl = road_meta["approach_length"]

    # ---- frame timing --------------------------------------------------------
    dt_approach = _avl / speed_ms
    dt_exit = _evl / speed_ms
    dt_box_frames = delta_t_frames(turn, speed_ms, fps)

    appear_frame = depart_frame
    disappear_frame = depart_frame + int(round(dt_approach * fps))
    reappear_frame = disappear_frame + dt_box_frames
    leave_frame = reappear_frame + int(round(dt_exit * fps))

    # ---- positions: ENTRY segment --------------------------------------------
    fx, fy = approach_forward(approach)
    if appear_anchor is not None:
        appear_pos = (appear_anchor[0], appear_anchor[1])
        if road_meta is not None:
            disappear_pos = lane_entry_box_edge(approach, lane)
        else:
            disappear_pos = (appear_pos[0] + fx * _avl,
                             appear_pos[1] + fy * _avl)
    else:
        disappear_pos = lane_entry_box_edge(approach, lane)
        appear_pos = (disappear_pos[0] - fx * _avl,
                      disappear_pos[1] - fy * _avl)

    # ---- positions: EXIT segment ---------------------------------------------
    outbound, ex_lane = exit_lane_for_movement(approach, lane, turn)
    ofx, ofy = outbound.vec
    if reappear_anchor is not None:
        reappear_pos = (reappear_anchor[0], reappear_anchor[1])
        leave_pos = (reappear_pos[0] + ofx * _evl,
                     reappear_pos[1] + ofy * _evl)
    else:
        reappear_pos = lane_exit_box_edge(outbound, ex_lane)
        leave_pos = (reappear_pos[0] + ofx * _evl,
                     reappear_pos[1] + ofy * _evl)

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
