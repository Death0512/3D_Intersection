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
from typing import Dict, List, Optional, Set, Tuple


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

TURN_FRICTION_MU = 0.7
TURN_ACCEL_MS2 = 2.5

# Comfortable longitudinal accel/decel for visible-segment motion profiles
# (Workstream A — believable stop/go imagery).  Chosen inside the envelope
# of real-world passenger-car comfortable braking (~2.5 m/s^2) and
# acceleration from a stop (~2.0 m/s^2), and well under emergency limits.
ACCEL_MS2 = 2.0
DECEL_MS2 = 2.5


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


# Strict lane-use control matrix. Lane 0 is median-side; lane NUM_LANES-1 is
# curb-side. Middle lanes are through-only.
LANE_TURN_RESTRICTIONS: Dict[int, Set[Turn]] = {
    0: {Turn.LEFT, Turn.STRAIGHT},
    1: {Turn.STRAIGHT},
    2: {Turn.STRAIGHT},
    3: {Turn.STRAIGHT, Turn.RIGHT},
}


def allowed_turns(lane_index: int) -> Set[Turn]:
    if lane_index not in LANE_TURN_RESTRICTIONS:
        raise ValueError(f"invalid lane index: {lane_index}")
    return set(LANE_TURN_RESTRICTIONS[lane_index])


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
            for turn in allowed_turns(lane):
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

def turn_radius(turn: Turn, lane_index: Optional[int] = None) -> Optional[float]:
    """Approximate lane-dependent turn radius inside the black box.

    ``None`` means straight/no arc. Lane 0 is median-side, lane 3 curb-side.
    Legal strict usage gives lane0-left and lane3-right, but the formula also
    behaves sensibly for permissive/test calls.
    """
    if turn == Turn.STRAIGHT:
        return None
    lane = 0 if lane_index is None else lane_index
    if turn == Turn.RIGHT:
        return 6.0 + (NUM_LANES - 1 - lane) * LANE_WIDTH
    if turn == Turn.LEFT:
        return 12.0 + lane * LANE_WIDTH


def intersection_path_length(turn: Turn, lane_index: Optional[int] = None) -> float:
    """Distance travelled inside the 30x30 box.

    Legacy fallback (lane_index is None) preserves the original fixed radii.
    When lane_index is supplied, turn radii depend on the entry lane.
    """
    if turn == Turn.STRAIGHT:
        return BOX_SIZE
    if lane_index is None:
        if turn == Turn.RIGHT:
            return (math.pi / 2) * 6.0
        if turn == Turn.LEFT:
            return (math.pi / 2) * 12.0
    radius = turn_radius(turn, lane_index)
    return (math.pi / 2) * radius


def delta_t_seconds(turn: Turn, speed_ms: float,
                    lane_index: Optional[int] = None) -> float:
    """Blind-zone delay time (s).

    Straight movement uses constant speed. Turning movement with lane_index
    supplied slows to a friction-limited curve speed and includes simple
    decel/accel time; lane_index=None preserves the legacy fixed-speed model.
    """
    if speed_ms <= 0:
        raise ValueError("speed must be > 0")
    path = intersection_path_length(turn, lane_index)
    if turn == Turn.STRAIGHT or lane_index is None:
        return path / speed_ms
    radius = turn_radius(turn, lane_index)
    curve_speed = min(speed_ms, math.sqrt(TURN_FRICTION_MU * 9.81 * radius))
    if curve_speed >= speed_ms:
        return path / speed_ms
    decel_time = (speed_ms - curve_speed) / TURN_ACCEL_MS2
    accel_time = decel_time
    return (path / curve_speed) + decel_time + accel_time


def delta_t_frames(turn: Turn, speed_ms: float, fps: int = FPS,
                   lane_index: Optional[int] = None) -> int:
    """Integer frame count of invisibility (rounded)."""
    return int(round(delta_t_seconds(turn, speed_ms, lane_index) * fps))


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

    # Per-segment keyframe tracks (for multi-state motion like queue/platoon).
    # Legacy (no road_meta) → each track has 2 LINEAR points matching the
    # scalar fields above.
    track_in: list = field(default_factory=list)   # List[TrackPoint]
    track_out: list = field(default_factory=list)  # List[TrackPoint]


@dataclass
class TrackPoint:
    """A single point in a vehicle's visible-segment keyframe track.

    ``interp`` is "LINEAR" or "BEZIER" for the segment FROM this point to the
    next.  For BEZIER segments ``cp1`` and ``cp2`` are the two explicit cubic
    Bézier control points (in world coordinates) used by both the Blender
    keyframer and the Python metadata interpolator, guaranteeing
    render==metadata.
    """
    frame: int
    x: float
    y: float
    visible: bool = True
    interp: str = "LINEAR"
    cp1: Optional[Tuple[float, float]] = None   # handle near current point
    cp2: Optional[Tuple[float, float]] = None   # handle near next point


def bezier_point(t: float, p0: Tuple[float, float],
                 p1: Tuple[float, float], p2: Tuple[float, float],
                 p3: Tuple[float, float]) -> Tuple[float, float]:
    """Evaluate a cubic Bézier curve B(t) for 0 ≤ t ≤ 1."""
    u = 1.0 - t
    x = u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0]
    y = u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]
    return (x, y)


def sample_track(track: List[TrackPoint], frame: int) -> Optional[Tuple[float, float]]:
    """Interpolate the track at ``frame`` using the per-segment interpolation mode.

    Returns (x, y) or None if frame is outside the track's frame range.
    """
    if not track or frame < track[0].frame or frame > track[-1].frame:
        return None
    for i in range(len(track) - 1):
        if track[i].frame <= frame <= track[i + 1].frame:
            t0 = track[i]
            t1 = track[i + 1]
            span = t1.frame - t0.frame
            if span == 0:
                return (t0.x, t0.y)
            t = (frame - t0.frame) / span
            if t0.interp == "BEZIER" and t0.cp1 is not None:
                return bezier_point(t, (t0.x, t0.y), t0.cp1, t0.cp2, (t1.x, t1.y))
            else:
                return (t0.x + (t1.x - t0.x) * t, t0.y + (t1.y - t0.y) * t)
    return None


def compute_motion(vehicle_id: str, approach: Direction, lane: int, turn: Turn,
                   speed_ms: float, depart_frame: int,
                   approach_visible_length: float = 40.0,
                   exit_visible_length: float = 40.0,
                   fps: int = FPS,
                   appear_anchor: Optional[Tuple[float, float]] = None,
                   reappear_anchor: Optional[Tuple[float, float]] = None,
                   road_meta: Optional[dict] = None,
                   stop_frame: Optional[int] = None,
                   release_frame: Optional[int] = None) -> VehicleMotion:
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

    stop_frame / release_frame: for queued vehicles. stop_frame is when the
        vehicle reaches the stop line (box near edge). release_frame is when
        it enters the box after waiting. When both are provided and
        release_frame > stop_frame, a multi-point IN track with idle segment
        is built.
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
    dt_box_frames = delta_t_frames(
        turn, speed_ms, fps,
        lane_index=lane if road_meta is not None else None)

    appear_frame = depart_frame
    free_disappear_frame = depart_frame + int(round(dt_approach * fps))

    is_queued = (stop_frame is not None and release_frame is not None
                 and release_frame > stop_frame)
    if is_queued:
        disappear_frame = release_frame
    else:
        disappear_frame = free_disappear_frame

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

    # ---- build keyframe tracks ------------------------------------------------
    if is_queued and road_meta is not None:
        track_in = _build_queue_track(
            appear_frame, stop_frame, release_frame,
            appear_pos, disappear_pos, speed_ms=speed_ms, fps=fps)
    else:
        track_in = _build_segment_track(
            appear_frame, disappear_frame, appear_pos, disappear_pos,
            turn, road_meta, is_out=False)

    # OUT track: queued vehicles launch from rest at the box edge (accel-from-
    # stop ease-out) so the green-light launch is visible; turning vehicles
    # also ease-out because they exit the curve at low speed.
    track_out = _build_segment_track(
        reappear_frame, leave_frame, reappear_pos, leave_pos,
        turn, road_meta, is_out=True, queued=is_queued,
        accel_from_stop=is_queued)

    return VehicleMotion(
        vehicle_id=vehicle_id, approach=approach, lane=lane, turn=turn,
        speed_ms=speed_ms, depart_frame=depart_frame, fps=fps,
        appear_frame=appear_frame, disappear_frame=disappear_frame,
        reappear_frame=reappear_frame, leave_frame=leave_frame,
        appear_pos=appear_pos, disappear_pos=disappear_pos,
        reappear_pos=reappear_pos, leave_pos=leave_pos,
        exit_direction=outbound, exit_lane=ex_lane,
        track_in=track_in, track_out=track_out,
    )


def _bezier_ease_in(start_pos: Tuple, end_pos: Tuple,
                    ease_in_frac: float = 0.25) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """Cubic Bézier control points for an *ease-in* decel segment (fast→slow).

    The curve leaves the start at near-full tangent (free-flow cruise) and
    decelerates to a stop at ``end_pos``. ``ease_in_frac`` is the fraction of
    the segment length reserved for the decel taper at the END. Returns
    ``(cp1, cp2)`` in world coordinates.
    """
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    # cp1 near start, almost at the start (cruise tangent preserved).
    cp1 = (start_pos[0] + dx * (1.0 - ease_in_frac),
           start_pos[1] + dy * (1.0 - ease_in_frac))
    # cp2 hugs the end (tangent → 0 ⇒ decel to rest).
    cp2 = (end_pos[0], end_pos[1])
    return cp1, cp2


def _bezier_ease_out(start_pos: Tuple, end_pos: Tuple,
                     ease_out_frac: float = 0.25) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """Cubic Bézier control points for an *ease-out* accel segment (slow→fast).

    The vehicle starts from rest at ``start_pos``, accelerates, and reaches
    full cruise speed by the end. ``ease_out_frac`` is the fraction of the
    segment length spent ramping up (at the START).
    """
    # cp1 hugs the start (tangent 0 ⇒ start from rest).
    cp1 = (start_pos[0], start_pos[1])
    # cp2 near end, almost at the end (full cruise tangent restored).
    cp2 = (start_pos[0] + (end_pos[0] - start_pos[0]) * ease_out_frac,
           start_pos[1] + (end_pos[1] - start_pos[1]) * ease_out_frac)
    return cp1, cp2


def _build_segment_track(start_frame: int, end_frame: int,
                         start_pos: Tuple, end_pos: Tuple,
                         turn: Turn, road_meta: Optional[dict],
                         is_out: bool, queued: bool = False,
                         accel_from_stop: bool = False) -> List[TrackPoint]:
    """Build a 2-point keyframe track for one visible segment.

    BEZIER ease-out (acceleration from rest at the box edge) is applied to
    the OUT segment when the vehicle starts from rest — i.e. when it was
    queued at a red light (``queued`` / ``accel_from_stop``) OR when it is a
    turning movement (exits the curve at low speed and accelerates back to
    cruise). This makes the launch visible on the out-camera instead of an
    instant jump to full speed. Free-flow straight vehicles on green do not
    ease-out (they reappear already at cruise).
    """
    use_bezier = (is_out and road_meta is not None
                  and (turn != Turn.STRAIGHT or queued or accel_from_stop))
    if use_bezier:
        p0 = start_pos
        p3 = end_pos
        if accel_from_stop or queued:
            cp1, cp2 = _bezier_ease_out(p0, p3, ease_out_frac=0.30)
        else:
            # turning vehicle: mild ease-out from curve-exit speed
            cp1 = (p0[0] + (p3[0] - p0[0]) * 0.03, p0[1] + (p3[1] - p0[1]) * 0.03)
            cp2 = (p3[0] - (p3[0] - p0[0]) * 0.15, p3[1] - (p3[1] - p0[1]) * 0.15)
    else:
        cp1 = cp2 = None

    return [
        TrackPoint(frame=start_frame, x=start_pos[0], y=start_pos[1],
                   visible=True,
                   interp="LINEAR" if not use_bezier else "BEZIER",
                   cp1=cp1, cp2=cp2),
        TrackPoint(frame=end_frame, x=end_pos[0], y=end_pos[1],
                   visible=True, interp="LINEAR"),
    ]


def _build_queue_track(start_frame: int, stop_frame: int, release_frame: int,
                       start_pos: Tuple, stop_pos: Tuple,
                       speed_ms: float, fps: int = FPS) -> List[TrackPoint]:
    """Build a 4-point keyframe track for a queued IN segment with realistic
    deceleration into the stop line.

    Profile: cruise (LINEAR) → BEZIER ease-in decel → idle (stationary at
    stop line) until release.

    Points:
      0. ``start_frame`` at ``start_pos`` — vehicle appears at the env anchor.
      1. ``decel_end_frame`` at ``decel_end_pos`` — end of the cruise; the
         vehicle has travelled the cruise portion of the approach at free-flow
         speed and is about to brake. The segment from point 0 to here is LINEAR.
      2. ``stop_frame`` at ``stop_pos`` — the vehicle is now stationary at the
         stop line. The segment from point 1 to here is BEZIER ease-in (decel).
      3. ``release_frame`` at ``stop_pos`` — the vehicle idles until release.
         The segment from point 2 to here is LINEAR with identical positions.

    Braking physics: the vehicle decelerates from ``speed_ms`` to 0 at
    ``DECEL_MS2``; the braking distance is ``d_brake = v^2 / (2*DECEL)`` and
    the braking time ``t_brake = v / DECEL``. The cruise covers the remaining
    approach distance; its hold time is what adjusts to make the total
    travel time land exactly on ``stop_frame``.

    Fallback: the decel phase is only inserted when the approach is long
    enough to fit a meaningful brake (>1 frame at the configured decel);
    otherwise the legacy constant-speed-then-snap 3-point track is used.
    """
    v = speed_ms
    approach_dx = stop_pos[0] - start_pos[0]
    approach_dy = stop_pos[1] - start_pos[1]
    approach_len = math.sqrt(approach_dx * approach_dx + approach_dy * approach_dy)
    available_time_s = max(0.0, (stop_frame - start_frame) / fps)

    # Nominal brake: full-speed cruise then brake at DECEL_MS2 to a stop.
    # If the brake time does not fit in the available approach time, fall back
    # to a gentler single-ramp decel that spans the whole approach (the
    # vehicle eases down from v to 0 over the entire visible segment). This
    # keeps the stop landing exactly on stop_frame with no pre-appear braking.
    t_brake_s = v / DECEL_MS2
    d_brake = (v * v) / (2.0 * DECEL_MS2)
    use_full_ramp = (t_brake_s >= available_time_s
                     or d_brake >= approach_len * 0.95
                     or available_time_s <= 0.0
                     or approach_len <= 0.0)
    if use_full_ramp:
        # Whole approach is one BEZIER ease-in (decel from v to 0).
        cp1, cp2 = _bezier_ease_in(start_pos, stop_pos, ease_in_frac=0.50)
        return [
            TrackPoint(frame=start_frame, x=start_pos[0], y=start_pos[1],
                       visible=True, interp="BEZIER", cp1=cp1, cp2=cp2),
            TrackPoint(frame=stop_frame, x=stop_pos[0], y=stop_pos[1],
                       visible=True, interp="LINEAR"),
            TrackPoint(frame=release_frame, x=stop_pos[0], y=stop_pos[1],
                       visible=True, interp="LINEAR"),
        ]

    t_brake_frames = int(round(t_brake_s * fps))
    if t_brake_frames < 2:
        # Too few frames to render a decel — keep legacy 3-point snap track.
        return [
            TrackPoint(frame=start_frame, x=start_pos[0], y=start_pos[1],
                       visible=True, interp="LINEAR"),
            TrackPoint(frame=stop_frame, x=stop_pos[0], y=stop_pos[1],
                       visible=True, interp="LINEAR"),
            TrackPoint(frame=release_frame, x=stop_pos[0], y=stop_pos[1],
                       visible=True, interp="LINEAR"),
        ]

    # Cruise covers (approach_len - d_brake) at speed v; brake covers d_brake
    # in t_brake_frames. Both anchored so their sum lands on stop_frame.
    cruise_dist = approach_len - d_brake
    cruise_time_s = cruise_dist / v if v > 0 else 0.0
    cruise_frames = int(round(cruise_time_s * fps))
    cruise_end_frame = start_frame + cruise_frames
    decel_end_frame = stop_frame - t_brake_frames
    # Clamp monotonicity: never let cruise_end run past where the decel must
    # begin (high-speed / short-approach boundary). A small overlap is OK —
    # the BEZIER shape handles the transition; just keep frames ordered.
    if cruise_end_frame > decel_end_frame:
        cruise_end_frame = decel_end_frame
    if cruise_end_frame < start_frame:
        cruise_end_frame = start_frame

    unit_x = approach_dx / approach_len if approach_len > 0 else 0.0
    unit_y = approach_dy / approach_len if approach_len > 0 else 0.0
    cruise_end_pos = (start_pos[0] + unit_x * cruise_dist,
                      start_pos[1] + unit_y * cruise_dist)

    cp1, cp2 = _bezier_ease_in(cruise_end_pos, stop_pos, ease_in_frac=0.30)

    return [
        # 0 → 1 : cruise at free-flow speed (LINEAR).
        TrackPoint(frame=start_frame, x=start_pos[0], y=start_pos[1],
                   visible=True, interp="LINEAR"),
        TrackPoint(frame=cruise_end_frame, x=cruise_end_pos[0], y=cruise_end_pos[1],
                   visible=True, interp="BEZIER", cp1=cp1, cp2=cp2),
        # 1 → 2 : decelerate to rest at the stop line (BEZIER ease-in).
        TrackPoint(frame=stop_frame, x=stop_pos[0], y=stop_pos[1],
                   visible=True, interp="LINEAR"),
        # 2 → 3 : idle at the stop line until release (LINEAR, stationary).
        TrackPoint(frame=release_frame, x=stop_pos[0], y=stop_pos[1],
                   visible=True, interp="LINEAR"),
    ]


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
