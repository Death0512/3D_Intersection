"""Phase 4 — Render all 8 camera videos + export metadata.json.

For each of the 8 cameras (in_N, out_N, in_E, out_E, ...), builds the scene
(via build_scene) and renders an .mp4. Then writes metadata.json containing:
  - per-vehicle identity (id, class, color, plate, approach, turn, speed)
  - identity linkage (in_cam -> out_cam, disappear/reappear frames)
  - per-frame XYZ position + rotation + visibility flag per vehicle
  - 2D bounding box per camera (projected from world XYZ)

Run:
    blender -b --python scripts/render.py -- --scenario output/run1/scenario.json --out output/run1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
sys.path.insert(0, HERE)

# bpy / build_scene are only needed for rendering. Import lazily so that the
# pure-python metadata helpers (compute_metadata, bbox projection) can be used
# from a non-Blender python (e.g. conda) for tests/validation.
try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None
    Vector = None

import geometry as G
import kinematics as K


# ---------------------------------------------------------------------------
# 2D bbox projection (pure-python, matches build_scene.place_camera geometry)
# ---------------------------------------------------------------------------

# Camera params must match build_scene.py + road.json
CAM_HEIGHT = 7.0
CAM_BACK_DIST = 45.0   # used for out-camera look target distance
LENS_MM = 60.0
SENSOR_MM = 36.0       # default full-frame width
RES_X = 1920
RES_Y = 1080

# Road geometry (from road.json) — must stay in sync with assets/road.json
_CROSSWALK_Y = 27.846
_APPROACH_LENGTH = 54.751
_ARM_BACK = _APPROACH_LENGTH - _CROSSWALK_Y   # ≈ 26.905 (back-end offset past origin)


def _camera_pose(approach: G.Direction, is_in: bool):
    """Return (cam_loc, look_at) matching build_scene.place_camera exactly.

    Entry (in_<D>): camera at the outer/back end of the arm, looking toward box.
    Exit  (out_<D>): camera at the box-edge/crosswalk, looking outward.
    No lateral offset — each arm is centred on its own branch axis.
    """
    fx, fy = approach.vec
    half = G.BOX_SIZE / 2

    if is_in:
        dist_from_box = _CROSSWALK_Y + _ARM_BACK   # full arm length ≈ 54.75
        cam_ground = (-fx * (half + dist_from_box),
                      -fy * (half + dist_from_box))
        look_ground = (-fx * (half - 2.0),
                       -fy * (half - 2.0), 0.0)
    else:
        cam_ground = (-fx * half,
                      -fy * half)
        look_ground = (-fx * half + fx * CAM_BACK_DIST,
                       -fy * half + fy * CAM_BACK_DIST, 0.0)

    cam_loc = (cam_ground[0], cam_ground[1], CAM_HEIGHT)
    return cam_loc, look_ground


def _project_point(world_xyz, cam_loc, look_at, res_x=RES_X, res_y=RES_Y,
                   lens=LENS_MM, sensor=SENSOR_MM):
    """Project a world point to pixel coords (u, v) with top-left origin.
    Returns (u, v) or None if behind camera. Pinhole model with camera
    looking along (look_at - cam_loc), world up = +Z."""
    # camera basis
    cx, cy, cz = cam_loc
    fwd = (look_at[0] - cx, look_at[1] - cy, look_at[2] - cz)
    fl = math.sqrt(sum(c*c for c in fwd))
    if fl == 0:
        return None
    fwd = tuple(c / fl for c in fwd)
    # right = forward x world_up(0,0,1)
    rx = fwd[1] * 1 - fwd[2] * 0
    ry = fwd[2] * 0 - fwd[0] * 1
    rz = fwd[0] * 0 - fwd[1] * 0
    rl = math.sqrt(rx*rx + ry*ry)
    if rl == 0:
        rx, ry, rz = 1.0, 0.0, 0.0
    else:
        rx, ry, rz = rx/rl, ry/rl, 0.0
    # up = right x forward
    ux = ry * fwd[2] - rz * fwd[1]
    uy = rz * fwd[0] - rx * fwd[2]
    uz = rx * fwd[1] - ry * fwd[0]
    # point relative to camera
    d = (world_xyz[0] - cx, world_xyz[1] - cy, world_xyz[2] - cz)
    depth = d[0]*fwd[0] + d[1]*fwd[1] + d[2]*fwd[2]
    if depth <= 0:
        return None
    x_cam = d[0]*rx + d[1]*ry + d[2]*rz
    y_cam = d[0]*ux + d[1]*uy + d[2]*uz
    # pinhole
    focal_px = lens * res_x / sensor
    u = res_x / 2 + (x_cam / depth) * focal_px
    v = res_y / 2 - (y_cam / depth) * focal_px
    return (u, v)


def _vehicle_bbox(world_xyz, dims, cam_loc, look_at):
    """Project the 8 corners of a vehicle bbox (world_xyz centre, dims
    [w,l,h]) and return (u_min, v_min, u_max, v_max) or None."""
    w, l, h = dims
    corners = [
        (world_xyz[0]-w/2, world_xyz[1]-l/2, 0),
        (world_xyz[0]+w/2, world_xyz[1]-l/2, 0),
        (world_xyz[0]-w/2, world_xyz[1]+l/2, 0),
        (world_xyz[0]+w/2, world_xyz[1]+l/2, 0),
        (world_xyz[0]-w/2, world_xyz[1]-l/2, h),
        (world_xyz[0]+w/2, world_xyz[1]-l/2, h),
        (world_xyz[0]-w/2, world_xyz[1]+l/2, h),
        (world_xyz[0]+w/2, world_xyz[1]+l/2, h),
    ]
    us, vs = [], []
    for c in corners:
        p = _project_point(c, cam_loc, look_at)
        if p is None:
            return None
        us.append(p[0]); vs.append(p[1])
    return (min(us), min(vs), max(us), max(vs))


# ---------------------------------------------------------------------------
# Metadata assembly (pure-python, sparse + bbox)
# ---------------------------------------------------------------------------

def compute_metadata(scenario: dict) -> dict:
    """Build the full metadata structure from the scenario + kinematics.
    Per-frame data is SPARSE: only visible frames are listed, each with pose
    and a 2D bbox projected into the relevant camera."""
    fps = scenario["fps"]
    duration = scenario["duration_frames"]
    vehicles_meta = []
    for veh in scenario["vehicles"]:
        approach = G.Direction(veh["approach"])
        turn = G.Turn(veh["turn"])
        motion = K.plan_motion(veh["id"], approach, veh["lane"], turn,
                               veh["speed_ms"], veh["depart_frame"], fps=fps)
        in_cam_tag = f"in_{approach.value}"
        out_cam_tag = f"out_{motion.exit_direction.value}"
        in_cam_loc, in_look = _camera_pose(approach, is_in=True)
        out_cam_loc, out_look = _camera_pose(motion.exit_direction, is_in=False)
        dims = veh["length"] and (1.9, veh["length"], 1.4)  # approx w,h

        frames = []
        # In segment
        for f in range(motion.appear_frame, min(motion.disappear_frame, duration) + 1):
            t = (f - motion.appear_frame) / max(1, motion.disappear_frame - motion.appear_frame)
            x = motion.appear_pos[0] + (motion.disappear_pos[0] - motion.appear_pos[0]) * t
            y = motion.appear_pos[1] + (motion.disappear_pos[1] - motion.appear_pos[1]) * t
            bbox = _vehicle_bbox((x, y, 0.0), dims, in_cam_loc, in_look)
            frames.append({
                "frame": f, "visible": True, "camera": in_cam_tag,
                "pose": {"x": round(x, 3), "y": round(y, 3), "z": 0.0,
                         "rot_z": round(G.approach_rotation(approach), 4)},
                "bbox": [round(b, 1) for b in bbox] if bbox else None,
            })
        # Out segment
        for f in range(max(motion.reappear_frame, 0), min(motion.leave_frame, duration) + 1):
            t = (f - motion.reappear_frame) / max(1, motion.leave_frame - motion.reappear_frame)
            x = motion.reappear_pos[0] + (motion.leave_pos[0] - motion.reappear_pos[0]) * t
            y = motion.reappear_pos[1] + (motion.leave_pos[1] - motion.reappear_pos[1]) * t
            bbox = _vehicle_bbox((x, y, 0.0), dims, out_cam_loc, out_look)
            frames.append({
                "frame": f, "visible": True, "camera": out_cam_tag,
                "pose": {"x": round(x, 3), "y": round(y, 3), "z": 0.0,
                         "rot_z": round(G.approach_rotation(motion.exit_direction), 4)},
                "bbox": [round(b, 1) for b in bbox] if bbox else None,
            })
        frames.sort(key=lambda d: d["frame"])

        vehicles_meta.append({
            "id": veh["id"],
            "class": veh["class"],
            "color": veh["color"],
            "color_name": veh.get("color_name"),
            "plate": veh["plate"],
            "approach": veh["approach"],
            "lane": veh["lane"],
            "turn": veh["turn"],
            "speed_ms": veh["speed_ms"],
            "speed_kmh": veh["speed_kmh"],
            "length": veh["length"],
            "depart_frame": veh["depart_frame"],
            "in_camera": in_cam_tag,
            "out_camera": out_cam_tag,
            "appear_frame": motion.appear_frame,
            "disappear_frame": motion.disappear_frame,
            "reappear_frame": motion.reappear_frame,
            "leave_frame": motion.leave_frame,
            "exit_direction": motion.exit_direction.value,
            "exit_lane": motion.exit_lane,
            "delta_t_frames": motion.reappear_frame - motion.disappear_frame,
            "frames": frames,
        })
    return {
        "seed": scenario["seed"],
        "fps": fps,
        "duration_frames": duration,
        "box_size": G.BOX_SIZE,
        "num_vehicles": len(vehicles_meta),
        "vehicles": vehicles_meta,
    }


# ---------------------------------------------------------------------------
# Rendering loop
# ---------------------------------------------------------------------------

def render_one(scenario: dict, camera_tag: str, out_dir: str):
    """Build + render one camera shot to <out_dir>/video_<tag>.mp4.

    Renders a PNG frame sequence into a temp subdir, then encodes to mp4 with
    ffmpeg (Blender's built-in FFMPEG container can be finicky across builds).
    """
    import shutil
    import subprocess
    import build_scene as BS  # requires bpy (only available inside Blender)

    scene_blend = os.path.join(out_dir, f"scene_{camera_tag}.blend")
    BS.build_shot(scenario, camera_tag, scene_blend)

    # Re-configure GPU here too: Cycles addon prefs live in user preferences,
    # not the .blend, so they must be set in the active session before render.
    BS.configure_gpu()
    BS.setup_render()

    scene = bpy.context.scene
    duration = scenario["duration_frames"]
    scene.frame_start = 0
    scene.frame_end = duration

    # PNG sequence into a frames subdir
    frames_dir = os.path.join(out_dir, f"frames_{camera_tag}")
    os.makedirs(frames_dir, exist_ok=True)
    # clear any old frames
    for fn in os.listdir(frames_dir):
        if fn.endswith(".png"):
            os.remove(os.path.join(frames_dir, fn))
    scene.render.filepath = os.path.join(frames_dir, "f_")  # produces f_0001.png ...
    scene.render.image_settings.file_format = "PNG"

    bpy.ops.render.render(animation=True)

    # encode to mp4
    video_path = os.path.join(out_dir, f"video_{camera_tag}.mp4")
    fps = scenario["fps"]
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "f_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ffmpeg FAILED for {camera_tag}: {proc.stderr[-300:]}")
    else:
        print(f"  rendered: {video_path}")
    # optionally clean up frames dir to save space
    try:
        shutil.rmtree(frames_dir)
    except Exception:
        pass
    return video_path


def main():
    args = sys.argv
    if "--" not in args:
        raise SystemExit("Usage: blender -b --python render.py -- --scenario X --out Y [--only in_N]")
    post = args[args.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="render only this camera tag (e.g. in_N)")
    ns = ap.parse_args(post)

    with open(ns.scenario) as f:
        scenario = json.load(f)
    os.makedirs(ns.out, exist_ok=True)

    # pre-generate plates (in conda python normally; here we try via bpy fallback)
    plates_dir = os.path.join(ns.out, "plates")
    os.makedirs(plates_dir, exist_ok=True)

    cameras = G.camera_names()
    if ns.only:
        cameras = [ns.only]

    rendered = []
    for tag in cameras:
        try:
            p = render_one(scenario, tag, ns.out)
            rendered.append(p)
        except Exception as e:
            print(f"  FAILED {tag}: {e}")

    # metadata
    meta = compute_metadata(scenario)
    meta["videos"] = rendered
    meta_path = os.path.join(ns.out, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {meta_path}")
    print(f"Rendered {len(rendered)}/{len(cameras)} videos")


if __name__ == "__main__":
    main()
