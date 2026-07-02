"""Phase 4 — Render all 8 camera videos + export metadata.json.

For each of the 8 cameras (in_N, out_N, in_E, out_E, ...), builds the scene
(via build_scene) and renders an .mp4. Then writes metadata.json containing:
  - per-vehicle identity (id, class, color, plate, approach, turn, speed)
  - identity linkage (in_cam -> out_cam, disappear/reappear frames)
  - per-frame world pose (x, y, z, rot_z) + visibility flag + camera tag

No 2D bounding box is emitted: the metadata is pure ground truth for
stress-testing detection / LPR models, which must localise vehicles themselves.
Camera placement and per-lane vehicle start anchors come from the required env
files (assets/envs/<tag>.json), loaded by envfile.load_env; build_scene reads
the SAME files, so the rendered view and the metadata always agree.

Run:
    blender -b --python scripts/render.py -- --scenario output/run1/scenario.json --out output/run1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "lib"))
sys.path.insert(0, HERE)

# bpy / build_scene are only needed for rendering. Import lazily so that the
# pure-python metadata helper (compute_metadata) can be used from a non-Blender
# python (e.g. conda) for tests/validation.
try:
    import bpy
except ImportError:
    bpy = None

import geometry as G
import kinematics as K
import envfile as ENV


# ---------------------------------------------------------------------------
# Metadata assembly (pure-python, sparse pose ground truth)
# ---------------------------------------------------------------------------

def compute_metadata(scenario: dict, root: str) -> dict:
    """Build the full metadata structure from the scenario + kinematics + env.

    Per-frame data is SPARSE: only visible frames are listed, each carrying the
    vehicle's world pose (x, y, z, rot_z), visibility flag, and the camera tag
    that films it. Poses are derived by linear interpolation of the kinematics
    motion plan (the same plan build_scene keyframes), so metadata and render
    stay consistent.

    Vehicle START positions come from the required env files
    (``assets/envs/<tag>.json`` lane_defaults) — the SAME anchors build_scene
    uses — so the render and the ground-truth metadata agree exactly. The env
    files are loaded once and validated (hard-fail if missing/invalid).
    """
    fps = scenario["fps"]
    duration = scenario["duration_frames"]
    envs = {tag: ENV.load_env(tag, root) for tag in G.camera_names()}

    road_path = os.path.join(root, "assets", "road.json")
    if not os.path.exists(road_path):
        raise SystemExit(f"FAIL: road.json not found: {road_path}")
    with open(road_path) as f:
        road_meta = json.load(f)

    vehicles_meta = []
    for veh in scenario["vehicles"]:
        approach = G.Direction(veh["approach"])
        turn = G.Turn(veh["turn"])
        exit_dir, ex_lane = G.exit_lane_for_movement(approach, veh["lane"], turn)
        in_cam_tag = f"in_{approach.value}"
        out_cam_tag = f"out_{exit_dir.value}"
        # Required env anchors for this vehicle's in/out segments.
        in_anchor, in_rot_z = ENV.lane_default_anchor(envs[in_cam_tag], veh["lane"])
        out_anchor, out_rot_z = ENV.lane_default_anchor(envs[out_cam_tag], ex_lane)
        motion = K.plan_motion(veh["id"], approach, veh["lane"], turn,
                               veh["speed_ms"], veh["depart_frame"], fps=fps,
                               appear_anchor=in_anchor[:2],
                               reappear_anchor=out_anchor[:2],
                               road_meta=road_meta)

        frames = []
        # In segment — rot_z is the env anchor heading (true vehicle heading;
        # equals approach_rotation for an unedited file).
        for f in range(motion.appear_frame, min(motion.disappear_frame, duration) + 1):
            t = (f - motion.appear_frame) / max(1, motion.disappear_frame - motion.appear_frame)
            x = motion.appear_pos[0] + (motion.disappear_pos[0] - motion.appear_pos[0]) * t
            y = motion.appear_pos[1] + (motion.disappear_pos[1] - motion.appear_pos[1]) * t
            frames.append({
                "frame": f, "visible": True, "camera": in_cam_tag,
                "pose": {"x": round(x, 3), "y": round(y, 3), "z": 0.0,
                         "rot_z": round(in_rot_z, 4)},
            })
        # Out segment — rot_z is the env anchor heading for the exit lane.
        for f in range(max(motion.reappear_frame, 0), min(motion.leave_frame, duration) + 1):
            t = (f - motion.reappear_frame) / max(1, motion.leave_frame - motion.reappear_frame)
            x = motion.reappear_pos[0] + (motion.leave_pos[0] - motion.reappear_pos[0]) * t
            y = motion.reappear_pos[1] + (motion.leave_pos[1] - motion.reappear_pos[1]) * t
            frames.append({
                "frame": f, "visible": True, "camera": out_cam_tag,
                "pose": {"x": round(x, 3), "y": round(y, 3), "z": 0.0,
                         "rot_z": round(out_rot_z, 4)},
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
    meta = compute_metadata(scenario, ROOT)
    meta["videos"] = rendered
    meta_path = os.path.join(ns.out, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {meta_path}")
    print(f"Rendered {len(rendered)}/{len(cameras)} videos")


if __name__ == "__main__":
    main()
