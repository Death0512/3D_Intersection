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
                               road_meta=road_meta,
                               stop_frame=veh.get("stop_frame"),
                               release_frame=veh.get("release_frame"))

        frames = []
        # In segment — rot_z is the env anchor heading (true vehicle heading;
        # equals approach_rotation for an unedited file).
        for f in range(motion.appear_frame, min(motion.disappear_frame, duration) + 1):
            p = G.sample_track(motion.track_in, f)
            if p is None:
                continue
            frames.append({
                "frame": f, "visible": True, "camera": in_cam_tag,
                "pose": {"x": round(p[0], 3), "y": round(p[1], 3), "z": 0.0,
                         "rot_z": round(in_rot_z, 4)},
            })
        # Out segment — rot_z is the env anchor heading for the exit lane.
        for f in range(max(motion.reappear_frame, 0), min(motion.leave_frame, duration) + 1):
            p = G.sample_track(motion.track_out, f)
            if p is None:
                continue
            frames.append({
                "frame": f, "visible": True, "camera": out_cam_tag,
                "pose": {"x": round(p[0], 3), "y": round(p[1], 3), "z": 0.0,
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
            # Signal-stop ground truth (mirrors scenario_gen).  Present even
            # for free-flow vehicles (wait_frames == 0) so downstream labels
            # can rely on the keys always existing.
            "stop_frame": veh.get("stop_frame"),
            "release_frame": veh.get("release_frame"),
            "wait_frames": veh.get("wait_frames", 0),
            "queue_slot": veh.get("queue_slot", -1),
            "frames": frames,
        })
    meta = {
        "seed": scenario["seed"],
        "fps": fps,
        "duration_frames": duration,
        "box_size": G.BOX_SIZE,
        "num_vehicles": len(vehicles_meta),
        "vehicles": vehicles_meta,
        # Resolved camera spec per tag — the EXACT values build_scene.place_camera
        # applies (env override of geometry default, via envfile.resolve_camera).
        # Lets any consumer project world poses (above) into 2D pixels without
        # needing Blender: pinhole with lens_mm/sensor_mm/sensor_fit="HORIZONTAL",
        # extrinsic derived from location + look_at (rotation_euler null) or the
        # explicit rotation_euler.  Resolution matches setup_render's RES_X/RES_Y.
        "cameras": {
            tag: {
                **ENV.resolve_camera(envs[tag], road_meta),
                "sensor_fit": "HORIZONTAL",
                "resolution": [G.RES_X, G.RES_Y],
            }
            for tag in G.camera_names()
        },
    }
    # Emit the signal timeline + mode when the scenario carried signal info,
    # so per-frame "why did this car stop" ground truth is available for
    # downstream vision / behaviour labels.
    if "signal_mode" in scenario or "signal_timeline" in scenario:
        meta["signal"] = {
            "mode": scenario.get("signal_mode", "fixed"),
            "cycle_frames": scenario.get("signal_cycle_frames"),
            "timeline": scenario.get("signal_timeline", []),
            "clearances": scenario.get("signal_clearances", []),
        }
    return meta


# ---------------------------------------------------------------------------
# ffmpeg encoding — GPU (NVENC) with CPU (libx264) fallback
# ---------------------------------------------------------------------------
_NVENC_AVAILABLE = None  # cached probe result (None = not yet probed)


def _nvenc_available() -> bool:
    """Probe once per process whether ffmpeg can actually use h264_nvenc.

    Checks both that the encoder is compiled in AND that it can init on this
    GPU (e.g. missing NVENC driver bits would fail at encode time, not at
    -encoders listing time).
    """
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    import subprocess
    try:
        probe = subprocess.run(
            # NVENC requires at least ~145x49 (varies by GPU); use 256x256 to
            # stay safely above the minimum on all supported hardware.
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=15,
        )
        _NVENC_AVAILABLE = probe.returncode == 0
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def _ffmpeg_encode(frames_dir: str, video_path: str, fps: float) -> bool:
    """Encode the PNG frame sequence to mp4. Prefers GPU NVENC (frees the CPU
    for other parallel render workers); falls back to CPU libx264 if NVENC
    is unavailable or fails at runtime. Inherits CUDA_VISIBLE_DEVICES from
    the parent process environment (set per-worker by run_pipeline.py), so
    NVENC binds to the same GPU this Blender instance rendered on.
    """
    import subprocess
    import os as _os

    base = [
        "ffmpeg", "-y", "-framerate", str(fps),
        # frames start at f_0000.png (scene.frame_start = 0); ffmpeg's %04d
        # glob defaults to -start_number 1, which would drop frame 0.
        "-start_number", "0",
        "-i", _os.path.join(frames_dir, "f_%04d.png"),
    ]

    if _nvenc_available():
        cmd = base + [
            "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p",
            "-cq", "20", video_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return True
        print(f"  [ffmpeg] NVENC encode failed, falling back to CPU: "
              f"{proc.stderr[-200:]}")

    cmd = base + [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0


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

    # encode to mp4 (GPU NVENC when available, else CPU libx264 fallback —
    # keeps encoding off the CPU so it doesn't bottleneck/contend with
    # parallel Blender-GPU render workers).
    video_path = os.path.join(out_dir, f"video_{camera_tag}.mp4")
    fps = scenario["fps"]
    ok = _ffmpeg_encode(frames_dir, video_path, fps)
    if ok:
        print(f"  rendered: {video_path}")
    else:
        print(f"  ffmpeg FAILED for {camera_tag}")
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
    ap.add_argument("--no-metadata", action="store_true",
                    help="skip metadata compute (used in parallel render workers)")
    ap.add_argument("--metadata-only", action="store_true",
                    help="write metadata.json from scenario + existing videos (no render)")
    ns = ap.parse_args(post)

    with open(ns.scenario) as f:
        scenario = json.load(f)
    os.makedirs(ns.out, exist_ok=True)

    # --metadata-only mode: pure-python metadata pass (no bpy/render).
    if ns.metadata_only:
        _write_metadata(scenario, ns.out)
        return

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

    if not ns.no_metadata:
        _write_metadata(scenario, ns.out)
    print(f"Rendered {len(rendered)}/{len(cameras)} videos")


def _write_metadata(scenario, out_dir):
    """Write metadata.json — scans out_dir for existing video_*.mp4 files."""
    import glob
    videos = sorted(os.path.relpath(p, out_dir)
                    for p in glob.glob(os.path.join(out_dir, "video_*.mp4")))
    meta = compute_metadata(scenario, ROOT)
    meta["videos"] = videos
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
