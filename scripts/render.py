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
import time

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
from sim.trajectory import (
    apply_samples_to_motion,
    complete_trajectory_with_metadata,
    load_trajectory_index,
)


# ---------------------------------------------------------------------------
# Metadata assembly (pure-python, sparse pose ground truth)
# ---------------------------------------------------------------------------

def compute_metadata(scenario: dict, root: str, run_dir: str | None = None) -> dict:
    """Build the full metadata structure from the scenario + kinematics + env.

    Per-frame data is SPARSE: only visible frames are listed, each carrying the
    vehicle's world pose (x, y, z, rot_z), visibility flag, and the camera tag
    that films it. Research scenarios prefer trajectory.json-backed approach
    tracks via the same adapter build_scene uses; legacy scenarios fall back to
    the kinematics motion plan.

    Vehicle START positions come from the required env files
    (``assets/envs/<tag>.json`` lane_defaults) — the SAME anchors build_scene
    uses — so the render and the ground-truth metadata agree exactly. The env
    files are loaded once and validated (hard-fail if missing/invalid).
    """
    fps = scenario["fps"]
    duration = scenario["duration_frames"]
    last_frame = max(0, duration - 1)
    envs = {tag: ENV.load_env(tag, root) for tag in G.camera_names()}

    road_path = os.path.join(root, "assets", "road.json")
    if not os.path.exists(road_path):
        raise SystemExit(f"FAIL: road.json not found: {road_path}")
    with open(road_path) as f:
        road_meta = json.load(f)
    traj_index = load_trajectory_index(scenario, run_dir) if run_dir else {}

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
                               release_frame=veh.get("release_frame"),
                               queue_slot=veh.get("queue_slot", -1))
        motion = apply_samples_to_motion(
            motion,
            traj_index.get(veh["id"], []),
            in_anchor[:2],
            road_meta=road_meta,
        )

        frames = []
        # In segment — rot_z is the env anchor heading (true vehicle heading;
        # equals approach_rotation for an unedited file).
        for f in range(max(motion.appear_frame, 0),
                       min(motion.disappear_frame, last_frame) + 1):
            p = G.sample_track(motion.track_in, f)
            if p is None:
                continue
            frames.append({
                "frame": f, "visible": True, "camera": in_cam_tag,
                "pose": {"x": round(p[0], 3), "y": round(p[1], 3), "z": 0.0,
                         "rot_z": round(in_rot_z, 4)},
            })
        # Out segment — rot_z is the env anchor heading for the exit lane.
        for f in range(max(motion.reappear_frame, 0), min(motion.leave_frame, last_frame) + 1):
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
# ffmpeg encoding — GPU-only NVENC
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


def _print_ffmpeg_stderr(encoder: str, rc: int, stderr: str):
    """Print the head + tail of an ffmpeg stderr dump for diagnosis.

    The full ffmpeg log can be thousands of lines of per-frame progress. The
    useful diagnostics are at the very top (NVENC init / format negotiation /
    "no capable devices found") and the very bottom (the final error summary
    + the encode-stats line). Print both, labelled, instead of either claiming
    "full stderr" while truncating or dumping the whole multi-thousand-line log.
    """
    lines = stderr.splitlines()
    print(f"  [ffmpeg] {encoder} failed (rc={rc}):", flush=True)
    # Short logs: print whole thing. Long logs (real ffmpeg failure dumps can
    # be thousands of per-frame lines): print the first + last 20 with an
    # omitted-count marker. The 40-line boundary avoids the head/tail overlap
    # that would otherwise duplicate lines and print a negative omit count.
    if len(lines) <= 40:
        for line in lines:
            print(f"    | {line}", flush=True)
    else:
        for line in lines[:20]:
            print(f"    | {line}", flush=True)
        print(f"    | ... ({len(lines) - 40} lines omitted) ...", flush=True)
        for line in lines[-20:]:
            print(f"    | {line}", flush=True)


def _ffmpeg_encode(frames_dir: str, video_path: str, fps: float,
                   timeout_s: int = 1800) -> bool:
    """Encode the PNG frame sequence to mp4 with GPU NVENC only.

    Inherits CUDA_VISIBLE_DEVICES from
    the parent process environment (set per-worker by run_pipeline.py), so
    NVENC binds to the same GPU this Blender instance rendered on.

    ``timeout_s`` bounds the encode (default 30 min) — ffmpeg can itself hang
    on NVENC init contention or a malformed PNG sequence, and without a
    timeout the worker silently stalls (the 8h-silent-hang scenario). On
    timeout the process is killed and full stderr is printed for diagnosis.
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

    def _run(cmd, tag):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout_s)
            return p
        except subprocess.TimeoutExpired:
            print(f"  [ffmpeg] {tag} encode timed out after {timeout_s}s "
                  f"— process killed.", flush=True)
            return None
        except Exception as e:
            print(f"  [ffmpeg] {tag} encode raised {type(e).__name__}: {e}",
                  flush=True)
            return None

    if not _nvenc_available():
        print("  [ffmpeg] h264_nvenc unavailable — GPU encoding required; "
              "no CPU fallback.", flush=True)
        return False
    cmd = base + [
        "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p",
        "-cq", "20", video_path,
    ]
    proc = _run(cmd, "NVENC")
    if proc is None:
        return False
    if proc.returncode != 0 and proc.stderr:
        _print_ffmpeg_stderr("NVENC", proc.returncode, proc.stderr)
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Rendering loop
# ---------------------------------------------------------------------------

# Module-level render-samples override. Set from --samples on the render.py
# CLI (when invoked by run_pipeline.py); falls back to build_scene's
# CYCLES_SAMPLES constant if None (preserves the default-48 behaviour when
# render.py is invoked directly without --samples).
RENDER_SAMPLES = None

# Per-frame progress: emit a progress line during the Cycles animation render
# (the single longest operation). Without this, headless Blender produces zero
# stdout during the render, which — combined with a block-buffered stdout under
# non-TTY — is the root cause of the multi-hour silent-hang scenario. The
# handler prints to stdout (flush=True) so the parent process's
# `for line in proc.stdout` gets a regular heartbeat and the watchdog knows we
# are alive.
#
# CRITICAL: the handler registers on `render_write` (fires after each frame's
# PNG is written to disk), NOT `frame_change_post`. `frame_change_post` does
# NOT fire reliably per-frame during `bpy.ops.render.render(animation=True)` in
# headless Blender — it only fires for UI/depsgraph frame changes. Using it
# produced exactly one line (`frame 0`) then silence for the entire 13,000+
# frame render, which the watchdog killed as a false-positive hang at 600s.
# `render_write` is guaranteed to fire once per written frame during animation
# render.
RENDER_PROGRESS_EVERY = 10  # print every Nth frame
HEARTBEAT_SECONDS = 30     # also print if >= this long since last print,
                           # so a slow frame still emits before the watchdog


def _install_render_progress_handler(scene):
    """Register a `render_write` callback that prints per-frame progress.

    Called once before bpy.ops.render.render(animation=True) in render_one.
    The handler is unregistered after the render to avoid leaking across
    multiple camera renders in a single Blender process.

    Fires on `render_write` (after each frame's PNG hits disk) — the canonical
    per-frame signal during animation render. Prints `frame N/M (elapsed Xs)`
    every RENDER_PROGRESS_EVERY frames, OR if HEARTBEAT_SECONDS or more have
    elapsed since the last print (wall-clock fallback so a slow-but-progressing
    frame can't silently trip the 600s watchdog). A genuinely stalled render
    writes nothing → the watchdog correctly kills it.
    """
    import bpy as _bpy
    try:
        _bpy.app.handlers.render_write.clear()
    except Exception:
        pass

    frame_end = _bpy.context.scene.frame_end
    t_start = time.time()
    last_print_t = [t_start]
    last_printed_frame = [-1]

    def _on_render_write(scene, *args):
        f = scene.frame_current
        now = time.time()
        elapsed = now - t_start
        since_last = now - last_print_t[0]
        # Print on the Nth-frame cadence, or the wall-clock heartbeat,
        # or always on the first/last frame. Avoid duplicate prints for the
        # same frame (render_write shouldn't double-fire, but be safe).
        due = (f % RENDER_PROGRESS_EVERY == 0
               or elapsed >= last_print_t[0] - t_start + HEARTBEAT_SECONDS
               or f == frame_end)
        if due and f != last_printed_frame[0]:
            print(f"  frame {f}/{frame_end} ({elapsed:.1f}s)", flush=True)
            last_print_t[0] = now
            last_printed_frame[0] = f

    _bpy.app.handlers.render_write.append(_on_render_write)


def render_one(scenario: dict, camera_tag: str, out_dir: str):
    """Build + render one camera shot to <out_dir>/video_<tag>.mp4.

    Renders a PNG frame sequence into a temp subdir, then encodes to mp4 with
    ffmpeg (Blender's built-in FFMPEG container can be finicky across builds).

    Emits explicit phase markers (D1) so each long step (buildshot → GPU →
    render → encode) is visible in the parent's streamed output, and a
    per-frame progress handler (C2) so the Cycles render itself is not a
    multi-minute silent gap.
    """
    import shutil
    import subprocess
    import build_scene as BS  # requires bpy (only available inside Blender)

    print(f"  [{camera_tag}] building scene...", flush=True)
    scene_blend = os.path.join(out_dir, f"scene_{camera_tag}.blend")
    BS.build_shot(scenario, camera_tag, scene_blend)

    # Re-configure GPU here too: Cycles addon prefs live in user preferences,
    # not the .blend, so they must be set in the active session before render.
    print(f"  [{camera_tag}] configuring GPU...", flush=True)
    BS.configure_gpu()
    # M9: pass --samples through to setup_render (which sets scene.cycles.samples).
    # Done here, AFTER configure_gpu, so the override is applied to the live
    # scene in one place rather than patched in after the fact.
    samples_override = RENDER_SAMPLES if RENDER_SAMPLES is not None else None
    BS.setup_render(samples=samples_override)
    if samples_override is not None:
        print(f"  [{camera_tag}] Cycles samples = {samples_override}", flush=True)

    scene = bpy.context.scene
    duration = scenario["duration_frames"]
    last_frame = max(0, duration - 1)
    scene.frame_start = 0
    scene.frame_end = last_frame

    # PNG sequence into a frames subdir
    frames_dir = os.path.join(out_dir, f"frames_{camera_tag}")
    os.makedirs(frames_dir, exist_ok=True)
    # clear any old frames
    for fn in os.listdir(frames_dir):
        if fn.endswith(".png"):
            os.remove(os.path.join(frames_dir, fn))
    scene.render.filepath = os.path.join(frames_dir, "f_")  # produces f_0001.png ...
    scene.render.image_settings.file_format = "PNG"

    # Install per-frame progress handler before the long render (C2).
    print(f"  [{camera_tag}] rendering frames 0..{last_frame} "
          f"({scene.cycles.samples} samples)...", flush=True)
    _install_render_progress_handler(scene)
    try:
        bpy.ops.render.render(animation=True)
    finally:
        # Unregister handler so the next render_one in the same Blender
        # process starts clean (we render one camera per Blender subprocess
        # via run_pipeline.py, but render.py:main loops all cameras when
        # invoked directly).
        try:
            bpy.app.handlers.render_write.clear()
        except Exception:
            pass

    # encode to mp4 with GPU NVENC only; fail fast instead of CPU fallback.
    print(f"  [{camera_tag}] encoding...", flush=True)
    video_path = os.path.join(out_dir, f"video_{camera_tag}.mp4")
    fps = scenario["fps"]
    ok = _ffmpeg_encode(frames_dir, video_path, fps)
    if ok:
        print(f"  [{camera_tag}] rendered: {video_path}", flush=True)
    else:
        print(f"  [{camera_tag}] ffmpeg FAILED", flush=True)
        raise RuntimeError(f"ffmpeg encode failed for {camera_tag}")
    # optionally clean up frames dir to save space — D6: warn on failure
    # (silent pass here hides NFS/permission issues that eat disk space).
    try:
        shutil.rmtree(frames_dir)
    except Exception as e:
        print(f"  [{camera_tag}] [WARN] rmtree frames_dir failed: {e}",
              flush=True)
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
    ap.add_argument("--samples", type=int, default=None,
                    help="Cycles render samples (default: build_scene.CYCLES_SAMPLES=48; "
                         "lower = faster, noisier — denoiser compensates. "
                         "Use 16-24 for quick test runs, 48 for production.")
    ns = ap.parse_args(post)

    # Apply --samples override to the module global so render_one picks it up.
    global RENDER_SAMPLES
    RENDER_SAMPLES = ns.samples

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
    failed = []
    for tag in cameras:
        try:
            p = render_one(scenario, tag, ns.out)
            rendered.append(p)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED {tag}: {e}", flush=True)
            failed.append(tag)

    if not ns.no_metadata:
        _write_metadata(scenario, ns.out)
    print(f"Rendered {len(rendered)}/{len(cameras)} videos")
    if failed:
        raise SystemExit(
            f"FAILED cameras ({len(failed)}/{len(cameras)}): "
            f"{', '.join(failed)}")


def _write_metadata(scenario, out_dir):
    """Write metadata.json — scans out_dir for existing video_*.mp4 files."""
    import glob
    videos = sorted(os.path.relpath(p, out_dir)
                    for p in glob.glob(os.path.join(out_dir, "video_*.mp4")))
    meta = compute_metadata(scenario, ROOT, run_dir=out_dir)
    meta["videos"] = videos
    complete_trajectory_with_metadata(scenario, out_dir, meta)
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
