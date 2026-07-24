#!/usr/bin/env python3
"""Render all cameras from one unified Blender scene and encode MP4s."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "lib"))

import geometry as G

BLENDER = (os.environ.get("DOAN_BLENDER") or os.environ.get("BLENDER")
             or shutil.which("blender") or "/root/.local/bin/blender" or "blender")


def _camera_tags(only: str | None) -> list[str]:
    if not only:
        return G.camera_names()
    tags = [t.strip() for t in only.split(",") if t.strip()]
    valid = set(G.camera_names())
    bad = [t for t in tags if t not in valid]
    if bad:
        raise SystemExit(f"FAIL: invalid camera tag(s): {', '.join(bad)}")
    return tags


def _run_render(scene, scenario, out_dir, tag, gpu_id, samples, frame_reuse=True):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [BLENDER, "-b", "--python", os.path.join(HERE, "render_unified_camera.py"), "--",
           "--scene", scene, "--camera", tag, "--out", out_dir,
           "--samples", str(samples), "--scenario", scenario]
    if not frame_reuse:
        cmd.append("--no-frame-reuse")
    print(f"[unified:{tag}] $ {' '.join(cmd)} (GPU {gpu_id})", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line and "Saved:" not in line:
            print(f"  [unified:{tag}] {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"render failed for {tag} (rc={proc.returncode})")
    return tag


def _encode(frames_dir: str, video_path: str, fps: int) -> None:
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-start_number", "0",
           "-i", os.path.join(frames_dir, "f_%04d.jpg"),
           "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-cq", "20", video_path]
    print(f"[encode] {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        tail = "\n".join((p.stderr or "").splitlines()[-30:])
        raise RuntimeError(f"ffmpeg failed for {video_path}:\n{tail}")


def _video_valid(video_path: str, expected_fps: int,
                 expected_frames: int, frame_tol: int = 1) -> bool:
    """Return True if video exists and matches expected fps/frame count."""
    if not os.path.isfile(video_path):
        return False
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_frames,r_frame_rate",
                            "-of", "csv=p=0", video_path],
                           capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if p.returncode != 0 or not p.stdout.strip():
        return False
    parts = p.stdout.strip().split(",")
    if len(parts) < 2:
        return False
    fps_str, frames_str = parts[0], parts[-1]
    # parse rational fps e.g. "30/1" or "30000/1001"
    try:
        num, den = fps_str.split("/", 1)
        actual_fps = float(int(num)) / float(int(den))
    except (ValueError, ZeroDivisionError):
        return False
    try:
        actual_frames = int(frames_str)
    except ValueError:
        return False
    if abs(actual_fps - float(expected_fps)) > 0.51:
        return False
    if abs(actual_frames - expected_frames) > frame_tol:
        return False
    return True


def _gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    except FileNotFoundError:
        return 0
    if out.returncode != 0:
        return 1  # ponytail: driver loaded but query failed → assume 1
    lines = [l for l in out.stdout.splitlines() if l.strip().startswith("GPU")]
    return max(1, len(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-frame-reuse", action="store_true",
                    help="render every frame instead of copying unchanged frames")
    ns = ap.parse_args()
    with open(ns.scenario) as f:
        scenario = json.load(f)
    fps = int(scenario.get("fps", 30) or 30)
    tags = _camera_tags(ns.only)
    os.makedirs(ns.out, exist_ok=True)
    phys_gpus = _gpu_count()
    jobs = max(1, min(int(ns.jobs), len(tags), phys_gpus if phys_gpus > 0 else 1))
    print(f"[unified] requested --jobs={ns.jobs}, "
          f"detected {phys_gpus} nvidia GPU{'s' if phys_gpus != 1 else ''}, "
          f"effective jobs={jobs}", flush=True)
    duration_frames = int(scenario.get("duration_frames", 0) or 0)
    for i, tag in enumerate(tags):
        video = os.path.join(ns.out, f"video_{tag}.mp4")
        if duration_frames > 0 and _video_valid(video, fps, duration_frames):
            print(f"[unified:{tag}] SKIP: valid video exists ({duration_frames}f@{fps}fps)", flush=True)
            frames = os.path.join(ns.out, f"frames_{tag}")
            shutil.rmtree(frames, ignore_errors=True)
            continue
        _run_render(ns.scene, ns.scenario, ns.out, tag, i % jobs, ns.samples,
                    frame_reuse=not ns.no_frame_reuse)
        frames = os.path.join(ns.out, f"frames_{tag}")
        _encode(frames, video, fps)
        shutil.rmtree(frames, ignore_errors=True)
    print(f"[unified] rendered+encoded {len(tags)} camera(s)", flush=True)


if __name__ == "__main__":
    main()
