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

BLENDER = shutil.which("blender") or "blender"


def _run_render(scene, out_dir, tag, gpu_id, samples):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [BLENDER, "-b", "--python", os.path.join(HERE, "render_unified_camera.py"), "--",
           "--scene", scene, "--camera", tag, "--out", out_dir, "--samples", str(samples)]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--only", default=None)
    ns = ap.parse_args()
    with open(ns.scenario) as f:
        scenario = json.load(f)
    fps = int(scenario.get("fps", 30) or 30)
    tags = [ns.only] if ns.only else G.camera_names()
    os.makedirs(ns.out, exist_ok=True)
    jobs = max(1, min(int(ns.jobs), len(tags)))
    for i, tag in enumerate(tags):
        _run_render(ns.scene, ns.out, tag, i % jobs, ns.samples)
        frames = os.path.join(ns.out, f"frames_{tag}")
        video = os.path.join(ns.out, f"video_{tag}.mp4")
        _encode(frames, video, fps)
        shutil.rmtree(frames, ignore_errors=True)
    print(f"[unified] rendered+encoded {len(tags)} camera(s)", flush=True)


if __name__ == "__main__":
    main()
