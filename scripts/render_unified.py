#!/usr/bin/env python3
"""Render all cameras from one unified Blender scene in parallel with
GPU round-robin assignment. Each Blender camera subprocess renders,
batch-encodes segments (CBR, -fs bounded), concatenates, and validates its
own final MP4."""
from __future__ import annotations

import argparse
import concurrent.futures
import math
import itertools
import json
import os
import shutil
import subprocess
import sys
import traceback
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "lib"))

import geometry as G

BLENDER = (os.environ.get("DOAN_BLENDER") or os.environ.get("BLENDER")
           or shutil.which("blender") or "/root/.local/bin/blender" or "blender")

_GIB = 1_073_741_824


def _camera_tags(only: str | None) -> list[str]:
    if not only:
        return G.camera_names()
    tags = [t.strip() for t in only.split(",") if t.strip()]
    valid = set(G.camera_names())
    bad = [t for t in tags if t not in valid]
    if bad:
        raise SystemExit(f"FAIL: invalid camera tag(s): {', '.join(bad)}")
    return tags


def _gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    if out.returncode != 0:
        return 1  # ponytail: driver loaded but query failed → assume 1
    lines = [l for l in out.stdout.splitlines() if l.strip().startswith("GPU")]
    return max(1, len(lines))


def _run_one_camera(scene, scenario, out_dir, tag, gpu_id, samples,
                     batch_size, frame_reuse, bitrate: str,
                     storage_cap_bytes: int,
                     segment_limit_bytes: int,
                     concat_limit_bytes: int) -> str:
    """Run one Blender camera subprocess on specified GPU.
    Returns the camera tag on success, raises on failure."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    # WSL2: NVIDIA GPU accessible via /usr/lib/wsl/lib; Snap isolates env so inject here
    wsl_lib = "/usr/lib/wsl/lib"
    if os.path.isdir(wsl_lib):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{wsl_lib}:{existing}" if existing else wsl_lib
    blender_args = [
        "-b", "--python",
        os.path.join(ROOT, "scripts", "render_unified_camera.py"), "--",
        "--scene", scene, "--camera", tag, "--out", out_dir,
        "--samples", str(samples), "--scenario", scenario,
        "--batch-size", str(batch_size), "--bitrate", bitrate,
        "--storage-cap-bytes", str(storage_cap_bytes),
        "--segment-limit-bytes", str(segment_limit_bytes),
        "--concat-limit-bytes", str(concat_limit_bytes),
    ]
    cmd = [BLENDER] + blender_args
    if not frame_reuse:
        cmd.append("--no-frame-reuse")
    print(f"[unified:{tag}] $ {' '.join(cmd)} (GPU {gpu_id})", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line and "Saved:" not in line:
            print(f"  [unified:{tag}] {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"render failed for {tag} (rc={proc.returncode})")
    return tag


def _dir_usage_bytes(path: str) -> int:
    """Total bytes in path via du -sb; fallback to walk."""
    p = subprocess.run(["du", "-sb", path],
                       capture_output=True, text=True, timeout=30)
    if p.returncode == 0:
        try:
            return int(p.stdout.strip().split()[0])
        except (ValueError, IndexError):
            pass
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return total


def _existing_usage(out_dir: str) -> int:
    """Bytes already consumed inside out_dir (0 if doesn't exist yet)."""
    if not os.path.isdir(out_dir):
        return 0
    return _dir_usage_bytes(out_dir)


def _bitrate_kbps(total_bytes: int, duration_s: float) -> str:
    """Return h264_nvenc bitrate as kilobits/s string from byte budget."""
    if duration_s <= 0:
        duration_s = 1.0
    # ponytail: leave room below the 85% per-segment hard ceiling for muxing.
    bps = int(total_bytes * 8 * 0.75 / duration_s)
    kbps = max(50, min(50_000, bps // 1_000))  # h264_nvenc: H.264 level 5.2 max ~240 Mbps; cap at 50 Mbps for 1080p
    return f"{kbps}k"


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
    ap.add_argument("--batch-size", type=int, default=1000,
                    help="frames per batch segment (default 1000)")
    ap.add_argument("--storage-limit-gib", type=int, default=50,
                    help="hard storage cap for the entire output dir in GiB (default 50)")
    ns = ap.parse_args()

    if ns.storage_limit_gib <= 0:
        raise SystemExit("--storage-limit-gib must be positive")

    with open(ns.scenario) as f:
        scenario = json.load(f)

    fps = int(scenario.get("fps", 30) or 30)
    duration_frames = int(scenario.get("duration_frames", 0) or 0)
    if duration_frames <= 0:
        raise SystemExit(f"scenario {ns.scenario} has missing/zero duration_frames")
    duration_s = duration_frames / max(fps, 1)

    tags = _camera_tags(ns.only)
    n_cameras = len(tags)
    os.makedirs(ns.out, exist_ok=True)

    phys_gpus = _gpu_count()
    jobs = max(1, min(int(ns.jobs), n_cameras))
    frame_reuse = not ns.no_frame_reuse
    batch_size = max(1, int(ns.batch_size))
    total_batches = max(1, math.ceil(duration_frames / batch_size))

    # ---- Storage preflight ----
    # Invariant: total_limit = existing + reserve + camera_media * (n_cameras + jobs)
    # where n_cameras persistent segments/finals + jobs transient concat copies
    # must all fit below the cap. Segment/ffmpeg filesystem overhead is absorbed
    # by the 5% bitrate headroom and -fs bounds below.
    total_limit = ns.storage_limit_gib * _GIB
    existing = _existing_usage(ns.out)

    # Conservative JPEG peak: 1080p RGB raw-size bound (worst-case)
    # ponytail: 8 MiB/frame is the absolute ceiling for a 1920×1080×3 RRB buffer;
    # actual JPEG quality 95 is ~250-500 KiB but we use the raw bound as guarantee.
    reserve = 2 * _GIB + jobs * batch_size * 8 * 1024 * 1024
    remaining = total_limit - existing - reserve
    if remaining <= (n_cameras + jobs) * 10_000_000:
        # Floor: each camera needs at least ~10 MB for a tiny final video
        raise SystemExit(
            f"Storage preflight FAILED for {ns.out}:\n"
            f"  limit   = {total_limit >> 20} MiB ({ns.storage_limit_gib} GiB)\n"
            f"  existing= {existing >> 20} MiB\n"
            f"  reserve = {reserve >> 20} MiB "
            f"(2 GiB base + {jobs}×{batch_size}×8 MiB peak JPEGs)\n"
            f"  remaining = {remaining >> 20} MiB for {n_cameras} cameras + "
            f"{jobs} transient concat copies\n"
            f"  → per-camera media < 10 MiB — cannot encode video."
        )

    # Static pre-partition: each camera owns a persistent segments+final budget;
    # `jobs` simultaneous concat outputs at peak also need headroom.
    # camera_media_budget covers all segment MP4s AND the final video
    # (they coexist briefly before cleanup deletes segments).
    camera_media_budget = remaining // (n_cameras + jobs)

    # per-segment -fs: ceil(budget/total_batches) with 15% headroom so CBR +
    # muxer overshoot cannot burst past the alloc
    segment_limit_bytes = max(500_000, int(
        math.ceil(camera_media_budget / total_batches) * 0.85))

    # Concat limit: bound the final MP4 to camera_media_budget (-fs defense-in-depth)
    concat_limit_bytes = camera_media_budget

    # CBR bitrate from shared camera persistent budget
    per_camera_bitrate = _bitrate_kbps(camera_media_budget, duration_s)

    # Per-camera scoped backstop allows its transient segments + final concat.
    # Global pre-partition above still bounds all camera peaks below total_limit.
    per_camera_cap_bytes = 2 * camera_media_budget + batch_size * 8 * 1024 * 1024

    free = shutil.disk_usage(ns.out).free
    if free < reserve:
        raise SystemExit(
            f"Not enough disk space on {ns.out}: "
            f"{free >> 20} MiB free, need >= {reserve >> 20} MiB "
            f"(reserved: 2 GiB + {jobs}×{batch_size}×8 MiB peak JPEGs)")

    print(f"[unified] requested --jobs={ns.jobs}, "
          f"detected {phys_gpus} nvidia GPU{'s' if phys_gpus != 1 else ''}, "
          f"effective jobs={jobs}, batch-size={batch_size}", flush=True)
    print(f"[unified] storage: limit={ns.storage_limit_gib} GiB, "
          f"existing={existing >> 20} MiB, reserve={reserve >> 20} MiB, "
          f"remaining={remaining >> 20} MiB, "
          f"cam_media={camera_media_budget >> 20} MiB, "
          f"seg_limit={segment_limit_bytes >> 10} KiB, "
          f"bitrate={per_camera_bitrate} "
          f"({duration_frames}f @{fps}fps = {duration_s:.0f}s, "
          f"{total_batches} batches)",
          flush=True)

    # GPU round-robin assignment: cycle through detected GPUs
    gpu_cycle = itertools.cycle(
        range(phys_gpus) if phys_gpus > 0 else [0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {}
        for tag in tags:
            gpu_id = next(gpu_cycle)
            f = ex.submit(_run_one_camera, ns.scene, ns.scenario, ns.out,
                          tag, gpu_id, ns.samples, batch_size, frame_reuse,
                          per_camera_bitrate, per_camera_cap_bytes,
                          segment_limit_bytes, concat_limit_bytes)
            futures[f] = (tag, gpu_id)

        for f in concurrent.futures.as_completed(futures):
            tag, gpu_id = futures[f]
            try:
                result = f.result()
                print(f"[unified:{result}] DONE (GPU {gpu_id})", flush=True)
            except Exception as e:
                # Cancel remaining queued work on first failure
                for remaining in futures:
                    remaining.cancel()
                print(f"[unified:{tag}] FAILED: {e}", flush=True)
                traceback.print_exc()
                raise RuntimeError(f"render failure for {tag}: {e}") from e

    print(f"[unified] rendered {n_cameras} camera(s)", flush=True)


if __name__ == "__main__":
    main()
