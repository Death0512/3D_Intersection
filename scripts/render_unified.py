#!/usr/bin/env python3
"""Render all cameras from time-chunk .blend files sequentially, one MP4
segment per camera per chunk, then concatenate per-camera segments into
final videos. Supports GPU round-robin across cameras within each chunk."""
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

CHUNK_SIZE = 500  # global frames per chunk


def _stream_scenario_meta(path: str) -> tuple[int, int]:
    """Stream-parse only fps, duration_frames from scenario JSON."""
    import ijson
    with open(path, "rb") as f:
        fps = 30
        duration_frames = 0
        for prefix, event, value in ijson.parse(f, use_float=True):
            if prefix == "fps" and event == "number":
                fps = int(value)
            elif prefix == "duration_frames" and event == "number":
                duration_frames = int(value)
            elif prefix.startswith("vehicles."):
                break
    return fps, duration_frames


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
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    if out.returncode != 0:
        return 1
    uuids = {l.strip() for l in out.stdout.splitlines() if l.strip()}
    return max(1, len(uuids))


def _chunk_path(chunks_dir: str, idx: int) -> str:
    return os.path.join(chunks_dir, f"chunk_{idx:04d}.blend")


def _chunk_exists(chunks_dir: str, idx: int) -> bool:
    return os.path.isfile(_chunk_path(chunks_dir, idx))


def _run_one_camera_chunk(scene_path, out_dir, tag, chunk_idx, gpu_id, samples,
                           chunk_frames, bitrate: str,
                           storage_cap_bytes: int,
                           segment_limit_bytes: int,
                           scenario_path: str = "") -> str:
    """Run one Blender camera subprocess for ONE chunk → ONE segment."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    wsl_lib = "/usr/lib/wsl/lib"
    if os.path.isdir(wsl_lib):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{wsl_lib}:{existing}" if existing else wsl_lib

    blender_args = [
        "-b", "--python",
        os.path.join(ROOT, "scripts", "render_unified_camera.py"), "--",
        "--scene", scene_path, "--camera", tag, "--out", out_dir,
        "--samples", str(samples),
        "--batch-size", str(chunk_frames),
        "--bitrate", bitrate,
        "--storage-cap-bytes", str(storage_cap_bytes),
        "--segment-limit-bytes", str(segment_limit_bytes),
        "--concat-limit-bytes", "0",
        "--chunk-idx", str(chunk_idx),
    ]
    if scenario_path:
        blender_args += ["--scenario", scenario_path]
    cmd = [BLENDER] + blender_args
    print(f"[unified.chunk{chunk_idx}:{tag}] $ {' '.join(cmd)} (GPU {gpu_id})", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line and "Saved:" not in line:
            print(f"  [unified.chunk{chunk_idx}:{tag}] {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"render failed for {tag} chunk {chunk_idx} (rc={proc.returncode})")
    return tag


def _dir_usage_bytes(path: str) -> int:
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
    if not os.path.isdir(out_dir):
        return 0
    return _dir_usage_bytes(out_dir)


def _bitrate_kbps(total_bytes: int, duration_s: float) -> str:
    if duration_s <= 0:
        duration_s = 1.0
    bps = int(total_bytes * 8 * 0.75 / duration_s)
    kbps = max(50, min(50_000, bps // 1_000))
    return f"{kbps}k"


def _video_valid(video_path: str, expected_fps: int,
                 expected_frames: int, frame_tol: int = 1) -> bool:
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


def _segment_valid(seg_path: str, expected_fps: int,
                   expected_frames: int) -> bool:
    """Check if a segment MP4 exists and matches expected frame count."""
    return _video_valid(seg_path, expected_fps, expected_frames)


def _concat_segments(segments: list[str], video_path: str,
                     concat_limit_bytes: int = 0) -> str:
    concat_list = os.path.join(
        os.path.dirname(video_path), f"_{os.path.basename(video_path)}.concat.txt")
    with open(concat_list, "w") as f:
        for seg in segments:
            f.write(f"file '{os.path.relpath(seg, os.path.dirname(video_path))}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", concat_list, "-c", "copy"]
    if concat_limit_bytes > 0:
        cmd += ["-fs", str(concat_limit_bytes)]
    cmd.append(video_path)
    print(f"  [encode] concat {len(segments)} segments -> {video_path}: {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=600)
    if p.returncode != 0:
        tail = "\n".join((p.stderr or "").splitlines()[-30:])
        raise RuntimeError(f"ffmpeg concat failed for {video_path}:\n{tail}")
    return concat_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True,
                    help="chunks directory (directory, NOT a single .blend)")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--only", default=None)
    ap.add_argument("--batch-size", type=int, default=1000,
                    help="frames per chunk (default 1000)")
    ap.add_argument("--storage-limit-gib", type=int, default=50)
    ap.add_argument("--segment-limit-bytes", type=int, default=0,
                    help="ffmpeg -fs per segment (auto-computed if 0)")
    ns = ap.parse_args()

    if ns.storage_limit_gib <= 0:
        raise SystemExit("--storage-limit-gib must be positive")

    chunk_size = max(1, int(ns.batch_size))

    fps, duration_frames = _stream_scenario_meta(ns.scenario)
    if duration_frames <= 0:
        raise SystemExit(f"scenario {ns.scenario} has missing/zero duration_frames")
    duration_s = duration_frames / max(fps, 1)

    tags = _camera_tags(ns.only)
    n_cameras = len(tags)
    os.makedirs(ns.out, exist_ok=True)

    # A completed final video is authoritative: do not require chunk files or
    # re-render its already-concatenated camera stream on a resumed run.
    pending_tags = [
        tag for tag in tags
        if not _video_valid(os.path.join(ns.out, f"video_{tag}.mp4"),
                            fps, duration_frames)
    ]
    if not pending_tags:
        print(f"[unified] all {n_cameras} final video(s) already valid — skipping render",
              flush=True)
        return

    chunks_dir = ns.scene  # --scene points to chunks/ directory
    total_chunks = max(1, math.ceil(duration_frames / chunk_size))

    phys_gpus = _gpu_count()
    jobs = max(1, min(int(ns.jobs), len(pending_tags)))

    # ---- Storage preflight ----
    total_limit = ns.storage_limit_gib * _GIB
    existing = _existing_usage(ns.out)
    _JPEG_BYTES_PER_FRAME = 512 * 1024
    reserve = 2 * _GIB + jobs * chunk_size * _JPEG_BYTES_PER_FRAME
    remaining = total_limit - existing - reserve
    if remaining <= (n_cameras + jobs) * 10_000_000:
        raise SystemExit(
            f"Storage preflight FAILED for {ns.out}:\n"
            f"  limit   = {total_limit >> 20} MiB ({ns.storage_limit_gib} GiB)\n"
            f"  existing= {existing >> 20} MiB\n"
            f"  reserve = {reserve >> 20} MiB\n"
            f"  remaining = {remaining >> 20} MiB for {n_cameras} cameras + "
            f"{jobs} transient concat copies\n"
            f"  → per-camera media < 10 MiB — cannot encode video."
        )

    camera_media_budget = remaining // (n_cameras + jobs)
    segment_limit_bytes = ns.segment_limit_bytes if ns.segment_limit_bytes > 0 else max(500_000,
        int(math.ceil(camera_media_budget / total_chunks) * 0.85))
    concat_limit_bytes = camera_media_budget
    per_camera_bitrate = _bitrate_kbps(camera_media_budget, duration_s)
    per_camera_cap_bytes = 2 * camera_media_budget + chunk_size * _JPEG_BYTES_PER_FRAME

    free = shutil.disk_usage(ns.out).free
    if free < reserve:
        raise SystemExit(
            f"Not enough disk space on {ns.out}: "
            f"{free >> 20} MiB free, need >= {reserve >> 20} MiB")

    print(f"[unified] chunks={total_chunks}, cameras={n_cameras}, "
          f"effective jobs={jobs}, chunk-size={chunk_size}", flush=True)
    print(f"[unified] storage: limit={ns.storage_limit_gib} GiB, "
          f"existing={existing >> 20} MiB, reserve={reserve >> 20} MiB, "
          f"cam_media={camera_media_budget >> 20} MiB, "
          f"seg_limit={segment_limit_bytes >> 10} KiB, "
          f"bitrate={per_camera_bitrate} "
          f"({duration_frames}f@{fps}fps = {duration_s:.0f}s)",
          flush=True)

    gpu_cycle = itertools.cycle(range(phys_gpus) if phys_gpus > 0 else [0])

    # ---- Render each chunk sequentially ----
    for ci in range(total_chunks):
        c_start = ci * chunk_size
        c_end = min(duration_frames - 1, (ci + 1) * chunk_size - 1)
        chunk_frames = c_end - c_start + 1
        chunk_blend = _chunk_path(chunks_dir, ci)

        if not _chunk_exists(chunks_dir, ci):
            raise SystemExit(
                f"FAIL: chunk blend not found: {chunk_blend}\n"
                f"  Run the build phase first: "
                f"blender -b --python scripts/build_unified_scene.py -- "
                f"--chunk-start {c_start} --chunk-end {c_end} "
                f"--out {chunk_blend} --scenario {ns.scenario} --only ...")

        # Show which cameras already have a valid segment for this chunk
        done_tags = set()
        for tag in pending_tags:
            seg_path = os.path.join(ns.out, f"segments_{tag}", f"seg_{ci:04d}.mp4")
            if _segment_valid(seg_path, fps, chunk_frames):
                done_tags.add(tag)
        remaining_tags = [t for t in pending_tags if t not in done_tags]

        if not remaining_tags:
            print(f"[unified] chunk {ci}/{total_chunks - 1} [{c_start},{c_end}] "
                  f"all cameras already complete — skipping", flush=True)
            continue

        if done_tags:
            print(f"[unified] chunk {ci}/{total_chunks - 1} [{c_start},{c_end}] "
                  f"resuming: {len(remaining_tags)}/{len(pending_tags)} cameras remaining "
                  f"(done: {', '.join(sorted(done_tags))})", flush=True)
        else:
            print(f"[unified] chunk {ci}/{total_chunks - 1} [{c_start},{c_end}] "
                  f"rendering {len(remaining_tags)} camera(s)", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {}
            for tag in remaining_tags:
                gpu_id = next(gpu_cycle)
                f_idx = ex.submit(_run_one_camera_chunk,
                                   chunk_blend, ns.out, tag, ci, gpu_id,
                                   ns.samples, chunk_frames,
                                   per_camera_bitrate, per_camera_cap_bytes,
                                   segment_limit_bytes,
                                   ns.scenario)
                futures[f_idx] = (tag, gpu_id)

            for f in concurrent.futures.as_completed(futures):
                tag, gpu_id = futures[f]
                try:
                    result = f.result()
                    print(f"[unified.chunk{ci}:{result}] DONE (GPU {gpu_id})", flush=True)
                except Exception as e:
                    for remaining in futures:
                        remaining.cancel()
                    print(f"[unified.chunk{ci}:{tag}] FAILED: {e}", flush=True)
                    traceback.print_exc()
                    raise RuntimeError(f"render failure chunk {ci} {tag}: {e}") from e

    # ---- Concatenate per-camera segments into final videos ----
    print("\n[unified] concat phase: merging per-camera segments...", flush=True)
    for tag in tags:
        segments_dir = os.path.join(ns.out, f"segments_{tag}")
        video_path = os.path.join(ns.out, f"video_{tag}.mp4")

        # Resume: skip if final video already valid
        if os.path.isfile(video_path) and _video_valid(video_path, fps, duration_frames):
            print(f"[unified:{tag}] SKIP: valid final video exists ({duration_frames}f@{fps}fps)",
                  flush=True)
            continue

        segs = []
        for ci in range(total_chunks):
            sp = os.path.join(segments_dir, f"seg_{ci:04d}.mp4")
            if not os.path.isfile(sp):
                raise SystemExit(f"FAIL: missing segment for {tag} chunk {ci}: {sp}")
            segs.append(sp)

        concat_list = _concat_segments(segs, video_path, concat_limit_bytes)

        # Validate final video
        if not _video_valid(video_path, fps, duration_frames):
            raise RuntimeError(
                f"final video validation failed: {video_path} "
                f"(expected {duration_frames}f@{fps}fps)")

        os.remove(concat_list)
        print(f"[unified:{tag}] final video ready: {video_path}", flush=True)

    print(f"[unified] rendered {n_cameras} camera(s), {total_chunks} chunks", flush=True)


if __name__ == "__main__":
    main()
