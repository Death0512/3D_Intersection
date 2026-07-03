#!/usr/bin/env python3
"""Phase 5 — Pipeline driver.

Orchestrates the full dataset generation:
  [0/5] Validate env files       (conda/venv python: envfile)
  [1/5] (Optional) Validate assets  (blender headless: validate_assets.py)
  [2/5] Generate scenario         (conda/venv python: scenario_gen.py)
  [3/5] Pre-generate plate PNGs   (conda/venv python: gen_plate batch)
  [4/5] Render all 8 cameras      (blender headless: render.py, parallel)
  [5/5] Metadata + run validation (conda/venv python)

The scenario duration auto-extends to fit all vehicles (``--seconds`` is the
minimum floor).

Run (from the project root, with venv python):
    python3 scripts/run_pipeline.py --seed 42 --num-vehicles 10 --seconds 12.0 --out output/run1

Blender is invoked via subprocess; the python interpreter with Pillow is
resolved from the DOAN_PYTHON env var (set by scripts/env.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
_DOAN_PY = os.environ.get("DOAN_PYTHON")
PYTHON = _DOAN_PY if _DOAN_PY and os.path.exists(_DOAN_PY) else sys.executable
BLENDER = shutil.which("blender") or "blender"

sys.path.insert(0, os.path.join(HERE, "lib"))
import envfile as ENV
import geometry as G

# Per-job VRAM budget estimate (MiB).  Covers scene + BVH + OptiX denoiser +
# output buffer + headroom.  Conservative for the RTX 3050 Ti at 1080p/48 samples.
VRAM_PER_JOB_MIB = 1600

# Minimum free VRAM to attempt any GPU rendering (MiB).
MIN_FREE_VRAM_MIB = 1200


def run(cmd: list, cwd=ROOT, check=True):
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    dt = time.time() - t0
    if proc.stdout:
        for line in proc.stdout.splitlines()[-15:]:
            print("  ", line)
    if proc.returncode != 0 and check:
        print("STDERR:", proc.stderr[-1500:])
        raise SystemExit(f"Command failed (code {proc.returncode}) after {dt:.1f}s: {' '.join(cmd)}")
    print(f"  (ok, {dt:.1f}s)")
    return proc


def step_assets_validate():
    """Run asset validation (fast)."""
    run([BLENDER, "-b", "--python", os.path.join(HERE, "validate_assets.py")])


def step_scenario(seed, num_vehicles, seconds, out_dir, fps=None,
                   signal=False, signal_mode="fixed", demand=None):
    """Run scenario_gen.py.

    ``signal_mode`` is forwarded only when ``signal`` is True (mirrors
    scenario_gen.py, where --signal-mode is meaningful only together with
    --signal).  ``demand`` is forwarded as-is: ``None`` → default demand
    model, a path → custom JSON, the string ``"none"`` → legacy uniform
    scheduler.
    """
    cmd = [PYTHON, os.path.join(HERE, "scenario_gen.py"),
           "--seed", str(seed),
           "--num-vehicles", str(num_vehicles),
           "--seconds", str(seconds),
           "--out", out_dir]
    if fps is not None:
        cmd += ["--fps", str(fps)]
    if signal:
        cmd += ["--signal"]
        cmd += ["--signal-mode", str(signal_mode)]
    if demand is not None:
        cmd += ["--demand", str(demand)]
    run(cmd)
    return os.path.join(out_dir, "scenario.json")


def step_plates(scenario_path, out_dir):
    """Pre-generate all plate PNGs in conda python (Pillow available)."""
    plates_dir = os.path.join(out_dir, "plates")
    os.makedirs(plates_dir, exist_ok=True)
    script = (
        "import sys,json; sys.path.insert(0,'scripts'); "
        "from gen_plate import pregenerate_plates; "
        f"d=json.load(open({scenario_path!r})); "
        "pregenerate_plates([v['plate'] for v in d['vehicles']], "
        f"{plates_dir!r}); print('plates done')"
    )
    run([PYTHON, "-c", script])


def _gpu_count():
    """Return the number of NVIDIA GPUs, or 0 if nvidia-smi is unavailable."""
    return len(_gpu_info())


def _gpu_info():
    """Return a list of (index, free_mib) tuples for all GPUs, or empty list."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        result = []
        for line in out.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                idx = int(parts[0].strip())
                free = int(parts[1].strip())
                if free > 0:
                    result.append((idx, free))
        return result
    except Exception:
        return []


def _free_vram_mib():
    """Query free GPU VRAM from nvidia-smi.  Returns integer MiB or None.

    Uses the first GPU (index 0) for single-GPU compat — the multi-GPU path
    in _detect_jobs uses _gpu_info() instead.
    """
    info = _gpu_info()
    if not info:
        return None
    # Find GPU 0 specifically (backward compat with single-GPU query).
    for idx, free in info:
        if idx == 0:
            return free
    return info[0][1]  # fallback: first GPU, whatever its index


def _detect_jobs(camera_count: int, explicit: int = 0):
    """Determine how many parallel Blender jobs to run AND which GPU each binds to.

    Multi-GPU hosts (e.g. Kaggle T4×2): each GPU runs one Blender worker,
    giving near-linear throughput. Single-GPU: jobs based on VRAM budget
    (all bind to GPU 0). Returns (job_count, gpu_assignment) where
    gpu_assignment is a list of GPU-id integers, length == job_count.

    ``explicit`` > 0 overrides auto-detection — all workers share GPU 0
    (user-chosen parallelism, not per-GPU).
    """
    assignment: list[int] = []
    if explicit and explicit > 0:
        n = min(explicit, camera_count)
        return n, [0] * n

    n_gpu = _gpu_count()
    if n_gpu == 0:
        print("[GPU] nvidia-smi unavailable — defaulting to --jobs 1")
        return 1, [0]

    # Multi-GPU: pack multiple workers per GPU (VRAM-limited, no fixed cap),
    # interleaved so capping by camera_count spreads load evenly across GPUs.
    if n_gpu >= 2:
        info = _gpu_info()
        # Per-GPU worker slots based on VRAM budget.
        per_gpu_slots: list[tuple[int, int]] = []  # (gpu_id, n_slots)
        for gid, free in info:
            if free < MIN_FREE_VRAM_MIB:
                continue
            w = max(1, free // VRAM_PER_JOB_MIB)
            per_gpu_slots.append((gid, int(w)))
        if len(per_gpu_slots) < 2:
            print(f"[GPU] {len(info)} GPU(s) detected but only "
                  f"{len(per_gpu_slots)} meet the {MIN_FREE_VRAM_MIB} MiB "
                  f"VRAM floor → serial render")
            return 1, [0]
        # Interleave slots across GPUs: [g0, g1, g0, g1, ...] so truncating
        # to camera_count keeps both GPUs busy rather than stacking on GPU 0.
        max_slots = max(s for _, s in per_gpu_slots)
        interleaved: list[int] = []
        for slot in range(max_slots):
            for gid, n_slots in per_gpu_slots:
                if slot < n_slots:
                    interleaved.append(gid)
        n = min(len(interleaved), camera_count)
        assignment = interleaved[:n]
        # Banner summarising per-GPU worker counts.
        from collections import Counter
        per_gpu = Counter(assignment)
        hdr = " ".join(f"GPU{g}×{per_gpu.get(g, 0)}" for g, _ in per_gpu_slots)
        print(f"[GPU] {n_gpu} GPU(s), VRAM-limited → {n} parallel "
              f"render worker{'' if n==1 else 's'} ({hdr})")
        return n, assignment

    # Single GPU: VRAM-budget job count (original behaviour).
    free = _free_vram_mib()
    if free is None:
        print("[GPU] nvidia-smi unavailable — defaulting to --jobs 1")
        return 1, [0]
    jobs = max(1, free // VRAM_PER_JOB_MIB)
    jobs = min(jobs, camera_count)
    if free < MIN_FREE_VRAM_MIB:
        print(f"[GPU] free VRAM {free} MiB < {MIN_FREE_VRAM_MIB} — forcing --jobs 1")
        jobs = 1
    assignment = [0] * jobs
    print(f"[GPU] free VRAM {free} MiB → {jobs} parallel "
          f"render job{'' if jobs==1 else 's'} "
          f"(budget {VRAM_PER_JOB_MIB} MiB/job)")
    return jobs, assignment


def _render_worker(args):
    """Run one Blender render worker for a single camera. Blocks.

    ``args`` is (scenario_path, out_dir, tag, gpu_id).  ``gpu_id`` is the
    NVIDIA GPU index this worker should bind to via CUDA_VISIBLE_DEVICES,
    so multi-GPU hosts run one Blender per physical GPU.

    Output is streamed in real-time (prefixed by [tag]) so the user can see
    progress during long renders.  Returns (camera_tag, success, wall_s).
    Metadata is deferred to a separate pass.
    """
    import sys as _sys

    scenario_path, out_dir, tag, gpu_id = args
    cmd = [
        BLENDER, "-b", "--python", os.path.join(HERE, "render.py"), "--",
        "--scenario", scenario_path, "--out", out_dir,
        "--only", tag, "--no-metadata",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tag_label = f"[{tag}]"
    print(f"\n{tag_label} $ {' '.join(cmd)}  (GPU {gpu_id})", flush=True)
    t0 = time.time()
    ok = True
    # Stream output in real-time so long renders show progress.
    with subprocess.Popen(
        cmd, cwd=ROOT, text=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1,
    ) as proc:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                # Suppress noisy warnings and frame save messages
                if "HIPEW initialization failed" in line or "Saved:" in line:
                    continue
                _sys.stdout.write(f"  {tag_label} {line}\n")
                _sys.stdout.flush()
        proc.wait()
        dt = time.time() - t0
        ok = proc.returncode == 0
    status = "OK" if ok else "FAILED"
    print(f"{tag_label} {status} ({dt:.1f}s)", flush=True)
    return tag, ok, dt


def step_render_parallel(scenario_path, out_dir, jobs=2, gpu_assignment=None,
                         only=None):
    """Render all 8 (or ``only``) cameras. If ``gpu_assignment`` is set, each
    worker binds to a different GPU via CUDA_VISIBLE_DEVICES (round-robins
    when there are more cameras than GPUs). Otherwise all workers share GPU 0.
    """
    camera_tags = G.camera_names()
    if only:
        camera_tags = [only]
    n_cams = len(camera_tags)
    n_gpus = len(gpu_assignment) if gpu_assignment else 0

    # gpu_assignment is the exact per-worker GPU id list (length == jobs);
    # pair each camera tag with its GPU id in order. If we have more cameras
    # than assignments, fall back to GPU 0 for the overflow.
    tasks = [(scenario_path, out_dir, tag,
              gpu_assignment[i] if i < n_gpus else 0)
             for i, tag in enumerate(camera_tags)]
    results = {}

    # Serial fallback for 1 job or 1 camera — avoid thread overhead.
    if jobs <= 1 or n_cams == 1:
        for args in tasks:
            tag, ok, dt = _render_worker(args)
            results[tag] = (ok, dt)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_render_worker, t): i for i, t in enumerate(tasks)}
            for fut in as_completed(futs):
                tag, ok, dt = fut.result()
                results[tag] = (ok, dt)

    print()
    print(f"  Render summary ({len(results)}/{n_cams} cameras):")
    for tag in camera_tags:
        ok, dt = results.get(tag, (False, 0))
        print(f"    {tag:6s} {'OK' if ok else 'FAILED'}  ({dt:.1f}s)")
    failed = sum(1 for ok, _ in results.values() if not ok)
    if failed:
        print(f"  {failed} camera(s) FAILED — metadata will be partial")


def step_metadata(scenario_path, out_dir):
    cmd = [PYTHON, os.path.join(HERE, "render.py"), "--",
           "--scenario", scenario_path, "--out", out_dir,
           "--metadata-only"]
    run(cmd, check=False)


def step_validate_run(out_dir):
    run([PYTHON, os.path.join(HERE, "validate_run.py"), "--out", out_dir], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-vehicles", type=int, default=10)
    ap.add_argument("--fps", type=int, default=None,
                    help="frames per second (default: geometry.FPS)")
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="minimum video length in seconds (actual duration auto-extends "
                         "to fit all vehicles)")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "output", "run1"))
    ap.add_argument("--only", help="render only this camera (debug)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel render workers (0 = auto-detect from free VRAM)")
    ap.add_argument("--skip-asset-check", action="store_true")
    ap.add_argument("--signal", action="store_true",
                    help="enable traffic signal SPaT gating + queue")
    ap.add_argument("--signal-mode", type=str, default="fixed",
                    choices=["fixed", "adaptive"],
                    help="signal controller type when --signal is set: "
                         "'fixed' (default, 70s cycle permissive-left) or "
                         "'adaptive' (NEMA 8-phase MaxPressure, closed-loop "
                         "on realised arrivals)")
    ap.add_argument("--demand", type=str, default=None,
                    help="path to a demand JSON (per-approach flow veh/h + "
                         "turning split). When omitted, the default demand "
                         "model is used. Pass 'none' to disable demand and "
                         "use the legacy uniform-random scheduler.")
    args = ap.parse_args()

    fps = args.fps
    seconds = args.seconds
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    fps_label = f"{fps} fps" if fps else "default fps"
    print("=" * 60)
    print(f"PIPELINE  seed={args.seed} n={args.num_vehicles}  "
          f"min={seconds}s ({fps_label})  out={out_dir}")
    print(f"  python : {PYTHON}")
    print(f"  blender: {BLENDER}")
    print("=" * 60)

    ENV.validate_all_envs(ROOT)
    print("[0/5] Env files OK")

    if not args.skip_asset_check:
        print("\n[1/5] Asset validation")
        step_assets_validate()

    print("\n[2/5] Scenario generation")
    scn = step_scenario(args.seed, args.num_vehicles, seconds, out_dir, fps=fps,
                        signal=args.signal, signal_mode=args.signal_mode,
                        demand=args.demand)

    print("\n[3/5] Plate pre-generation")
    step_plates(scn, out_dir)

    print("\n[4/5] Render cameras (parallel)")
    n_jobs, gpu_assign = _detect_jobs(8 if not args.only else 1, explicit=args.jobs)
    step_render_parallel(scn, out_dir, jobs=n_jobs, gpu_assignment=gpu_assign,
                         only=args.only)

    print("\n[5/5] Metadata + run validation")
    step_metadata(scn, out_dir)
    step_validate_run(out_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  output dir: {out_dir}")
    print(f"  metadata: {os.path.join(out_dir, 'metadata.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
