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


def step_scenario(seed, num_vehicles, seconds, out_dir, fps=None, signal=False):
    cmd = [PYTHON, os.path.join(HERE, "scenario_gen.py"),
           "--seed", str(seed),
           "--num-vehicles", str(num_vehicles),
           "--seconds", str(seconds),
           "--out", out_dir]
    if fps is not None:
        cmd += ["--fps", str(fps)]
    if signal:
        cmd += ["--signal"]
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


def _free_vram_mib():
    """Query free GPU VRAM from nvidia-smi.  Returns integer MiB or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        val = int(out.stdout.strip())
        return val if val > 0 else None
    except Exception:
        return None


def _detect_jobs(camera_count: int, explicit: int = 0):
    """Determine how many parallel Blender processes to run.

    ``explicit`` > 0 overrides auto-detection (user-supplied --jobs).
    Auto-detection divides free VRAM by VRAM_PER_JOB_MIB (conservative per-job
    budget), capped by the number of cameras. Returns at least 1.
    """
    if explicit and explicit > 0:
        return min(explicit, camera_count)
    free = _free_vram_mib()
    if free is None:
        print("[GPU] nvidia-smi unavailable — defaulting to --jobs 1")
        return 1
    jobs = max(1, free // VRAM_PER_JOB_MIB)
    jobs = min(jobs, camera_count)
    if free < MIN_FREE_VRAM_MIB:
        print(f"[GPU] free VRAM {free} MiB < {MIN_FREE_VRAM_MIB} — forcing --jobs 1")
        jobs = 1
    print(f"[GPU] free VRAM {free} MiB → {jobs} parallel "
          f"render job{'' if jobs==1 else 's'} "
          f"(budget {VRAM_PER_JOB_MIB} MiB/job)")
    return jobs


def _render_worker(args):
    """Run one Blender render worker for a single camera. Blocks.

    Output is streamed in real-time (prefixed by [tag]) so the user can see
    progress during long renders.  Returns (camera_tag, success, wall_s).
    Metadata is deferred to a separate pass.
    """
    # Import sys here so this works in a ThreadPoolExecutor.
    import sys as _sys

    scenario_path, out_dir, tag = args
    cmd = [
        BLENDER, "-b", "--python", os.path.join(HERE, "render.py"), "--",
        "--scenario", scenario_path, "--out", out_dir,
        "--only", tag, "--no-metadata",
    ]
    tag_label = f"[{tag}]"
    print(f"\n{tag_label} $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    ok = True
    # Stream output in real-time so long renders show progress.
    with subprocess.Popen(
        cmd, cwd=ROOT, text=True,
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


def step_render_parallel(scenario_path, out_dir, jobs=2, only=None):
    camera_tags = G.camera_names()
    if only:
        camera_tags = [only]
    n_cams = len(camera_tags)

    tasks = [(scenario_path, out_dir, tag) for tag in camera_tags]
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
                        signal=args.signal)

    print("\n[3/5] Plate pre-generation")
    step_plates(scn, out_dir)

    print("\n[4/5] Render cameras (parallel)")
    jobs = _detect_jobs(8 if not args.only else 1, explicit=args.jobs)
    step_render_parallel(scn, out_dir, jobs=jobs, only=args.only)

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
