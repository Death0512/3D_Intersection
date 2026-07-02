#!/usr/bin/env python3
"""Phase 5 — Pipeline driver.

Orchestrates the full dataset generation:
  1. Generate scenario        (conda python: scenario_gen.py)
  2. Pre-generate plate PNGs  (conda python: gen_plate batch)
  3. Validate assets          (blender headless: validate_assets.py)
  4. Render all 8 cameras     (blender headless: render.py)
  5. Validate the run         (conda python: validate_run.py)

Run (from the project root, with conda python):
    python3 scripts/run_pipeline.py --seed 42 --num-vehicles 10 --duration 200 --out output/run1

Blender is invoked via subprocess; conda python is the current interpreter.
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
# Python interpreter with Pillow (for plate pre-generation). Override with the
# DOAN_PYTHON env var; otherwise fall back to a common conda env path, else the
# current interpreter.
_DOAN_PY = (os.environ.get("DOAN_PYTHON")
            or os.path.expanduser("~/miniconda3/envs/DoAn/bin/python"))
PYTHON = _DOAN_PY if os.path.exists(_DOAN_PY) else sys.executable
BLENDER = shutil.which("blender") or "blender"

sys.path.insert(0, os.path.join(HERE, "lib"))
import envfile as ENV


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


def step_scenario(seed, num_vehicles, duration, out_dir, fps=None):
    cmd = [PYTHON, os.path.join(HERE, "scenario_gen.py"),
           "--seed", str(seed),
           "--num-vehicles", str(num_vehicles),
           "--duration", str(duration),
           "--out", out_dir]
    if fps is not None:
        cmd += ["--fps", str(fps)]
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


def step_render(scenario_path, out_dir, only=None):
    cmd = [BLENDER, "-b", "--python", os.path.join(HERE, "render.py"), "--",
           "--scenario", scenario_path, "--out", out_dir]
    if only:
        cmd += ["--only", only]
    run(cmd, check=False)


def step_validate_run(out_dir):
    run([PYTHON, os.path.join(HERE, "validate_run.py"), "--out", out_dir], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-vehicles", type=int, default=10)
    ap.add_argument("--fps", type=int, default=None,
                    help="frames per second (default: geometry.FPS)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="video length in seconds; overrides --duration")
    ap.add_argument("--duration", type=int, default=200,
                    help="duration in frames (ignored if --seconds given)")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "output", "run1"))
    ap.add_argument("--only", help="render only this camera (debug)")
    ap.add_argument("--skip-asset-check", action="store_true")
    args = ap.parse_args()

    fps = args.fps  # may be None -> scenario_gen uses G.FPS default
    if args.seconds is not None:
        effective_fps = fps if fps is not None else 30
        duration = int(round(args.seconds * effective_fps))
    else:
        duration = args.duration

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    fps_label = f"{fps} fps" if fps else "default fps"
    print("=" * 60)
    print(f"PIPELINE  seed={args.seed} n={args.num_vehicles} dur={duration}f ({fps_label})  out={out_dir}")
    print(f"  python : {PYTHON}")
    print(f"  blender: {BLENDER}")
    print("=" * 60)

    # Env files are REQUIRED inputs (camera/road/sun/vehicle anchors). Validate
    # them up front, unconditionally — NOT skippable via --skip-asset-check.
    print("\n[0/5] Env file validation (required)")
    ENV.validate_all_envs(ROOT)

    if not args.skip_asset_check:
        print("\n[1/5] Asset validation")
        step_assets_validate()

    print("\n[2/5] Scenario generation")
    scn = step_scenario(args.seed, args.num_vehicles, duration, out_dir, fps=fps)

    print("\n[3/5] Plate pre-generation")
    step_plates(scn, out_dir)

    print("\n[4/5] Render 8 cameras")
    step_render(scn, out_dir, only=args.only)

    print("\n[5/5] Run validation")
    step_validate_run(out_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  output dir: {out_dir}")
    videos = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
    print(f"  videos: {len(videos)}  {videos}")
    print(f"  metadata: {os.path.join(out_dir, 'metadata.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
