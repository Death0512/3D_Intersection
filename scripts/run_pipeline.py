#!/usr/bin/env python3
"""Phase 5 — Pipeline driver.

Orchestrates the full dataset generation:
  1. Validate env files       (conda/venv python: envfile)
  2. (Optional) Validate assets  (blender headless: validate_assets.py)
  3. Generate scenario         (conda/venv python: scenario_gen.py)
  4. Pre-generate plate PNGs   (conda/venv python: gen_plate batch)
  5. Render all 8 cameras      (blender headless: render.py)
  6. Validate the run          (conda/venv python: validate_run.py)

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
# Python interpreter with Pillow (for plate pre-generation). Set by the
# DOAN_PYTHON env var (exported by scripts/env.sh) or fall back to the current
# interpreter.
_DOAN_PY = os.environ.get("DOAN_PYTHON")
PYTHON = _DOAN_PY if _DOAN_PY and os.path.exists(_DOAN_PY) else sys.executable
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
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="minimum video length in seconds (actual duration auto-extends "
                         "to fit all vehicles)")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "output", "run1"))
    ap.add_argument("--only", help="render only this camera (debug)")
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
