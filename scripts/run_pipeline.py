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
# Use the DoAn conda env python (has Pillow) if available; else sys.executable.
_DOAN_PY = os.path.expanduser("~/miniconda3/envs/DoAn/bin/python")
PYTHON = _DOAN_PY if os.path.exists(_DOAN_PY) else sys.executable
BLENDER = shutil.which("blender") or "blender"


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


def step_scenario(seed, num_vehicles, duration, out_dir):
    run([PYTHON, os.path.join(HERE, "scenario_gen.py"),
         "--seed", str(seed),
         "--num-vehicles", str(num_vehicles),
         "--duration", str(duration),
         "--out", out_dir])
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
    ap.add_argument("--duration", type=int, default=200, help="frames")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "output", "run1"))
    ap.add_argument("--only", help="render only this camera (debug)")
    ap.add_argument("--skip-asset-check", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print(f"PIPELINE  seed={args.seed} n={args.num_vehicles} dur={args.duration}f  out={out_dir}")
    print(f"  python : {PYTHON}")
    print(f"  blender: {BLENDER}")
    print("=" * 60)

    if not args.skip_asset_check:
        print("\n[1/5] Asset validation")
        step_assets_validate()

    print("\n[2/5] Scenario generation")
    scn = step_scenario(args.seed, args.num_vehicles, args.duration, out_dir)

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
