"""Pure-Python pipeline smoke test.

Exercises the cheap end-to-end path without Blender/GPU: scenario generation →
dummy rendered videos → metadata-only export → validate_run. This catches CLI
plumbing/import regressions while leaving real rendering to manual/GPU runs.

Run:
    cd /home/death/Documents/3D_Intersection_Video
    python3 -m pytest scripts/tests/test_pipeline_smoke.py -v
  or:
    python3 scripts/tests/test_pipeline_smoke.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lib import geometry as G  # noqa: E402


def _run(cmd, cwd=ROOT):
    subprocess.run(cmd, cwd=cwd, check=True, text=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_pipeline_metadata_validate_smoke():
    with tempfile.TemporaryDirectory() as td:
        demand_path = os.path.join(td, "demand.json")
        with open(demand_path, "w") as f:
            json.dump({
                "flows": {d.value: 400.0 for d in G.Direction},
                "turn_split": {"left": 0.0, "straight": 1.0, "right": 0.0},
            }, f)

        _run([
            sys.executable, "scripts/scenario_gen.py",
            "--seed", "123",
            "--seconds", "12",
            "--demand", demand_path,
            "--out", td,
        ])

        for tag in G.camera_names():
            with open(os.path.join(td, f"video_{tag}.mp4"), "wb") as f:
                f.write(b"smoke")

        _run([
            sys.executable, "scripts/render.py", "--",
            "--scenario", os.path.join(td, "scenario.json"),
            "--out", td,
            "--metadata-only",
        ])
        _run([sys.executable, "scripts/validate_run.py", "--out", td])


def test_research_trajectory_integrity_validate_smoke():
    with tempfile.TemporaryDirectory() as td:
        _run([
            sys.executable, "scripts/scenario_gen.py",
            "--seed", "7",
            "--seconds", "1",
            "--out", td,
            "--simulator", "research",
        ])

        for tag in G.camera_names():
            with open(os.path.join(td, f"video_{tag}.mp4"), "wb") as f:
                f.write(b"smoke")

        _run([
            sys.executable, "scripts/render.py", "--",
            "--scenario", os.path.join(td, "scenario.json"),
            "--out", td,
            "--metadata-only",
        ])
        _run([sys.executable, "scripts/validate_run.py", "--out", td])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
        except Exception as e:
            print(f"  [ERR ] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
