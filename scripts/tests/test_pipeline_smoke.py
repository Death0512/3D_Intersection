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
import run_pipeline as RP  # noqa: E402


def _run(cmd, cwd=ROOT):
    subprocess.run(cmd, cwd=cwd, check=True, text=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)





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
