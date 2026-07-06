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

# Hard cap on Blender render workers per GPU. VRAM is not the only constraint
# on per-GPU packing: each Cycles+OptiX context also contends for the GPU's
# compute units, the host-side texture upload bandwidth, and the CPU-side
# denoiser (OIDN runs on the CPU when the OptiX-denox weights file
# /usr/share/nvidia/nvoptix.bin is unavailable — the case on Kaggle's T4
# container). The prior packing used only VRAM (free // VRAM_PER_JOB_MIB) and
# on a 15 GB T4 could schedule 4+ heavy scenes (40+ vehicle scenes, each with
# hundreds of remapped textures — see the `in_E` build of 42 vehicles / 378
# textures in run 1) onto one card. That exhausted VRAM and stalled the OptiX
# render on frame 0 — silent for 600 s until the watchdog aborted the pool
# (see PIPELINE ABORTED — worker silent for Ns). 2 per GPU keeps each Cycles
# context at ~7 GB headroom on a 15 GB card and prevents the stall while still
# using every GPU. Override with --max-workers-per-gpu for denser/lighter
# scenes; the working n=50 baseline ran at 2/GPU on Kaggle 2×T4 without issue.
MAX_WORKERS_PER_GPU = 2


def run(cmd: list, cwd=ROOT, check=True, timeout=None):
    """Run a subprocess step, streaming stdout+stderr live.

    Replaces the buffered `capture_output=True` pattern (which hid all output
    until exit, leaving long steps like scenario-gen / validate-assets silent
    for their whole duration when run under a non-TTY). Now each line is
    forwarded to stdout immediately, prefixed with 2 spaces, so the user can
    see progress in real time regardless of buffering.

    ``timeout`` (seconds) kills the subprocess and aborts the pipeline if
    exceeded — pass concrete values per caller (e.g. 120s for asset-validate,
    600s for scenario-gen) to bound headless-Blender / driver hangs.
    """
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"FAIL: command not found: {cmd[0]} ({e})")
    last_lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                print(" ", line, flush=True)
                last_lines.append(line)
                if len(last_lines) > 50:
                    last_lines.pop(0)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        dt = time.time() - t0
        raise SystemExit(
            f"FAIL: command timed out after {timeout}s (killed): {' '.join(cmd)}\n"
            f"  last output:\n    " + "\n    ".join(last_lines[-10:]))
    dt = time.time() - t0
    if proc.returncode != 0 and check:
        raise SystemExit(
            f"Command failed (code {proc.returncode}) after {dt:.1f}s: {' '.join(cmd)}\n"
            f"  last output:\n    " + "\n    ".join(last_lines[-15:]))
    print(f"  (ok, {dt:.1f}s)", flush=True)
    return proc


def step_assets_validate():
    """Run asset validation (fast). Bounded to 120s — headless Blender should
    start and exit within seconds; a hang here means a corrupt .blend / driver."""
    run([BLENDER, "-b", "--python", os.path.join(HERE, "validate_assets.py")],
        timeout=120)


def step_scenario(seed, num_vehicles, seconds, out_dir, fps=None,
                   signal=False, signal_mode="fixed", demand=None,
                   demand_scale=None):
    """Run scenario_gen.py. Bounded to 600s — the `_resolve_all` fixpoint is
    capped at 20 rounds, so even with 200 vehicles this stays well under 60s;
    a hang here means a non-converging signal/exit loop."""
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
    if demand_scale is not None:
        cmd += ["--demand-scale", str(demand_scale)]
    run(cmd, timeout=600)
    return os.path.join(out_dir, "scenario.json")


def step_plates(scenario_path, out_dir):
    """Pre-generate all plate PNGs in conda python (Pillow available).
    Bounded to 300s — plate rendering is fast per plate (~10ms)."""
    plates_dir = os.path.join(out_dir, "plates")
    os.makedirs(plates_dir, exist_ok=True)
    script = (
        "import sys,json; sys.path.insert(0,'scripts'); "
        "from gen_plate import pregenerate_plates; "
        f"d=json.load(open({scenario_path!r})); "
        "pregenerate_plates([v['plate'] for v in d['vehicles']], "
        f"{plates_dir!r}); print('plates done')"
    )
    run([PYTHON, "-c", script], timeout=300)


def _gpu_count():
    """Return the number of NVIDIA GPUs, or 0 if nvidia-smi is unavailable."""
    return len(_gpu_info())


# Module-level cache: `_detect_jobs` calls _gpu_info up to 3× otherwise; cache
# the result of the first probe for the whole process lifetime. nvidia-smi is
# cheap (~50 ms) but redundant 3× adds latency + multiplies driver-hiccup risk.
_GPU_INFO_CACHE = None


def _gpu_info():
    """Return a list of (index, free_mib) tuples for all GPUs, or empty list.

    Single source of truth for both GPU count and per-GPU free VRAM. Cache is
    safe for a single pipeline run (VRAM only changes across render launches).
    Distinguishes "nvidia-smi binary missing" (silent fallback) from "binary
    exists but output parse failed" (warns loudly so a silent serial-fallback
    doesn't eat 8× wall time with no clue why).
    """
    global _GPU_INFO_CACHE
    if _GPU_INFO_CACHE is not None:
        return _GPU_INFO_CACHE
    # Cache miss → probe.
    import shutil as _sh
    if _sh.which("nvidia-smi") is None:
        _GPU_INFO_CACHE = []
        return []
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
        _GPU_INFO_CACHE = result
        return result
    except subprocess.TimeoutExpired:
        print(f"[WARN] nvidia-smi timed out (>10s) — defaulting to serial",
              file=sys.stderr, flush=True)
        _GPU_INFO_CACHE = []
        return []
    except (ValueError, OSError) as e:
        # Binary present but parse failed — the most insidious case: silent
        # fallback to --jobs 1 with no clue why. Warn loudly.
        print(f"[WARN] nvidia-smi output parse failed ({type(e).__name__}: {e}) "
              f"— defaulting to serial render. raw stdout: {out.stdout!r}",
              file=sys.stderr, flush=True)
        _GPU_INFO_CACHE = []
        return []


def _free_vram_mib():
    """Query free GPU VRAM from nvidia-smi.  Returns integer MiB or None.

    Uses the first GPU (index 0) for single-GPU compat — the multi-GPU path
    in _detect_jobs uses _gpu_info() instead. If GPU 0 is filtered out (free=
    0), warns rather than silently returning a different GPU's VRAM (which
    would underbook the real GPU 0 and oversubscribe it).
    """
    info = _gpu_info()
    if not info:
        return None
    for idx, free in info:
        if idx == 0:
            return free
    print(f"[WARN] GPU 0 not in nvidia-smi free-VRAM list (it may be full); "
          f"available GPUs: {info} — using {info[0][1]} MiB budget on GPU {info[0][0]}",
          file=sys.stderr, flush=True)
    return info[0][1]  # fallback: first available GPU, whatever its index


def _detect_jobs(camera_count: int, explicit: int = 0,
                 max_workers_per_gpu: int = MAX_WORKERS_PER_GPU):
    """Determine how many parallel Blender jobs to run AND which GPU each binds to.

    Multi-GPU hosts (e.g. Kaggle T4×2): each GPU runs up to
    ``max_workers_per_gpu`` Blender workers (default 2 — see MAX_WORKERS_PER_GPU
    for the rationale), giving good throughput without exhausting VRAM /
    compute on dense scenes. Single-GPU: same cap applies (avoids oversubscribing
    one card). Returns (job_count, gpu_assignment) where gpu_assignment is a
    list of GPU-id integers, length == job_count.

    ``explicit`` > 0 overrides auto-detection — all workers share GPU 0
    (user-chosen parallelism, not per-GPU).  Calls ``_gpu_info`` exactly once
    (cached), so the prior 3× nvidia-smi spawn is gone.

    ``max_workers_per_gpu`` caps how many workers each GPU hosts regardless of
    the VRAM-derived estimate. The VRAM budget alone over-packs on big cards
    (a 15 GB T4 lets ~9 jobs fit by VRAM, but 4+ heavy Cycles+OptiX contexts
    stall the render — see MAX_WORKERS_PER_GPU docstring). Raise it via
    --max-workers-per-gpu for lighter scenes (few vehicles / no signal) where
    the per-scene VRAM footprint is small.
    """
    if max_workers_per_gpu < 1:
        max_workers_per_gpu = 1
    assignment: list[int] = []
    if explicit and explicit > 0:
        n = min(explicit, camera_count)
        return n, [0] * n

    info = _gpu_info()
    n_gpu = len(info)
    if n_gpu == 0:
        print("[GPU] nvidia-smi unavailable — defaulting to --jobs 1")
        return 1, [0]

    # Multi-GPU: pack up to max_workers_per_gpu workers per GPU (VRAM-limited
    # AND cap-limited), interleaved so capping by camera_count spreads load
    # evenly across GPUs.
    if n_gpu >= 2:
        # Per-GPU worker slots based on VRAM budget, capped per-GPU.
        per_gpu_slots: list[tuple[int, int]] = []  # (gpu_id, n_slots)
        for gid, free in info:
            if free < MIN_FREE_VRAM_MIB:
                continue
            w = max(1, min(free // VRAM_PER_JOB_MIB, max_workers_per_gpu))
            per_gpu_slots.append((gid, int(w)))
        if len(per_gpu_slots) < 2:
            print(f"[GPU] {n_gpu} GPU(s) detected but only "
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
        print(f"[GPU] {n_gpu} GPU(s), cap={max_workers_per_gpu}/GPU → {n} "
              f"parallel render worker{'' if n==1 else 's'} ({hdr})")
        return n, assignment

    # Single GPU: VRAM-budget job count, capped per-GPU (original behaviour
    # plus the cap so a huge single card doesn't oversubscribe).
    free = info[0][1] if info[0][0] == 0 else _free_vram_mib()
    if free is None:
        print("[GPU] nvidia-smi unavailable — defaulting to --jobs 1")
        return 1, [0]
    jobs = max(1, min(free // VRAM_PER_JOB_MIB, max_workers_per_gpu))
    jobs = min(jobs, camera_count)
    if free < MIN_FREE_VRAM_MIB:
        print(f"[GPU] free VRAM {free} MiB < {MIN_FREE_VRAM_MIB} — forcing --jobs 1")
        jobs = 1
    assignment = [0] * jobs
    print(f"[GPU] free VRAM {free} MiB, cap={max_workers_per_gpu}/GPU → "
          f"{jobs} parallel render job{'' if jobs==1 else 's'} "
          f"(budget {VRAM_PER_JOB_MIB} MiB/job)")
    return jobs, assignment


# ---------------------------------------------------------------------------
# Render workers + watchdog
# ---------------------------------------------------------------------------

# Default silence timeout: if a Blender worker produces no stdout for this
# many seconds, the pipeline is assumed hung (Cycles stuck on a black frame,
# GPU init deadlock, driver timeout, NVENC init hang) and is aborted with
# diagnostics. Tunable via --silence-timeout.
DEFAULT_SILENCE_TIMEOUT_S = 600  # 10 min


def _render_worker(args, watchdog_state=None, samples=None):
    """Run one Blender render worker for a single camera. Blocks.

    ``args`` is (scenario_path, out_dir, tag, gpu_id).  ``gpu_id`` is the
    NVIDIA GPU index this worker should bind to via CUDA_VISIBLE_DEVICES,
    so multi-GPU hosts run one Blender per physical GPU.

    ``samples`` (optional int) threads the --samples flag through to render.py
    so Cycles uses the user's sample count (lower = faster, noisier).

    Output is streamed in real-time (prefixed by [tag]) so the user can see
    progress during long renders.  Returns (camera_tag, success, wall_s).
    Metadata is deferred to a separate pass.

    ``watchdog_state`` is an optional shared-dict slot used by the watchdog
    in ``step_render_parallel`` to detect silent workers: this function updates
    ``watchdog_state['last_output_time']`` and ``watchdog_state['last_line']``
    on every received stdout line, so a separate thread can detect "no output
    for N seconds" and abort the whole pool (hard-kill policy chosen by user).
    """
    import sys as _sys

    scenario_path, out_dir, tag, gpu_id = args
    cmd = [
        BLENDER, "-b", "--python", os.path.join(HERE, "render.py"), "--",
        "--scenario", scenario_path, "--out", out_dir,
        "--only", tag, "--no-metadata",
    ]
    if samples is not None:
        cmd += ["--samples", str(samples)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Force Python-level stdout unbuffered in the Blender child so every
    # `print(..., flush=True)` from render.py / build_scene.py reaches the
    # OS pipe immediately. Without this, the 8-hour silent-hang scenario
    # (block-buffered stdout under non-TTY) reappears.
    env["PYTHONUNBUFFERED"] = "1"
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
        if watchdog_state is not None:
            watchdog_state["proc"] = proc
            watchdog_state["last_output_time"] = time.time()
            watchdog_state["last_line"] = ""
            watchdog_state["started_at"] = t0
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                # Suppress noisy warnings and frame save messages
                if "HIPEW initialization failed" in line or "Saved:" in line:
                    continue
                _sys.stdout.write(f"  {tag_label} {line}\n")
                _sys.stdout.flush()
                if watchdog_state is not None:
                    watchdog_state["last_output_time"] = time.time()
                    watchdog_state["last_line"] = line
        proc.wait()
        dt = time.time() - t0
        ok = proc.returncode == 0
    status = "OK" if ok else "FAILED"
    print(f"{tag_label} {status} ({dt:.1f}s)", flush=True)
    if watchdog_state is not None:
        watchdog_state["done"] = True
    return tag, ok, dt


def _watchdog(workers_state, silence_timeout_s, stop_event):
    """Background thread that aborts the pool if any worker goes silent.

    ``workers_state`` is a dict tag → state-dict (with keys: last_output_time,
    last_line, started_at, proc, optional 'done').  If any *non-done* worker's
    ``time.time() - last_output_time`` exceeds ``silence_timeout_s``, the
    watchdog kills ALL worker procs and signals the main thread via
    ``stop_event``, which causes ``step_render_parallel`` to raise SystemExit
    with a diagnostic banner (hard-kill policy per user choice).

    Also times out workers whose TOTAL wall time exceeds 6× the silence
    timeout (catches the case where a worker emits periodic short lines but
    takes punitively long).
    """
    while not stop_event.is_set():
        now = time.time()
        for tag, st in workers_state.items():
            if st.get("done"):
                continue
            last_t = st.get("last_output_time", st.get("started_at", now))
            silent_for = now - last_t
            if silent_for > silence_timeout_s:
                # Hard-kill whole pool. Kill every live proc.
                for _, st2 in workers_state.items():
                    p = st2.get("proc")
                    if p and p.poll() is None:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                # Give procs 5s to terminate, then SIGKILL.
                time.sleep(5)
                for _, st2 in workers_state.items():
                    p = st2.get("proc")
                    if p and p.poll() is None:
                        try:
                            p.kill()
                        except Exception:
                            pass
                stop_event.set()
                print("\n" + "=" * 70, file=sys.stderr, flush=True)
                print(f"PIPELINE ABORTED — worker [{tag}] silent for "
                      f"{silent_for:.0f}s (> {silence_timeout_s}s timeout)",
                      file=sys.stderr, flush=True)
                for t, s in workers_state.items():
                    last = s.get("last_line", "")[:80]
                    print(f"  [{t}] last: {last!r}", file=sys.stderr, flush=True)
                print("=" * 70, file=sys.stderr, flush=True)
                return
        time.sleep(5)


def step_render_parallel(scenario_path, out_dir, jobs=2, gpu_assignment=None,
                         only=None, silence_timeout_s=DEFAULT_SILENCE_TIMEOUT_S,
                         samples=None):
    """Render all 8 (or ``only``) cameras. If ``gpu_assignment`` is set, each
    worker binds to a different GPU via CUDA_VISIBLE_DEVICES (round-robins
    when there are more cameras than GPUs). Otherwise all workers share GPU 0.

    ``samples`` threads --samples through to render.py (None = use the
    build_scene default of 48).

    A background watchdog kills the entire pool (hard-kill policy) if any
    worker is silent for ``silence_timeout_s`` seconds — converts a
    multi-hour silent hang into a fast, diagnosable abort.
    """
    camera_tags = G.camera_names()
    if only:
        camera_tags = [only]
    n_cams = len(camera_tags)

    # D12: short-circuit BEFORE building the parallel banner, so a single-
    # camera render doesn't print a misleading "N parallel workers" line.
    if jobs <= 1 or n_cams == 1:
        print(f"[render] serial/single-camera render — 1 worker")
        results = {}
        for tag in camera_tags:
            args = (scenario_path, out_dir, tag,
                    gpu_assignment[0] if gpu_assignment else 0)
            tag_r, ok, dt = _render_worker(args, samples=samples)
            results[tag_r] = (ok, dt)
        _print_render_summary(results, camera_tags)
        failed = sum(1 for ok, _ in results.values() if not ok)
        if failed:
            raise SystemExit(
                f"PIPELINE ABORTED — {failed} render worker(s) failed. "
                f"Partial renders may exist in {out_dir}.")
        return

    n_gpus = len(gpu_assignment) if gpu_assignment else 0
    # gpu_assignment is the exact per-worker GPU id list (length == jobs);
    # pair each camera tag with its GPU id in order. If we have more cameras
    # than assignments, fall back to GPU 0 for the overflow.
    tasks = [(scenario_path, out_dir, tag,
              gpu_assignment[i] if i < n_gpus else 0)
             for i, tag in enumerate(camera_tags)]
    results = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Shared watchdog state: tag → mutable dict of progress markers.
    workers_state = {tag: {} for tag in camera_tags}
    stop_event = threading.Event()
    wd = threading.Thread(
        target=_watchdog,
        args=(workers_state, silence_timeout_s, stop_event),
        daemon=True,
        name="render-watchdog",
    )
    wd.start()
    aborted = False
    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {}
            for i, t in enumerate(tasks):
                tag = t[2]
                # Pass the per-tag watchdog slot so the worker updates it.
                futs[pool.submit(_render_worker, t, workers_state[tag],
                                 samples)] = i
            try:
                for fut in as_completed(futs, timeout=None):
                    if stop_event.is_set():
                        aborted = True
                        break
                    try:
                        tag, ok, dt = fut.result(timeout=5)
                        results[tag] = (ok, dt)
                    except Exception:
                        # Worker raised (e.g. proc killed by watchdog). Mark
                        # the corresponding tag FAILED; don't abort the pool
                        # mid-collection (the watchdog already decided).
                        i = futs[fut]
                        tag = tasks[i][2]
                        results[tag] = (False, 0.0)
            except Exception:
                aborted = True
    finally:
        stop_event.set()
        wd.join(timeout=2)

    if aborted:
        # Any worker not yet recorded as a result is presumed killed.
        for tag in camera_tags:
            if tag not in results:
                results[tag] = (False, 0.0)
        _print_render_summary(results, camera_tags)
        raise SystemExit(
            f"PIPELINE ABORTED — render watchdog killed the pool (see stderr "
            f"above for which worker went silent and its last line). "
            f"Partial renders may exist in {out_dir}.")
    _print_render_summary(results, camera_tags)
    failed = sum(1 for ok, _ in results.values() if not ok)
    if failed:
        raise SystemExit(
            f"PIPELINE ABORTED — {failed} render worker(s) failed. "
            f"Partial renders may exist in {out_dir}.")


def _print_render_summary(results, camera_tags):
    print()
    print(f"  Render summary ({len(results)}/{len(camera_tags)} cameras):")
    for tag in camera_tags:
        ok, dt = results.get(tag, (False, 0))
        print(f"    {tag:6s} {'OK' if ok else 'FAILED'}  ({dt:.1f}s)")
    failed = sum(1 for ok, _ in results.values() if not ok)
    if failed:
        print(f"  {failed} camera(s) FAILED — pipeline will abort")


def step_metadata(scenario_path, out_dir):
    """Generate metadata.json. FATAL — the whole point of the run is the
    metadata ground-truth; a silent partial write here wastes all the render
    compute. Bounded to 600s (pure-Python per-frame pose loop, bounded by
    vehicles × visible frames)."""
    cmd = [PYTHON, os.path.join(HERE, "render.py"), "--",
           "--scenario", scenario_path, "--out", out_dir,
           "--metadata-only"]
    run(cmd, check=True, timeout=600)


def step_validate_run(out_dir):
    """Validate the run output. FATAL — a partial dataset (missing metadata,
    missing videos, malformed JSON) must bail rather than print "complete"."""
    run([PYTHON, os.path.join(HERE, "validate_run.py"), "--out", out_dir],
        check=True, timeout=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-vehicles", type=int, default=10)
    ap.add_argument("--fps", type=int, default=None,
                    help="frames per second (default: geometry.FPS)")
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="MINIMUM video length in seconds. The actual video "
                         "auto-extends to fit all vehicles (it can be longer "
                         "than this, never shorter). At default demand "
                         "(400 veh/h/approach), ~25 vehicles fit in 12s; "
                         "100+ vehicles will spread across minutes.")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "output", "run1"))
    ap.add_argument("--only", help="render only this camera (debug)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel render workers (0 = auto-detect from free "
                         "VRAM capped by --max-workers-per-gpu)")
    ap.add_argument("--max-workers-per-gpu", type=int,
                    default=MAX_WORKERS_PER_GPU,
                    help=f"cap on Blender render workers per GPU regardless of "
                         f"free VRAM (default {MAX_WORKERS_PER_GPU}). The VRAM "
                         f"budget alone over-packs big cards (a 15 GB T4 lets "
                         f"~9 jobs fit by VRAM, but 4+ heavy Cycles+OptiX "
                         f"contexts stall the render — silent hang). Raise for "
                         f"light scenes (few vehicles / no signal) only.")
    ap.add_argument("--silence-timeout", type=int,
                    default=DEFAULT_SILENCE_TIMEOUT_S,
                    help=f"seconds a render worker may go silent before the "
                         f"watchdog kills the whole pool (default "
                         f"{DEFAULT_SILENCE_TIMEOUT_S})")
    ap.add_argument("--samples", type=int, default=48,
                    help="Cycles render samples per frame (default 48; lower "
                         "= faster, noisier — denoiser compensates. Use 16-24 "
                         "for quick test runs, 48 for production.")
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
    ap.add_argument("--demand-scale", type=float, default=None,
                    help="density multiplier on the default demand model "
                         "(default: 1.0 when --demand is not given). E.g. "
                         "--demand-scale 3 makes ~1200 veh/h/approach -> denser "
                         "on-screen traffic, no JSON file needed. Ignored when "
                         "--demand is a path or 'none'.")
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
                        demand=args.demand, demand_scale=args.demand_scale)

    print("\n[3/5] Plate pre-generation")
    step_plates(scn, out_dir)

    print("\n[4/5] Render cameras (parallel)")
    n_jobs, gpu_assign = _detect_jobs(
        8 if not args.only else 1, explicit=args.jobs,
        max_workers_per_gpu=args.max_workers_per_gpu)
    step_render_parallel(scn, out_dir, jobs=n_jobs, gpu_assignment=gpu_assign,
                         only=args.only, silence_timeout_s=args.silence_timeout,
                         samples=args.samples)

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
