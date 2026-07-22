#!/usr/bin/env python3
"""Phase 5 — Pipeline driver.

Orchestrates the full dataset generation:
  [0/5] Validate env files       (conda/venv python: envfile)
  [1/5] (Optional) Validate assets  (blender headless: validate_assets.py)
  [2/5] Generate scenario         (conda/venv python: scenario_gen.py)
  [3/5] Pre-generate plate PNGs   (conda/venv python: gen_plate batch)
  [4/5] Render all 8 cameras      (blender headless: render.py, parallel)
  [5/5] Metadata + run validation (conda/venv python)

``--seconds`` sets the rendered clip length. Scenario generation uses a
steady-state warm-up stream and admits vehicles whose motion intersects that
fixed window; render/metadata clamp per-frame outputs to the clip bounds.

Run (from the project root, with venv python):
    python3 scripts/run_pipeline.py --seed 42 --seconds 12.0 --out output/run1

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

# Per-job VRAM budget estimate (MiB) for a LIGHT camera scene (~3-5 vehicles,
# 1 road segment, 1080p, 48 Cycles samples + OPTIX denoiser).  Empirically the
# per-context footprint for such scenes is ~500-700 MiB; we budget 900 MiB to
# include OS + driver overhead and headroom for BVH spikes.
# Adjust down for very sparse scenes (e.g. --demand-scale 0.5) or up for
# heavy-demand runs with many vehicles in frame.
VRAM_PER_JOB_MIB = 900

# Minimum free VRAM to attempt any GPU rendering (MiB).
MIN_FREE_VRAM_MIB = 1200

# Default cap on Blender render workers per GPU.  Raised to 2 because each
# camera scene is very lightweight (3-5 vehicles vs. a full city scene), so two
# Cycles contexts co-reside comfortably within typical NVIDIA consumer VRAM (4 GB+).
# Hard limit: consumer NVIDIA GPUs allow at most 3 concurrent NVENC sessions;
# we stay at 2/GPU to leave headroom for the OS encoder and avoid driver-level
# NVENC contention.  Raise via --max-workers-per-gpu only for tested hardware.
MAX_WORKERS_PER_GPU = 2

# Building cached .blend scenes launches Blender processes too. Keep this
# modest to preserve reliability on laptops/Kaggle while still overlapping the
# CPU-heavy scene construction phase.
MAX_BUILD_WORKERS = 2

# NVENC concurrent encode limit.  Consumer NVIDIA GPUs support at most 3
# concurrent NVENC sessions.  The encode semaphore (created per render phase)
# caps simultaneous ffmpeg NVENC calls to this value, preventing encoder
# contention when multiple workers finish their Cycles render at the same time.
MAX_NVENC_CONCURRENT = 2


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


def step_scenario(seed, seconds, out_dir, fps=None,
                   signal=False, signal_mode="fixed", demand=None,
                   demand_scale=None, simulator=None):
    """Run scenario_gen.py. Bounded to 600s; v2 simulation should finish fast,
    so a hang here means the event loop horizon/queue release logic regressed."""
    cmd = [PYTHON, os.path.join(HERE, "scenario_gen.py"),
           "--seed", str(seed),
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
    if simulator is not None:
        cmd += ["--simulator", str(simulator)]
    run(cmd, timeout=600)
    return os.path.join(out_dir, "scenario.json")


def step_sumo_scenario(seed, seconds, out_dir, fps=None, demand_scale=None):
    """Run SUMO/TraCI once and write scenario.json with per-frame trajectories."""
    cmd = [PYTHON, os.path.join(HERE, "run_sumo_unified.py"),
           "--seed", str(seed), "--seconds", str(seconds), "--out", out_dir]
    if fps is not None:
        cmd += ["--fps", str(fps)]
    if demand_scale is not None:
        cmd += ["--demand-scale", str(demand_scale)]
    run(cmd, timeout=900)
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
    """Return a list of (index, free_mib) tuples for all GPUs, or raise
    SystemExit if no NVIDIA GPU is available. Fail-fast — this pipeline
    cannot render on CPU.

    Single source of truth for both GPU count and per-GPU free VRAM. Cache is
    safe for a single pipeline run (VRAM only changes across render launches).
    """
    global _GPU_INFO_CACHE
    if _GPU_INFO_CACHE is not None:
        return _GPU_INFO_CACHE
    # Cache miss → probe.
    import shutil as _sh
    if _sh.which("nvidia-smi") is None:
        raise SystemExit(
            "FAIL: nvidia-smi not found — no NVIDIA GPU detected. "
            "This pipeline requires an NVIDIA GPU (OptiX or CUDA) for Cycles rendering. "
            "CPU rendering is not supported.")
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
        if not result:
            raise SystemExit(
                "FAIL: nvidia-smi returned but no GPUs with free VRAM > 0. "
                "GPU rendering cannot proceed.")
        _GPU_INFO_CACHE = result
        return result
    except subprocess.TimeoutExpired:
        raise SystemExit(
            "FAIL: nvidia-smi timed out (>10s). NVIDIA GPU probe failed — "
            "rendering cannot proceed.")
    except (ValueError, OSError) as e:
        raise SystemExit(
            f"FAIL: nvidia-smi output parse failed ({type(e).__name__}: {e}). "
            f"raw stdout: {out.stdout!r}. NVIDIA GPU probe failed — "
            f"rendering cannot proceed.")


def _free_vram_mib():
    """Query free GPU VRAM from nvidia-smi.  Returns integer MiB.

    Uses the first GPU (index 0) for single-GPU compat — the multi-GPU path
    in _detect_jobs uses _gpu_info() instead.
    """
    info = _gpu_info()
    for idx, free in info:
        if idx == 0:
            return free
    # GPU 0 not in the list (unusual but possible) — use first available.
    print(f"[WARN] GPU 0 not in nvidia-smi free-VRAM list; "
          f"using GPU {info[0][0]} with {info[0][1]} MiB",
          file=sys.stderr, flush=True)
    return info[0][1]


def _detect_jobs(camera_count: int, explicit: int = 0,
                 max_workers_per_gpu: int = MAX_WORKERS_PER_GPU,
                 vram_budget: int = VRAM_PER_JOB_MIB):
    """Determine how many parallel Blender jobs to run AND which GPU each binds to.

    Multi-GPU hosts (e.g. Kaggle T4×2): each GPU runs up to
    ``max_workers_per_gpu`` Blender workers (default 1 — see MAX_WORKERS_PER_GPU
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

    ``vram_budget`` overrides the module-level VRAM_PER_JOB_MIB estimate —
    callers scale this up for heavy scenes (many vehicles) so the worker
    count is computed correctly in a single pass instead of a post-hoc
    second banner.
    """
    if max_workers_per_gpu < 1:
        max_workers_per_gpu = 1
    assignment: list[int] = []
    if explicit and explicit > 0:
        n = min(explicit, camera_count)
        return n, [0] * n

    info = _gpu_info()
    n_gpu = len(info)

    # Multi-GPU: pack up to max_workers_per_gpu workers per GPU (VRAM-limited
    # AND cap-limited), interleaved so capping by camera_count spreads load
    # evenly across GPUs.
    if n_gpu >= 2:
        # Per-GPU worker slots based on VRAM budget, capped per-GPU.
        per_gpu_slots: list[tuple[int, int]] = []  # (gpu_id, n_slots)
        for gid, free in info:
            if free < MIN_FREE_VRAM_MIB:
                continue
            w = max(1, min(free // vram_budget, max_workers_per_gpu))
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
    jobs = max(1, min(free // vram_budget, max_workers_per_gpu))
    jobs = min(jobs, camera_count)
    if free < MIN_FREE_VRAM_MIB:
        raise SystemExit(
            f"FAIL: free VRAM {free} MiB < {MIN_FREE_VRAM_MIB} MiB minimum. "
            f"GPU rendering cannot proceed with insufficient VRAM.")
    assignment = [0] * jobs
    print(f"[GPU] free VRAM {free} MiB, cap={max_workers_per_gpu}/GPU → "
          f"{jobs} parallel render job{'' if jobs==1 else 's'} "
          f"(budget {vram_budget} MiB/job)")
    return jobs, assignment


# ---------------------------------------------------------------------------
# Render workers + watchdog
# ---------------------------------------------------------------------------

# Default silence timeout: if a Blender worker produces no stdout for this
# many seconds, the pipeline is assumed hung (Cycles stuck on a black frame,
# GPU init deadlock, driver timeout, NVENC init hang) and is aborted with
# diagnostics. Tunable via --silence-timeout.
DEFAULT_SILENCE_TIMEOUT_S = 600  # 10 min


def _camera_worker(args, watchdog_state=None, samples=None, mode="render",
                   skip_encode=False):
    """Run one Blender worker for a single camera. Blocks.

    ``args`` is (scenario_path, out_dir, tag, gpu_id).  ``gpu_id`` is the
    NVIDIA GPU index this worker should bind to via CUDA_VISIBLE_DEVICES,
    so multi-GPU hosts run one Blender per physical GPU for render mode.

    ``mode`` is either ``"build"`` (create cached scene_<tag>.blend) or
    ``"render"`` (render from that cached scene). Splitting these phases keeps
    each camera's scene isolated while avoiding repeated build work during the
    GPU-bound render stage.

    ``skip_encode`` (render mode only): when True, the Blender worker writes
    JPEG frames only and exits without calling ffmpeg.  The encode step is then
    handled by ``step_encode_all`` with a semaphore so at most
    ``MAX_NVENC_CONCURRENT`` ffmpeg NVENC sessions run simultaneously — avoiding
    NVENC driver contention when multiple workers finish their Cycles render at
    the same time.

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
    if mode == "build":
        cmd.append("--build-only")
    elif mode == "render":
        cmd.append("--render-only")
    else:
        raise ValueError(f"unknown camera worker mode: {mode}")
    if samples is not None:
        cmd += ["--samples", str(samples)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Force Python-level stdout unbuffered in the Blender child so every
    # `print(..., flush=True)` from render.py / build_scene.py reaches the
    # OS pipe immediately. Without this, the 8-hour silent-hang scenario
    # (block-buffered stdout under non-TTY) reappears.
    env["PYTHONUNBUFFERED"] = "1"
    if skip_encode and mode == "render":
        # Signal the child to leave JPEG frames on disk; the pipeline encodes
        # in a dedicated step with NVENC semaphore.
        env["RENDER_SKIP_ENCODE"] = "1"
    tag_label = f"[{mode}:{tag}]"
    gpu_note = f"GPU {gpu_id}" if mode == "render" else "build"
    print(f"\n{tag_label} $ {' '.join(cmd)}  ({gpu_note})", flush=True)
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


def _render_worker(args, watchdog_state=None, samples=None, skip_encode=False):
    return _camera_worker(args, watchdog_state=watchdog_state,
                          samples=samples, mode="render",
                          skip_encode=skip_encode)


def _build_worker(args, watchdog_state=None, samples=None):
    return _camera_worker(args, watchdog_state=watchdog_state,
                          samples=samples, mode="build")


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


def step_encode_all(out_dir, camera_tags, scenario_path, fps=None):
    """Encode all cameras' JPEG frame directories to MP4 with an NVENC semaphore.

    Called after step_render_parallel when ``skip_encode=True`` — at that
    point each camera has a ``frames_<tag>/`` directory on disk with all JPEGs
    rendered.  This step encodes them one-at-a-time up to MAX_NVENC_CONCURRENT
    at once using a threading.Semaphore to avoid saturating the NVENC encoder.

    Consumer NVIDIA GPUs allow at most 3 concurrent NVENC sessions; keeping
    sessions ≤ MAX_NVENC_CONCURRENT prevents the driver-level stall that occurs
    when multiple ffmpeg processes compete for the same encoder chip.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import json as _json

    # Reuse render.py's ffmpeg helper — it already has the correct
    # "-start_number 0" flag (scene.frame_start = 0) and the NVENC
    # availability probe with proper diagnostics; duplicating that command
    # here previously dropped frame 0 and skipped the NVENC probe.
    sys.path.insert(0, HERE)
    import render as _render

    # Derive FPS from scenario.json if not provided.
    if fps is None:
        try:
            with open(scenario_path) as f:
                fps = _json.load(f).get("fps", 30)
        except Exception:
            fps = 30

    sema = threading.Semaphore(MAX_NVENC_CONCURRENT)

    def _encode_one(tag):
        frames_dir = os.path.join(out_dir, f"frames_{tag}")
        video_path = os.path.join(out_dir, f"video_{tag}.mp4")
        if not os.path.isdir(frames_dir):
            print(f"  [encode:{tag}] SKIP — frames dir not found: {frames_dir}",
                  flush=True)
            return tag, False
        with sema:
            print(f"  [encode:{tag}] encoding {frames_dir} → {video_path}",
                  flush=True)
            t0 = time.time()
            ok = _render._ffmpeg_encode(frames_dir, video_path, fps)
            dt = time.time() - t0
            if ok:
                print(f"  [encode:{tag}] OK ({dt:.1f}s)", flush=True)
                try:
                    shutil.rmtree(frames_dir)
                except Exception as e:
                    print(f"  [encode:{tag}] [WARN] rmtree failed: {e}",
                          flush=True)
                return tag, True
            else:
                print(f"  [encode:{tag}] FAILED after {dt:.1f}s "
                      f"(see ffmpeg diagnostics above)", flush=True)
                return tag, False

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_NVENC_CONCURRENT) as pool:
        futs = {pool.submit(_encode_one, tag): tag for tag in camera_tags}
        for fut in as_completed(futs):
            tag, ok = fut.result()
            results[tag] = ok

    failed = [t for t, ok in results.items() if not ok]
    if failed:
        raise SystemExit(
            f"PIPELINE ABORTED — encode failed for: {', '.join(failed)}. "
            f"JPEG frames preserved in {out_dir}/frames_* for manual recovery.")
    print(f"  [encode] all {len(camera_tags)} camera(s) encoded OK", flush=True)


def step_render_parallel(scenario_path, out_dir, jobs=2, gpu_assignment=None,
                         only=None, silence_timeout_s=DEFAULT_SILENCE_TIMEOUT_S,
                         samples=None, skip_encode=False, camera_tags=None):
    """Render all 8 (or ``only``, or explicit ``camera_tags``) cameras. If
    ``gpu_assignment`` is set, each worker binds to a different GPU via
    CUDA_VISIBLE_DEVICES (round-robins when there are more cameras than
    GPUs). Otherwise all workers share GPU 0.

    ``samples`` threads --samples through to render.py (None = use the
    build_scene default of 48).

    ``skip_encode`` defers ffmpeg encoding out of the Blender workers so the
    pipeline can control NVENC concurrency via a semaphore (see
    ``step_encode_all``).  Workers write JPEG frames to ``frames_<tag>/`` and
    exit; the caller then runs ``step_encode_all`` after all workers finish.

    ``camera_tags`` overrides the camera set entirely (used by the GPU-error
    retry path to re-render only the cameras that failed in the first pass,
    instead of re-rendering all 8).

    A background watchdog kills the entire pool (hard-kill policy) if any
    worker is silent for ``silence_timeout_s`` seconds — converts a
    multi-hour silent hang into a fast, diagnosable abort.
    """
    if camera_tags is None:
        camera_tags = G.camera_names()
        if only:
            camera_tags = [only]

    def _worker_with_skip(args, watchdog_state=None, samples=None):
        return _render_worker(args, watchdog_state=watchdog_state,
                              samples=samples, skip_encode=skip_encode)

    _step_camera_phase(
        "render", _worker_with_skip, scenario_path, out_dir, camera_tags,
        jobs=jobs, gpu_assignment=gpu_assignment,
        silence_timeout_s=silence_timeout_s, samples=samples)


def step_build_scenes_parallel(scenario_path, out_dir, only=None,
                               silence_timeout_s=DEFAULT_SILENCE_TIMEOUT_S):
    """Build cached per-camera .blend scenes before GPU rendering.

    These are still isolated one-scene-per-camera builds, preserving the current
    data integrity model. The split lets the later render phase open an already
    built .blend and focus on GPU rendering/encoding.
    """
    camera_tags = G.camera_names()
    if only:
        camera_tags = [only]
    # Build processes are CPU/startup bound. Cap at camera count; GPU assignment
    # is irrelevant, but keep a stable zero for the worker environment.
    _step_camera_phase(
        "build", _build_worker, scenario_path, out_dir, camera_tags,
        jobs=min(len(camera_tags), MAX_BUILD_WORKERS),
        gpu_assignment=[0] * max(1, min(len(camera_tags), MAX_BUILD_WORKERS)),
        silence_timeout_s=silence_timeout_s, samples=None)


def _step_camera_phase(phase, worker, scenario_path, out_dir, camera_tags,
                       jobs=2, gpu_assignment=None,
                       silence_timeout_s=DEFAULT_SILENCE_TIMEOUT_S,
                       samples=None):
    """Shared executor for per-camera build/render phases."""
    n_cams = len(camera_tags)

    # D12: short-circuit BEFORE building the parallel banner, so a single-
    # camera render doesn't print a misleading "N parallel workers" line.
    if jobs <= 1 or n_cams == 1:
        print(f"[{phase}] serial/single-camera phase — 1 worker")
        results = {}
        for tag in camera_tags:
            args = (scenario_path, out_dir, tag,
                    gpu_assignment[0] if gpu_assignment else 0)
            tag_r, ok, dt = worker(args, samples=samples)
            results[tag_r] = (ok, dt)
        _print_phase_summary(phase, results, camera_tags)
        failed = sum(1 for ok, _ in results.values() if not ok)
        if failed:
            raise SystemExit(
                f"PIPELINE ABORTED — {failed} {phase} worker(s) failed. "
                f"Partial renders may exist in {out_dir}.")
        return

    n_gpus = len(gpu_assignment) if gpu_assignment else 0
    # gpu_assignment is the exact per-worker GPU id list (length == jobs);
    # pair each camera tag with its GPU id in order. If we have more cameras
    # than assignments, fall back to GPU 0 for the overflow.
    tasks = [(scenario_path, out_dir, tag,
              gpu_assignment[i % n_gpus] if n_gpus else 0)
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
                futs[pool.submit(worker, t, workers_state[tag], samples)] = i
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
        _print_phase_summary(phase, results, camera_tags)
        raise SystemExit(
            f"PIPELINE ABORTED — {phase} watchdog killed the pool (see stderr "
            f"above for which worker went silent and its last line). "
            f"Partial renders may exist in {out_dir}.")
    _print_phase_summary(phase, results, camera_tags)
    failed = sum(1 for ok, _ in results.values() if not ok)
    if failed:
        raise SystemExit(
            f"PIPELINE ABORTED — {failed} {phase} worker(s) failed. "
            f"Partial renders may exist in {out_dir}.")


def _print_phase_summary(phase, results, camera_tags):
    print()
    print(f"  {phase.title()} summary ({len(results)}/{len(camera_tags)} cameras):")
    for tag in camera_tags:
        ok, dt = results.get(tag, (False, 0))
        print(f"    {tag:6s} {'OK' if ok else 'FAILED'}  ({dt:.1f}s)")
    failed = sum(1 for ok, _ in results.values() if not ok)
    if failed:
        print(f"  {failed} camera(s) FAILED — pipeline will abort")


def step_metadata(scenario_path, out_dir, only=None, expected_videos=False):
    """Generate metadata.json. FATAL — the whole point of the run is the
    metadata ground-truth; a silent partial write here wastes all the render
    compute. Bounded to 600s (pure-Python per-frame pose loop, bounded by
    vehicles × visible frames)."""
    cmd = [PYTHON, os.path.join(HERE, "render.py"), "--",
           "--scenario", scenario_path, "--out", out_dir,
           "--metadata-only"]
    if only:
        cmd += ["--only", only]
    if expected_videos:
        cmd.append("--metadata-expected-videos")
    run(cmd, check=True, timeout=600)


def step_sumo_unified_build(scenario_path, out_dir):
    scene_path = os.path.join(out_dir, "unified_scene.blend")
    cmd = [BLENDER, "-b", "--python", os.path.join(HERE, "build_unified_scene.py"), "--",
           "--scenario", scenario_path, "--out", scene_path]
    run(cmd, check=True, timeout=1800)
    return scene_path


def step_sumo_unified_render(scenario_path, out_dir, jobs, samples, only=None):
    scene_path = os.path.join(out_dir, "unified_scene.blend")
    if not os.path.exists(scene_path):
        raise SystemExit(f"FAIL: unified scene not found: {scene_path}")
    cmd = [PYTHON, os.path.join(HERE, "render_unified.py"),
           "--scene", scene_path, "--scenario", scenario_path, "--out", out_dir,
           "--jobs", str(max(1, jobs)), "--samples", str(samples)]
    if only:
        cmd += ["--only", only]
    run(cmd, check=True, timeout=7200)


def step_sumo_metadata(scenario_path, out_dir, only=None):
    cmd = [PYTHON, os.path.join(HERE, "compute_sumo_metadata.py"),
           "--scenario", scenario_path, "--out", out_dir, "--expected-videos"]
    if only:
        cmd += ["--only", only]
    run(cmd, check=True, timeout=600)


def step_validate_run(out_dir):
    """Validate the run output. FATAL — a partial dataset (missing metadata,
    missing videos, malformed JSON) must bail rather than print "complete"."""
    run([PYTHON, os.path.join(HERE, "validate_run.py"), "--out", out_dir],
        check=True, timeout=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=None,
                    help="frames per second (default: geometry.FPS)")
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="video length in seconds (HARD ceiling). At default demand "
                         "(400 veh/h/approach), ~5-7 vehicles appear in 12s; "
                         "increase --demand-scale for denser traffic.")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "output", "run1"))
    ap.add_argument("--only", help="render only this camera (debug)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel render workers (0 = auto-detect from free "
                         "VRAM capped by --max-workers-per-gpu)")
    ap.add_argument("--max-workers-per-gpu", type=int,
                    default=MAX_WORKERS_PER_GPU,
                    help=f"cap on Blender render workers per GPU regardless of "
                         f"free VRAM (default {MAX_WORKERS_PER_GPU}). One worker "
                         f"per GPU avoids VRAM/context contention.")
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
                         "model is used.")
    ap.add_argument("--demand-scale", type=float, default=None,
                    help="density multiplier on the default demand model "
                         "(default: 1.0 when --demand is not given). E.g. "
                         "--demand-scale 3 makes ~1200 veh/h/approach -> denser "
                         "on-screen traffic, no JSON file needed. "
                         "Scale <= 0 produces zero vehicles. "
                         "Ignored when --demand is a path.")
    ap.add_argument("--simulator", type=str, default=None,
                    choices=["legacy", "micro", "research", "sumo"],
                    help="simulation engine: 'legacy' (event-driven, default) "
                         "'micro' (IDM prototype), 'research' "
                         "(formal state-based simulation kernel), or 'sumo' "
                         "(SUMO/TraCI unified trajectory pipeline)")
    ap.add_argument("--phase", type=str, default="all",
                    choices=["all", "cpu1", "gpu", "cpu2"],
                    help="pipeline phase to run: "
                         "'all' (default, full pipeline), "
                         "'cpu1' (steps 0-3: env+assets+scenario+plates, CPU only), "
                         "'gpu' (step 4: build+render+encode, GPU required), "
                         "'cpu2' (steps 5-6: metadata+validation, CPU only)")
    args = ap.parse_args()

    fps = args.fps
    seconds = args.seconds
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    fps_label = f"{fps} fps" if fps else "default fps"
    sim_label = args.simulator or "legacy"
    print("=" * 60)
    print(f"PIPELINE  seed={args.seed}  "
          f"seconds={seconds}s ({fps_label})  out={out_dir}")
    print(f"  python    : {PYTHON}")
    print(f"  blender   : {BLENDER}")
    print(f"  simulator : {sim_label}")
    print("=" * 60)

    phase = args.phase
    sumo_unified = args.simulator == "sumo"

    ENV.validate_all_envs(ROOT)
    print("[0/5] Env files OK")

    # ---- cpu1: steps 0-3 ---------------------------------------------------
    if phase in ("all", "cpu1"):
        if not args.skip_asset_check:
            print("\n[1/5] Asset validation")
            step_assets_validate()

        print("\n[2/5] Scenario generation")
        if sumo_unified:
            if args.signal:
                print("[sumo] --signal ignored: SUMO build uses a traffic-light junction")
            if args.demand:
                print("[sumo] --demand JSON ignored in unified mode; use --demand-scale")
            scn = step_sumo_scenario(args.seed, seconds, out_dir, fps=fps,
                                     demand_scale=args.demand_scale)
        else:
            scn = step_scenario(args.seed, seconds, out_dir, fps=fps,
                                signal=args.signal, signal_mode=args.signal_mode,
                                demand=args.demand, demand_scale=args.demand_scale,
                                simulator=args.simulator)

        print("\n[3/5] Plate pre-generation")
        step_plates(scn, out_dir)

        if phase == "cpu1":
            print("\n" + "=" * 60)
            print("PHASE cpu1 COMPLETE — copy output dir to GPU host, then run:")
            print(f"  bash scripts/run_gpu.sh --out {out_dir} [render options]")
            print("=" * 60)
            return

    # For gpu / cpu2 phases, scenario.json must already exist
    scn = os.path.join(out_dir, "scenario.json")
    if phase in ("gpu", "cpu2") and not os.path.exists(scn):
        raise SystemExit(
            f"FAIL: scenario.json not found at {scn}\n"
            f"  Run cpu1 phase first: bash scripts/run_cpu1.sh --out {out_dir} ...")

    # ---- gpu: step 4 --------------------------------------------------------
    if phase in ("all", "gpu"):
        if sumo_unified:
            print("\n[4a/6] Build unified SUMO scene")
            step_sumo_unified_build(scn, out_dir)

            print("\n[4b/6] Render unified SUMO scene")
            n_jobs, _ = _detect_jobs(
                8 if not args.only else 1, explicit=args.jobs,
                max_workers_per_gpu=args.max_workers_per_gpu,
                vram_budget=VRAM_PER_JOB_MIB * 2)
            step_sumo_unified_render(scn, out_dir, n_jobs, args.samples, args.only)

            if phase == "gpu":
                print("\n" + "=" * 60)
                print("PHASE gpu COMPLETE — copy output dir back to CPU host, then run:")
                print(f"  bash scripts/run_cpu2.sh --out {out_dir} --simulator sumo")
                print("=" * 60)
                return
        else:
            print("\n[4a/6] Build cached camera scenes")
            step_build_scenes_parallel(scn, out_dir, only=args.only,
                                       silence_timeout_s=args.silence_timeout)

            # Per-scene VRAM estimate: scale budget up if the scenario has many
            # vehicles (>8 means a heavy scene; each extra vehicle adds BVH nodes,
            # textures, and frame buffer pressure).  This lets _detect_jobs choose
            # fewer workers automatically on heavy runs without user intervention.
            _veh_count = 0
            try:
                with open(scn) as _f:
                    _veh_count = len(json.load(_f).get("vehicles", []))
            except Exception:
                pass
            _vram_budget = VRAM_PER_JOB_MIB if _veh_count <= 8 else (
                VRAM_PER_JOB_MIB + (_veh_count - 8) * 60)

            print("\n[4b/6] Render cached camera scenes sequentially "
                  "(JPEG frames; encode+cleanup after each camera)")
            _n_jobs_detected, _gpu_assign_detected = _detect_jobs(
                8 if not args.only else 1, explicit=args.jobs,
                max_workers_per_gpu=args.max_workers_per_gpu,
                vram_budget=_vram_budget)
            # Storage-optimized Option A: force one render worker at a time.
            # With skip_encode=False, render.py encodes the camera immediately
            # and removes frames before the next camera starts.
            n_jobs = 1
            gpu_assign = [_gpu_assign_detected[0] if _gpu_assign_detected else 0]

            render_error = None
            try:
                step_render_parallel(scn, out_dir, jobs=n_jobs,
                                     gpu_assignment=gpu_assign,
                                     only=args.only,
                                     silence_timeout_s=args.silence_timeout,
                                     samples=args.samples,
                                      skip_encode=False)
            except SystemExit as e:
                render_error = e
            except Exception as e:
                render_error = e

            if render_error is not None:
                raise SystemExit(render_error)

            if phase == "gpu":
                print("\n" + "=" * 60)
                print("PHASE gpu COMPLETE — copy output dir back to CPU host, then run:")
                print(f"  bash scripts/run_cpu2.sh --out {out_dir}")
                print("=" * 60)
                return

    # ---- cpu2: steps 5-6 ---------------------------------------------------
    if phase in ("all", "cpu2"):
        # In split mode metadata wasn't run alongside render, so run it now.
        if phase == "cpu2":
            print("\n[5/6] Metadata generation")
            if sumo_unified:
                step_sumo_metadata(scn, out_dir, args.only)
            else:
                step_metadata(scn, out_dir, args.only, True)
        elif phase == "all":
            # 'all' mode: metadata ran in parallel with render above, but the
            # parallel ThreadPoolExecutor was removed in the gpu block refactor.
            # Run it sequentially here so 'all' mode still produces metadata.
            print("\n[5/6] Metadata generation")
            if sumo_unified:
                step_sumo_metadata(scn, out_dir, args.only)
            else:
                step_metadata(scn, out_dir, args.only, True)

        print("\n[6/6] Run validation")
        step_validate_run(out_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  output dir: {out_dir}")
    print(f"  metadata: {os.path.join(out_dir, 'metadata.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
