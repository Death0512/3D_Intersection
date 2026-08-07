#!/usr/bin/env python3
"""Phase 5 — Pipeline driver (SUMO unified).

Orchestrates the full dataset generation:
  [0/5] Validate env files       (venv python: envfile)
  [1/5] (Optional) Validate assets  (blender headless: validate_assets.py)
  [2/5] SUMO simulation + trajectory export  (run_sumo_unified.py)
  [3/5] Pre-generate plate PNGs   (conda/venv python: gen_plate batch)
  [4/5] Build + render all cameras (blender headless: build_unified_scene + render_unified)
  [5/5] Metadata + run validation (python)

--seconds sets the rendered clip length. SUMO generates a steady-state
warm-up stream and admits vehicles whose motion intersects the fixed window;
render/metadata clamp per-frame outputs to the clip bounds.

Called by scripts/run_all.sh, the sole supported entry point.

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
MAX_WORKERS_PER_GPU = 1

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


def camera_tags_from_only(only):
    if not only:
        return G.camera_names()
    tags = [t.strip() for t in str(only).split(",") if t.strip()]
    valid = set(G.camera_names())
    bad = [t for t in tags if t not in valid]
    if bad:
        raise SystemExit(f"FAIL: invalid --only camera tag(s): {', '.join(bad)}")
    return tags


def vehicle_visible_for_camera(veh, camera_tag):
    role, direction_s = camera_tag.split("_", 1)
    direction = G.Direction(direction_s)
    if role == "in":
        return veh.get("approach") == direction.value
    ex_dir, _ex_lane = G.exit_lane_for_movement(
        G.Direction(veh["approach"]), veh["lane"], G.Turn(veh["turn"]))
    return ex_dir == direction


def filter_vehicles_for_cameras(scenario, only):
    tags = camera_tags_from_only(only)
    if not only:
        return list(scenario.get("vehicles", []))
    visible = []
    for veh in scenario.get("vehicles", []):
        if any(vehicle_visible_for_camera(veh, tag) for tag in tags):
            visible.append(veh)
    return visible


def write_sumo_blender_scenario(scenario_path, out_dir, only=None):
    """Write a Blender-only scenario subset for selected cameras.

    SUMO 300s dense scenarios can contain hundreds of MB of trajectories across
    all four directions.  When rendering only `in_N,out_N`, making Blender load
    every unrelated vehicle is pure RAM/build-time waste.  Metadata/validation
    still use the original `scenario.json`; this filtered copy is only for scene
    build/render FPS lookup.
    """
    if not only:
        return scenario_path
    with open(scenario_path) as f:
        scenario = json.load(f)
    total = len(scenario.get("vehicles", []))
    selected = filter_vehicles_for_cameras(scenario, only)
    filtered = dict(scenario)
    filtered["vehicles"] = selected
    filtered["filtered_from"] = os.path.basename(scenario_path)
    filtered["filtered_for_cameras"] = camera_tags_from_only(only)
    filtered_path = os.path.join(out_dir, "scenario_blender.json")
    with open(filtered_path, "w") as f:
        json.dump(filtered, f, separators=(",", ":"))
    print(f"  [sumo] blender scenario subset: {len(selected)}/{total} vehicles -> {filtered_path}", flush=True)
    return filtered_path


def step_sumo_scenario(seed, seconds, out_dir, fps=None, demand_scale=None,
                       demand_profile=None):
    """Run SUMO/TraCI once and write scenario.json with per-frame trajectories."""
    cmd = [PYTHON, os.path.join(HERE, "run_sumo_unified.py"),
           "--seed", str(seed), "--seconds", str(seconds), "--out", out_dir]
    if fps is not None:
        cmd += ["--fps", str(fps)]
    if demand_scale is not None:
        cmd += ["--demand-scale", str(demand_scale)]
    if demand_profile:
        cmd += ["--demand-profile", str(demand_profile)]
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


CHUNK_SIZE = 7500  # global frames per chunk


def _stream_scenario_duration_frames(path: str) -> int:
    """Stream-parse only duration_frames from scenario JSON."""
    import ijson
    with open(path, "rb") as f:
        for prefix, event, value in ijson.parse(f, use_float=True):
            if prefix == "duration_frames" and event == "number":
                return int(value)
            if prefix.startswith("vehicles."):
                break
    return 0


def step_sumo_unified_build(scenario_path, out_dir, only=None,
                            keyframe_stride=6,
                            heading_threshold_deg=1.0,
                            speed_threshold=0.8,
                            force_rebuild=False):
    """Build ONE .blend per CHUNK_SIZE global-frame chunk containing ALL
    selected cameras and every vehicle whose trajectory overlaps that window."""
    tags = camera_tags_from_only(only)
    chunks_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    duration_frames = _stream_scenario_duration_frames(scenario_path)
    if duration_frames <= 0:
        raise SystemExit(f"{scenario_path} has missing/zero duration_frames")
    total_chunks = max(1, (duration_frames + CHUNK_SIZE - 1) // CHUNK_SIZE)

    print(f"  [build] {total_chunks} chunk(s) ({CHUNK_SIZE} frames each, "
          f"{len(tags)} camera(s))", flush=True)

    for ci in range(total_chunks):
        c_start = ci * CHUNK_SIZE
        c_endl = min(duration_frames - 1, (ci + 1) * CHUNK_SIZE - 1)
        chunk_out = os.path.join(chunks_dir, f"chunk_{ci:04d}.blend")

        if not force_rebuild and os.path.isfile(chunk_out) and os.path.getsize(chunk_out) > 0:
            print(f"  [build] chunk {ci}/{total_chunks - 1} "
                  f"[{c_start},{c_endl}] already exists — skip "
                  f"({os.path.getsize(chunk_out) // 1024 // 1024} MB)", flush=True)
            continue

        print(f"  [build] chunk {ci}/{total_chunks - 1} "
              f"[{c_start},{c_endl}] building...", flush=True)
        cmd = [BLENDER, "-b", "--python-exit-code", "1", "--python", os.path.join(HERE, "build_unified_scene.py"), "--",
               "--scenario", scenario_path, "--out", chunk_out,
               "--keyframe-stride", str(keyframe_stride),
               "--heading-threshold-deg", str(heading_threshold_deg),
               "--speed-threshold", str(speed_threshold),
               "--only", ",".join(tags),
               "--chunk-start", str(c_start),
               "--chunk-end", str(c_endl)]
        run(cmd, check=True, timeout=7200)

    return chunks_dir


def step_sumo_unified_render(scenario_path, out_dir, jobs, samples,
                              only=None):
    chunks_dir = os.path.join(out_dir, "chunks")
    cmd = [PYTHON, os.path.join(HERE, "render_unified.py"),
           "--scene", chunks_dir, "--scenario", scenario_path, "--out", out_dir,
           "--jobs", str(max(1, jobs)), "--samples", str(samples),
           "--batch-size", str(CHUNK_SIZE)]
    if only:
        cmd += ["--only", only]
    run(cmd, check=True, timeout=None)


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
    ap.add_argument("--only", help="render only this camera or comma-separated cameras, e.g. in_N,out_N")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel render workers (0 = auto-detect from free "
                         "VRAM capped by --max-workers-per-gpu)")
    ap.add_argument("--max-workers-per-gpu", type=int,
                    default=MAX_WORKERS_PER_GPU,
                    help=f"cap on Blender render workers per GPU regardless of "
                         f"free VRAM (default {MAX_WORKERS_PER_GPU}). One worker "
                         f"per GPU avoids VRAM/context contention.")
    ap.add_argument("--samples", type=int, default=48,
                    help="Cycles render samples per frame (default 48; lower "
                         "= faster, noisier — denoiser compensates. Use 16-24 "
                         "for quick test runs, 48 for production.)")
    ap.add_argument("--skip-asset-check", action="store_true")
    ap.add_argument("--demand-scale", type=float, default=None,
                    help="density multiplier on the default demand model "
                         "(default: 1.0 when --demand is not given). E.g. "
                         "--demand-scale 3 makes ~1200 veh/h/approach -> denser "
                         "on-screen traffic, no JSON file needed. "
                         "Scale <= 0 produces zero vehicles. "
                         "Ignored when --demand is a path.")
    ap.add_argument("--demand-profile", type=str, default=None,
                    help="SUMO-only time-varying demand profile. Currently supports "
                         "spike:start=55,end=65,scale=20 (base demand outside, "
                         "base*scale inside).")
    ap.add_argument("--keyframe-stride", type=int, default=6,
                    help="SUMO unified Blender build: fallback keyframe spacing "
                         "for straight/steady trajectory runs (default 6 = 5 FPS at 30 FPS).")
    ap.add_argument("--heading-threshold-deg", type=float, default=1.0,
                    help="SUMO unified Blender build: keep extra keyframes when "
                         "heading changes by more than this many degrees.")
    ap.add_argument("--speed-threshold", type=float, default=0.8,
                    help="SUMO unified Blender build: keep extra keyframes when "
                         "speed changes by more than this many m/s.")
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
    print("=" * 60)
    print(f"PIPELINE  seed={args.seed}  "
          f"seconds={seconds}s ({fps_label})  out={out_dir}")
    print(f"  python    : {PYTHON}")
    print(f"  blender   : {BLENDER}")
    print("  simulator : sumo")
    print("=" * 60)

    phase = args.phase

    ENV.validate_all_envs(ROOT)
    print("[0/5] Env files OK")

    # ---- cpu1: steps 0-3 ---------------------------------------------------
    if phase in ("all", "cpu1"):
        if not args.skip_asset_check:
            print("\n[1/5] Asset validation")
            step_assets_validate()

        print("\n[2/5] Scenario generation")
        scn = step_sumo_scenario(args.seed, seconds, out_dir, fps=fps,
                                 demand_scale=args.demand_scale,
                                 demand_profile=args.demand_profile)
        scn_blender = write_sumo_blender_scenario(scn, out_dir, args.only)

        print("\n[3/5] Plate pre-generation")
        step_plates(scn_blender, out_dir)

        if phase == "cpu1":
            print("\n" + "=" * 60)
            print("PHASE cpu1 COMPLETE — copy output dir to GPU host, then run:")
            print(f"  bash scripts/run_all.sh --out {out_dir} --phase vm [render options]")
            print("=" * 60)
            return

    # For gpu / cpu2 phases, scenario.json must already exist
    scn = os.path.join(out_dir, "scenario.json")
    scn_blender = os.path.join(out_dir, "scenario_blender.json")
    scn_render = scn_blender if os.path.exists(scn_blender) else scn
    if phase in ("gpu", "cpu2") and not os.path.exists(scn):
        raise SystemExit(
            f"FAIL: scenario.json not found at {scn}\n"
            f"  Run cpu1 phase first: bash scripts/run_all.sh --out {out_dir} --phase cpu1 ...")

    # ---- gpu: step 4 --------------------------------------------------------
    if phase in ("all", "gpu"):
        camera_tags = camera_tags_from_only(args.only)
        print("\n[4a/6] Build time-chunk .blend scenes")
        step_sumo_unified_build(scn_render, out_dir, args.only,
                                keyframe_stride=args.keyframe_stride,
                                heading_threshold_deg=args.heading_threshold_deg,
                                speed_threshold=args.speed_threshold)

        print("\n[4b/6] Render chunk scenes sequentially")
        n_jobs, _ = _detect_jobs(
            len(camera_tags), explicit=args.jobs,
            max_workers_per_gpu=args.max_workers_per_gpu,
            vram_budget=VRAM_PER_JOB_MIB * 2)
        step_sumo_unified_render(scn_render, out_dir, n_jobs, args.samples,
                                 only=args.only)

        if phase == "gpu":
            print("\n" + "=" * 60)
            print("PHASE gpu COMPLETE — run metadata/validation with:")
            print(f"  bash scripts/run_all.sh --out {out_dir} --phase cpu2")
            print("=" * 60)
            return

    # ---- cpu2: steps 5-6 ---------------------------------------------------
    if phase in ("all", "cpu2"):
        print("\n[5/6] Metadata generation")
        step_sumo_metadata(scn, out_dir, args.only)

        print("\n[6/6] Run validation")
        step_validate_run(out_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  output dir: {out_dir}")
    print(f"  metadata: {os.path.join(out_dir, 'metadata.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
