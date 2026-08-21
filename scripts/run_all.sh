#!/usr/bin/env bash
# run_all.sh — Full synthetic CCTV intersection video pipeline (SUMO only).
#
# ponytail: One shell entry point — owns the scenario loop, phase dispatch,
# and all passthrough to run_pipeline.py. No sub-scripts, no legacy simulators.
#
# Required inputs:
#   assets/envs/{in,out}_{N,S,E,W}.json — per-camera env files that drive the
#   camera, road, sun, and per-lane vehicle spawn anchors. The pipeline
#   hard-fails if any of these 8 files is missing or has a null required field.
#
# Usage:
#   bash scripts/run_all.sh [OPTIONS]
#
# Options:
#   --phase PHASE        Pipeline phase: all (default), cpu1, gpu, cpu2, vm.
#                        vm runs --phase gpu then --phase cpu2.
#   --seed N            RNG seed (default: 42)
#   --fps N             Frames per second (default: 30)
#   --seconds F         Video length in seconds (default: 60)
#   --samples N         Cycles render samples (default: 24, pipeline default)
#   --out DIR           Output directory (default: output/run1)
#   --scenarios         Run six SUMO North-camera scenarios (empty, sparse,
#                       moderate, dense, surge_spike, signal_cycle).
#                       Presets: --seconds 300 and --samples 12; renders all
#                       8 cameras unless --only is supplied.
#                       Accepts --seconds/--fps/--samples/--seed/--out/--jobs.
#   --only CAMS         Render only this camera or comma-separated cameras,
#                       e.g. in_N or in_N,out_N
#   --skip-asset-check  Skip blender asset validation step
#   --demand-scale F    Density multiplier on the default demand model
#                       (default: 1.0). E.g. 3 makes ~1200 veh/h/approach.
#   --demand-profile P  SUMO time-varying demand profile, e.g.
#                       spike:start=55,end=65,scale=30
#   --jobs N            Parallel render workers (default: auto-detect from usg
#                       free VRAM, capped by --max-workers-per-gpu)
#   --max-workers-per-gpu N
#                       Cap on Blender workers per GPU (default: 2)
#   --silence-timeout S Render watchdog: kill the pool if a worker is silent
#                       for S seconds (default: 600)
#   --keyframe-stride N SUMO unified only: fallback Blender keyframe spacing
#                       for straight/steady vehicle tracks (default: 6).
#   --heading-threshold-deg F
#                       SUMO unified only: keep extra keyframes when heading
#                       changes more than F degrees (default: 1.0).
#   --speed-threshold F
#                       SUMO unified only: keep extra keyframes when speed
#                       changes more than F m/s (default: 0.8).
#   --blender PATH      Path to blender binary (default: auto-detect)
#   --python PATH       Path to python binary (default: DOAN_PYTHON env or $PATH)
#   -h, --help          Show this help
#
# Logging: all output (stdout+stderr) is tee'd to <out>/run_<timestamp>.log
# and a stable <out>/latest.log symlink. The log captures the command, host,
# timestamps, and the pipeline's exit code for post-mortem.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SEED=42
FPS=30
SECONDS_VAL=60
SECONDS_EXPLICIT=0
OUT_DIR=""            # empty → auto-selected per host (see below)
ONLY=""
SKIP_ASSET_CHECK=0
DEMAND_SCALE=""
DEMAND_PROFILE=""
BLENDER_BIN=""
PYTHON_BIN=""
JOBS=0   # 0 = auto-detect from free VRAM
MAX_WORKERS_PER_GPU=1
SILENCE_TIMEOUT=0    # 0 = use run_pipeline.py default (600s)
SAMPLES=0            # 0 = use pipeline default (24)
KEYFRAME_STRIDE=""
HEADING_THRESHOLD_DEG=""
SPEED_THRESHOLD=""
PHASE="all"
SCENARIOS=0
# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)             SEED="$2";           shift 2 ;;
        --fps)              FPS="$2";            shift 2 ;;
        --seconds)          SECONDS_VAL="$2"; SECONDS_EXPLICIT=1; shift 2 ;;
        --samples)          SAMPLES="$2";        shift 2 ;;
        --out)              OUT_DIR="$2";        shift 2 ;;
        --phase)            PHASE="$2";          shift 2 ;;
        --scenarios)        SCENARIOS=1;         shift ;;
        --only)             ONLY="$2";           shift 2 ;;
        --skip-asset-check) SKIP_ASSET_CHECK=1;  shift ;;
        --demand-scale)     DEMAND_SCALE="$2";   shift 2 ;;
        --demand-profile)   DEMAND_PROFILE="$2"; shift 2 ;;
        --blender)          BLENDER_BIN="$2";    shift 2 ;;
        --python)           PYTHON_BIN="$2";     shift 2 ;;
        --jobs)             JOBS="$2";           shift 2 ;;
        --max-workers-per-gpu) MAX_WORKERS_PER_GPU="$2"; shift 2 ;;
        --silence-timeout)  SILENCE_TIMEOUT="$2";  shift 2 ;;
        --keyframe-stride)  KEYFRAME_STRIDE="$2";  shift 2 ;;
        --heading-threshold-deg) HEADING_THRESHOLD_DEG="$2"; shift 2 ;;
        --speed-threshold)  SPEED_THRESHOLD="$2";   shift 2 ;;
        -h|--help)          usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths + detect host (Kaggle vs local Ubuntu)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ponytail: scenario loop formerly in run_north_scenarios.sh, now owned here.
if [[ "$SCENARIOS" -eq 1 ]]; then
    # Presets when user didn't override.
    [[ "$SECONDS_EXPLICIT" -eq 0 ]] && SECONDS_VAL=300
    [[ "$SAMPLES" -eq 0 ]] && SAMPLES=12
    [[ -z "$KEYFRAME_STRIDE" ]] && KEYFRAME_STRIDE=10
    [[ -z "$HEADING_THRESHOLD_DEG" ]] && HEADING_THRESHOLD_DEG=1.5
    [[ -z "$SPEED_THRESHOLD" ]] && SPEED_THRESHOLD=1.0

    OUT_ROOT="${OUT_DIR:-$ROOT_DIR/output/north}"
    mkdir -p "$OUT_ROOT"

    _base_flags=(
        --seconds "$SECONDS_VAL" --fps "$FPS" --seed "$SEED"
        --samples "$SAMPLES" --phase "$PHASE" --only "$ONLY"
    )
    [[ "$SKIP_ASSET_CHECK" -eq 1 ]] && _base_flags+=(--skip-asset-check)
    [[ "$JOBS" -gt 0 ]] && _base_flags+=(--jobs "$JOBS")
    [[ "$MAX_WORKERS_PER_GPU" -gt 0 ]] && _base_flags+=(--max-workers-per-gpu "$MAX_WORKERS_PER_GPU")
    [[ "$SILENCE_TIMEOUT" -gt 0 ]] && _base_flags+=(--silence-timeout "$SILENCE_TIMEOUT")
    [[ -n "$KEYFRAME_STRIDE" ]] && _base_flags+=(--keyframe-stride "$KEYFRAME_STRIDE")
    [[ -n "$HEADING_THRESHOLD_DEG" ]] && _base_flags+=(--heading-threshold-deg "$HEADING_THRESHOLD_DEG")
    [[ -n "$SPEED_THRESHOLD" ]] && _base_flags+=(--speed-threshold "$SPEED_THRESHOLD")
    [[ -n "$BLENDER_BIN" ]] && _base_flags+=(--blender "$BLENDER_BIN")
    [[ -n "$PYTHON_BIN" ]] && _base_flags+=(--python "$PYTHON_BIN")

    _run_one() {
        local name="$1" scale="$2" profile="$3"
        local out="$OUT_ROOT/$name"
        echo "============================================================"
        echo "SCENARIO: $name  scale=$scale  profile=${profile:-none}"
        echo "out: $out"
        echo "============================================================"
        set +e
        bash "$SCRIPT_DIR/run_all.sh" \
            --out "$out" --demand-scale "$scale" \
            ${profile:+--demand-profile "$profile"} \
            "${_base_flags[@]}"
        local rc=$?
        set -e
        if [[ "$rc" -ne 0 ]]; then
            echo "SCENARIO $name FAILED (exit $rc)" >&2
            exit "$rc"
        fi
    }

    SPIKE_START=$(( SECONDS_VAL * 2 / 5 ))
    SPIKE_END=$(( SPIKE_START + SECONDS_VAL / 15 ))

    _run_one empty        0 ""
    _run_one sparse       1 ""
    _run_one moderate     3 ""
    _run_one dense        8 ""
    _run_one surge_spike  1 "spike:start=$SPIKE_START,end=$SPIKE_END,scale=8"
    _run_one signal_cycle 5 ""

    echo ""
    echo "All scenarios complete: $OUT_ROOT"
    exit 0
fi

# Host detection: Kaggle notebooks set KAGGLE_KERNEL_RUN_TYPE and create a
# /kaggle tree. We use either signal to switch the default out dir and the
# install hint below.
if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] || [[ -d /kaggle ]]; then
    IS_KAGGLE=1
else
    IS_KAGGLE=0
fi

# Auto-source the installer-written env.sh if it exists (sets DOAN_PYTHON, puts
# blender on PATH). User-supplied --blender / --python still take priority.
_ENV_SH="$SCRIPT_DIR/env.sh"
if [[ -f "$_ENV_SH" ]]; then
    source "$_ENV_SH"
fi

if [[ -z "$OUT_DIR" ]]; then
    if [[ "$IS_KAGGLE" -eq 1 ]]; then
        OUT_DIR="/kaggle/working/run1"
    else
        OUT_DIR="$ROOT_DIR/output/run1"
    fi
fi
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

# Python interpreter: --python flag > DOAN_PYTHON env > DoAn conda env > sys.executable
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -n "${DOAN_PYTHON:-}" && -x "${DOAN_PYTHON}" ]]; then
        PYTHON_BIN="$DOAN_PYTHON"
    elif [[ -x "$HOME/miniconda3/envs/DoAn/bin/python3" ]]; then
        PYTHON_BIN="$HOME/miniconda3/envs/DoAn/bin/python3"
    else
        PYTHON_BIN="$(command -v python3 || command -v python)"
    fi
fi

# Blender binary: --blender flag > PATH (which env.sh may have prepended)
if [[ -z "$BLENDER_BIN" ]]; then
    BLENDER_BIN="$(command -v blender 2>/dev/null || true)"
    if [[ -z "$BLENDER_BIN" ]]; then
        echo "ERROR: blender not found on PATH. Run scripts/install.sh first or use --blender /path/to/blender" >&2
        if [[ "$IS_KAGGLE" -eq 1 ]]; then
            echo "        (On Kaggle: bash scripts/install.sh --yes  — installs Blender 5.2.x under \$HOME/.local)" >&2
        fi
        exit 1
    fi
fi

# Resolve --blender to an absolute path so later os.chdir() inside Blender/Python
# processes can't break the relative PATH we export below.
if [[ -n "$BLENDER_BIN" ]]; then
    BLENDER_BIN="$(readlink -f "$BLENDER_BIN" 2>/dev/null || realpath "$BLENDER_BIN")"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
if [[ "$IS_KAGGLE" -eq 1 ]]; then
    echo "PIPELINE [Kaggle]  seed=$SEED  seconds=${SECONDS_VAL}s @ ${FPS}fps  out=$OUT_DIR  phase=$PHASE"
else
    echo "PIPELINE  seed=$SEED  seconds=${SECONDS_VAL}s @ ${FPS}fps  out=$OUT_DIR  phase=$PHASE"
fi
echo "  python : $PYTHON_BIN"
echo "  blender: $BLENDER_BIN"
echo "  simulator: sumo"
[[ -n "$ONLY" ]] && echo "  only   : $ONLY"
[[ -n "$DEMAND_SCALE" ]] && echo "  demand : default (scale=${DEMAND_SCALE})"
echo "============================================================"

# ---------------------------------------------------------------------------
# Build run_pipeline.py args and delegate
# ---------------------------------------------------------------------------
PIPELINE_ARGS=(
    --seed "$SEED"
    --fps "$FPS"
    --seconds "$SECONDS_VAL"
    --out "$OUT_DIR"
)
[[ -n "$ONLY" ]]             && PIPELINE_ARGS+=(--only "$ONLY")
[[ "$SKIP_ASSET_CHECK" -eq 1 ]] && PIPELINE_ARGS+=(--skip-asset-check)
[[ "$JOBS" -gt 0 ]]         && PIPELINE_ARGS+=(--jobs "$JOBS")
[[ "$MAX_WORKERS_PER_GPU" -gt 0 ]] && PIPELINE_ARGS+=(--max-workers-per-gpu "$MAX_WORKERS_PER_GPU")
[[ "$SILENCE_TIMEOUT" -gt 0 ]] && PIPELINE_ARGS+=(--silence-timeout "$SILENCE_TIMEOUT")
[[ "$SAMPLES" -gt 0 ]]       && PIPELINE_ARGS+=(--samples "$SAMPLES")
[[ -n "$DEMAND_SCALE" ]]    && PIPELINE_ARGS+=(--demand-scale "$DEMAND_SCALE")
if [[ -n "$DEMAND_PROFILE" ]]; then
    # Auto-prefix file: if user passed a bare .json path instead of the
    # full file:/path syntax.
    case "$DEMAND_PROFILE" in
        file:*) : ;;
        *.json) DEMAND_PROFILE="file:${DEMAND_PROFILE}" ;;
    esac
    PIPELINE_ARGS+=(--demand-profile "$DEMAND_PROFILE")
fi
[[ -n "$KEYFRAME_STRIDE" ]]  && PIPELINE_ARGS+=(--keyframe-stride "$KEYFRAME_STRIDE")
[[ -n "$HEADING_THRESHOLD_DEG" ]] && PIPELINE_ARGS+=(--heading-threshold-deg "$HEADING_THRESHOLD_DEG")
[[ -n "$SPEED_THRESHOLD" ]]  && PIPELINE_ARGS+=(--speed-threshold "$SPEED_THRESHOLD")

# C5: force Python stdout unbuffered so every print(..., flush=True) in
# render.py / build_scene.py / run_pipeline.py reaches the terminal/log
# immediately. Without this, Python block-buffers stdout (4 KB) when not a
# TTY, and a long render appears silent for minutes even when progressing.
export PYTHONUNBUFFERED=1

# Pass blender/python overrides to the pipeline via env so subprocesses pick
# them up without needing extra flags on run_pipeline.py.
export DOAN_PYTHON="$PYTHON_BIN"
# run_pipeline.py resolves BLENDER via shutil.which; override PATH if needed.
BLENDER_DIR="$(dirname "$BLENDER_BIN")"
export PATH="$BLENDER_DIR:$PATH"

# ---------------------------------------------------------------------------
# VM phase: run gpu then cpu2 sequentially (no Python "vm" phase)
# ---------------------------------------------------------------------------
if [[ "$PHASE" == "vm" ]]; then
    LOG_TS="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$OUT_DIR/run_${LOG_TS}.log"
    LATEST_LINK="$OUT_DIR/latest.log"
    {
        echo "run_all.sh log (vm) — started $(date -Iseconds)"
        echo "host: $(hostname)  user: ${USER:-?}  kaggle: $IS_KAGGLE"
        echo "python : $PYTHON_BIN"
        echo "blender: $BLENDER_BIN"
        echo "args   : $*"
        echo "out    : $OUT_DIR"
        echo "============================================================"
    } > "$LOG_FILE"

    run_pipeline_step() {
        local phase="$1"
        echo ""
        echo "$ $PYTHON_BIN $SCRIPT_DIR/run_pipeline.py ${PIPELINE_ARGS[*]} --phase $phase"
        set +e
        "$PYTHON_BIN" "$SCRIPT_DIR/run_pipeline.py" "${PIPELINE_ARGS[@]}" --phase "$phase" 2>&1 | tee -a "$LOG_FILE"
        local rc=${PIPESTATUS[0]}
        set -e
        if [[ "$rc" -ne 0 ]]; then
            echo "FAILED phase $phase (exit $rc)  log: $LOG_FILE" | tee -a "$LOG_FILE"
            exit "$rc"
        fi
    }
    run_pipeline_step gpu
    run_pipeline_step cpu2

    {
        echo "============================================================"
        echo "ended $(date -Iseconds) (exit 0)"
    } >> "$LOG_FILE"
    ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LINK"
    echo ""
    echo "PHASE VM COMPLETE. Output: $OUT_DIR"
    echo "log: $LOG_FILE  (latest: $LATEST_LINK)"
    echo "============================================================"
    exit 0
fi

# ---------------------------------------------------------------------------
# Logging: tee all output (stdout+stderr) to a timestamped log file inside the
# output dir, plus a stable `latest.log` symlink. Keeps the terminal live while
# producing a persistent record for post-mortem (watchdog aborts, per-frame
# progress, ffmpeg stderr, [WARN] lines). The log captures the exact command,
# start/end timestamps, and the pipeline's exit code.
# ---------------------------------------------------------------------------
LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/run_${LOG_TS}.log"
LATEST_LINK="$OUT_DIR/latest.log"
# Header written only to the log (the terminal already has the summary box).
{
    echo "run_all.sh log — started $(date -Iseconds)"
    echo "host: $(hostname)  user: ${USER:-?}  kaggle: $IS_KAGGLE"
    echo "python : $PYTHON_BIN"
    echo "blender: $BLENDER_BIN"
    echo "args   : $*"
    echo "out    : $OUT_DIR"
    echo "============================================================"
} > "$LOG_FILE"

echo ""
echo "$ $PYTHON_BIN $SCRIPT_DIR/run_pipeline.py ${PIPELINE_ARGS[*]} --phase $PHASE"
# 2>&1 merges stderr into the stream so [WARN]/tracebacks land in the log too.
# `tee -a` appends (the header above already opened the file); the pipeline's
# own stdout/stderr pass through to the terminal unchanged.
# PIPESTATUS[0] captures the pipeline's real exit code (tee at PIPESTATUS[1]
# would otherwise mask a non-zero rc as 0). pipefail (set at line 35) also
# propagates the failing rc out of the pipe, but we record PIPESTATUS in the
# log regardless so the exact pipeline rc (not tee's) is preserved.
set +e
"$PYTHON_BIN" "$SCRIPT_DIR/run_pipeline.py" "${PIPELINE_ARGS[@]}" --phase "$PHASE" 2>&1 | tee -a "$LOG_FILE"
PIPELINE_RC=${PIPESTATUS[0]}
set -e
{
    echo "============================================================"
    echo "run_all.sh log — ended $(date -Iseconds) (pipeline exit $PIPELINE_RC)"
} >> "$LOG_FILE"
# Refresh the latest.log symlink to point at this run's log (atomic replace).
ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LINK"
echo ""
echo "log: $LOG_FILE  (latest: $LATEST_LINK)"

echo ""
echo "============================================================"
if [[ "$PIPELINE_RC" -eq 0 ]]; then
    echo "DONE.  Output: $OUT_DIR"
    echo "  log: $LOG_FILE"
    echo "============================================================"
else
    echo "FAILED (exit $PIPELINE_RC).  Output: $OUT_DIR"
    echo "  log: $LOG_FILE  (see latest: $LATEST_LINK)"
    echo "============================================================"
    exit "$PIPELINE_RC"
fi
