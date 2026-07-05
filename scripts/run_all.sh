#!/usr/bin/env bash
# run_all.sh — Full synthetic CCTV intersection video pipeline
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
#   --seed N            RNG seed (default: 42)
#   --num-vehicles N    Number of vehicles (default: 120)
#   --fps N             Frames per second (default: 30)
#   --seconds F         MINIMUM video length in seconds (default: 12).
#                       The actual video auto-extends to fit all vehicles
#                       (it can be LONGER than this, never shorter). At default
#                       demand (~400 veh/h/approach), ~25 vehicles fit in 12s;
#                       100+ vehicles will spread across minutes.
#   --out DIR           Output directory (default: output/run_car)
#   --only CAM          Render only this camera e.g. in_N (debug)
#   --skip-asset-check  Skip blender asset validation step
#   --signal             Enable traffic signal SPaT gating + queue
#   --signal-mode MODE   Signal controller when --signal is set:
#                        'fixed' (default, 70s cycle permissive-left) or
#                        'adaptive' (NEMA 8-phase MaxPressure, closed-loop on
#                        realised arrivals). Implies --signal.
#   --demand SPEC        Demand model: path to a demand JSON (per-approach
#                        flow veh/h + turning split), or 'none' for the legacy
#                        uniform-random scheduler. Default: built-in demand
#                        model (~400 veh/h/approach, straight-heavy split).
#   --demand-scale F     Density multiplier on the default demand model when
#                        --demand is not given (default: 1.0). E.g. 3 makes
#                        ~1200 veh/h/approach -> denser on-screen traffic in a
#                        shorter clip, no JSON file needed.
#   --jobs N            Parallel render workers (default: auto-detect from free
#                       VRAM, capped by --max-workers-per-gpu)
#   --max-workers-per-gpu N
#                       Cap on Blender workers per GPU (default: 2). The VRAM
#                       budget alone over-packs big cards (a 15 GB T4 lets ~9
#                       jobs fit by VRAM, but 4+ heavy Cycles+OptiX contexts
#                       stall the render — silent hang). Raise for light scenes
#                       only (few vehicles / no signal).
#   --silence-timeout S Render watchdog: kill the pool if a worker is silent
#                       for S seconds (default: 600)
#   --samples N         Cycles render samples (default: 48). Lower = faster,
#                       noisier (denoiser compensates). Try 16-24 for tests.
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
NUM_VEHICLES=120
FPS=30
SECONDS_VAL=5
OUT_DIR=""            # empty → auto-selected per host (see below)
ONLY=""
SKIP_ASSET_CHECK=0  
SIGNAL=0
SIGNAL_MODE="fixed"
DEMAND=""
DEMAND_SCALE=""
BLENDER_BIN=""
PYTHON_BIN=""
JOBS=0   # 0 = auto-detect from free VRAM
MAX_WORKERS_PER_GPU=2   # cap on Blender workers per GPU (prevents VRAM oversubscription hang)
SILENCE_TIMEOUT=0   # 0 = use run_pipeline.py default (600s)
SAMPLES=0           # 0 = use build_scene default (48)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)           SEED="$2";          shift 2 ;;
        --num-vehicles)   NUM_VEHICLES="$2";  shift 2 ;;
        --fps)            FPS="$2";           shift 2 ;;
        --seconds)        SECONDS_VAL="$2";   shift 2 ;;
        --out)            OUT_DIR="$2";       shift 2 ;;
        --only)           ONLY="$2";          shift 2 ;;
        --skip-asset-check) SKIP_ASSET_CHECK=1; shift ;;
        --signal)         SIGNAL=1;           shift ;;
        --signal-mode)    SIGNAL_MODE="$2";   shift 2 ;;
        --demand)         DEMAND="$2";        shift 2 ;;
        --demand-scale)   DEMAND_SCALE="$2";   shift 2 ;;
        --blender)        BLENDER_BIN="$2";   shift 2 ;;
        --python)         PYTHON_BIN="$2";    shift 2 ;;
        --jobs)           JOBS="$2";          shift 2 ;;
        --max-workers-per-gpu) MAX_WORKERS_PER_GPU="$2"; shift 2 ;;
        --silence-timeout) SILENCE_TIMEOUT="$2"; shift 2 ;;
        --samples)        SAMPLES="$2";       shift 2 ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths + detect host (Kaggle vs local Ubuntu)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

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

# Python interpreter: --python flag > DOAN_PYTHON env > sys.executable
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -n "${DOAN_PYTHON:-}" && -x "${DOAN_PYTHON}" ]]; then
        PYTHON_BIN="$DOAN_PYTHON"
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
            echo "        (On Kaggle: bash scripts/install.sh --yes  — installs Blender 5.1.x under \$HOME/.local)" >&2
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
    echo "PIPELINE [Kaggle]  seed=$SEED  n=$NUM_VEHICLES  min=${SECONDS_VAL}s @ ${FPS}fps  out=$OUT_DIR"
else
    echo "PIPELINE  seed=$SEED  n=$NUM_VEHICLES  min=${SECONDS_VAL}s @ ${FPS}fps  out=$OUT_DIR"
fi
echo "  python : $PYTHON_BIN"
echo "  blender: $BLENDER_BIN"
[[ -n "$ONLY" ]] && echo "  only   : $ONLY"
if [[ "$SIGNAL" -eq 1 ]]; then
    echo "  signal : on (mode=$SIGNAL_MODE)"
else
    echo "  signal : off"
fi
if [[ -n "$DEMAND" ]]; then
    echo "  demand : $DEMAND"
elif [[ -n "$DEMAND_SCALE" ]]; then
    echo "  demand : default model (scale=${DEMAND_SCALE})"
else
    echo "  demand : default model"
fi
echo "============================================================"

# ---------------------------------------------------------------------------
# Build run_pipeline.py args and delegate
# ---------------------------------------------------------------------------
PIPELINE_ARGS=(
    --seed "$SEED"
    --num-vehicles "$NUM_VEHICLES"
    --fps "$FPS"
    --seconds "$SECONDS_VAL"
    --out "$OUT_DIR"
)
[[ -n "$ONLY" ]]             && PIPELINE_ARGS+=(--only "$ONLY")
[[ "$SKIP_ASSET_CHECK" -eq 1 ]] && PIPELINE_ARGS+=(--skip-asset-check)
[[ "$JOBS" -gt 0 ]]         && PIPELINE_ARGS+=(--jobs "$JOBS")
[[ "$MAX_WORKERS_PER_GPU" -gt 0 ]] && PIPELINE_ARGS+=(--max-workers-per-gpu "$MAX_WORKERS_PER_GPU")
[[ "$SIGNAL" -eq 1 ]]        && PIPELINE_ARGS+=(--signal)
[[ "$SIGNAL" -eq 1 ]]        && PIPELINE_ARGS+=(--signal-mode "$SIGNAL_MODE")
[[ -n "$DEMAND" ]]           && PIPELINE_ARGS+=(--demand "$DEMAND")
[[ -n "$DEMAND_SCALE" ]]    && PIPELINE_ARGS+=(--demand-scale "$DEMAND_SCALE")
[[ "$SILENCE_TIMEOUT" -gt 0 ]] && PIPELINE_ARGS+=(--silence-timeout "$SILENCE_TIMEOUT")
[[ "$SAMPLES" -gt 0 ]]       && PIPELINE_ARGS+=(--samples "$SAMPLES")

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
echo "$ $PYTHON_BIN $SCRIPT_DIR/run_pipeline.py ${PIPELINE_ARGS[*]}"
# 2>&1 merges stderr into the stream so [WARN]/tracebacks land in the log too.
# `tee -a` appends (the header above already opened the file); the pipeline's
# own stdout/stderr pass through to the terminal unchanged.
# PIPESTATUS[0] captures the pipeline's real exit code (tee at PIPESTATUS[1]
# would otherwise mask a non-zero rc as 0). pipefail (set at line 35) also
# propagates the failing rc out of the pipe, but we record PIPESTATUS in the
# log regardless so the exact pipeline rc (not tee's) is preserved.
set +e
"$PYTHON_BIN" "$SCRIPT_DIR/run_pipeline.py" "${PIPELINE_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
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
