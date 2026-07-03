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
#   --seconds F         Minimum video length in seconds (default: 12).
#                       The actual duration auto-extends to fit all vehicles.
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
#   --jobs N            Parallel render workers (default: auto-detect from free VRAM)
#   --blender PATH      Path to blender binary (default: auto-detect)
#   --python PATH       Path to python binary (default: DOAN_PYTHON env or $PATH)
#   -h, --help          Show this help

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
BLENDER_BIN=""
PYTHON_BIN=""
JOBS=0   # 0 = auto-detect from free VRAM

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
        --blender)        BLENDER_BIN="$2";   shift 2 ;;
        --python)         PYTHON_BIN="$2";    shift 2 ;;
        --jobs)           JOBS="$2";          shift 2 ;;
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
[[ "$SIGNAL" -eq 1 ]]        && PIPELINE_ARGS+=(--signal)
[[ "$SIGNAL" -eq 1 ]]        && PIPELINE_ARGS+=(--signal-mode "$SIGNAL_MODE")
[[ -n "$DEMAND" ]]           && PIPELINE_ARGS+=(--demand "$DEMAND")

# Pass blender/python overrides to the pipeline via env so subprocesses pick
# them up without needing extra flags on run_pipeline.py.
export DOAN_PYTHON="$PYTHON_BIN"
# run_pipeline.py resolves BLENDER via shutil.which; override PATH if needed.
BLENDER_DIR="$(dirname "$BLENDER_BIN")"
export PATH="$BLENDER_DIR:$PATH"

echo ""
echo "$ $PYTHON_BIN $SCRIPT_DIR/run_pipeline.py ${PIPELINE_ARGS[*]}"
"$PYTHON_BIN" "$SCRIPT_DIR/run_pipeline.py" "${PIPELINE_ARGS[@]}"

echo ""
echo "============================================================"
echo "DONE.  Output: $OUT_DIR"
echo "============================================================"
