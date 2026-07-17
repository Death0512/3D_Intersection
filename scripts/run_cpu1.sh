#!/usr/bin/env bash
# run_cpu1.sh — Pipeline phase cpu1: steps 0-3 (CPU only)
#
#   [0/5] Env file validation
#   [1/5] Asset validation  (headless Blender, no GPU needed)
#   [2/5] Scenario generation
#   [3/5] Plate pre-generation
#
# Produces scenario.json + plates/ in --out dir. Transfer that dir to the
# GPU host, then run: bash scripts/run_gpu.sh --out <same-dir> [render opts]
#
# All options mirror run_all.sh (render/GPU flags are accepted but ignored).
#
# Usage:
#   bash scripts/run_cpu1.sh [OPTIONS]
#
# Options:
#   --seed N            RNG seed (default: 42)
#   --fps N             Frames per second (default: 30)
#   --seconds F         Video length in seconds (default: 60)
#   --out DIR           Output directory (default: output/run1)
#   --skip-asset-check  Skip blender asset validation step
#   --signal            Enable traffic signal SPaT gating + queue
#   --signal-mode MODE  'fixed' or 'adaptive' (default: adaptive)
#   --demand SPEC       Path to demand JSON
#   --demand-scale F    Density multiplier on default demand model
#   --simulator MODE    'legacy', 'micro', or 'research' (default: research)
#   --python PATH       Path to python binary
#   --blender PATH      Path to blender binary (only needed for asset check)
#   -h, --help          Show this help

set -euo pipefail

SEED=42
FPS=30
SECONDS_VAL=60
OUT_DIR=""
SKIP_ASSET_CHECK=0
SIGNAL=0
SIGNAL_MODE="adaptive"
DEMAND=""
DEMAND_SCALE=""
BLENDER_BIN=""
PYTHON_BIN=""
SIMULATOR="research"

usage() { grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)            SEED="$2";          shift 2 ;;
        --fps)             FPS="$2";           shift 2 ;;
        --seconds)         SECONDS_VAL="$2";   shift 2 ;;
        --out)             OUT_DIR="$2";       shift 2 ;;
        --skip-asset-check) SKIP_ASSET_CHECK=1; shift ;;
        --signal)          SIGNAL=1;           shift ;;
        --signal-mode)     SIGNAL_MODE="$2";   shift 2 ;;
        --demand)          DEMAND="$2";        shift 2 ;;
        --demand-scale)    DEMAND_SCALE="$2";  shift 2 ;;
        --blender)         BLENDER_BIN="$2";   shift 2 ;;
        --python)          PYTHON_BIN="$2";    shift 2 ;;
        --simulator)       SIMULATOR="$2";     shift 2 ;;
        # render/gpu flags silently accepted so callers can pass the same args to all three scripts
        --jobs|--max-workers-per-gpu|--silence-timeout|--samples|--only) shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] || [[ -d /kaggle ]]; then IS_KAGGLE=1; else IS_KAGGLE=0; fi

_ENV_SH="$SCRIPT_DIR/env.sh"
[[ -f "$_ENV_SH" ]] && source "$_ENV_SH"

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$([[ "$IS_KAGGLE" -eq 1 ]] && echo "/kaggle/working/run1" || echo "$ROOT_DIR/output/run1")"
fi
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -n "${DOAN_PYTHON:-}" && -x "${DOAN_PYTHON}" ]]; then
        PYTHON_BIN="$DOAN_PYTHON"
    else
        PYTHON_BIN="$(command -v python3 || command -v python)"
    fi
fi

if [[ -z "$BLENDER_BIN" ]]; then
    BLENDER_BIN="$(command -v blender 2>/dev/null || true)"
fi
[[ -n "$BLENDER_BIN" ]] && BLENDER_BIN="$(readlink -f "$BLENDER_BIN" 2>/dev/null || realpath "$BLENDER_BIN")"

echo "============================================================"
echo "PHASE cpu1  seed=$SEED  seconds=${SECONDS_VAL}s @ ${FPS}fps  out=$OUT_DIR"
echo "  python : $PYTHON_BIN"
[[ -n "$BLENDER_BIN" ]] && echo "  blender: $BLENDER_BIN"
[[ "$SIGNAL" -eq 1 ]] && echo "  signal : on (mode=$SIGNAL_MODE)" || echo "  signal : off"
[[ -n "$DEMAND_SCALE" ]] && echo "  demand : default model (scale=${DEMAND_SCALE})" || echo "  demand : ${DEMAND:-default model}"
echo "  simulator: $SIMULATOR"
echo "============================================================"

PIPELINE_ARGS=(
    --seed "$SEED" --fps "$FPS" --seconds "$SECONDS_VAL" --out "$OUT_DIR"
    --phase cpu1
)
[[ "$SKIP_ASSET_CHECK" -eq 1 ]] && PIPELINE_ARGS+=(--skip-asset-check)
[[ "$SIGNAL" -eq 1 ]]           && PIPELINE_ARGS+=(--signal --signal-mode "$SIGNAL_MODE")
[[ -n "$DEMAND" ]]               && PIPELINE_ARGS+=(--demand "$DEMAND")
[[ -n "$DEMAND_SCALE" ]]         && PIPELINE_ARGS+=(--demand-scale "$DEMAND_SCALE")
[[ -n "$SIMULATOR" ]]            && PIPELINE_ARGS+=(--simulator "$SIMULATOR")

export PYTHONUNBUFFERED=1
export DOAN_PYTHON="$PYTHON_BIN"
[[ -n "$BLENDER_BIN" ]] && export PATH="$(dirname "$BLENDER_BIN"):$PATH"

LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/cpu1_${LOG_TS}.log"
{
    echo "run_cpu1.sh log — started $(date -Iseconds)"
    echo "host: $(hostname)  user: ${USER:-?}"
    echo "python : $PYTHON_BIN"
    echo "out    : $OUT_DIR"
    echo "============================================================"
} > "$LOG_FILE"

echo ""
echo "$ $PYTHON_BIN $SCRIPT_DIR/run_pipeline.py ${PIPELINE_ARGS[*]}"
set +e
"$PYTHON_BIN" "$SCRIPT_DIR/run_pipeline.py" "${PIPELINE_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]}
set -e
{ echo "============================================================"; echo "ended $(date -Iseconds) (exit $RC)"; } >> "$LOG_FILE"
ln -sfn "$(basename "$LOG_FILE")" "$OUT_DIR/latest.log"
echo ""
echo "log: $LOG_FILE"
echo ""
echo "============================================================"
if [[ "$RC" -eq 0 ]]; then
    echo "PHASE cpu1 DONE.  Transfer output dir to GPU host:"
    echo "  rsync -av $OUT_DIR/ <gpu-host>:<path>/output/run1/"
    echo "Then on the GPU host:"
    echo "  bash scripts/run_gpu.sh --out <path>/output/run1 [--samples N ...]"
    echo "============================================================"
else
    echo "FAILED (exit $RC)  log: $LOG_FILE"
    echo "============================================================"
    exit "$RC"
fi
