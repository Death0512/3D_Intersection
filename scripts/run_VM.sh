#!/usr/bin/env bash
# run_VM.sh — GPU server phase: render/encode, then metadata + validation.
#
# Requires the output dir produced by run_cpu1.sh / run_north_scenarios.sh
# --phase cpu1. The directory must contain scenario.json and plate assets.
#
# Typical GPU-server usage for the North batch copied from the CPU host:
#   nohup bash scripts/run_north_scenarios.sh --phase vm --out output/north --samples 12 > output/north/run_VM.nohup.log 2>&1 &
#
# Usage:
#   bash scripts/run_VM.sh [OPTIONS]
#
# Options:
#   --out DIR               Output directory (must contain scenario.json)
#   --only CAMS             Render/metadata cameras, e.g. in_N,out_N
#   --jobs N                Parallel render workers (default: auto from VRAM)
#   --max-workers-per-gpu N Cap on Blender workers per GPU (default: 2)
#   --samples N             Cycles render samples (default: pipeline default)
#   --keyframe-stride N     Unified scene keyframe stride
#   --heading-threshold-deg F
#   --speed-threshold F
#   --blender PATH          Path to blender binary
#   --python PATH           Path to python binary
#   -h, --help              Show this help
#
# Scenario-generation flags are accepted but ignored here because scenario.json
# is already fixed by the CPU1 phase.

set -euo pipefail

OUT_DIR=""
ONLY=""
JOBS=0
MAX_WORKERS_PER_GPU=2
SAMPLES=0
KEYFRAME_STRIDE=""
HEADING_THRESHOLD_DEG=""
SPEED_THRESHOLD=""
BLENDER_BIN=""
PYTHON_BIN=""

usage() { grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)                 OUT_DIR="$2";             shift 2 ;;
        --only)                ONLY="$2";                shift 2 ;;
        --jobs)                JOBS="$2";                shift 2 ;;
        --max-workers-per-gpu) MAX_WORKERS_PER_GPU="$2"; shift 2 ;;
        --samples)             SAMPLES="$2";             shift 2 ;;
        --keyframe-stride)     KEYFRAME_STRIDE="$2";     shift 2 ;;
        --heading-threshold-deg) HEADING_THRESHOLD_DEG="$2"; shift 2 ;;
        --speed-threshold)     SPEED_THRESHOLD="$2";     shift 2 ;;
        --blender)             BLENDER_BIN="$2";         shift 2 ;;
        --python)              PYTHON_BIN="$2";          shift 2 ;;
        --seed|--fps|--seconds|--signal-mode|--demand|--demand-scale|--demand-profile|--simulator) shift 2 ;;
        --signal|--skip-asset-check) shift ;;
        --silence-timeout) shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

_ENV_SH="$SCRIPT_DIR/env.sh"
[[ -f "$_ENV_SH" ]] && source "$_ENV_SH"

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$ROOT_DIR/output/run1"
fi
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

SCN="$OUT_DIR/scenario.json"
if [[ ! -f "$SCN" ]]; then
    echo "ERROR: scenario.json not found at $SCN" >&2
    echo "  Run CPU1 first, then copy that output dir to this GPU server." >&2
    exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -n "${DOAN_PYTHON:-}" && -x "${DOAN_PYTHON}" ]]; then
        PYTHON_BIN="$DOAN_PYTHON"
    else
        PYTHON_BIN="$(command -v python3 || command -v python)"
    fi
fi

if [[ -z "$BLENDER_BIN" ]]; then
    BLENDER_BIN="$(command -v blender 2>/dev/null || true)"
    if [[ -z "$BLENDER_BIN" ]]; then
        echo "ERROR: blender not found on PATH. Use --blender /path/to/blender." >&2
        exit 1
    fi
fi
BLENDER_BIN="$(readlink -f "$BLENDER_BIN" 2>/dev/null || realpath "$BLENDER_BIN")"

echo "============================================================"
echo "PHASE VM  gpu+cpu2  out=$OUT_DIR"
echo "  python : $PYTHON_BIN"
echo "  blender: $BLENDER_BIN"
echo "  ffmpeg : $(command -v ffmpeg || echo missing)"
[[ -n "$ONLY" ]] && echo "  only   : $ONLY"
echo "============================================================"

COMMON_ARGS=(--out "$OUT_DIR")
[[ -n "$ONLY" ]] && COMMON_ARGS+=(--only "$ONLY")
[[ "$JOBS" -gt 0 ]] && COMMON_ARGS+=(--jobs "$JOBS")
[[ "$MAX_WORKERS_PER_GPU" -gt 0 ]] && COMMON_ARGS+=(--max-workers-per-gpu "$MAX_WORKERS_PER_GPU")
[[ "$SAMPLES" -gt 0 ]] && COMMON_ARGS+=(--samples "$SAMPLES")
[[ -n "$KEYFRAME_STRIDE" ]] && COMMON_ARGS+=(--keyframe-stride "$KEYFRAME_STRIDE")
[[ -n "$HEADING_THRESHOLD_DEG" ]] && COMMON_ARGS+=(--heading-threshold-deg "$HEADING_THRESHOLD_DEG")
[[ -n "$SPEED_THRESHOLD" ]] && COMMON_ARGS+=(--speed-threshold "$SPEED_THRESHOLD")
COMMON_ARGS+=(--simulator sumo)

export PYTHONUNBUFFERED=1
export DOAN_PYTHON="$PYTHON_BIN"
export PATH="$(dirname "$BLENDER_BIN"):$PATH"

LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/run_VM_${LOG_TS}.log"
{
    echo "run_VM.sh log — started $(date -Iseconds)"
    echo "host: $(hostname)  user: ${USER:-?}"
    echo "python : $PYTHON_BIN"
    echo "blender: $BLENDER_BIN"
    echo "ffmpeg : $(command -v ffmpeg || echo missing)"
    echo "out    : $OUT_DIR"
    echo "============================================================"
} > "$LOG_FILE"

run_phase() {
    local phase="$1"
    echo ""
    echo "$ $PYTHON_BIN $SCRIPT_DIR/run_pipeline.py ${COMMON_ARGS[*]} --phase $phase"
    set +e
    "$PYTHON_BIN" "$SCRIPT_DIR/run_pipeline.py" "${COMMON_ARGS[@]}" --phase "$phase" 2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ "$rc" -ne 0 ]]; then
        echo "FAILED phase $phase (exit $rc)  log: $LOG_FILE" | tee -a "$LOG_FILE"
        exit "$rc"
    fi
}

run_phase gpu
run_phase cpu2

{
    echo "============================================================"
    echo "run_VM.sh log — ended $(date -Iseconds) (exit 0)"
} >> "$LOG_FILE"
ln -sfn "$(basename "$LOG_FILE")" "$OUT_DIR/latest.log"

echo ""
echo "============================================================"
echo "PHASE VM COMPLETE. Output: $OUT_DIR"
echo "log: $LOG_FILE"
echo "============================================================"
