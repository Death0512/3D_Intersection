#!/usr/bin/env bash
# run_gpu.sh — Pipeline phase gpu: step 4 (GPU required)
#
#   [4a/6] Build cached camera scenes
#   [4b/6] Render all 8 cameras (Blender Cycles + OptiX/CUDA)
#   [4c/6] NVENC encode PNG frames → MP4
#
# Requires scenario.json + plates/ already present in --out (produced by
# run_cpu1.sh). After this step, transfer --out back to the CPU host and run:
#   bash scripts/run_cpu2.sh --out <same-dir>
#
# Usage:
#   bash scripts/run_gpu.sh [OPTIONS]
#
# Options:
#   --out DIR               Output directory (must contain scenario.json)
#   --only CAM              Render only this camera (debug)
#   --jobs N                Parallel render workers (default: auto from VRAM)
#   --max-workers-per-gpu N Cap on Blender workers per GPU (default: 4)
#   --silence-timeout S     Watchdog timeout in seconds (default: 600)
#   --samples N             Cycles render samples (default: 48)
#   --blender PATH          Path to blender binary
#   --python PATH           Path to python binary
#   -h, --help              Show this help
#
# Scenario/demand flags (--seed, --seconds, --signal, etc.) are accepted but
# ignored — scenario.json is already fixed at this point.

set -euo pipefail

OUT_DIR=""
ONLY=""
JOBS=0
MAX_WORKERS_PER_GPU=4
SILENCE_TIMEOUT=0
SAMPLES=0
BLENDER_BIN=""
PYTHON_BIN=""

usage() { grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)                 OUT_DIR="$2";             shift 2 ;;
        --only)                ONLY="$2";                shift 2 ;;
        --jobs)                JOBS="$2";                shift 2 ;;
        --max-workers-per-gpu) MAX_WORKERS_PER_GPU="$2"; shift 2 ;;
        --silence-timeout)     SILENCE_TIMEOUT="$2";     shift 2 ;;
        --samples)             SAMPLES="$2";             shift 2 ;;
        --blender)             BLENDER_BIN="$2";         shift 2 ;;
        --python)              PYTHON_BIN="$2";          shift 2 ;;
        # cpu phase flags silently accepted so callers can pass the same args to all three scripts
        --seed|--fps|--seconds|--signal-mode|--demand|--demand-scale|--simulator) shift 2 ;;
        --signal|--skip-asset-check) shift ;;
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

SCN="$OUT_DIR/scenario.json"
if [[ ! -f "$SCN" ]]; then
    echo "ERROR: scenario.json not found at $SCN" >&2
    echo "  Run run_cpu1.sh first and transfer the output dir to this host." >&2
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
        echo "ERROR: blender not found on PATH. Run scripts/install.sh or use --blender." >&2
        exit 1
    fi
fi
BLENDER_BIN="$(readlink -f "$BLENDER_BIN" 2>/dev/null || realpath "$BLENDER_BIN")"

echo "============================================================"
echo "PHASE gpu  out=$OUT_DIR"
echo "  python : $PYTHON_BIN"
echo "  blender: $BLENDER_BIN"
[[ -n "$ONLY" ]] && echo "  only   : $ONLY"
echo "============================================================"

PIPELINE_ARGS=(--out "$OUT_DIR" --phase gpu)
[[ -n "$ONLY" ]]                   && PIPELINE_ARGS+=(--only "$ONLY")
[[ "$JOBS" -gt 0 ]]                && PIPELINE_ARGS+=(--jobs "$JOBS")
[[ "$MAX_WORKERS_PER_GPU" -gt 0 ]] && PIPELINE_ARGS+=(--max-workers-per-gpu "$MAX_WORKERS_PER_GPU")
[[ "$SILENCE_TIMEOUT" -gt 0 ]]     && PIPELINE_ARGS+=(--silence-timeout "$SILENCE_TIMEOUT")
[[ "$SAMPLES" -gt 0 ]]             && PIPELINE_ARGS+=(--samples "$SAMPLES")

export PYTHONUNBUFFERED=1
export DOAN_PYTHON="$PYTHON_BIN"
export PATH="$(dirname "$BLENDER_BIN"):$PATH"

LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/gpu_${LOG_TS}.log"
{
    echo "run_gpu.sh log — started $(date -Iseconds)"
    echo "host: $(hostname)  user: ${USER:-?}"
    echo "python : $PYTHON_BIN"
    echo "blender: $BLENDER_BIN"
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
    echo "PHASE gpu DONE.  Transfer output dir back to CPU host:"
    echo "  rsync -av $OUT_DIR/ <cpu-host>:<path>/output/run1/"
    echo "Then on the CPU host:"
    echo "  bash scripts/run_cpu2.sh --out <path>/output/run1"
    echo "============================================================"
else
    echo "FAILED (exit $RC)  log: $LOG_FILE"
    echo "============================================================"
    exit "$RC"
fi
