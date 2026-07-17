#!/usr/bin/env bash
# run_cpu2.sh — Pipeline phase cpu2: steps 5-6 (CPU only)
#
#   [5/6] Metadata generation  (per-frame pose + ground-truth JSON)
#   [6/6] Run validation       (sanity-check all output files)
#
# Requires the output dir produced by run_cpu1.sh + run_gpu.sh:
#   scenario.json, plates/, trajectory.json, video_*.mp4 must all be present.
#
# Usage:
#   bash scripts/run_cpu2.sh [OPTIONS]
#
# Options:
#   --out DIR       Output directory (default: output/run1)
#   --only CAM      Restrict metadata to one camera (debug)
#   --python PATH   Path to python binary
#   -h, --help      Show this help
#
# All other flags (--seed, --samples, GPU flags, etc.) are accepted but
# ignored — scenario and videos are already fixed at this point.

set -euo pipefail

OUT_DIR=""
ONLY=""
PYTHON_BIN=""

usage() { grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)    OUT_DIR="$2";   shift 2 ;;
        --only)   ONLY="$2";     shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        # all other flags silently accepted so callers can pass the same args to all three scripts
        --seed|--fps|--seconds|--signal-mode|--demand|--demand-scale|--simulator) shift 2 ;;
        --jobs|--max-workers-per-gpu|--silence-timeout|--samples|--blender) shift 2 ;;
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
    echo "  Run run_cpu1.sh + run_gpu.sh first." >&2
    exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -n "${DOAN_PYTHON:-}" && -x "${DOAN_PYTHON}" ]]; then
        PYTHON_BIN="$DOAN_PYTHON"
    else
        PYTHON_BIN="$(command -v python3 || command -v python)"
    fi
fi

echo "============================================================"
echo "PHASE cpu2  out=$OUT_DIR"
echo "  python : $PYTHON_BIN"
echo "============================================================"

PIPELINE_ARGS=(--out "$OUT_DIR" --phase cpu2)
[[ -n "$ONLY" ]] && PIPELINE_ARGS+=(--only "$ONLY")

export PYTHONUNBUFFERED=1
export DOAN_PYTHON="$PYTHON_BIN"

LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/cpu2_${LOG_TS}.log"
{
    echo "run_cpu2.sh log — started $(date -Iseconds)"
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
    echo "PIPELINE COMPLETE.  Output: $OUT_DIR"
    echo "============================================================"
else
    echo "FAILED (exit $RC)  log: $LOG_FILE"
    echo "============================================================"
    exit "$RC"
fi
