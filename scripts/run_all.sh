#!/usr/bin/env bash
# run_all.sh — Full synthetic CCTV intersection video pipeline
#
# Usage:
#   bash scripts/run_all.sh [OPTIONS]
#
# Options:
#   --seed N            RNG seed (default: 42)
#   --num-vehicles N    Number of vehicles (default: 10)
#   --fps N             Frames per second (default: 30)
#   --seconds F         Video length in seconds (overrides --duration)
#   --duration N        Duration in frames (default: 300; ignored if --seconds given)
#   --out DIR           Output directory (default: output/run1)
#   --only CAM          Render only this camera e.g. in_N (debug)
#   --skip-asset-check  Skip blender asset validation step
#   --blender PATH      Path to blender binary (default: auto-detect)
#   --python PATH       Path to python binary (default: DOAN_PYTHON env or conda env)
#   -h, --help          Show this help

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SEED=42
NUM_VEHICLES=30
FPS=30
SECONDS_VAL=12
DURATION=12
OUT_DIR="output/run_car"
ONLY=""
SKIP_ASSET_CHECK=0
BLENDER_BIN=""
PYTHON_BIN=""

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
        --duration)       DURATION="$2";      shift 2 ;;
        --out)            OUT_DIR="$2";       shift 2 ;;
        --only)           ONLY="$2";          shift 2 ;;
        --skip-asset-check) SKIP_ASSET_CHECK=1; shift ;;
        --blender)        BLENDER_BIN="$2";   shift 2 ;;
        --python)         PYTHON_BIN="$2";    shift 2 ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$ROOT_DIR/output/run1"
fi
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

# Python interpreter: --python flag > DOAN_PYTHON env > conda env > sys.executable
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -n "${DOAN_PYTHON:-}" && -x "${DOAN_PYTHON}" ]]; then
        PYTHON_BIN="$DOAN_PYTHON"
    elif [[ -x "$HOME/miniconda3/envs/DoAn/bin/python" ]]; then
        PYTHON_BIN="$HOME/miniconda3/envs/DoAn/bin/python"
    else
        PYTHON_BIN="$(command -v python3 || command -v python)"
    fi
fi

# Blender binary
if [[ -z "$BLENDER_BIN" ]]; then
    BLENDER_BIN="$(command -v blender 2>/dev/null || true)"
    if [[ -z "$BLENDER_BIN" ]]; then
        echo "ERROR: blender not found on PATH. Use --blender /path/to/blender" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Compute duration_frames from --seconds if given
# ---------------------------------------------------------------------------
if [[ -n "$SECONDS_VAL" ]]; then
    # Use awk for float arithmetic (bash doesn't do floats)
    DURATION=$(awk "BEGIN { printf \"%d\", int($SECONDS_VAL * $FPS + 0.5) }")
    echo "  --seconds $SECONDS_VAL x --fps $FPS -> --duration $DURATION frames"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "PIPELINE  seed=$SEED  n=$NUM_VEHICLES  ${DURATION}f @ ${FPS}fps  out=$OUT_DIR"
echo "  python : $PYTHON_BIN"
echo "  blender: $BLENDER_BIN"
[[ -n "$ONLY" ]] && echo "  only   : $ONLY"
echo "============================================================"

# ---------------------------------------------------------------------------
# Build run_pipeline.py args and delegate
# ---------------------------------------------------------------------------
PIPELINE_ARGS=(
    --seed "$SEED"
    --num-vehicles "$NUM_VEHICLES"
    --fps "$FPS"
    --duration "$DURATION"
    --out "$OUT_DIR"
)
[[ -n "$ONLY" ]]             && PIPELINE_ARGS+=(--only "$ONLY")
[[ "$SKIP_ASSET_CHECK" -eq 1 ]] && PIPELINE_ARGS+=(--skip-asset-check)

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
