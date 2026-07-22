#!/usr/bin/env bash
# Run six SUMO unified scenarios for the North road only (in_N + out_N).
#
# Scenarios:
#   empty, sparse, moderate, dense, surge_spike, signal_cycle
#
# Example:
#   bash scripts/run_north_scenarios.sh --out output/north_set --samples 24

set -euo pipefail

SECONDS_VAL=120
FPS=30
SAMPLES=24
SEED=42
OUT_ROOT="output/north_scenarios"
ONLY="in_N,out_N"
SKIP_ASSET_CHECK=0
EXTRA_ARGS=()

usage() {
  sed -n '1,18p' "$0" | sed 's/^# \?//; /^#!\|^set /d'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seconds) SECONDS_VAL="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --skip-asset-check) SKIP_ASSET_CHECK=1; shift ;;
    --jobs|--max-workers-per-gpu|--silence-timeout|--blender|--python)
      EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$OUT_ROOT"

run_one() {
  local name="$1" scale="$2" profile="${3:-}"
  local out_dir="$OUT_ROOT/$name"
  local args=(
    --simulator sumo
    --seconds "$SECONDS_VAL"
    --fps "$FPS"
    --samples "$SAMPLES"
    --seed "$SEED"
    --demand-scale "$scale"
    --only "$ONLY"
    --out "$out_dir"
  )
  [[ "$SKIP_ASSET_CHECK" -eq 1 ]] && args+=(--skip-asset-check)
  [[ -n "$profile" ]] && args+=(--demand-profile "$profile")
  args+=("${EXTRA_ARGS[@]}")
  echo "============================================================"
  echo "NORTH SCENARIO: $name  scale=$scale  profile=${profile:-none}"
  echo "out: $out_dir"
  echo "============================================================"
  bash "$SCRIPT_DIR/run_all.sh" "${args[@]}"
}

run_one empty        0
run_one sparse       1
run_one moderate     2.5
run_one dense        6
# Brief overload near the middle, followed by enough time for recovery.
SPIKE_START=$(( SECONDS_VAL * 2 / 5 ))
SPIKE_END=$(( SPIKE_START + SECONDS_VAL / 15 ))
run_one surge_spike  1 "spike:start=$SPIKE_START,end=$SPIKE_END,scale=8"
run_one signal_cycle 2

echo ""
echo "All North scenarios complete: $OUT_ROOT"
