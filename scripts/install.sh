#!/usr/bin/env bash
# install.sh — One-shot setup for Blender 5.1.x + Python venv + ffmpeg + fonts.
#
# After running this, `bash scripts/run_all.sh` just works. The installer writes
# scripts/env.sh which run_all.sh auto-sources. You can also `source scripts/env.sh`
# to activate the environment manually for development.
#
# Usage:
#   bash scripts/install.sh [OPTIONS]
#
# Options:
#   --yes               Non-interactive: pass -y to apt, skip prompts.
#   --blender PATH      Skip download; use this existing blender binary (must be 5.1.x).
#   --python PYTHON3    Base python3 for venv creation (default: python3 on PATH).
#   --venv DIR          Venv directory (default: ./venv, project root).
#   --env-file FILE     Path to write the activation env file (default: scripts/env.sh).
#   --no-blender        Skip blender setup entirely (assume already on PATH, must be 5.x).
#   --no-ffmpeg         Skip ffmpeg apt install (assume already present).
#   --gpu-tune          (sudo) Enable GPU persistence mode + raise power limit to max.
#                       Requires NVIDIA driver + sudo. Opt-in; does NOT run by default.
#   -h, --help          Show this help.

set -euo pipefail

usage() {
  grep '^#' "$0" | grep -v '#!/\|set -' | sed 's/^# //; /^$/d'
  exit 0
}

# ---- defaults ---------------------------------------------------------------
YES=""
BLENDER_OVERRIDE=""
PYTHON3_BASE=""
VENV_DIR=""
ENV_FILE=""
NO_BLENDER=0
NO_FFMPEG=0
GPU_TUNE=0

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
BLENDER_VERSION="5.1.2"
BLENDER_URL="https://download.blender.org/release/Blender5.1/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_INSTALL_DIR="$HOME/.local/opt/blender-${BLENDER_VERSION}"
BLENDER_SYMLINK="$HOME/.local/bin/blender"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
ENV_FILE="${ENV_FILE:-$SCRIPTS_DIR/env.sh}"
PYTHON3_BASE="${PYTHON3_BASE:-$(command -v python3 || command -v python)}"
BLENDER_INSTALLED_BIN=""

APT_DEPS=(ffmpeg fonts-dejavu-core fonts-liberation
          libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libxkbcommon0 libsm6
          curl xz-utils)

# ---- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)         YES="-y";                    shift   ;;
    --blender)     BLENDER_OVERRIDE="$2";       shift 2 ;;
    --python)      PYTHON3_BASE="$2";           shift 2 ;;
    --venv)        VENV_DIR="$2";               shift 2 ;;
    --env-file)    ENV_FILE="$2";               shift 2 ;;
    --no-blender)  NO_BLENDER=1;                shift   ;;
    --no-ffmpeg)   NO_FFMPEG=1;                 shift   ;;
    --gpu-tune)    GPU_TUNE=1;                  shift   ;;
    -h|--help)     usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

# ---- host detection (Kaggle vs local) ---------------------------------------
# Kaggle notebooks set KAGGLE_KERNEL_RUN_TYPE and create a /kaggle tree. When
# detected we force non-interactive mode (no TTY prompts) and default the venv
# to a writable location. Explicit flags above still win.
if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] || [[ -d /kaggle ]]; then
  IS_KAGGLE=1
  YES="-y"
else
  IS_KAGGLE=0
fi

# ---- helpers ----------------------------------------------------------------
_sudo() {
  if command -v sudo &>/dev/null; then sudo "$@"; else "$@"; fi
}

assert_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: '$1' not found on PATH. Please install it first." >&2
    exit 1
  fi
}

echo "============================================================"
echo "INSTALL  blender=${BLENDER_VERSION}  venv=${VENV_DIR}"
echo "  project  : $ROOT_DIR"
echo "  env-file : $ENV_FILE"
[[ "$IS_KAGGLE" -eq 1 ]] && echo "  host     : Kaggle (non-interactive)"
echo "============================================================"

# ---- 1. System packages (apt) -----------------------------------------------
if [[ "$NO_FFMPEG" -ne 1 ]]; then
  echo ""
  echo "--- [1/3] System packages ---"
  _sudo apt-get ${YES:--y} update -qq
  _sudo apt-get install ${YES:--y} -qq "${APT_DEPS[@]}"
  echo "  apt: OK"
fi

# ---- 2. Blender 5.1.x -------------------------------------------------------
echo ""
echo "--- [2/3] Blender ${BLENDER_VERSION} ---"

if [[ "$NO_BLENDER" -eq 1 ]]; then
  assert_cmd blender
  BL="$(command -v blender)"
elif [[ -n "$BLENDER_OVERRIDE" ]]; then
  if [[ ! -x "$BLENDER_OVERRIDE" ]]; then
    echo "ERROR: --blender '$BLENDER_OVERRIDE' is not executable" >&2; exit 1
  fi
  BL="$(readlink -f "$BLENDER_OVERRIDE")"
else
  # Already installed?
  if [[ -f "$BLENDER_SYMLINK" ]] \
     && "$BLENDER_SYMLINK" --version 2>/dev/null | grep -q "^Blender ${BLENDER_VERSION%.*}"; then
    echo "  blender ${BLENDER_VERSION} already at $BLENDER_SYMLINK"
    BL="$BLENDER_SYMLINK"
  elif command -v blender &>/dev/null \
       && blender --version 2>/dev/null | grep -q "^Blender ${BLENDER_VERSION%.*}"; then
    echo "  using system blender: $(command -v blender)"
    BL="$(command -v blender)"
  else
    assert_cmd curl
    assert_cmd tar
    mkdir -p "$HOME/.local/bin" "$HOME/.local/opt"
    echo "  downloading Blender ${BLENDER_VERSION} ..."
    TARBALL="/tmp/blender-${BLENDER_VERSION}.tar.xz"
    curl -fsSL --retry 3 -o "$TARBALL" "$BLENDER_URL"
    echo "  extracting ..."
    rm -rf "$BLENDER_INSTALL_DIR"
    tar -xf "$TARBALL" -C "$HOME/.local/opt"
    mv "$HOME/.local/opt/blender-${BLENDER_VERSION}-linux-x64" "$BLENDER_INSTALL_DIR"
    ln -sf "$BLENDER_INSTALL_DIR/blender" "$BLENDER_SYMLINK"
    rm -f "$TARBALL"
    BL="$BLENDER_SYMLINK"
  fi
fi

BV="$("$BL" --version 2>/dev/null | head -1 || true)"
if [[ ! "$BV" =~ ^Blender\ 5\. ]]; then
  echo "ERROR: blender '$BL' reports '$BV' — must be Blender 5.x." >&2
  exit 1
fi
echo "  blender : $BV"
BLENDER_INSTALLED_BIN="$BL"

BLENDER_BIN_DIR="$(dirname "$BL")"

# ---- NVIDIA GPU check (non-fatal) -------------------------------------------
if command -v nvidia-smi &>/dev/null; then
  echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
  echo "  WARNING : nvidia-smi not found — no NVIDIA GPU detected."
  echo "            Rendering requires an NVIDIA GPU with OptiX support."
  echo "            The render step WILL FAIL without it."
fi

# ---- 3. Python venv with Pillow ---------------------------------------------
echo ""
echo "--- [3/3] Python venv + Pillow ---"

if [[ ! -x "$PYTHON3_BASE" ]]; then
  echo "ERROR: python3 interpreter '$PYTHON3_BASE' is not executable." >&2; exit 1
fi

PY_VERSION="$("$PYTHON3_BASE" --version 2>&1)"
echo "  base python: $PY_VERSION"
echo "  venv        : $VENV_DIR"

if [[ -d "$VENV_DIR" ]] && [[ -x "$VENV_DIR/bin/python" ]]; then
  echo "  venv already exists, updating Pillow ..."
else
  "$PYTHON3_BASE" -m venv "$VENV_DIR" --clear
fi

"$VENV_DIR/bin/pip" install --upgrade -q pip setuptools wheel
"$VENV_DIR/bin/pip" install -q Pillow
echo "  pip: Pillow $("$VENV_DIR/bin/python" -c "import PIL; print(PIL.__version__)" 2>&1)"

VENV_PYTHON="$VENV_DIR/bin/python"

# ---- font check (for gen_plate) ---------------------------------------------
FONT_FOUND=0
for f in "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" \
         "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"; do
  if [[ -f "$f" ]]; then FONT_FOUND=1; break; fi
done
if [[ "$FONT_FOUND" -eq 0 ]]; then
  echo "  WARNING : no DejaVu/Liberation Bold Mono font found."
  echo "            License plate text may render with the default font (ugly)."
  echo "            Install: apt-get install fonts-dejavu-core fonts-liberation"
else
  echo "  font    : OK"
fi

# ---- write env.sh -----------------------------------------------------------
# ---- write env.sh (MUST succeed before anything else) -----------------------
cat > "$ENV_FILE" <<EOF
# Generated by scripts/install.sh — $(date '+%Y-%m-%d %H:%M')
#
# Source this file to activate the project environment:
#   source scripts/env.sh
#
# run_all.sh auto-sources it, so \`bash scripts/run_all.sh\` just works.
#
# After sourcing:
#   - blender  →  $BLENDER_INSTALLED_BIN  (Blender 5.1.x)
#   - python3  →  $VENV_PYTHON  (venv with Pillow)
#   - \$DOAN_PYTHON is set  →  $VENV_PYTHON

export DOAN_PYTHON="$VENV_PYTHON"

# Prepend Blender dir to PATH so 'blender' and shutil.which pick it up.
BLENDER_DIR="${BLENDER_BIN_DIR}"
if [[ -d "\$BLENDER_DIR" ]]; then
  export PATH="\$BLENDER_DIR:\$PATH"
fi

# Convenience: also export a dedicated var so scripts can use it directly.
export DOAN_BLENDER="$BLENDER_INSTALLED_BIN"
EOF

echo ""
echo "============================================================"
echo "INSTALL COMPLETE"
echo "============================================================"
echo ""
echo "  blender : $BLENDER_INSTALLED_BIN   ($BV)"
echo "  python  : $VENV_PYTHON"
echo "  env     : $ENV_FILE"
echo ""
echo "Quick start:"
echo "  source $ENV_FILE"
echo "  python3 scripts/scenario_gen.py --seed 42 --num-vehicles 2 --seconds 5 --out output/test"
echo ""
echo "Or run the full pipeline:"
echo "  bash scripts/run_all.sh --num-vehicles 2 --seconds 5 --out output/test"
echo ""

# ---- GPU tuning (opt-in, --gpu-tune flag required) --------------------------
if [[ "$GPU_TUNE" -eq 1 ]]; then
  echo ""
  echo "--- GPU tuning ---"
  if ! command -v nvidia-smi &>/dev/null; then
    echo "  SKIP: nvidia-smi not found — no NVIDIA GPU detected."
  else
    echo "  NVIDIA GPU detected.  Optional performance tuning:"
    echo ""
    echo "  (1) Persistence mode — keeps GPU driver loaded across process starts,"
    echo "      reducing clock-ramp latency between parallel render workers."
    echo "  (2) Power limit → max — allows the GPU to draw its full rated power"
    echo "      (default is usually 80 W; max on this card is 95 W)."
    echo ""
    echo "  Both are session/boot-scoped and not permanently saved."
    echo ""
    # Persistence mode
    if _sudo nvidia-smi -pm 1 2>/dev/null; then
      echo "  persistence mode: ON"
    else
      echo "  persistence mode: SKIP (sudo may have failed)"
    fi
    # Max power limit
    MAX_PL=$(nvidia-smi -q -d POWER 2>/dev/null | awk '/Max Power Limit/{gsub(/[^0-9.]/,"",$0); if(int($0+0)>0) print int($0+0)}' | tail -1)
    if [[ -n "$MAX_PL" ]] && [[ "$MAX_PL" -gt 0 ]]; then
      if _sudo nvidia-smi -pl "$MAX_PL" 2>/dev/null; then
        echo "  power limit   : raised to ${MAX_PL} W"
      else
        echo "  power limit   : SKIP (sudo may have failed, or card rejected -pl)"
      fi
    else
      echo "  power limit   : SKIP (could not query max)"
    fi
    echo ""
    echo "  GPU tuning complete.  Run the pipeline with:"
    echo "    bash scripts/run_all.sh --num-vehicles 80 --seconds 60"
  fi
fi

# ---- smoke tests (quick, with timeouts) ------------------------------------
echo ""
echo "--- Smoke tests ---"

# Blender test: 20 s timeout in case gpu init is slow in headless.
timeout 20 "$BL" -b --python-expr "import bpy" 2>/dev/null \
  && echo "  blender  : OK" || echo "  blender  : WARN (non-fatal)"

"$VENV_PYTHON" -c "import PIL" 2>/dev/null \
  && echo "  Pillow   : OK" || echo "  Pillow   : FAIL"

command -v ffmpeg &>/dev/null \
  && echo "  ffmpeg   : OK" || echo "  ffmpeg   : WARN"

echo ""
echo "Done."