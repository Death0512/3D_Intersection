#!/usr/bin/env bash
# install.sh — One-shot setup for Blender 5.2.x + Python deps + ffmpeg + SUMO.
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
#   --blender PATH      Skip download; use this existing blender binary (must be 5.2.x).
#   --python PYTHON3    Base python3 for venv creation (default: python3 on PATH).
#   --venv DIR          Venv directory (default: ./venv, project root).
#   --env-file FILE     Path to write the activation env file (default: scripts/env.sh).
#   --no-blender        Skip blender setup entirely (assume already on PATH, must be 5.x).
#   --no-ffmpeg         Skip ffmpeg apt install (assume already present).
#   --no-sumo           Skip SUMO apt install (assume sumo + tools already present).
#   --gpu-tune          (sudo) Enable GPU persistence mode + raise power limit to max.
#                       Requires NVIDIA driver + sudo. Opt-in; does NOT run by default.
#   -h, --help          Show this help.

set -euo pipefail

usage() {
  cat <<'EOF'
install.sh — One-shot setup for Blender 5.2.x + Python deps + ffmpeg + SUMO.

Usage:
  bash scripts/install.sh [OPTIONS]

Options:
  --yes               Non-interactive: pass -y to apt, skip prompts.
  --blender PATH      Skip download; use this existing blender binary (must be 5.2.x).
  --python PYTHON3    Base python3 for venv creation (default: python3 on PATH).
  --venv DIR          Venv directory (default: ./venv, project root).
  --env-file FILE     Path to write activation env file (default: scripts/env.sh).
  --no-blender        Skip blender setup entirely (assume already on PATH, must be 5.x).
  --no-ffmpeg         Skip ffmpeg apt install (assume already present).
  --no-sumo           Skip SUMO apt install (assume sumo + tools already present).
  --gpu-tune          Enable optional NVIDIA persistence mode + max power limit.
  -h, --help          Show this help.

Kaggle notes:
  - Kaggle is auto-detected via KAGGLE_KERNEL_RUN_TYPE or /kaggle.
  - On Kaggle this installer skips venv and installs Python deps with --user.
  - If Blender download is blocked, upload Blender as a Kaggle Dataset and pass:
      --blender /kaggle/input/<dataset>/blender
EOF
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
NO_SUMO=0
GPU_TUNE=0

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
BLENDER_VERSION="5.2.0"
BLENDER_URL="https://download.blender.org/release/Blender5.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_INSTALL_DIR="$HOME/.local/opt/blender-${BLENDER_VERSION}"
BLENDER_SYMLINK="$HOME/.local/bin/blender"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
ENV_FILE="${ENV_FILE:-$SCRIPTS_DIR/env.sh}"
PYTHON3_BASE="${PYTHON3_BASE:-$(command -v python3 || command -v python)}"
BLENDER_INSTALLED_BIN=""

APT_DEPS=(fonts-dejavu-core fonts-liberation ca-certificates curl xz-utils tar
          python3-pip python3-venv
          libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libxkbcommon0 libsm6
          libx11-6 libxrandr2 libxinerama1 libxcursor1 libfontconfig1 libfreetype6)
FFMPEG_APT_DEPS=(ffmpeg)
SUMO_APT_DEPS=(sumo sumo-tools)
PY_DEPS=(Pillow traci sumolib ijson)

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
    --no-sumo)     NO_SUMO=1;                   shift   ;;
    --gpu-tune)    GPU_TUNE=1;                  shift   ;;
    -h|--help)     usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

# ---- host detection (Kaggle vs local, WSL2) ---------------------------------
# Kaggle notebooks set KAGGLE_KERNEL_RUN_TYPE and create a /kaggle tree. When
# detected we force non-interactive mode (no TTY prompts) and default the venv
# to a writable location. Explicit flags above still win.
if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] || [[ -d /kaggle ]]; then
  IS_KAGGLE=1
  YES="-y"
else
  IS_KAGGLE=0
fi

# WSL2: /proc/version contains "microsoft" string; /usr/lib/wsl/lib has CUDA stubs.
IS_WSL2=0
if grep -qi microsoft /proc/version 2>/dev/null && [[ -d /usr/lib/wsl/lib ]]; then
  IS_WSL2=1
  echo "  host: WSL2 detected (will inject LD_LIBRARY_PATH=/usr/lib/wsl/lib into env.sh)"
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

find_kaggle_blender() {
  # Common pattern when the Blender tarball/binary is uploaded as a Kaggle
  # Dataset to bypass flaky/geo-blocked download.blender.org access.
  if [[ ! -d /kaggle/input ]]; then
    return 1
  fi
  local cand
  while IFS= read -r -d '' cand; do
    if [[ -x "$cand" ]] && "$cand" --version 2>/dev/null | grep -q '^Blender 5\.'; then
      printf '%s\n' "$cand"
      return 0
    fi
  done < <(find /kaggle/input -type f -name blender -perm -111 -print0 2>/dev/null)
  return 1
}

detect_sumo_home() {
  if [[ -n "${SUMO_HOME:-}" && -d "$SUMO_HOME" ]]; then
    printf '%s\n' "$SUMO_HOME"
  elif [[ -d /usr/share/sumo ]]; then
    printf '%s\n' /usr/share/sumo
  elif [[ -d /usr/local/share/sumo ]]; then
    printf '%s\n' /usr/local/share/sumo
  else
    return 1
  fi
}

repair_cuda_compat() {
  # NVIDIA Container Toolkit may register a newer CUDA compatibility library
  # ahead of the host driver. On older host drivers this makes cuInit return
  # CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE even though /dev/nvidia0 works.
  command -v nvidia-smi &>/dev/null || return 0
  [[ -d /etc/ld.so.conf.d ]] || return 0

  local rc compat_file backup
  rc="$($PYTHON3_BASE - <<'PY'
import ctypes
try:
    lib = ctypes.CDLL("libcuda.so.1")
    rc = int(lib.cuInit(0))
except Exception:
    rc = -1
print(rc)
PY
  )"
  [[ "$rc" == "804" ]] || return 0

  compat_file=""
  while IFS= read -r candidate; do
    if grep -qE '^/usr/local/cuda[^/]*/compat/?$' "$candidate" 2>/dev/null; then
      compat_file="$candidate"
      break
    fi
  done < <(find /etc/ld.so.conf.d -maxdepth 1 -type f -name '*.conf' -print 2>/dev/null)

  if [[ -z "$compat_file" ]]; then
    echo "  CUDA    : FAIL (forward-compatibility library active; no config file found)" >&2
    return 1
  fi

  backup="${compat_file}.disabled"
  echo "  CUDA    : disabling incompatible compatibility config $compat_file"
  _sudo cp -a "$compat_file" "$backup"
  _sudo rm -f "$compat_file"
  _sudo ldconfig

  rc="$($PYTHON3_BASE - <<'PY'
import ctypes
try:
    print(int(ctypes.CDLL("libcuda.so.1").cuInit(0)))
except Exception:
    print(-1)
PY
  )"
  if [[ "$rc" != "0" ]]; then
    echo "  CUDA    : FAIL (cuInit=$rc after compatibility repair)" >&2
    return 1
  fi
  echo "  CUDA    : host driver usable after compatibility repair"
}

echo "============================================================"
echo "INSTALL  blender=${BLENDER_VERSION}  venv=${VENV_DIR}"
echo "  project  : $ROOT_DIR"
echo "  env-file : $ENV_FILE"
[[ "$IS_KAGGLE" -eq 1 ]] && echo "  host     : Kaggle (non-interactive)"
echo "============================================================"

# ---- 1. System packages (apt) -----------------------------------------------
echo ""
echo "--- [1/3] System packages ---"

# SUMO is not in the default Ubuntu apt repos on older distros; add the official
# PPA if sumo is missing and we haven't opted out.
if [[ "$NO_SUMO" -ne 1 ]] && ! apt-cache show sumo &>/dev/null 2>&1; then
  echo "  SUMO not found in apt — adding official SUMO PPA ..."
  _sudo apt-get install ${YES:--y} -qq software-properties-common
  _sudo add-apt-repository ${YES} ppa:sumo/stable
fi

APT_INSTALL=("${APT_DEPS[@]}")
if [[ "$NO_FFMPEG" -ne 1 ]]; then
  APT_INSTALL+=("${FFMPEG_APT_DEPS[@]}")
fi
if [[ "$NO_SUMO" -ne 1 ]]; then
  APT_INSTALL+=("${SUMO_APT_DEPS[@]}")
fi
if [[ "${#APT_INSTALL[@]}" -gt 0 ]]; then
  echo ""
  _sudo apt-get ${YES:--y} update -qq
  _sudo apt-get install ${YES:--y} -qq "${APT_INSTALL[@]}"
  echo "  apt: OK"
fi

# ---- 2. Blender 5.2.x -------------------------------------------------------
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
  elif [[ "$IS_KAGGLE" -eq 1 ]] && KG_BLENDER="$(find_kaggle_blender)"; then
    echo "  using Kaggle dataset blender: $KG_BLENDER"
    mkdir -p "$HOME/.local/bin"
    ln -sf "$KG_BLENDER" "$BLENDER_SYMLINK"
    BL="$BLENDER_SYMLINK"
  else
    assert_cmd curl
    assert_cmd tar
    mkdir -p "$HOME/.local/bin" "$HOME/.local/opt"
    echo "  downloading Blender ${BLENDER_VERSION} ..."
    TARBALL="/tmp/blender-${BLENDER_VERSION}.tar.xz"
    if ! curl -fL --retry 5 --retry-delay 2 --connect-timeout 30 -o "$TARBALL" "$BLENDER_URL"; then
      echo "ERROR: Blender download failed: $BLENDER_URL" >&2
      if [[ "$IS_KAGGLE" -eq 1 ]]; then
        echo "       Kaggle tip: upload a Blender 5.x linux-x64 folder/tarball as a" >&2
        echo "       Kaggle Dataset, then rerun with:" >&2
        echo "         bash scripts/install.sh --blender /kaggle/input/<dataset>/blender" >&2
        echo "       or place an executable named 'blender' under /kaggle/input." >&2
      fi
      exit 1
    fi
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

# ---- Blender bundled Python: install ijson for build_unified_scene.py ---------
# build_unified_scene.py runs inside Blender's embedded Python and uses ijson
# to stream scenario JSON. The project .venv ijson is NOT visible to Blender.
# Resolve the bundled Python that ships inside the Blender tarball and pip-install
# ijson into it. Hard-fail if it cannot be found (Blender won't render anyway).
_resolve_blender_python() {
  # Strategy: derive from the binary's containing directory + version subfolder.
  # Standard Blender 5.2 tarball layout:
  #   <install>/blender  +  <install>/5.2/python/bin/python3.13
  # (Blender 3.x used python3.10, 4.x used python3.11, 5.x uses python3.13)
  local bpy d cand
  bpy=""

  _try_py_candidates() {
    local base="$1"
    local v
    for v in python3.13 python3.12 python3.11 python3.10 python3; do
      cand="$base/bin/$v"
      [[ -x "$cand" ]] && { bpy="$cand"; return 0; }
    done
    return 1
  }

  # 1. Try to resolve from BLENDER_INSTALL_DIR (set by the download path).
  if [[ -n "${BLENDER_INSTALL_DIR:-}" ]]; then
    for d in "$BLENDER_INSTALL_DIR"/*/; do
      [[ "$(basename "$d")" =~ ^[0-9]+\.[0-9]+$ ]] || continue
      _try_py_candidates "${d}python" && break
    done
  fi
  # 2. Derive from the resolved blender binary: <dir containing blender>/<ver>/python/bin/...
  if [[ -z "$bpy" ]]; then
    local install_dir resolved
    resolved="$(readlink -f "$BL" 2>/dev/null)" || resolved="$BL"
    install_dir="$(dirname "$resolved")"
    for d in "$install_dir"/*/; do
      [[ "$(basename "$d")" =~ ^[0-9]+\.[0-9]+$ ]] || continue
      _try_py_candidates "${d}python" && break
    done
  fi
  # 3. Last resort: ask Blender itself.
  if [[ -z "$bpy" ]]; then
    bpy="$("$BL" --python-expr "import sys; print(sys.executable)" 2>/dev/null)" || true
    [[ -n "$bpy" && -x "$bpy" ]] || bpy=""
  fi
  printf '%s\n' "$bpy"
}

BLENDER_PYTHON="$(_resolve_blender_python)"

if [[ -z "$BLENDER_PYTHON" ]]; then
  echo "ERROR: could not resolve Blender bundled Python." >&2
  echo "       build_unified_scene.py requires ijson in Blender's Python." >&2
  exit 1
else
  echo "  blender python : $BLENDER_PYTHON"

  # Blender 5.2 ships Python 3.11 with ensurepip, but pip may not be bootstrapped.
  # Bootstrap pip if needed, then install ijson.  Hard-fail on failure — ijson is
  # required for build_unified_scene.py.
  "$BLENDER_PYTHON" -m ensurepip --default-pip 2>/dev/null || true
  "$BLENDER_PYTHON" -m pip install --upgrade -q pip 2>/dev/null || true

  if "$BLENDER_PYTHON" -m pip install -q ijson 2>&1; then
    echo "  ijson          : $( "$BLENDER_PYTHON" -c "import ijson; print(getattr(ijson, '__version__', 'OK'))" 2>&1 )"
  else
    echo "ERROR: ijson install into Blender Python ($BLENDER_PYTHON) failed." >&2
    echo "       build_unified_scene.py requires ijson to stream scenario JSON." >&2
    exit 1
  fi
fi

# ---- NVIDIA GPU check (non-fatal) -------------------------------------------
if command -v nvidia-smi &>/dev/null; then
  echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
  echo "  WARNING : nvidia-smi not found — no NVIDIA GPU detected."
  echo "            Rendering requires an NVIDIA GPU with OptiX support."
  echo "            The render step WILL FAIL without it."
fi

repair_cuda_compat

# ---- 3. Python environment ---------------------------------------------------
echo ""
echo "--- [3/3] Python deps ---"

if [[ ! -x "$PYTHON3_BASE" ]]; then
  echo "ERROR: python3 interpreter '$PYTHON3_BASE' is not executable." >&2; exit 1
fi

PY_VERSION="$("$PYTHON3_BASE" --version 2>&1)"
echo "  base python: $PY_VERSION"

# On Kaggle the venv ensurepip step is often stripped/broken, and the container
# is ephemeral anyway, so skip the venv and use the system Python directly.
if [[ "$IS_KAGGLE" -eq 1 ]]; then
  echo "  host     : Kaggle — skipping venv (using system python)"
  VENV_PYTHON="$PYTHON3_BASE"
  VENV_DIR=""
  # Ensure pip is usable; Kaggle ships it but make --user installs safe.
  "$VENV_PYTHON" -m pip install -q --upgrade pip --user 2>/dev/null || true
  "$VENV_PYTHON" -m pip install -q --user "${PY_DEPS[@]}"
  echo "  pip: Pillow $("$VENV_PYTHON" -c "import PIL; print(PIL.__version__)" 2>&1)"
  echo "  pip: traci  $("$VENV_PYTHON" -c "import traci; print(getattr(traci, '__version__', 'OK'))" 2>&1)"
else
  echo "  venv        : $VENV_DIR"
  if [[ -d "$VENV_DIR" ]] && [[ -x "$VENV_DIR/bin/python" ]]; then
    echo "  venv already exists, updating Pillow ..."
  else
    "$PYTHON3_BASE" -m venv "$VENV_DIR" --clear
  fi

  "$VENV_DIR/bin/pip" install --upgrade -q pip setuptools wheel
  "$VENV_DIR/bin/pip" install -q "${PY_DEPS[@]}"
  echo "  pip: Pillow $("$VENV_DIR/bin/python" -c "import PIL; print(PIL.__version__)" 2>&1)"
  echo "  pip: traci  $("$VENV_DIR/bin/python" -c "import traci; print(getattr(traci, '__version__', 'OK'))" 2>&1)"
  VENV_PYTHON="$VENV_DIR/bin/python"
fi

# ---- SUMO check --------------------------------------------------------------
SUMO_HOME_DETECTED=""
if SUMO_HOME_DETECTED="$(detect_sumo_home)"; then
  echo "  SUMO_HOME: $SUMO_HOME_DETECTED"
else
  echo "  WARNING : SUMO_HOME not detected. SUMO unified mode needs SUMO_HOME=/usr/share/sumo."
fi
if command -v sumo &>/dev/null; then
  echo "  sumo    : $(sumo --version 2>/dev/null | head -1)"
else
  echo "  WARNING : sumo binary not found — use apt install sumo sumo-tools or rerun without --no-sumo."
fi

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
cat > "$ENV_FILE" <<EOF
# Generated by scripts/install.sh — $(date '+%Y-%m-%d %H:%M')
#
# Source this file to activate the project environment:
#   source scripts/env.sh
#
# run_all.sh auto-sources it, so \`bash scripts/run_all.sh\` just works.
#
# After sourcing:
#   - blender  →  $BLENDER_INSTALLED_BIN  (Blender 5.2.x)
#   - python3  →  $VENV_PYTHON  $([[ "$IS_KAGGLE" -eq 1 ]] && echo "(system, with Pillow)" || echo "(venv with Pillow)")
#   - \$DOAN_PYTHON is set  →  $VENV_PYTHON

export DOAN_PYTHON="$VENV_PYTHON"

SUMO_HOME_DETECTED="$SUMO_HOME_DETECTED"
if [[ -n "\$SUMO_HOME_DETECTED" && -d "\$SUMO_HOME_DETECTED" ]]; then
  export SUMO_HOME="\$SUMO_HOME_DETECTED"
  export PATH="\$SUMO_HOME/tools:\$PATH"
fi

# Prepend Blender dir to PATH so 'blender' and shutil.which pick it up.
BLENDER_DIR="${BLENDER_BIN_DIR}"
if [[ -d "\$BLENDER_DIR" ]]; then
  export PATH="\$BLENDER_DIR:\$PATH"
fi

# Convenience: also export a dedicated var so scripts can use it directly.
export DOAN_BLENDER="$BLENDER_INSTALLED_BIN"

# WSL2: inject NVIDIA CUDA stub libs so Blender Snap and ffmpeg can find the GPU.
if grep -qi microsoft /proc/version 2>/dev/null && [[ -d /usr/lib/wsl/lib ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
fi
EOF

echo ""
echo "============================================================"
echo "INSTALL COMPLETE"
echo "============================================================"
echo ""
echo "  blender : $BLENDER_INSTALLED_BIN   ($BV)"
echo "  python  : $VENV_PYTHON"
if [[ -n "$SUMO_HOME_DETECTED" ]]; then
  echo "  sumo    : $(command -v sumo || true)  (SUMO_HOME=$SUMO_HOME_DETECTED)"
fi
echo "  env     : $ENV_FILE"
echo ""
echo "Quick start:"
echo "  bash $SCRIPTS_DIR/run_all.sh --seconds 60 --demand-scale 3 --out output/run1"
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
    echo "    bash scripts/run_all.sh --seconds 60 --demand-scale 3"
  fi
fi

# ---- smoke tests (quick, with timeouts) ------------------------------------
echo ""
echo "--- Smoke tests ---"

# On WSL2/headless, inject LD_LIBRARY_PATH before calling Blender so the Snap
# can reach the NVIDIA CUDA stubs at /usr/lib/wsl/lib.
_bl_env() {
  if [[ "$IS_WSL2" -eq 1 ]]; then
    env LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$@"
  else
    "$@"
  fi
}

# Blender basic import test — non-fatal; just warns if Blender is broken.
if _bl_env timeout 20 "$BL" -b --python-expr "import bpy" 2>/dev/null; then
  echo "  blender  : OK"
else
  echo "  blender  : WARN (bpy import failed — non-fatal, may work at render time)"
fi

# Blender GPU test — only run when nvidia-smi sees a GPU; non-fatal WARN so
# install completes even if the GPU driver isn't set up yet (e.g. fresh VM).
if command -v nvidia-smi &>/dev/null; then
  _BL_GPU_SCRIPT="$(mktemp /tmp/doan_gpu_check_XXXXXX.py)"
  cat > "$_BL_GPU_SCRIPT" <<'PYEOF'
import bpy, sys
p = bpy.context.preferences.addons['cycles'].preferences
p.compute_device_type = 'OPTIX'
p.refresh_devices()
ok = any(d.type in {'CUDA', 'OPTIX'} for d in p.devices)
if not ok:
    p.compute_device_type = 'CUDA'
    p.refresh_devices()
    ok = any(d.type in {'CUDA', 'OPTIX'} for d in p.devices)
sys.exit(0 if ok else 1)
PYEOF
  if _bl_env timeout 30 "$BL" -b --python "$_BL_GPU_SCRIPT" 2>/dev/null; then
    echo "  blender GPU: CUDA/OptiX OK"
  else
    echo "  blender GPU: WARN (NVIDIA visible but Blender could not use CUDA/OptiX)" >&2
    echo "               → on WSL2 make sure /usr/lib/wsl/lib is on LD_LIBRARY_PATH" >&2
    echo "               → re-run install after sourcing scripts/env.sh" >&2
  fi
  rm -f "$_BL_GPU_SCRIPT"
fi

"$VENV_PYTHON" -c "import PIL" 2>/dev/null \
  && echo "  Pillow   : OK" || echo "  Pillow   : FAIL"

"$VENV_PYTHON" -c "import traci, sumolib" 2>/dev/null \
  && echo "  traci    : OK" || echo "  traci    : FAIL"

command -v sumo &>/dev/null \
  && echo "  sumo     : OK" || echo "  sumo     : WARN"

command -v ffmpeg &>/dev/null \
  && echo "  ffmpeg   : OK" || echo "  ffmpeg   : WARN"

if command -v ffmpeg &>/dev/null; then
  if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q 'h264_nvenc'; then
    NVENC_TEST="${TMPDIR:-/tmp}/doan_nvenc_test.mp4"
    if ffmpeg -y -v error -f lavfi -i color=size=256x256:rate=1 \
        -frames:v 1 -c:v h264_nvenc -pix_fmt yuv420p "$NVENC_TEST" 2>/dev/null; then
      rm -f "$NVENC_TEST"
      echo "  nvenc    : OK"
    else
      rm -f "$NVENC_TEST"
      echo "  nvenc    : WARN (h264_nvenc listed but test encode failed — GPU may not be ready)" >&2
    fi
  else
    echo "  nvenc    : WARN (h264_nvenc not in ffmpeg — render will fail without NVENC)" >&2
    echo "             → install ffmpeg with NVENC support, or pass --no-ffmpeg and use system ffmpeg" >&2
  fi
fi

echo ""
echo "Done."
