#!/usr/bin/env bash
# =============================================================================
# jetson_setup_and_run.sh — Full setup + video generation for Jetson Orin 32GB
#
# USAGE:
#   bash scripts/jetson_setup_and_run.sh [OPTIONS]
#
# OPTIONS:
#   --seed N           RNG seed (default: 42)
#   --seconds N        Video length in seconds (default: 12)
#   --out DIR          Output directory (default: output/jetson_run)
#   --samples N        Blender Cycles samples per frame (default: 16)
#   --demand-scale N   Traffic density multiplier (default: 1.0)
#   --signal           Enable traffic signal gating
#   --skip-setup       Skip dependency installation (re-run only)
#   --skip-blender     Skip Blender installation check
#   --background       Run in background (SSH-safe): tmux session if available,
#                      otherwise nohup. Attach later with:
#                        tmux attach -t jetson_gen
#                      or watch the log:
#                        tail -f ~/jetson_gen.log
#   --attach           Attach to a running background session (tmux only)
#   --status           Show status of a background run (log tail + process check)
#   --help             Show this help
#
# REQUIREMENTS (manual steps before running):
#   - JetPack 5.x or 6.x installed (provides CUDA, V4L2 HW encoder)
#   - Internet access for apt/pip packages
#   - Run as a user with sudo privileges
#
# Jetson-specific adaptations vs desktop pipeline:
#   1. GPU detection via /proc/meminfo (unified memory) instead of nvidia-smi
#   2. Video encoding via h264_v4l2m2m (Jetson HW encoder) or libx264 fallback
#   3. Blender Cycles uses CUDA backend (sm_87 Ampere, iGPU on Orin)
#   4. Parallel render workers capped at 1 (single iGPU, avoid VRAM thrash)
#   5. OPTIX denoiser disabled (no RT cores in Orin iGPU); sample count
#      compensates (use --samples 24-32 for cleaner output)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SEED=42
SECONDS_VAL=12
OUT_DIR=""
SAMPLES=16
DEMAND_SCALE=""
SIGNAL_FLAG=""
SKIP_SETUP=false
SKIP_BLENDER=false
BACKGROUND=false
ATTACH=false
STATUS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)           SEED="$2"; shift 2 ;;
        --seconds)        SECONDS_VAL="$2"; shift 2 ;;
        --out)            OUT_DIR="$2"; shift 2 ;;
        --samples)        SAMPLES="$2"; shift 2 ;;
        --demand-scale)   DEMAND_SCALE="$2"; shift 2 ;;
        --signal)         SIGNAL_FLAG="--signal"; shift ;;
        --skip-setup)     SKIP_SETUP=true; shift ;;
        --skip-blender)   SKIP_BLENDER=true; shift ;;
        --background)     BACKGROUND=true; shift ;;
        --attach)         ATTACH=true; shift ;;
        --status)         STATUS=true; shift ;;
        --help)
            sed -n '3,31p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$PROJECT_ROOT/output/jetson_run"
fi

VENV_DIR="$PROJECT_ROOT/.venv"
BLENDER_INSTALL_DIR="$HOME/blender-jetson"
BLENDER_BIN="$BLENDER_INSTALL_DIR/blender"

# Background session config
BG_SESSION="jetson_gen"
BG_LOG="$HOME/jetson_gen.log"
BG_PID_FILE="$HOME/jetson_gen.pid"

# ---------------------------------------------------------------------------
# Background / attach / status — handle before anything else
# ---------------------------------------------------------------------------

# --status: print last 40 lines of log + whether process is still running
if [[ "$STATUS" == "true" ]]; then
    echo "=== Jetson background job status ==="
    if [[ -f "$BG_PID_FILE" ]]; then
        BG_PID=$(cat "$BG_PID_FILE")
        if kill -0 "$BG_PID" 2>/dev/null; then
            echo "  Process PID $BG_PID is RUNNING"
        else
            echo "  Process PID $BG_PID is NOT running (may have finished or crashed)"
        fi
    else
        echo "  No PID file found ($BG_PID_FILE) — no background job started yet"
    fi
    if command -v tmux &>/dev/null && tmux has-session -t "$BG_SESSION" 2>/dev/null; then
        echo "  tmux session '$BG_SESSION' is ACTIVE"
        echo "  Attach with: tmux attach -t $BG_SESSION"
    fi
    echo ""
    if [[ -f "$BG_LOG" ]]; then
        echo "=== Last 40 lines of $BG_LOG ==="
        tail -n 40 "$BG_LOG"
    else
        echo "  Log not found: $BG_LOG"
    fi
    exit 0
fi

# --attach: re-attach to the tmux session
if [[ "$ATTACH" == "true" ]]; then
    if ! command -v tmux &>/dev/null; then
        echo "tmux is not installed. Watch the log instead:"
        echo "  tail -f $BG_LOG"
        exit 1
    fi
    if ! tmux has-session -t "$BG_SESSION" 2>/dev/null; then
        echo "No active tmux session named '$BG_SESSION'."
        echo "The job may have finished. Check the log:"
        echo "  cat $BG_LOG"
        exit 1
    fi
    exec tmux attach -t "$BG_SESSION"
fi

# --background: re-launch this exact script inside tmux or nohup, then exit
if [[ "$BACKGROUND" == "true" ]]; then
    # Remove --background from the args we'll forward so we don't recurse
    FORWARD_ARGS=()
    for arg in "$@"; do
        [[ "$arg" == "--background" ]] && continue
        FORWARD_ARGS+=("$arg")
    done
    # Rebuild the full original argument list without --background
    # (we already parsed them above, reconstruct from variables)
    FORWARD_ARGS=(
        "--seed" "$SEED"
        "--seconds" "$SECONDS_VAL"
        "--out" "$OUT_DIR"
        "--samples" "$SAMPLES"
    )
    [[ -n "$DEMAND_SCALE" ]]    && FORWARD_ARGS+=("--demand-scale" "$DEMAND_SCALE")
    [[ -n "$SIGNAL_FLAG" ]]     && FORWARD_ARGS+=("$SIGNAL_FLAG")
    [[ "$SKIP_SETUP" == "true" ]] && FORWARD_ARGS+=("--skip-setup")
    [[ "$SKIP_BLENDER" == "true" ]] && FORWARD_ARGS+=("--skip-blender")

    SELF="$(realpath "${BASH_SOURCE[0]}")"
    CMD_LINE="bash \"$SELF\" ${FORWARD_ARGS[*]}"

    if command -v tmux &>/dev/null; then
        # Kill any existing stale session with same name
        tmux kill-session -t "$BG_SESSION" 2>/dev/null || true
        # Start new detached session; pipe-pane tees output to log file
        tmux new-session -d -s "$BG_SESSION" -x 220 -y 50 \
            "bash \"$SELF\" ${FORWARD_ARGS[*]} 2>&1 | tee \"$BG_LOG\"; echo '[DONE] Session ended. Press Enter to close.'; read"
        sleep 1
        echo "============================================================"
        echo " Background job started in tmux session: $BG_SESSION"
        echo "============================================================"
        echo ""
        echo "  Attach to watch live:    tmux attach -t $BG_SESSION"
        echo "  Detach once attached:    Ctrl-B then D"
        echo "  Watch log (no attach):   tail -f $BG_LOG"
        echo "  Check status:            bash $SELF --status"
        echo ""
        echo "  Log file: $BG_LOG"
        echo ""
        echo " The session survives SSH disconnect. Reconnect any time."
        echo "============================================================"
    else
        # tmux not available — fall back to nohup
        echo "tmux not found — using nohup fallback."
        echo "Log: $BG_LOG"
        nohup bash "$SELF" "${FORWARD_ARGS[@]}" > "$BG_LOG" 2>&1 &
        BG_PID=$!
        echo "$BG_PID" > "$BG_PID_FILE"
        echo "============================================================"
        echo " Background job started with nohup (PID $BG_PID)"
        echo "============================================================"
        echo ""
        echo "  Watch log:       tail -f $BG_LOG"
        echo "  Check status:    bash $SELF --status"
        echo "  Kill job:        kill $BG_PID"
        echo ""
        echo "  Log file: $BG_LOG"
        echo "============================================================"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[JETSON] $*"; }
ok()   { echo "[JETSON] OK: $*"; }
warn() { echo "[JETSON] WARN: $*" >&2; }
die()  { echo "[JETSON] FATAL: $*" >&2; exit 1; }

check_jetson() {
    if [[ ! -f /proc/device-tree/model ]]; then
        warn "Cannot confirm Jetson hardware (/proc/device-tree/model not found)."
        warn "This script is designed for NVIDIA Jetson Orin. Continuing anyway."
        return
    fi
    local model
    model=$(cat /proc/device-tree/model 2>/dev/null || true)
    if [[ "$model" != *"Jetson"* ]]; then
        warn "Device model: '$model' — does not look like a Jetson."
        warn "This script is designed for Jetson Orin. Continuing anyway."
    else
        ok "Detected: $model"
    fi
}

check_jetpack() {
    local jp_version
    # JetPack version file (JetPack 5+)
    if [[ -f /etc/nv_tegra_release ]]; then
        jp_version=$(cat /etc/nv_tegra_release | head -1)
        ok "JetPack: $jp_version"
    elif command -v jetson_release &>/dev/null; then
        jetson_release | head -3
    else
        warn "Cannot detect JetPack version. Ensure JetPack 5.x or 6.x is installed."
    fi

    # Check CUDA
    if [[ -f /usr/local/cuda/version.json ]]; then
        local cuda_ver
        cuda_ver=$(python3 -c "import json; d=json.load(open('/usr/local/cuda/version.json')); print(d['cuda']['version'])" 2>/dev/null || echo "unknown")
        ok "CUDA: $cuda_ver"
    elif command -v nvcc &>/dev/null; then
        ok "CUDA: $(nvcc --version | grep 'release' | awk '{print $5}' | tr -d ',')"
    else
        warn "nvcc not found — CUDA may not be properly installed or not in PATH."
        warn "Add /usr/local/cuda/bin to PATH in ~/.bashrc"
    fi
}

check_unified_memory() {
    # Jetson has unified memory; estimate available memory for GPU work
    local mem_avail_mib
    mem_avail_mib=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
    ok "Available unified memory: ${mem_avail_mib} MiB"
    if [[ $mem_avail_mib -lt 4096 ]]; then
        warn "Less than 4GB free RAM. Blender render may be slow or fail."
        warn "Close other applications before rendering."
    fi
}

# ---------------------------------------------------------------------------
# PHASE 1: System dependencies
# ---------------------------------------------------------------------------
phase_system_deps() {
    log "=== Phase 1: System dependencies ==="

    # Update apt cache
    sudo apt-get update -q

    # Python 3 + pip + venv
    sudo apt-get install -y -q python3 python3-pip python3-venv python3-dev

    # ffmpeg with V4L2 support (Jetson HW encoder)
    if ! command -v ffmpeg &>/dev/null; then
        log "Installing ffmpeg..."
        sudo apt-get install -y -q ffmpeg
    fi

    # Verify V4L2 encoder is available in ffmpeg
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_v4l2m2m; then
        ok "h264_v4l2m2m (Jetson HW encoder) available in ffmpeg"
    else
        warn "h264_v4l2m2m not found in ffmpeg. Will use libx264 (CPU) as fallback."
        # Make sure libx264 is available
        if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264; then
            sudo apt-get install -y -q ffmpeg libx264-dev || true
        fi
    fi

    # V4L2 utilities (optional, for HW encoder diagnostics)
    sudo apt-get install -y -q v4l-utils 2>/dev/null || true

    # Git (for cloning if needed)
    sudo apt-get install -y -q git

    # Libraries needed by Blender headless on ARM64
    sudo apt-get install -y -q \
        libx11-6 libxrender1 libxi6 libxkbcommon0 libxxf86vm1 \
        libxfixes3 libxext6 libgl1 libglu1-mesa \
        libsm6 libice6 libegl1 \
        libjpeg-turbo8 libpng16-16 libtiff5 \
        libfontconfig1 libfreetype6 \
        2>/dev/null || true

    ok "System dependencies installed."
}

# ---------------------------------------------------------------------------
# PHASE 2: Blender ARM64 installation
# ---------------------------------------------------------------------------
phase_blender() {
    log "=== Phase 2: Blender installation ==="

    if [[ -x "$BLENDER_BIN" ]]; then
        local ver
        ver=$("$BLENDER_BIN" --version 2>/dev/null | head -1 || echo "unknown")
        ok "Blender already installed: $ver"
        return
    fi

    # -------------------------------------------------------------------------
    # Blender 4.1 LTS ARM64 Linux build
    # The official Blender builds are x86_64 only. For Jetson (ARM64/aarch64),
    # use the community ARM64 build or build from source.
    #
    # Option A (auto): download a known-good community ARM64 build
    # Option B (fallback): prompt user to provide a Blender binary
    # -------------------------------------------------------------------------

    log "Blender not found at $BLENDER_BIN"
    log "Checking for ARM64 Blender build..."

    # Check if blender is somewhere on PATH (e.g. installed via apt or snap)
    if command -v blender &>/dev/null; then
        local sys_blender
        sys_blender=$(command -v blender)
        local sys_ver
        sys_ver=$(blender --version 2>/dev/null | head -1 || echo "unknown")
        log "Found system Blender: $sys_blender ($sys_ver)"
        BLENDER_BIN="$sys_blender"
        ok "Using system Blender: $BLENDER_BIN"
        return
    fi

    # Try apt (some Jetson distros have Blender in universe)
    log "Attempting: sudo apt-get install blender"
    if sudo apt-get install -y -q blender 2>/dev/null; then
        if command -v blender &>/dev/null; then
            BLENDER_BIN=$(command -v blender)
            ok "Blender installed via apt: $(blender --version 2>/dev/null | head -1)"
            return
        fi
    fi

    # -------------------------------------------------------------------------
    # Manual install path: download Blender ARM64 portable
    # Blender 4.1.1 ARM64 Linux (community build, Cycles CUDA enabled)
    # NOTE: Replace this URL with the latest verified ARM64 build URL.
    # Official ARM64 builds: https://builder.blender.org/download/daily/
    # -------------------------------------------------------------------------
    local BLENDER_VERSION="4.1.1"
    local BLENDER_ARCHIVE="blender-${BLENDER_VERSION}-linux.aarch64.tar.xz"
    # Try Blender builder download (daily/experimental ARM64)
    local BLENDER_URL="https://builder.blender.org/download/daily/archive/blender-${BLENDER_VERSION}+stable+v41.b9960d6b930f-linux.aarch64-release.tar.xz"

    log "Downloading Blender ${BLENDER_VERSION} ARM64..."
    log "URL: $BLENDER_URL"

    mkdir -p "$BLENDER_INSTALL_DIR"
    local TMPDIR_BLENDER
    TMPDIR_BLENDER=$(mktemp -d)
    trap "rm -rf $TMPDIR_BLENDER" EXIT

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$TMPDIR_BLENDER/$BLENDER_ARCHIVE" "$BLENDER_URL" || {
            die "Blender download failed. Please manually download a Blender ARM64 build and place the 'blender' binary at: $BLENDER_BIN
    
  Download options:
    1. https://builder.blender.org/download/daily/  (filter: Linux, ARM64)
    2. Build from source: https://wiki.blender.org/wiki/Building_Blender/Linux
  
  After placing the binary, re-run with: --skip-blender"
        }
    else
        curl -L --progress-bar -o "$TMPDIR_BLENDER/$BLENDER_ARCHIVE" "$BLENDER_URL" || {
            die "Blender download failed. See instructions above."
        }
    fi

    log "Extracting Blender..."
    tar -xJf "$TMPDIR_BLENDER/$BLENDER_ARCHIVE" -C "$BLENDER_INSTALL_DIR" --strip-components=1
    rm -rf "$TMPDIR_BLENDER"
    trap - EXIT

    if [[ ! -x "$BLENDER_BIN" ]]; then
        die "Blender extraction failed — '$BLENDER_BIN' not found after extract."
    fi

    local ver
    ver=$("$BLENDER_BIN" --version 2>/dev/null | head -1 || echo "unknown")
    ok "Blender installed: $ver"

    # Add to PATH for convenience
    if [[ -f "$HOME/.bashrc" ]] && ! grep -q "blender-jetson" "$HOME/.bashrc"; then
        echo "export PATH=\"$BLENDER_INSTALL_DIR:\$PATH\"  # added by jetson_setup_and_run.sh" >> "$HOME/.bashrc"
        log "Added Blender to PATH in ~/.bashrc (takes effect in new shells)"
    fi
}

# ---------------------------------------------------------------------------
# PHASE 3: Python virtual environment + pip packages
# ---------------------------------------------------------------------------
phase_python_env() {
    log "=== Phase 3: Python virtual environment ==="

    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtualenv at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    else
        ok "Virtualenv already exists at $VENV_DIR"
    fi

    # Activate
    source "$VENV_DIR/bin/activate"

    # Upgrade pip silently
    pip install --quiet --upgrade pip

    # Install required packages
    log "Installing Python packages..."
    pip install --quiet \
        Pillow \
        numpy \
        scipy

    # Optional but helpful for research sim
    pip install --quiet gymnasium 2>/dev/null || true

    ok "Python packages installed."
    ok "Virtualenv: $VENV_DIR"
}

# ---------------------------------------------------------------------------
# PHASE 4: Verify CUDA for Blender Cycles on Jetson
# ---------------------------------------------------------------------------
phase_verify_cuda() {
    log "=== Phase 4: Verifying CUDA for Blender Cycles ==="

    # Jetson Orin: iGPU is Ampere sm_87
    # Blender Cycles needs CUDA toolkit (not just runtime) — but the Blender
    # portable build ships its own CUDA kernels for supported sm_ architectures.
    # sm_87 support was added in Blender 3.5+ with CUDA 11.8+.

    # Check if sm_87 is in Blender's bundled CUDA kernels
    local CYCLES_CUDA_DIR="$BLENDER_INSTALL_DIR/$(ls "$BLENDER_INSTALL_DIR" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+$' | head -1)/scripts/../../../lib/linux_arm64/cuda"
    # Simpler: look for the kernels anywhere in the Blender dir
    local kernel_count
    kernel_count=$(find "$BLENDER_INSTALL_DIR" -name "*.cubin" -o -name "kernel_sm_*.cubin" 2>/dev/null | wc -l || echo 0)

    if [[ "$kernel_count" -gt 0 ]]; then
        ok "Blender CUDA kernels found ($kernel_count .cubin files)"
    else
        warn "No .cubin CUDA kernel files found in Blender install."
        warn "Blender will try to JIT-compile CUDA kernels on first render (slow, ~5-10 min)."
        warn "This is normal on first run. Compiled kernels are cached in ~/.cache/cycles/"
        # Check if nvcc is available for JIT compilation
        if ! command -v nvcc &>/dev/null && ! [[ -f /usr/local/cuda/bin/nvcc ]]; then
            warn "nvcc not found — CUDA kernel JIT may fail."
            warn "Install with: sudo apt-get install cuda-nvcc-11-4  (match your JetPack CUDA version)"
        fi
    fi

    # Verify Jetson CUDA driver is active
    if [[ -f /dev/nvhost-ctrl ]]; then
        ok "NVIDIA Tegra GPU device node present (/dev/nvhost-ctrl)"
    else
        warn "/dev/nvhost-ctrl not found. GPU may not be initialized."
    fi
}

# ---------------------------------------------------------------------------
# PHASE 5: Patch pipeline for Jetson (runtime env vars only — no file edits)
# ---------------------------------------------------------------------------
#
# We don't patch run_pipeline.py or render.py. Instead we set env vars that
# the Blender subprocess inherits:
#
#   JETSON_MODE=1          → render.py: use h264_v4l2m2m before h264_nvenc
#   DOAN_PYTHON            → run_pipeline.py: use our venv python
#   BLENDER               → (overridden on cmdline below)
#
# The actual encoder patch is applied to render.py via a small monkey-patch
# Python snippet that run_pipeline.py passes to Blender. BUT: render.py is
# a Blender Python script, so we can't easily inject into it without editing.
#
# Solution: We patch render.py TEMPORARILY with a Jetson encoder shim, and
# restore it after the run. The shim is a 3-line addition that checks
# JETSON_MODE=1 and reroutes _nvenc_available() to always return False so
# the encoder cascade hits V4L2 / libx264.
#
# This is done by writing a thin wrapper script that run_pipeline.py calls
# instead of the real render.py. The wrapper prepends the Jetson encoder
# override to sys.path via a stub module.
#
# Actually — simplest approach: write a jetson_render_shim.py that:
#   1. Monkey-patches _nvenc_available to False
#   2. Injects V4L2 / libx264 encoder cascade into _ffmpeg_encode
#   3. exec()s the real render.py
# Then set DOAN_RENDER_SCRIPT=jetson_render_shim.py to tell run_pipeline.py
# to use it. But run_pipeline.py doesn't have that hook.
#
# Cleanest: patch render.py in-place for Jetson, restore after.
# We do this with a sed command that's easy to invert.
# ---------------------------------------------------------------------------

RENDER_PY="$PROJECT_ROOT/scripts/render.py"
RENDER_PY_BACKUP="$PROJECT_ROOT/scripts/render.py.desktop_backup"

apply_jetson_render_patch() {
    if [[ -f "$RENDER_PY_BACKUP" ]]; then
        ok "Jetson render patch already applied (backup exists)."
        return
    fi

    log "=== Phase 5: Applying Jetson render patch ==="

    cp "$RENDER_PY" "$RENDER_PY_BACKUP"

    # Write the Jetson patch as a Python fragment to be inserted.
    # We replace the _nvenc_available function and _ffmpeg_encode to add
    # the V4L2 / libx264 cascade.
    python3 - <<'PATCH_SCRIPT'
import re, sys

render_py = sys.argv[1]

with open(render_py, 'r') as f:
    src = f.read()

# ---- Replace _nvenc_available -----------------------------------------------
old_nvenc = '''def _nvenc_available() -> bool:
    """Probe once per process whether ffmpeg can actually use h264_nvenc.

    Checks both that the encoder is compiled in AND that it can init on this
    GPU (e.g. missing NVENC driver bits would fail at encode time, not at
    -encoders listing time).
    """
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    import subprocess
    try:
        probe = subprocess.run(
            # NVENC requires at least ~145x49 (varies by GPU); use 256x256 to
            # stay safely above the minimum on all supported hardware.
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=15,
        )
        _NVENC_AVAILABLE = probe.returncode == 0
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE'''

new_nvenc = '''def _nvenc_available() -> bool:
    """Probe once per process whether ffmpeg can actually use h264_nvenc.

    On Jetson (JETSON_MODE=1 env var), NVENC is not available — Jetson uses
    the V4L2 HW encoder (h264_v4l2m2m) instead. Return False immediately.
    """
    # ponytail: Jetson fast-path — skip NVENC probe entirely
    import os as _os
    if _os.environ.get("JETSON_MODE") == "1":
        return False
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    import subprocess
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=15,
        )
        _NVENC_AVAILABLE = probe.returncode == 0
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE'''

# ---- Replace _ffmpeg_encode -------------------------------------------------
old_encode_tail = '''    if not _nvenc_available():
        print("  [ffmpeg] h264_nvenc unavailable — GPU encoding required; "
              "no CPU fallback.", flush=True)
        return False
    cmd = base + [
        "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p",
        "-cq", "20", video_path,
    ]
    proc = _run(cmd, "NVENC")
    if proc is None:
        return False
    if proc.returncode != 0 and proc.stderr:
        _print_ffmpeg_stderr("NVENC", proc.returncode, proc.stderr)
    return proc.returncode == 0'''

new_encode_tail = '''    # ponytail: encoder cascade — NVENC (desktop) → V4L2 (Jetson) → libx264 (CPU)
    import os as _os
    if _os.environ.get("JETSON_MODE") != "1" and _nvenc_available():
        cmd = base + [
            "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p",
            "-cq", "20", video_path,
        ]
        proc = _run(cmd, "NVENC")
        if proc is not None and proc.returncode == 0:
            return True
        if proc is not None and proc.stderr:
            _print_ffmpeg_stderr("NVENC", proc.returncode, proc.stderr)
        print("  [ffmpeg] NVENC failed — trying V4L2 HW encoder", flush=True)

    # Jetson HW encoder: h264_v4l2m2m (V4L2 Memory-to-Memory)
    import subprocess as _sp
    v4l2_probe = _sp.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
         "-c:v", "h264_v4l2m2m", "-frames:v", "1", "-f", "null", "-"],
        capture_output=True, text=True, timeout=15,
    )
    if v4l2_probe.returncode == 0:
        cmd = base + [
            "-c:v", "h264_v4l2m2m", "-pix_fmt", "yuv420p",
            "-b:v", "8M", "-maxrate", "10M", "-bufsize", "20M", video_path,
        ]
        proc = _run(cmd, "V4L2")
        if proc is not None and proc.returncode == 0:
            return True
        if proc is not None and proc.stderr:
            _print_ffmpeg_stderr("V4L2", proc.returncode, proc.stderr)
        print("  [ffmpeg] V4L2 failed — falling back to libx264 (CPU)", flush=True)
    else:
        if not _os.environ.get("JETSON_MODE") == "1":
            print("  [ffmpeg] h264_nvenc unavailable and V4L2 unavailable — trying libx264", flush=True)

    # CPU fallback: libx264 (slow but universally available)
    x264_probe = _sp.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True, timeout=10,
    )
    if "libx264" not in (x264_probe.stdout or ""):
        print("  [ffmpeg] libx264 not available — no encoder found. Install ffmpeg with x264 support.", flush=True)
        return False
    cmd = base + [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast", video_path,
    ]
    proc = _run(cmd, "libx264")
    if proc is None:
        return False
    if proc.returncode != 0 and proc.stderr:
        _print_ffmpeg_stderr("libx264", proc.returncode, proc.stderr)
    return proc.returncode == 0'''

if old_nvenc not in src:
    print("ERROR: Could not find _nvenc_available function to patch", file=sys.stderr)
    sys.exit(1)

if old_encode_tail not in src:
    print("ERROR: Could not find _ffmpeg_encode tail to patch", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_nvenc, new_nvenc)
src = src.replace(old_encode_tail, new_encode_tail)

with open(render_py, 'w') as f:
    f.write(src)

print("[patch] render.py patched for Jetson encoder cascade (NVENC→V4L2→libx264)")
PATCH_SCRIPT
    python3 "$PROJECT_ROOT/scripts/render.py.patch_helper.py" 2>/dev/null || \
    python3 - "$RENDER_PY" <<'INLINE_PATCH'
import sys
render_py = sys.argv[1]
with open(render_py, 'r') as f:
    src = f.read()

old_nvenc = '''def _nvenc_available() -> bool:
    """Probe once per process whether ffmpeg can actually use h264_nvenc.

    Checks both that the encoder is compiled in AND that it can init on this
    GPU (e.g. missing NVENC driver bits would fail at encode time, not at
    -encoders listing time).
    """
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    import subprocess
    try:
        probe = subprocess.run(
            # NVENC requires at least ~145x49 (varies by GPU); use 256x256 to
            # stay safely above the minimum on all supported hardware.
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=15,
        )
        _NVENC_AVAILABLE = probe.returncode == 0
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE'''

new_nvenc = '''def _nvenc_available() -> bool:
    """Probe once per process whether ffmpeg can actually use h264_nvenc.

    On Jetson (JETSON_MODE=1 env var), NVENC is not available.
    """
    # ponytail: Jetson fast-path
    import os as _os
    if _os.environ.get("JETSON_MODE") == "1":
        return False
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    import subprocess
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=15,
        )
        _NVENC_AVAILABLE = probe.returncode == 0
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE'''

old_encode_tail = '''    if not _nvenc_available():
        print("  [ffmpeg] h264_nvenc unavailable — GPU encoding required; "
              "no CPU fallback.", flush=True)
        return False
    cmd = base + [
        "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p",
        "-cq", "20", video_path,
    ]
    proc = _run(cmd, "NVENC")
    if proc is None:
        return False
    if proc.returncode != 0 and proc.stderr:
        _print_ffmpeg_stderr("NVENC", proc.returncode, proc.stderr)
    return proc.returncode == 0'''

new_encode_tail = '''    # ponytail: encoder cascade NVENC -> V4L2 (Jetson) -> libx264 (CPU)
    import os as _os, subprocess as _sp
    if _os.environ.get("JETSON_MODE") != "1" and _nvenc_available():
        cmd = base + ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-cq", "20", video_path]
        proc = _run(cmd, "NVENC")
        if proc is not None and proc.returncode == 0:
            return True
        if proc is not None and proc.stderr:
            _print_ffmpeg_stderr("NVENC", proc.returncode, proc.stderr)

    v4l2_ok = _sp.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
         "-c:v", "h264_v4l2m2m", "-frames:v", "1", "-f", "null", "-"],
        capture_output=True, text=True, timeout=15,
    ).returncode == 0
    if v4l2_ok:
        cmd = base + ["-c:v", "h264_v4l2m2m", "-pix_fmt", "yuv420p",
                      "-b:v", "8M", "-maxrate", "10M", "-bufsize", "20M", video_path]
        proc = _run(cmd, "V4L2")
        if proc is not None and proc.returncode == 0:
            return True
        if proc is not None and proc.stderr:
            _print_ffmpeg_stderr("V4L2", proc.returncode, proc.stderr)

    x264_ok = "libx264" in (_sp.run(["ffmpeg", "-hide_banner", "-encoders"],
                                     capture_output=True, text=True, timeout=10).stdout or "")
    if not x264_ok:
        print("  [ffmpeg] no encoder available (nvenc/v4l2/libx264)", flush=True)
        return False
    cmd = base + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast", video_path]
    proc = _run(cmd, "libx264")
    if proc is None:
        return False
    if proc.returncode != 0 and proc.stderr:
        _print_ffmpeg_stderr("libx264", proc.returncode, proc.stderr)
    return proc.returncode == 0'''

if old_nvenc not in src:
    print("ERROR: _nvenc_available not found in render.py", file=sys.stderr)
    sys.exit(1)
if old_encode_tail not in src:
    print("ERROR: _ffmpeg_encode tail not found in render.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_nvenc, new_nvenc)
src = src.replace(old_encode_tail, new_encode_tail)

with open(render_py, 'w') as f:
    f.write(src)
print("[patch] render.py patched for Jetson (NVENC->V4L2->libx264 cascade)")
INLINE_PATCH

    ok "render.py patched. Backup at: $RENDER_PY_BACKUP"
}

restore_render_patch() {
    if [[ -f "$RENDER_PY_BACKUP" ]]; then
        cp "$RENDER_PY_BACKUP" "$RENDER_PY"
        rm "$RENDER_PY_BACKUP"
        log "render.py restored to desktop version."
    fi
}

# ---------------------------------------------------------------------------
# PHASE 6: Jetson GPU detection shim (patches run_pipeline.py temporarily)
# ---------------------------------------------------------------------------
# run_pipeline.py calls _gpu_info() which requires nvidia-smi.
# On Jetson, nvidia-smi may not exist or may report 0 VRAM (unified memory).
# We patch _gpu_info to return a synthetic entry based on /proc/meminfo.
# ---------------------------------------------------------------------------

RUN_PIPELINE_PY="$PROJECT_ROOT/scripts/run_pipeline.py"
RUN_PIPELINE_BACKUP="$PROJECT_ROOT/scripts/run_pipeline.py.desktop_backup"

apply_jetson_pipeline_patch() {
    if [[ -f "$RUN_PIPELINE_BACKUP" ]]; then
        ok "Jetson pipeline patch already applied."
        return
    fi

    log "=== Phase 6: Applying Jetson GPU detection patch ==="

    cp "$RUN_PIPELINE_PY" "$RUN_PIPELINE_BACKUP"

    python3 - "$RUN_PIPELINE_PY" <<'PIPELINE_PATCH'
import sys

run_py = sys.argv[1]
with open(run_py, 'r') as f:
    src = f.read()

old_gpu_info = '''def _gpu_info():
    """Return a list of (index, free_mib) tuples for all GPUs, or raise
    SystemExit if no NVIDIA GPU is available. Fail-fast — this pipeline
    cannot render on CPU.

    Single source of truth for both GPU count and per-GPU free VRAM. Cache is
    safe for a single pipeline run (VRAM only changes across render launches).
    """
    global _GPU_INFO_CACHE
    if _GPU_INFO_CACHE is not None:
        return _GPU_INFO_CACHE
    # Cache miss → probe.
    import shutil as _sh
    if _sh.which("nvidia-smi") is None:
        raise SystemExit(
            "FAIL: nvidia-smi not found — no NVIDIA GPU detected. "
            "This pipeline requires an NVIDIA GPU (OptiX or CUDA) for Cycles rendering. "
            "CPU rendering is not supported.")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        result = []
        for line in out.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                idx = int(parts[0].strip())
                free = int(parts[1].strip())
                if free > 0:
                    result.append((idx, free))
        if not result:
            raise SystemExit(
                "FAIL: nvidia-smi returned but no GPUs with free VRAM > 0. "
                "GPU rendering cannot proceed.")
        _GPU_INFO_CACHE = result
        return result
    except subprocess.TimeoutExpired:
        raise SystemExit(
            "FAIL: nvidia-smi timed out (>10s). NVIDIA GPU probe failed — "
            "rendering cannot proceed.")
    except (ValueError, OSError) as e:
        raise SystemExit(
            f"FAIL: nvidia-smi output parse failed ({type(e).__name__}: {e}). "
            f"raw stdout: {out.stdout!r}. NVIDIA GPU probe failed — "
            f"rendering cannot proceed.")'''

new_gpu_info = '''def _gpu_info():
    """Return a list of (index, free_mib) tuples for all GPUs.

    On Jetson (JETSON_MODE=1 or nvidia-smi absent), uses /proc/meminfo to
    estimate available unified memory as a synthetic GPU entry.
    ponytail: unified-memory estimate — 60% of MemAvailable as GPU budget.
    """
    global _GPU_INFO_CACHE
    if _GPU_INFO_CACHE is not None:
        return _GPU_INFO_CACHE
    import shutil as _sh, os as _os

    # Jetson path: no nvidia-smi or explicitly forced
    is_jetson = (_os.environ.get("JETSON_MODE") == "1" or
                 _sh.which("nvidia-smi") is None)
    if is_jetson:
        # Read available RAM from /proc/meminfo; treat 60% as GPU budget
        # (conservative for Orin 32GB unified memory)
        mem_avail_mib = 8192  # safe fallback if /proc/meminfo unreadable
        try:
            with open("/proc/meminfo") as _mf:
                for _line in _mf:
                    if _line.startswith("MemAvailable:"):
                        mem_avail_mib = int(_line.split()[1]) // 1024
                        break
        except Exception:
            pass
        budget_mib = int(mem_avail_mib * 0.6)
        print(f"[GPU] Jetson unified memory: {mem_avail_mib} MiB available, "
              f"budget {budget_mib} MiB for GPU rendering", flush=True)
        _GPU_INFO_CACHE = [(0, budget_mib)]
        return _GPU_INFO_CACHE

    # Desktop path: use nvidia-smi
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        result = []
        for line in out.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                idx = int(parts[0].strip())
                free = int(parts[1].strip())
                if free > 0:
                    result.append((idx, free))
        if not result:
            raise SystemExit(
                "FAIL: nvidia-smi returned but no GPUs with free VRAM > 0. "
                "GPU rendering cannot proceed.")
        _GPU_INFO_CACHE = result
        return result
    except subprocess.TimeoutExpired:
        raise SystemExit(
            "FAIL: nvidia-smi timed out (>10s). NVIDIA GPU probe failed — "
            "rendering cannot proceed.")
    except (ValueError, OSError) as e:
        raise SystemExit(
            f"FAIL: nvidia-smi output parse failed ({type(e).__name__}: {e}). "
            f"raw stdout: {out.stdout!r}. NVIDIA GPU probe failed — "
            f"rendering cannot proceed.")'''

if old_gpu_info not in src:
    print("ERROR: _gpu_info not found in run_pipeline.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_gpu_info, new_gpu_info)

with open(run_py, 'w') as f:
    f.write(src)
print("[patch] run_pipeline.py patched for Jetson GPU detection")
PIPELINE_PATCH

    ok "run_pipeline.py patched. Backup at: $RUN_PIPELINE_BACKUP"
}

restore_pipeline_patch() {
    if [[ -f "$RUN_PIPELINE_BACKUP" ]]; then
        cp "$RUN_PIPELINE_BACKUP" "$RUN_PIPELINE_PY"
        rm "$RUN_PIPELINE_BACKUP"
        log "run_pipeline.py restored to desktop version."
    fi
}

# ---------------------------------------------------------------------------
# PHASE 7: Run the pipeline
# ---------------------------------------------------------------------------
phase_run() {
    log "=== Phase 7: Running pipeline ==="

    source "$VENV_DIR/bin/activate"

    mkdir -p "$OUT_DIR"

    # Export env vars for Jetson mode
    export JETSON_MODE=1
    export DOAN_PYTHON="$VENV_DIR/bin/python"
    # Override BLENDER in environment so run_pipeline.py picks it up
    # (run_pipeline.py uses shutil.which("blender") as default)
    export PATH="$BLENDER_INSTALL_DIR:$PATH"

    log "Config:"
    log "  seed          = $SEED"
    log "  seconds       = $SECONDS_VAL"
    log "  samples       = $SAMPLES (Cycles, lower = faster)"
    log "  output        = $OUT_DIR"
    log "  JETSON_MODE   = $JETSON_MODE"
    log "  DOAN_PYTHON   = $DOAN_PYTHON"
    log "  blender       = $(command -v blender 2>/dev/null || echo 'NOT FOUND')"
    log "  ffmpeg        = $(command -v ffmpeg 2>/dev/null || echo 'NOT FOUND')"

    # Build the pipeline command
    CMD=(
        "$VENV_DIR/bin/python"
        "$PROJECT_ROOT/scripts/run_pipeline.py"
        "--seed" "$SEED"
        "--seconds" "$SECONDS_VAL"
        "--out" "$OUT_DIR"
        "--samples" "$SAMPLES"
        "--jobs" "1"           # Single iGPU on Jetson: no parallel
        "--max-workers-per-gpu" "1"
        "--silence-timeout" "1800"  # 30 min (Jetson is slower)
    )

    [[ -n "$DEMAND_SCALE" ]] && CMD+=("--demand-scale" "$DEMAND_SCALE")
    [[ -n "$SIGNAL_FLAG"  ]] && CMD+=("$SIGNAL_FLAG")

    log "Running: ${CMD[*]}"
    echo ""

    "${CMD[@]}"
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo ""
        log "=== PIPELINE COMPLETE ==="
        log "Output: $OUT_DIR"
        log ""
        log "Videos:"
        find "$OUT_DIR" -name "*.mp4" | sort | while read -r f; do
            local size
            size=$(du -sh "$f" 2>/dev/null | cut -f1)
            log "  $f  ($size)"
        done
    else
        echo ""
        log "=== PIPELINE FAILED (exit code $exit_code) ==="
        log "Check output above for errors."
        return $exit_code
    fi
}

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "Interrupted or failed — restoring patched files..."
    fi
    restore_render_patch
    restore_pipeline_patch
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo "============================================================"
    echo " Jetson Orin 32GB — 3D Intersection Video Generation"
    echo "============================================================"
    echo ""

    # Tip: remind user how to run safely over SSH if they're not already in bg
    if [[ -z "${TMUX:-}" ]] && [[ -z "${JETSON_BG_CHILD:-}" ]]; then
        echo "  TIP: To run safely over SSH (survives disconnect), use:"
        echo "    bash $SCRIPT_DIR/jetson_setup_and_run.sh --background [your options]"
        echo ""
    fi

    check_jetson
    check_jetpack
    check_unified_memory
    echo ""

    if [[ "$SKIP_SETUP" == "false" ]]; then
        phase_system_deps
        echo ""
        if [[ "$SKIP_BLENDER" == "false" ]]; then
            phase_blender
            echo ""
        fi
        phase_python_env
        echo ""
        phase_verify_cuda
        echo ""
    else
        log "Skipping setup phases (--skip-setup)"
        # Still need to resolve BLENDER_BIN
        if [[ ! -x "$BLENDER_BIN" ]]; then
            if command -v blender &>/dev/null; then
                BLENDER_BIN=$(command -v blender)
            else
                die "Blender not found. Run without --skip-setup first, or install Blender manually."
            fi
        fi
        source "$VENV_DIR/bin/activate" 2>/dev/null || die "Virtualenv not found at $VENV_DIR. Run without --skip-setup first."
    fi

    apply_jetson_render_patch
    apply_jetson_pipeline_patch

    echo ""
    phase_run
}

main
