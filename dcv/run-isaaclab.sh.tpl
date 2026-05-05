#!/bin/bash
set -euo pipefail
CONTAINER_IMAGE="${ISAAC_LAB_IMAGE:-__CONTAINER_IMAGE__}"
SESSION_NAME="isaac-lab"
LEISAAC_COMMIT="__LEISAAC_COMMIT__"
PKGS_DIR="/home/ubuntu/isaaclab-pkgs"
MARKER="$PKGS_DIR/.leisaac-installed"
ASSETS_DIR="/home/ubuntu/leisaac-assets"
ASSETS_MARKER="$ASSETS_DIR/.assets-downloaded"
mkdir -p ~/docker/isaac-sim/cache/kit ~/docker/isaac-sim/cache/ov
mkdir -p ~/docker/isaac-sim/cache/pip ~/docker/isaac-sim/cache/glcache
mkdir -p ~/docker/isaac-sim/cache/computecache ~/docker/isaac-sim/logs
mkdir -p ~/docker/isaac-sim/data ~/docker/isaac-sim/documents
mkdir -p "$PKGS_DIR"
if [[ ! -f "$MARKER" ]]; then
  docker run --rm --gpus all \
    --entrypoint bash \
    -e ACCEPT_EULA=Y \
    -e PYTHONPATH=/workspace/isaaclab-pkgs \
    -v "$PKGS_DIR":/workspace/isaaclab-pkgs:rw \
    "$CONTAINER_IMAGE" \
    -c "/workspace/isaaclab/_isaac_sim/python.sh -m pip install --target /workspace/isaaclab-pkgs lerobot 'leisaac[gr00t] @ git+https://github.com/LightwheelAI/leisaac.git@${LEISAAC_COMMIT}#subdirectory=source/leisaac' && touch /workspace/isaaclab-pkgs/.leisaac-installed"
  # Fix root-owned files from docker run (container runs as root)
  sudo chown -R ubuntu:ubuntu "$PKGS_DIR"
fi
if [[ ! -f "$ASSETS_MARKER" ]]; then
  mkdir -p "$ASSETS_DIR/scenes" "$ASSETS_DIR/robots"
  curl -fsSL -o /tmp/kitchen_with_orange.zip \
    https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip
  unzip -o /tmp/kitchen_with_orange.zip -d "$ASSETS_DIR/scenes/"
  rm -f /tmp/kitchen_with_orange.zip
  curl -fsSL -o "$ASSETS_DIR/robots/so101_follower.usd" \
    https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd
  touch "$ASSETS_MARKER"
fi
SCRIPTS_DIR="/home/ubuntu/leisaac-repo"
if [[ ! -d "$SCRIPTS_DIR/scripts" ]]; then
  git clone https://github.com/LightwheelAI/leisaac.git "$SCRIPTS_DIR"
  cd "$SCRIPTS_DIR" && git checkout "$LEISAAC_COMMIT"
fi
# Patch policy_inference.py for headless keyboard support.
# omni.appwindow.get_default_app_window() returns None when running via SSH
# without a windowed display, crashing the Controller.__init__. This wraps
# the keyboard setup in None-checks so eval can run headless.
POLICY_PY="$PKGS_DIR/../leisaac-repo/scripts/evaluation/policy_inference.py"
if [[ -f "$POLICY_PY" ]] && grep -q 'self._appwindow.get_keyboard()' "$POLICY_PY"; then
  python3 - "$POLICY_PY" << 'PATCH_EOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

old_init = """\
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )"""

new_init = """\
        try:
            self._appwindow = omni.appwindow.get_default_app_window()
        except Exception:
            self._appwindow = None
        self._input = carb.input.acquire_input_interface()
        if self._appwindow is not None:
            self._keyboard = self._appwindow.get_keyboard()
        else:
            self._keyboard = None
        if self._keyboard is not None:
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                self._on_keyboard_event,
            )
        else:
            self._keyboard_sub = None"""

if old_init in src:
    src = src.replace(old_init, new_init)
    with open(path, "w") as f:
        f.write(src)
    print("Patched policy_inference.py for headless keyboard support")
else:
    print("policy_inference.py already patched or structure changed — skipping")
PATCH_EOF
fi

xhost +local:docker 2>/dev/null || true
XAUTH_FILE="/run/user/1000/dcv/console.xauth"
if [[ ! -f "$XAUTH_FILE" ]]; then XAUTH_FILE="$HOME/.Xauthority"; fi
XAUTH_MOUNT=""
if [[ -f "$XAUTH_FILE" ]]; then XAUTH_MOUNT="-v $XAUTH_FILE:/root/.Xauthority:ro"; fi
docker run \
  --name "$SESSION_NAME" \
  --entrypoint bash \
  -it \
  --gpus all \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -e DISPLAY \
  -e LEISAAC_ASSETS_ROOT=/assets \
  -e PYTHONPATH=/workspace/isaaclab-pkgs:${PYTHONPATH:-} \
  $XAUTH_MOUNT \
  -v "$ASSETS_DIR":/assets:ro \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -v "$PKGS_DIR":/workspace/isaaclab-pkgs:rw \
  --rm \
  --network=host \
  -v $HOME/leisaac-repo/scripts:/workspace/scripts:ro \
  "$CONTAINER_IMAGE" \
  "$@"
