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
    -c "/workspace/isaaclab/_isaac_sim/python.sh -m pip install --target /workspace/isaaclab-pkgs 'leisaac[gr00t] @ git+https://github.com/LightwheelAI/leisaac.git@${LEISAAC_COMMIT}#subdirectory=source/leisaac' && touch /workspace/isaaclab-pkgs/.leisaac-installed"
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
# Patch policy_inference.py to support gr00tn1.6 (upstream LeIsaac v0.3.0 only supports gr00tn1.5)
EVAL_SCRIPT="$SCRIPTS_DIR/scripts/evaluation/policy_inference.py"
if [[ -f "$EVAL_SCRIPT" ]] && ! grep -q "gr00tn1.6" "$EVAL_SCRIPT"; then
  sed -i 's/if model_type in \["gr00tn1.5", "lerobot", "openpi"\]/if model_type in ["gr00tn1.5", "gr00tn1.6", "lerobot", "openpi"]/' "$EVAL_SCRIPT"
  sed -i 's/if args_cli.policy_type == "gr00tn1.5"/if args_cli.policy_type in ["gr00tn1.5", "gr00tn1.6"]/' "$EVAL_SCRIPT"
fi
# Patch Controller class for headless mode (no app window / keyboard in headless IsaacSim)
if [[ -f "$EVAL_SCRIPT" ]] && grep -q 'self._keyboard = self._appwindow.get_keyboard()$' "$EVAL_SCRIPT"; then
  sed -i 's/self._keyboard = self._appwindow.get_keyboard()$/self._keyboard = self._appwindow.get_keyboard() if self._appwindow else None/' "$EVAL_SCRIPT"
  sed -i '/self._keyboard = self._appwindow.get_keyboard()/!{
    /self._keyboard_sub = self._input.subscribe_to_keyboard_events(/{
      N;N;N
      s/self._keyboard_sub = self._input.subscribe_to_keyboard_events(\n *self._keyboard,\n *self._on_keyboard_event,\n *)/if self._keyboard:\n            self._keyboard_sub = self._input.subscribe_to_keyboard_events(\n                self._keyboard,\n                self._on_keyboard_event,\n            )\n        else:\n            self._keyboard_sub = None/
    }
  }' "$EVAL_SCRIPT"
fi
# Patch Gr00tServicePolicyClient for N1.6 PolicyServer API compatibility.
# Upstream LeIsaac client was written for N1.5; the N1.6 PolicyServer expects:
#   - observation wrapped as {"observation": obs_dict} (server does **kwargs expansion)
#   - response is (action_dict, info_dict) tuple, not bare action dict
#   - language key "annotation.human.action.task_description" (not "annotation.human.task_description")
#   - video arrays with temporal dim (B,1,H,W,C) and state as float32 (B,1,D)
#   - with --use-sim-policy-wrapper, action keys have "action." prefix
POLICY_CLIENT="$PKGS_DIR/leisaac/policy/service_policy_clients.py"
if [[ -f "$POLICY_CLIENT" ]] && ! grep -q 'GR00T N1.6 SimPolicyWrapper' "$POLICY_CLIENT"; then
  python3 << 'PATCH_EOF'
import re

path = "/home/ubuntu/isaaclab-pkgs/leisaac/policy/service_policy_clients.py"
with open(path) as f:
    content = f.read()

old = '''    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs_dict = {f"video.{key}": observation_dict[key].cpu().numpy().astype(np.uint8) for key in self.camera_keys}

        if "single_arm" in self.modality_keys:
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            obs_dict["state.single_arm"] = joint_pos[:, 0:5].astype(np.float64)
            obs_dict["state.gripper"] = joint_pos[:, 5:6].astype(np.float64)
        # TODO: add bi-arm support

        obs_dict["annotation.human.task_description"] = [observation_dict["task_description"]]

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "state.single_arm": np.zeros((1, 5)),
                "state.gripper": np.zeros((1, 1)),
                "annotation.human.action.task_description": [observation_dict["task_description"]],
            }
        """

        # get the action chunk via the policy server
        action_chunk = self.call_endpoint("get_action", obs_dict)

        """
            Example of action_chunk for single arm task:
            action_chunk = {
                "action.single_arm": np.zeros((1, 5)),
                "action.gripper": np.zeros((1, 1)),
            }
        """
        concat_action = np.concatenate(
            [action_chunk["action.single_arm"], action_chunk["action.gripper"]],
            axis=1,
        )
        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        return torch.from_numpy(concat_action[:, None, :])'''

new = """    def get_action(self, observation_dict: dict) -> torch.Tensor:
        # Build flat-keyed observation for GR00T N1.6 SimPolicyWrapper
        obs_dict = {}
        for key in self.camera_keys:
            vid = observation_dict[key].cpu().numpy().astype(np.uint8)
            if vid.ndim == 4:  # (B, H, W, C) -> (B, 1, H, W, C)
                vid = vid[:, None, :, :, :]
            obs_dict[f"video.{key}"] = vid

        if "single_arm" in self.modality_keys:
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            arm = joint_pos[:, 0:5].astype(np.float32)
            grip = joint_pos[:, 5:6].astype(np.float32)
            if arm.ndim == 2:  # (B, D) -> (B, 1, D)
                arm = arm[:, None, :]
            if grip.ndim == 2:
                grip = grip[:, None, :]
            obs_dict["state.single_arm"] = arm
            obs_dict["state.gripper"] = grip

        # Language key must match model config; format as tuple (B,)
        obs_dict["annotation.human.action.task_description"] = (observation_dict["task_description"],)

        # GR00T PolicyServer expands data as **kwargs to get_action(observation, options)
        response = self.call_endpoint("get_action", {"observation": obs_dict})

        # Server returns (action_dict, info_dict) tuple via msgpack
        if isinstance(response, (list, tuple)):
            action_chunk = response[0]
        else:
            action_chunk = response

        # SimPolicyWrapper prefixes keys with "action."
        arm_key = "action.single_arm" if "action.single_arm" in action_chunk else "single_arm"
        grip_key = "action.gripper" if "action.gripper" in action_chunk else "gripper"

        arm_action = action_chunk[arm_key]
        grip_action = action_chunk[grip_key]

        # Actions are (B, T, D); concatenate arm + gripper along last dim
        concat_action = np.concatenate([arm_action, grip_action], axis=-1)
        if concat_action.ndim == 3:  # Remove batch dim -> (T, D)
            concat_action = concat_action[0]

        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        return torch.from_numpy(concat_action[:, None, :])"""

if old not in content:
    print("ERROR: old text not found — client may already be patched")
    raise SystemExit(1)

content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("OK")
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
