---
name: deploy-stack
description: >
  Guide for deploying the Embodied AI Platform CDK stacks (IsaacGr00tBatchStack + IsaacLabDcvStack)
  from scratch, including prerequisites, deployment, bootstrap monitoring, SSH setup via SSM,
  verification, W&B training visualization, and model evaluation. Also covers tearing down stacks
  and cleaning up retained resources. Use this skill whenever someone wants to deploy, redeploy,
  tear down the CDK infrastructure, visualize training metrics, or run model evaluations —
  even if they just say "set up the stack", "deploy to AWS", "get DCV running", "view training
  loss", "run evals", or "clean up AWS resources".
---

# Deploy Embodied AI Platform CDK Stacks

This skill walks through deploying the two CDK stacks that make up the platform's AWS
infrastructure, setting up SSH access to the GPU workstation, visualizing training metrics,
and running model evaluations. It also covers tearing everything down when you're done.

The two stacks are:
- **IsaacGr00tBatchStack** — VPC, EFS, ECR, CodeBuild, AWS Batch compute environment and job queue
- **IsaacLabDcvStack** — GPU-accelerated DCV workstation (EC2 instance with Elastic IP)

The DCV stack depends on the Batch stack (shares its VPC and EFS), so deploy order matters:
Batch first, DCV second. Destroy order is reversed: DCV first, Batch second.

## Phase 1: Prerequisites

Before deploying, verify these are in place. Run all checks and report any failures before
proceeding.

```bash
# AWS CLI configured with correct account
aws sts get-caller-identity --query '[Account, Arn]' --output text

# CDK available (via npx, no global install needed)
npx cdk --version

# Python venv exists at repo root
ls -la <repo-root>/.venv/bin/python

# If venv is missing, create it:
cd <repo-root> && python3 -m venv .venv
source .venv/bin/activate
pip install -r training/gr00t/infra/requirements.txt
pip install -r dcv/requirements.txt
```

Also confirm CDK has been bootstrapped in the target account/region:
```bash
aws cloudformation describe-stacks --stack-name CDKToolkit --query 'Stacks[0].StackStatus' --output text
```
If that fails, run `npx cdk bootstrap` from `training/gr00t/infra/`.

## Phase 2: Validate with CDK Synth

Always synth before deploying — it catches version mismatches, missing dependencies, and
code errors at zero cost (no AWS resources created).

```bash
cd training/gr00t/infra
npx cdk synth --quiet
```

**Expected output:** `Successfully synthesized to .../cdk.out` listing both stack names.

If synth fails, fix the error before proceeding. Common issues:
- Missing Python dependencies → `pip install -r requirements.txt`
- Version validation errors → check `dcv/versions.py` for supported IsaacSim versions

## Phase 3: Deploy Stacks

No context parameters needed — the stacks auto-create VPC, EFS, and ECR.

```bash
cd training/gr00t/infra

# Step 1: Batch stack (VPC, EFS, ECR, Batch — ~3 min)
npx cdk deploy IsaacGr00tBatchStack --require-approval=never

# Step 2: DCV stack (GPU instance — ~3 min for CFN, then ~15 min bootstrap)
npx cdk deploy IsaacLabDcvStack --require-approval=never
```

After each deploy, capture the stack outputs — you'll need them for SSH setup and later
operations. Key outputs include instance ID, Elastic IP, EFS ID, ECR URI, checkpoint S3
path, and DCV credentials.

## Phase 4: Monitor Bootstrap Completion

The DCV instance runs a container-based bootstrap script that installs NVIDIA drivers,
Docker, pulls the IsaacLab container from NGC, installs DCV, and mounts EFS. The
CloudFormation stack waits for a cfn-signal before marking CREATE_COMPLETE. Monitor
progress via SSM (no SSH needed yet).

```bash
INSTANCE_ID=<instance-id-from-stack-outputs>

# Check progress (repeat every 2-3 minutes)
CMD_ID=$(aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["cat /var/log/dcv-bootstrap.summary 2>/dev/null || echo BOOTSTRAP_NOT_STARTED"]' \
  --output text --query 'Command.CommandId') \
&& sleep 5 \
&& aws ssm get-command-invocation \
  --command-id $CMD_ID \
  --instance-id $INSTANCE_ID \
  --query 'StandardOutputContent' --output text
```

Steps appear one by one as `STEP_OK` entries. The full sequence (16 named steps):

**Prerequisites (configure_dcv_instance.sh):**
`baseline-update` → `set-ubuntu-password` → `disable-nouveau` (non-fatal) →
`install-nvidia-driver` → `install-docker-nvidia-toolkit` → `install-aws-cli-v2` →
`install-efs-utils` → `install-firefox` (non-fatal) → `install-cfn-bootstrap`

**Container + DCV layer (dcv_construct.py):**
`pull-isaaclab-container` (~3-5 min) → `create-helper-script` → `install-desktop` (~3 min) →
`disable-gnome-initial-setup` (non-fatal) → `install-nice-dcv` → `configure-dcv` (non-fatal) →
`install-auto-dcv-service`

After all steps, the script sends cfn-signal and writes an ALL_DONE marker. The instance
does **not** reboot — bootstrap completes in-place. Total time: ~15-20 minutes.

You may also see `STEP_WARN:nvidia-driver-loaded` — this is expected on first boot (the
NVIDIA kernel module loads after the next reboot). GPU containers will work after a manual
`sudo reboot`.

If any step shows `STEP_FAIL`, check the detailed log:

```bash
CMD_ID=$(aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["sudo grep -A 50 \"== START: <step-name> ==\" /var/log/dcv-bootstrap.log"]' \
  --output text --query 'Command.CommandId') \
&& sleep 5 \
&& aws ssm get-command-invocation \
  --command-id $CMD_ID \
  --instance-id $INSTANCE_ID \
  --query 'StandardOutputContent' --output text
```

## Phase 5: Set Up SSH via SSM

SSH through SSM is the recommended access method — it doesn't require port 22 to be open
in the security group. It tunnels through the AWS Session Manager plugin.

### 5a. Verify Session Manager plugin is installed locally

```bash
session-manager-plugin --version
```

If missing: Ubuntu/Debian `sudo dpkg -i session-manager-plugin.deb`, macOS `brew install --cask session-manager-plugin`.

### 5b. Push SSH public key to the instance

```bash
INSTANCE_ID=<instance-id>
PUBKEY=$(cat ~/.ssh/id_ed25519.pub)

aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters "commands=[\"mkdir -p /home/ubuntu/.ssh && echo '$PUBKEY' > /home/ubuntu/.ssh/authorized_keys && chmod 700 /home/ubuntu/.ssh && chmod 600 /home/ubuntu/.ssh/authorized_keys && chown -R ubuntu:ubuntu /home/ubuntu/.ssh\"]" \
  --output text --query 'Command.CommandId'
```

If the user doesn't have an SSH key pair yet: `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""`

### 5c. Configure SSH config

Add or update this entry in `~/.ssh/config`:

```
Host dcv-isaac
  HostName <instance-id>
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
  ProxyCommand aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p' --region us-west-2
```

### 5d. Test the connection

```bash
ssh -o StrictHostKeyChecking=accept-new dcv-isaac "echo 'SSH OK'"
```

If this prints `SSH OK`, subsequent connections just need `ssh dcv-isaac`.

## Phase 6: Verify Everything Works

```bash
# 1. Bootstrap summary — all named steps should be STEP_OK
ssh dcv-isaac "cat /var/log/dcv-bootstrap.summary"

# 2. EFS mounted
ssh dcv-isaac "mount | grep efs"
# Expected: 127.0.0.1:/ on /mnt/efs type nfs4 ...

# 3. Container image pulled
ssh dcv-isaac "docker images | grep isaac-lab"
# Expected: nvcr.io/nvidia/isaac-lab   2.3.0   ...

# 4. Helper script installed
ssh dcv-isaac "test -x /usr/local/bin/run-isaaclab.sh && echo 'Helper script OK'"
# Expected: Helper script OK

# 5. Host venv with tensorboard/wandb
ssh dcv-isaac "/home/ubuntu/.venv/bin/tensorboard --version"
# Expected: TensorBoard X.Y.Z

# 6. Persistent package directory exists
ssh dcv-isaac "ls -d /home/ubuntu/isaaclab-pkgs"
# Expected: /home/ubuntu/isaaclab-pkgs

# 7. NVIDIA GPU detected (requires reboot after first deploy)
ssh dcv-isaac "nvidia-smi --query-gpu=name --format=csv,noheader"
# Expected: NVIDIA L4 (or similar). If "failed to initialize", run: sudo reboot
```

The DCV web console is also available at `https://<elastic-ip>:8443` (accept the
self-signed certificate warning).

## Phase 6a: Container and LeIsaac Testing

Before running evaluations, verify the containerized IsaacLab environment works.

### 6a.1 Launch IsaacLab Container

The helper script wraps `docker run` with GPU access, X11 forwarding, cache volumes, and
persistent package mounts. On first launch, it auto-installs leisaac to the persistent
volume (~30 seconds).

```bash
ssh dcv-isaac
run-isaaclab.sh
```

**Expected:** First run prints `Installing leisaac @<commit> to persistent volume...`, then
drops into a container shell. Subsequent runs skip the install step.

> **Note:** The container Python is at `/workspace/isaaclab/_isaac_sim/python.sh` (an Isaac
> Sim wrapper, not on `$PATH` as `python`). All Python commands below use this wrapper.
> `import leisaac` requires the full IsaacSim runtime (omni.physics) to be initialised —
> use it only inside `policy_inference.py` or a full sim launch, not bare `python -c`.

### 6a.2 Verify IsaacLab Python Wrapper

```bash
/workspace/isaaclab/_isaac_sim/python.sh --version
# Expected: Python 3.10.x

/workspace/isaaclab/_isaac_sim/python.sh -c "import torch; print(f'torch {torch.__version__}')"
# Expected: torch 2.x.x
```

### 6a.3 Verify LeIsaac Installation

LeIsaac is installed to `/workspace/isaaclab-pkgs` (mounted from host). Verify the dist-info
and that the N1.6 policy client class is present:

```bash
grep "^Version" /workspace/isaaclab-pkgs/leisaac-*.dist-info/METADATA
# Expected: Version: 0.3.0

grep "class Gr00t16ServicePolicyClient" /workspace/isaaclab-pkgs/leisaac/policy/service_policy_clients.py
# Expected: class Gr00t16ServicePolicyClient(ZMQServicePolicy):
```

> **Why not `import leisaac`?** The `leisaac.__init__` imports `leisaac.tasks` which imports
> `isaaclab` which imports `omni.physics.tensors` — a Kit runtime extension. It is only
> available inside a running IsaacSim process (e.g. `policy_inference.py`). The policy
> *client* classes are self-contained and do not need the sim runtime.

### 6a.4 Test GPU Access Inside Container

```bash
nvidia-smi
# Expected: GPU table showing NVIDIA L4 or g6-series GPU

/workspace/isaaclab/_isaac_sim/python.sh -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
# Expected: CUDA: True, Devices: 1
```

Exit the container with `exit` or Ctrl-D. The leisaac packages persist at
`/home/ubuntu/isaaclab-pkgs/` on the host — subsequent `run-isaaclab.sh` invocations skip
the installation step.

## Phase 7: Visualize Training Metrics (W&B)

Training jobs log to W&B in **offline mode** — run data is written to EFS so it persists
after the Batch container exits. To view loss curves, sync the offline runs into a local
W&B server on the DCV instance.

### 7a. Start the W&B local server

```bash
ssh dcv-isaac "docker run -d --name wandb-local -p 8080:8080 -v wandb-data:/vol wandb/local:latest"
```

### 7b. Create a local account and API key

Open `http://<elastic-ip>:8080` in your browser, create an account, and generate an API
key from Settings → API Keys.

### 7c. Sync offline runs from EFS

```bash
ssh dcv-isaac "bash -l -c '
  export WANDB_BASE_URL=http://localhost:8080
  export WANDB_API_KEY=<your-local-api-key>
  /home/ubuntu/.venv/bin/wandb sync /mnt/efs/gr00t/checkpoints/<JOB_ID>/wandb/offline-run-*
'"
```

### 7d. View loss curves

Open `http://<elastic-ip>:8080` — the synced runs appear with training loss, learning
rate, and other logged metrics.

The W&B server only needs to run when viewing results. Stop with `docker stop wandb-local`,
restart with `docker start wandb-local` — data persists in the `wandb-data` Docker volume.

**Optional — online mode:** To stream metrics directly during training, override env vars
when submitting the Batch job:
```bash
aws batch submit-job ... --container-overrides '{
  "environment": [
    {"name": "WANDB_MODE", "value": "online"},
    {"name": "WANDB_BASE_URL", "value": "http://<DCV_PRIVATE_IP>:8080"},
    {"name": "WANDB_API_KEY", "value": "<your-local-api-key>"}
  ]
}'
```
Use the DCV instance's **private IP** since Batch jobs run in the same VPC.

## Phase 8: Model Evaluation

### Open-loop evaluation (action prediction metrics)

Computes MSE and cosine similarity between predicted and ground-truth actions on a
held-out dataset. Run inside the training container on the DCV instance:

```bash
ACCOUNT_ID=<aws-account-id>
ECR_URI=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest
CHECKPOINT=/mnt/efs/gr00t/checkpoints/<JOB_ID>/checkpoint-6000

# Authenticate to ECR
ssh dcv-isaac "aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com"

# Pull the training image
ssh dcv-isaac "docker pull $ECR_URI"

# Run open-loop eval
ssh dcv-isaac "docker run --gpus all --rm \
  -v /mnt/efs:/mnt/efs:ro \
  --shm-size=8g \
  $ECR_URI \
  python -m gr00t.eval.robot_eval \
    --model-path $CHECKPOINT \
    --embodiment-tag new_embodiment \
    --dataset-path /path/to/eval-dataset \
    --modality-config-path /workspace/scripts/so101_modality_config.py"
```

### Closed-loop evaluation (policy server)

Serves a trained checkpoint as a ZMQ policy server for real-time robot control or
simulation testing. The server accepts observations and returns action predictions over
TCP port 5555.

**Prerequisites:** Verify you have a trained checkpoint on EFS:
```bash
ssh dcv-isaac "ls /mnt/efs/gr00t/checkpoints/"
# Expected: one or more job directories containing checkpoint-<step> subdirs
```

**Start the policy server:**
```bash
CHECKPOINT=/mnt/efs/gr00t/checkpoints/<JOB_ID>/checkpoint-<step>
ECR_URI=<account-id>.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest

ssh dcv-isaac "docker run --gpus all -d \
  --name gr00t-policy-server \
  --shm-size=8g \
  --network host \
  --entrypoint /bin/sh \
  -v $CHECKPOINT:/workspace/checkpoint \
  $ECR_URI \
  -c '/workspace/gr00t-repo/.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model_path /workspace/checkpoint \
    --embodiment_tag NEW_EMBODIMENT \
    --host 0.0.0.0'"
```

> **Important:**
> - Use `--entrypoint /bin/sh -c '...'` — the NGC base image's `/usr/bin/bash` is broken
>   (cannot execute binary file). `/bin/sh` works fine.
> - The server CLI expects **uppercase** enum names: `NEW_EMBODIMENT`, not `new_embodiment`.
>   (The training API's `EmbodimentTag` uses lowercase values, but the `tyro` CLI parser for
>   `run_gr00t_server.py` expects the enum *name*.)
> - Use `--network host` instead of `-p 5555:5555` for simplicity.
> - Pass `--host 0.0.0.0` to allow remote clients. The default is `127.0.0.1` (localhost only).

### Verify the server is running

```bash
# Check container status
ssh dcv-isaac "docker ps | grep gr00t-policy-server"
# Expected: UP status

# Check server logs for startup confirmation
ssh dcv-isaac "docker logs gr00t-policy-server 2>&1 | tail -20"
# Expected: "ZMQ server listening on 0.0.0.0:5555" or similar

# Test port is open
ssh dcv-isaac "ss -tlnp | grep 5555"
# Expected: LISTEN on *:5555
```

If the container exits immediately, check `docker logs gr00t-policy-server`. Common issues:
- Missing checkpoint files — verify the CHECKPOINT path exists and contains model files
- GPU not available — check `nvidia-smi` works (may need `sudo reboot` after first deploy)
- Insufficient shared memory — ensure `--shm-size=8g` is set

### Client-side observation format (N1.6)

Observations are a nested dict with these keys:
- `video`: dict of camera arrays, each shape `(B, T, H, W, C)` uint8 — e.g. `video.front`, `video.wrist`
- `state`: dict of state arrays, each shape `(B, T, D)` float32 — e.g. `state.single_arm` (D=5), `state.gripper` (D=1)
- `language`: dict with key `annotation.human.action.task_description` as `list[list[str]]` (batch x time)

Numpy arrays must be serialized using the `MsgSerializer` protocol (np.save to BytesIO,
wrapped in `{"__ndarray_class__": True, "as_npy": bytes}`). See `gr00t/policy/server_client.py`.

**Client-side response format (N1.6):**
- The server returns `list[action_dict, info_dict]` via msgpack.
- `action_dict` contains `single_arm` shape `(1, 16, 5)` and `gripper` shape `(1, 16, 1)` as numpy arrays.
- The language instruction key is `annotation.human.action.task_description` (N1.5 used
  `annotation.human.task_description` — note the extra `.action.` segment).

### Enable remote client access

The DCV security group does **not** open port 5555 by default. If your client is outside
the VPC, add an ingress rule:

```bash
# Look up the security group from the instance
SG_ID=$(aws ec2 describe-instances \
  --instance-ids <instance-id> \
  --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
  --output text)

# Allow access from your client IP
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 5555 \
  --cidr <client-ip>/32 \
  --description "GR00T policy server access"

# Verify
aws ec2 describe-security-groups --group-ids $SG_ID \
  --query 'SecurityGroups[0].IpPermissions[?ToPort==`5555`]'
```

**Connectivity notes:**
- From within the VPC (Batch jobs, other EC2): use the DCV instance's **private IP**
- From outside the VPC: use the **Elastic IP** and ensure the security group rule is in place
- Test connectivity: `nc -zv <dcv-ip> 5555`

Stop the server: `ssh dcv-isaac "docker stop gr00t-policy-server"`

## Phase 8a: LeIsaac Closed-Loop Evaluation

Test trained N1.6 policies in simulation using [LeIsaac](https://github.com/LightwheelAI/leisaac),
which drives an IsaacSim environment and feeds observations to the policy server in a
closed loop. This requires the policy server to be running (Phase 8) and a DCV desktop
session for the IsaacSim GUI.

### 8a.1 Set Up LeIsaac on the DCV Instance

LeIsaac is **not** installed by default — set it up when you need to run evaluations.
There are two parts: the Python package (policy client library) and the evaluation
scripts (from the git repo).

**SSH into the DCV instance:**
```bash
ssh dcv-isaac
```

**Step 1 — Clone the leisaac repo** (provides `scripts/evaluation/policy_inference.py`):
```bash
LEISAAC_COMMIT=d2cbfd2e33517f2094e1904ff817aa17de6e8939
git clone https://github.com/LightwheelAI/leisaac.git ~/leisaac-repo
cd ~/leisaac-repo && git checkout $LEISAAC_COMMIT
```

> The commit SHA must match the version pinned in `dcv/versions.py` for your IsaacSim
> version. The `Gr00t16ServicePolicyClient` (N1.6) was added after the `v0.3.0` tag and
> only exists on `main`.

**Step 2 — Install the leisaac Python package** to a persistent directory that gets
mounted into the IsaacLab container:
```bash
mkdir -p ~/isaaclab-pkgs
docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -v ~/isaaclab-pkgs:/workspace/isaaclab-pkgs:rw \
  nvcr.io/nvidia/isaac-lab:2.3.0 \
  -c "/workspace/isaaclab/_isaac_sim/python.sh -m pip install \
    --target /workspace/isaaclab-pkgs \
    'leisaac[gr00t] @ git+https://github.com/LightwheelAI/leisaac.git@${LEISAAC_COMMIT}#subdirectory=source/leisaac'"
```

> **Why `python.sh`?** The IsaacLab container's Python is at
> `/workspace/isaaclab/_isaac_sim/python.sh` (an Isaac Sim wrapper). It is not on `$PATH`
> as `python`. All pip/python commands inside this container must use this wrapper.

**Step 3 — Update `run-isaaclab.sh`** to mount the evaluation scripts and fix
an unbound variable:
```bash
# Mount leisaac scripts into the container
sudo sed -i '/--network=host/a\  -v $HOME/leisaac-repo/scripts:/workspace/scripts:ro \\' \
  /usr/local/bin/run-isaaclab.sh

# Fix PYTHONPATH unbound variable (set -u trips on empty PYTHONPATH)
sudo sed -i 's|-e PYTHONPATH=/workspace/isaaclab-pkgs:$PYTHONPATH|-e PYTHONPATH=/workspace/isaaclab-pkgs:${PYTHONPATH:-}|' \
  /usr/local/bin/run-isaaclab.sh
```

**Verify the setup:**
```bash
grep 'leisaac-repo/scripts' /usr/local/bin/run-isaaclab.sh
# Expected: -v $HOME/leisaac-repo/scripts:/workspace/scripts:ro \

grep 'PYTHONPATH:-' /usr/local/bin/run-isaaclab.sh
# Expected: -e PYTHONPATH=/workspace/isaaclab-pkgs:${PYTHONPATH:-}

grep "class Gr00t16ServicePolicyClient" ~/isaaclab-pkgs/leisaac/policy/service_policy_clients.py
# Expected: class Gr00t16ServicePolicyClient(ZMQServicePolicy):
```

### 8a.2 Verify the Policy Server (Direct Test)

Before launching the full sim, verify the policy server responds to inference requests.
This uses GR00T's own `PolicyClient` from the training container (not the leisaac client,
which requires a running IsaacSim process to import — see note below).

```bash
ECR_URI=<account-id>.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest

ssh dcv-isaac "docker run --rm --network=host \
  --entrypoint /bin/sh \
  $ECR_URI \
  -c 'cd /workspace/gr00t-repo && /workspace/gr00t-repo/.venv/bin/python -c \"
import sys, numpy as np
sys.path.insert(0, \\\"/workspace/gr00t-repo\\\")
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host=\\\"localhost\\\", port=5555)
obs = {
    \\\"video\\\": {
        \\\"front\\\": np.random.randint(0, 255, (1, 1, 224, 224, 3), dtype=np.uint8),
        \\\"wrist\\\": np.random.randint(0, 255, (1, 1, 224, 224, 3), dtype=np.uint8),
    },
    \\\"state\\\": {
        \\\"single_arm\\\": np.zeros((1, 1, 5), dtype=np.float32),
        \\\"gripper\\\": np.zeros((1, 1, 1), dtype=np.float32),
    },
    \\\"language\\\": {
        \\\"annotation.human.action.task_description\\\": [[\\\"pick up the orange\\\"]],
    },
}
result = client.get_action(obs)
for k, v in result[0].items():
    print(f\\\"{k}: shape={v.shape}, dtype={v.dtype}\\\")
print(\\\"Policy server test PASSED\\\")
\"'"
```

**Expected:**
```
single_arm: shape=(1, 16, 5), dtype=float32
gripper: shape=(1, 16, 1), dtype=float32
Policy server test PASSED
```

> **Why not use the leisaac `Gr00t16ServicePolicyClient` for this test?**
> `leisaac.__init__` eagerly imports `leisaac.tasks` → `isaaclab_tasks` → `isaaclab` →
> `omni.physics.tensors`, which is a Kit runtime extension only available inside a
> running IsaacSim process. Even `from leisaac.policy.service_policy_clients import ...`
> triggers this chain because `service_policy_clients.py` imports `leisaac.utils.constant`.
> The leisaac client works correctly inside `policy_inference.py` (which runs under
> IsaacSim), but cannot be imported in a standalone Python script.

### 8a.3 Run Closed-Loop Simulation Evaluation

This requires the DCV desktop — open a terminal from the DCV web console
(`https://<elastic-ip>:8443`) or via `ssh dcv-isaac`.

**Launch the IsaacLab container:**
```bash
run-isaaclab.sh
```

**Inside the container, run the evaluation:**
```bash
# Verify DISPLAY is set (DCV provides the X11 session)
echo $DISPLAY
# Expected: :0 or :1

# Verify scripts are mounted
ls /workspace/scripts/evaluation/policy_inference.py
# Expected: file exists

# Run closed-loop evaluation with the N1.6 policy
/workspace/isaaclab/_isaac_sim/python.sh /workspace/scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 \
    --policy_type=gr00tn1.6 \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --device=cuda \
    --enable_cameras
```

**Expected output:** Per-episode success/failure and a final success rate:
```
[Evaluation] Evaluating episode 1...
[Evaluation] Episode 1 is successful!
...
[Evaluation] Final success rate: 0.700 [7/10]
```

**Key parameters:**
- `--task`: LeIsaac task name (see `leisaac` docs for available tasks)
- `--eval_rounds`: Number of episodes (0 = run indefinitely, press R to reset)
- `--policy_type`: `gr00tn1.6` for N1.6, `gr00tn1.5` for N1.5
- `--policy_action_horizon`: Number of action steps per inference (16 for GR00T)
- `--enable_cameras`: Required for vision-based policies

### 8a.4 Observation and Response Format Reference

The policy server (`run_gr00t_server.py`) expects nested-dict observations. The modality
keys come from the checkpoint's `processor_config.json` — for `new_embodiment`:

| Modality | Keys | Shape | Dtype |
|---|---|---|---|
| `video` | `front`, `wrist` | `(B, T, H, W, C)` | `uint8` |
| `state` | `single_arm`, `gripper` | `(B, T, D)` | `float32` |
| `language` | `annotation.human.action.task_description` | `list[list[str]]` | — |

**Response** (tuple of `action_dict, info_dict`):
- `action_dict["single_arm"]`: shape `(1, 16, 5)` float32
- `action_dict["gripper"]`: shape `(1, 16, 1)` float32

The leisaac `Gr00t16ServicePolicyClient` abstracts this format — it accepts a simpler
observation dict with `front`, `wrist` (camera images), `joint_pos` (6D state), and
`task_description` (string), and handles the conversion internally. The `policy_inference.py`
script uses this client under the hood.

> **N1.6 vs N1.5 key difference**: N1.6 uses `annotation.human.action.task_description`
> (N1.5 used `annotation.human.task_description` — note the extra `.action.` segment).

> **LeIsaac API reference** (`leisaac.policy.service_policy_clients`):
> - `Gr00t16ServicePolicyClient` — N1.6 policy client (ZMQ, requires `main` branch commit)
> - `Gr00tServicePolicyClient` — N1.5 policy client (ZMQ, in v0.3.0 tag)
> - `LeRobotServicePolicyClient` — LeRobot policy client (gRPC)
> - `OpenPIServicePolicyClient` — OpenPI policy client (WebSocket)

## Cleanup: Tearing Down Stacks

### Step 1: Destroy stacks (DCV first, then Batch)

The DCV instance has termination protection — disable it first, then destroy both stacks:

```bash
INSTANCE_ID=<dcv-instance-id>
aws ec2 modify-instance-attribute --instance-id $INSTANCE_ID --no-disable-api-termination

cd training/gr00t/infra
npx cdk destroy IsaacLabDcvStack --force
npx cdk destroy IsaacGr00tBatchStack --force
```

If DCV destroy fails with `DELETE_FAILED` (forgot to disable termination protection):
```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws cloudformation delete-stack --stack-name IsaacLabDcvStack
aws cloudformation wait stack-delete-complete --stack-name IsaacLabDcvStack
# Then retry: npx cdk destroy IsaacGr00tBatchStack --force
```

### Step 2: Clean up retained resources

Three resources have `RemovalPolicy.RETAIN` and survive stack deletion. Check the CDK
destroy output for `DELETE_SKIPPED` entries — these need manual deletion:

```bash
# S3 checkpoint bucket (versioned — force-delete empties and removes)
BUCKET=<bucket-name-from-stack-outputs>
aws s3 rb s3://$BUCKET --force
# If that fails with "BucketNotEmpty" (versioned objects), delete versions first:
# aws s3api delete-objects --bucket $BUCKET --delete "$(aws s3api list-object-versions --bucket $BUCKET --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' --output json)"
# aws s3api delete-objects --bucket $BUCKET --delete "$(aws s3api list-object-versions --bucket $BUCKET --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' --output json)"
# aws s3 rb s3://$BUCKET

# ECR repository
aws ecr delete-repository --repository-name gr00t-finetune --force --region us-west-2

# EFS file system (mount targets are deleted with the stack, but verify)
EFS_ID=<efs-id-from-stack-outputs>
aws efs delete-file-system --file-system-id $EFS_ID
# If that fails with "FileSystemInUse", delete lingering mount targets first:
# aws efs describe-mount-targets --file-system-id $EFS_ID --query 'MountTargets[].MountTargetId' --output text | xargs -n1 aws efs delete-mount-target --mount-target-id
# sleep 30 && aws efs delete-file-system --file-system-id $EFS_ID
```

## Troubleshooting

**Bootstrap failed partway:** The script is idempotent — re-running skips completed steps.
SSH in and check `sudo cat /var/lib/cloud/instance/scripts/part-001` for the full script.
Re-run with `sudo bash /var/lib/cloud/instance/scripts/part-001`.

**SSM "TargetNotConnected":** The instance is booting. Wait 60-90 seconds and retry.

**CDK destroy "Cannot delete export":** Happens when destroying Batch before DCV.
Always destroy DCV first.

**`--shm-size=8g` required:** Both open-loop eval and policy server need this flag or
DataLoader workers crash with a bus error.

**NGC base image bash broken:** The `nvcr.io/nvidia/pytorch:25.04-py3` base image ships a
`/usr/bin/bash` that fails with "cannot execute binary file". Use `--entrypoint /bin/sh`
when running `docker run` commands against the `gr00t-finetune:latest` image.

**Uppercase embodiment tag in server CLI:** The `run_gr00t_server.py` CLI (via `tyro`)
expects the `EmbodimentTag` enum **name** in uppercase (`NEW_EMBODIMENT`), not the enum
**value** in lowercase (`new_embodiment`). The training script uses the value form.

**nvidia-smi fails after first deploy:** The NVIDIA kernel module can't load via `modprobe`
on the same boot that installs the driver. Run `sudo reboot` and reconnect after ~60
seconds. This only affects the first deploy.

**Helper script not found:** If `run-isaaclab.sh` doesn't exist, check the bootstrap
summary for `create-helper-script` status. If missing, the container setup step failed —
check detailed logs: `grep -A 50 "pull-isaaclab-container" /var/log/dcv-bootstrap.log`.

**LeIsaac import fails inside container:** Verify the persistent package dir is mounted:
`ls /workspace/isaaclab-pkgs/.leisaac-installed`. If missing, the auto-install didn't run —
exit the container and run `run-isaaclab.sh` again (it retries on each launch).
