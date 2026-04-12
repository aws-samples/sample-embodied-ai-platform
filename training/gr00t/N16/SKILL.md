---
name: deploy-stack
description: >
  Guide for deploying the Embodied AI Platform CDK stacks (IsaacGr00tBatchStack + IsaacLabDcvStack)
  from scratch, including prerequisites, deployment, bootstrap monitoring, SSH setup via SSM,
  verification, training job submission, W&B visualization, and model evaluation. Also covers
  tearing down stacks and cleaning up retained resources. Use this skill whenever someone wants
  to deploy, redeploy, tear down the CDK infrastructure, visualize training metrics, submit
  training jobs, or run model evaluations — even if they just say "set up the stack", "deploy
  to AWS", "get DCV running", "view training loss", "run evals", or "clean up AWS resources".
---

# Deploy Embodied AI Platform CDK Stacks

> **Autonomous execution:** Execute all phases sequentially from Phase 1 through Phase 8a
> without pausing for user input. Only stop if a phase fails or requires information not
> available in this document. When a phase depends on a long-running process (e.g. CodeBuild,
> training job), poll until completion then proceed to the next phase automatically.

This skill walks through deploying the two CDK stacks that make up the platform's AWS
infrastructure, setting up SSH access to the GPU workstation, submitting training jobs,
visualizing metrics, and running model evaluations.

The two stacks are:
- **IsaacGr00tBatchStack** — VPC, EFS, ECR, CodeBuild, AWS Batch compute environment and job queue
- **IsaacLabDcvStack** — GPU-accelerated DCV workstation (EC2 instance, accessed via SSM)

The DCV stack depends on the Batch stack (shares its VPC and EFS), so deploy order matters:
Batch first, DCV second. Destroy order is reversed: DCV first, Batch second.

## Phase 1: Prerequisites

Before deploying, verify these are in place.

> **Region selection:** Deploy in a region with g6e GPU instances for training
> (e.g. `us-west-2`, `us-east-1`). The DCV instance uses g6 family by default.
> Set your region: `export AWS_DEFAULT_REGION=us-west-2` or use `--profile`
> with a configured AWS CLI profile.

```bash
# AWS CLI configured with correct account
aws sts get-caller-identity --query '[Account, Arn]' --output text

# CDK available (via npx, no global install needed)
npx cdk --version

# jq for parsing stack outputs
jq --version

# Python venv — CDK uses Python to synthesize CloudFormation templates.
# The venv isolates CDK dependencies from your system Python.
REPO_ROOT=$(git rev-parse --show-toplevel)
ls -la "$REPO_ROOT/.venv/bin/python"

# If venv is missing, create it:
cd "$REPO_ROOT" && python3 -m venv .venv
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

The default `app.py` deploys with IsaacSim 5.1.0 / IsaacLab v2.3.0.

Always synth before deploying — it catches version mismatches, missing dependencies, and
code errors at zero cost (no AWS resources created).

```bash
cd training/gr00t/infra
npx cdk synth --quiet
```

**Expected output:** `Successfully synthesized to .../cdk.out` listing both stack names.

If synth fails, fix the error before proceeding. Common issues:
- Missing Python dependencies -> `pip install -r requirements.txt`
- Version validation errors -> check `dcv/versions.py` for supported IsaacSim versions

## Phase 2.5: Probe GPU Instance Capacity

`--dry-run` does **not** test actual instance capacity — it only validates IAM permissions.
The only reliable check is a real launch + immediate terminate. Run this after
`IsaacGr00tBatchStack` deploys (so its VPC subnets exist), before deploying `IsaacLabDcvStack`.

```bash
# Requires $VpcId from the Batch stack outputs (captured below after Phase 3 Step 1)

UBUNTU_AMI=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)

SUBNETS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VpcId" "Name=mapPublicIpOnLaunch,Values=true" \
  --query 'Subnets[].{AZ:AvailabilityZone,SubnetId:SubnetId}' --output json)

# Probe in order: preferred first, then fallbacks (all ≥32GiB — minimum for GR00T + sim)
FOUND=false
for ITYPE in g6.4xlarge g6.2xlarge g5.2xlarge; do
  while IFS= read -r row; do
    AZ=$(echo "$row" | jq -r '.AZ')
    SUBNET=$(echo "$row" | jq -r '.SubnetId')
    RESULT=$(aws ec2 run-instances \
      --image-id "$UBUNTU_AMI" --instance-type "$ITYPE" \
      --subnet-id "$SUBNET" --count 1 \
      --no-associate-public-ip-address \
      --query 'Instances[0].InstanceId' --output text 2>&1)
    if [[ "$RESULT" == i-* ]]; then
      aws ec2 terminate-instances --instance-ids "$RESULT" > /dev/null
      echo "✅  $ITYPE in $AZ — available. Set in app.py:"
      echo "    availability_zone=\"$AZ\","
      echo "    instance_type=\"$ITYPE\","
      FOUND=true; break 2
    else
      echo "❌  $ITYPE / $AZ — $(echo "$RESULT" | grep -oE 'InsufficientInstanceCapacity|[A-Za-z]+Error')"
    fi
  done < <(echo "$SUBNETS" | jq -c '.[]')
done
$FOUND || echo "⚠️  No capacity found — try a different region or instance family"
```

Update `training/gr00t/infra/app.py` with the printed `availability_zone` and `instance_type`
values before running Step 2 below.

## Phase 3: Deploy Stacks

No context parameters needed — the stacks auto-create VPC, EFS, and ECR.

> **N1.6 build target:** The BatchStack deploy automatically triggers a CodeBuild run.
> Pass `--context build_target=n16` so it builds `N16/Dockerfile` (not the N1.5 default).

```bash
cd training/gr00t/infra

# Step 1: Batch stack (VPC, EFS, ECR, Batch — ~3 min)
# build_target=n16 selects N16/Dockerfile for the CodeBuild run triggered on deploy.
npx cdk deploy IsaacGr00tBatchStack --require-approval=never --context build_target=n16

# Capture VpcId output, then run Phase 2.5 capacity probe, then update app.py.

# Step 2: DCV stack (GPU instance — ~3 min for CFN, then ~15 min bootstrap)
# This blocks until cfn-signal is received from the bootstrap script.
npx cdk deploy IsaacLabDcvStack --require-approval=never
```

> **Parallel monitoring during Step 2:** Open a second terminal and poll CloudFormation
> events directly — avoids stdout buffering delays from CDK:
>
> ```bash
> while true; do
>   STATUS=$(aws cloudformation describe-stacks --stack-name IsaacLabDcvStack \
>     --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "CREATING")
>   printf '\n=== %s [%s] ===\n' "$(date +%H:%M:%S)" "$STATUS"
>   aws cloudformation describe-stack-events --stack-name IsaacLabDcvStack \
>     --query 'StackEvents[0:8].[Timestamp,ResourceStatus,LogicalResourceId]' \
>     --output table 2>/dev/null
>   [[ "$STATUS" =~ (COMPLETE|FAILED) ]] && break
>   sleep 20
> done
> ```

After both deploys complete, capture the stack outputs as shell variables for later phases.
The jq approach handles values with spaces safely (unlike `eval`-based approaches):

```bash
# Capture Batch stack outputs
BATCH_OUTPUTS=$(aws cloudformation describe-stacks --stack-name IsaacGr00tBatchStack \
  --query 'Stacks[0].Outputs' --output json)
export EcrImageUri=$(echo "$BATCH_OUTPUTS" | jq -r '.[] | select(.OutputKey=="EcrImageUri") | .OutputValue')
export EFSFileSystemId=$(echo "$BATCH_OUTPUTS" | jq -r '.[] | select(.OutputKey=="EFSFileSystemId") | .OutputValue')
export EFSSecurityGroupId=$(echo "$BATCH_OUTPUTS" | jq -r '.[] | select(.OutputKey=="EFSSecurityGroupId") | .OutputValue')
export VpcId=$(echo "$BATCH_OUTPUTS" | jq -r '.[] | select(.OutputKey=="VpcId") | .OutputValue')
export CodeBuildProjectName=$(echo "$BATCH_OUTPUTS" | jq -r '.[] | select(.OutputKey=="CodeBuildProjectName") | .OutputValue // empty')
export CheckpointS3UploadUri=$(echo "$BATCH_OUTPUTS" | jq -r '.[] | select(.OutputKey=="CheckpointS3UploadUri") | .OutputValue // empty')

# Capture DCV stack outputs
DCV_OUTPUTS=$(aws cloudformation describe-stacks --stack-name IsaacLabDcvStack \
  --query 'Stacks[0].Outputs' --output json)
export InstanceId=$(echo "$DCV_OUTPUTS" | jq -r '.[] | select(.OutputKey=="InstanceId") | .OutputValue')
export InstancePublicIP=$(echo "$DCV_OUTPUTS" | jq -r '.[] | select(.OutputKey=="InstancePublicIP") | .OutputValue')
export DCVWebURL=$(echo "$DCV_OUTPUTS" | jq -r '.[] | select(.OutputKey=="DCVWebURL") | .OutputValue')
export DCVCredentials=$(echo "$DCV_OUTPUTS" | jq -r '.[] | select(.OutputKey=="DCVCredentials") | .OutputValue')

# Verify key values are set
echo "Instance ID: $InstanceId"
echo "Elastic IP:  $InstancePublicIP"
echo "ECR URI:     $EcrImageUri"
echo "EFS ID:      $EFSFileSystemId"
echo "DCV URL:     $DCVWebURL"
echo "DCV Creds:   $DCVCredentials"
```

> The variable names match the CDK `CfnOutput` keys exactly (PascalCase). These are used
> in later phases as `$InstanceId`, `$InstancePublicIP`, `$EcrImageUri`, etc.

## Phase 4: Monitor Bootstrap Completion

The DCV instance runs a bootstrap that installs NVIDIA drivers, Docker, pulls the IsaacLab
container from NGC, installs DCV, and mounts EFS. CloudFormation waits for a cfn-signal
before marking CREATE_COMPLETE — so the Phase 3 CDK deploy does not return until bootstrap
finishes. Phase 4 monitoring is for a **parallel terminal** to track mid-progress.

```bash
# Auto-poll bootstrap summary every 60 seconds (run in a second terminal during Phase 3)
while true; do
  CMD_ID=$(aws ssm send-command \
    --instance-ids $InstanceId \
    --document-name AWS-RunShellScript \
    --parameters 'commands=["cat /var/log/dcv-bootstrap.summary 2>/dev/null || echo BOOTSTRAP_NOT_STARTED"]' \
    --output text --query 'Command.CommandId')
  sleep 5
  OUTPUT=$(aws ssm get-command-invocation \
    --command-id $CMD_ID --instance-id $InstanceId \
    --query 'StandardOutputContent' --output text)
  printf '\n=== %s ===\n%s\n' "$(date +%H:%M:%S)" "$OUTPUT"
  echo "$OUTPUT" | grep -q "STEP_FAIL" && echo "Bootstrap FAILED" && break
  sleep 55
done
```

Steps appear as `STEP_OK` entries. Total time: ~15-20 minutes. You may see
`STEP_WARN:nvidia-driver-loaded` — this is expected on first boot (the NVIDIA kernel
module loads after a reboot).

If any step shows `STEP_FAIL`, check the detailed log via SSM:
`sudo grep -A 50 "== START: <step-name> ==" /var/log/dcv-bootstrap.log`

## Phase 5: Set Up SSH via SSM

SSH through SSM tunnels through AWS Session Manager — it doesn't require port 22 to be
open in the security group.

### 5a. Verify Session Manager plugin is installed locally

```bash
session-manager-plugin --version
```

If missing: Ubuntu/Debian `sudo dpkg -i session-manager-plugin.deb`, macOS `brew install --cask session-manager-plugin`.

### 5b. Push SSH public key to the instance

```bash
# Generate SSH key if you don't have one
test -f ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

PUBKEY=$(cat ~/.ssh/id_ed25519.pub)
aws ssm send-command \
  --instance-ids $InstanceId \
  --document-name AWS-RunShellScript \
  --parameters "commands=[\"mkdir -p /home/ubuntu/.ssh && echo '$PUBKEY' > /home/ubuntu/.ssh/authorized_keys && chmod 700 /home/ubuntu/.ssh && chmod 600 /home/ubuntu/.ssh/authorized_keys && chown -R ubuntu:ubuntu /home/ubuntu/.ssh\"]" \
  --output text --query 'Command.CommandId'
```

### 5c. Configure SSH config

Add or update this entry in `~/.ssh/config`:

```
Host dcv-isaac
  HostName <$InstanceId>
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
  ProxyCommand aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p' --region <$AWS_DEFAULT_REGION>
```

> Replace `<$InstanceId>` and `<$AWS_DEFAULT_REGION>` with the values from Phase 3.

### 5d. Test the connection

```bash
ssh -o StrictHostKeyChecking=accept-new dcv-isaac "echo 'SSH OK'"
```

## Phase 5e: Wait for Auto-Reboot

The bootstrap automatically reboots the instance after sending cfn-signal. This loads the
NVIDIA kernel module (which can't load on the same boot that installs the driver). Wait
~2 minutes after `cdk deploy` returns, then verify:

```bash
ssh dcv-isaac "nvidia-smi --query-gpu=name --format=csv,noheader"
# Expected: NVIDIA L4 (or similar, depending on instance type)
```

> If `nvidia-smi` fails, the reboot may still be in progress. Wait another minute and retry.

## Phase 6: Verify Everything Works

```bash
# 1. Bootstrap summary — all named steps should be STEP_OK
ssh dcv-isaac "cat /var/log/dcv-bootstrap.summary"

# 2. EFS mounted
ssh dcv-isaac "mount | grep efs"

# 3. Container image pulled
ssh dcv-isaac "docker images | grep isaac-lab"

# 4. Helper script installed
ssh dcv-isaac "test -x /usr/local/bin/run-isaaclab.sh && echo 'Helper script OK'"

# 5. NVIDIA GPU detected (should work after auto-reboot)
ssh dcv-isaac "nvidia-smi --query-gpu=name --format=csv,noheader"
```

The DCV web console is also available via SSH port forwarding. No public ports are
exposed — all access is through SSM tunneling (configured in Phase 5).

To access DCV or W&B from a browser, start port forwards in the background:
```bash
ssh -f -N -L 8443:localhost:8443 -L 8080:localhost:8080 dcv-isaac
```
Then open `https://localhost:8443` for DCV or `http://localhost:8080` for W&B.
Claude Code does not need these port forwards — it accesses everything via SSH commands.

## Phase 6a: Container and LeIsaac Testing

### 6a.1 Launch IsaacLab Container

The helper script wraps `docker run` with GPU access, X11 forwarding, cache volumes, and
persistent package mounts. On first launch, it auto-installs leisaac and downloads scene
assets (~60 seconds total).

```bash
ssh dcv-isaac
run-isaaclab.sh
```

> The container Python is at `/workspace/isaaclab/_isaac_sim/python.sh` (an Isaac Sim
> wrapper). All Python commands below use this wrapper.

### 6a.2 Verify Inside Container

```bash
# Python wrapper works
/workspace/isaaclab/_isaac_sim/python.sh --version

# GPU accessible
nvidia-smi
```

Exit the container with `exit` or Ctrl-D.

### 6a.3 Verify Policy Client N1.6 Patch

The `run-isaaclab.sh` helper patches the LeIsaac policy client for N1.6 compatibility
on first launch. Confirm the patch applied:

```bash
ssh dcv-isaac "grep -q 'GR00T N1.6 SimPolicyWrapper' /home/ubuntu/isaaclab-pkgs/leisaac/policy/service_policy_clients.py && echo 'Policy client patch OK' || echo 'MISSING — re-run run-isaaclab.sh'"
```

If missing, re-run `run-isaaclab.sh -c 'echo patched'` and check again.

## Phase 7: Submit Training Job

### 7a. Verify container image is ready

The Batch stack triggers a CodeBuild project that builds and pushes the training container
to ECR. If you deployed with `--context build_target=n16` in Phase 3, the N1.6 container
is already building. This takes ~15 minutes for N1.6 (PyTorch3D compilation). **Note:** Redeploying
BatchStack auto-triggers a new CodeBuild run that may fail due to transient Docker Hub
rate limits — check for a recent successful build, not just the latest build status:

```bash
aws codebuild batch-get-projects --names "$CodeBuildProjectName" \
  --query 'projects[0].lastSuccessfulBuild.endTime' --output text

# If no successful build yet, check current build status:
BUILD_ID=$(aws codebuild list-builds-for-project --project-name "$CodeBuildProjectName" \
  --query 'ids[0]' --output text)
aws codebuild batch-get-builds --ids "$BUILD_ID" \
  --query 'builds[0].[currentPhase, buildStatus]' --output text
```

Do not submit a Batch job until the build shows `SUCCEEDED`.

### 7b. Submit the training job

```bash
JOB_ID=$(aws batch submit-job \
  --job-name "IsaacGr00tFinetuning" \
  --job-queue "IsaacGr00tJobQueue" \
  --job-definition "IsaacGr00tJobDefinition" \
  --query 'jobId' --output text)
echo "Job submitted: $JOB_ID"
```

Monitor progress:
```bash
aws batch describe-jobs --jobs $JOB_ID --query 'jobs[0].status' --output text

# Stream logs once RUNNING:
aws logs tail /aws/batch/job --follow \
  --log-stream-names "$(aws batch describe-jobs --jobs $JOB_ID \
  --query 'jobs[0].container.logStreamName' --output text)"
```

> Default: 6000 steps (~2 hours on g6e.4xlarge). Checkpoints saved every 2000 steps
> at `/mnt/efs/gr00t/checkpoints/$JOB_ID/`.

### 7c. Visualize Training Metrics (W&B)

Training logs to W&B in **offline mode** — run data persists on EFS after the container exits.

```bash
# Start local W&B server on DCV instance
ssh dcv-isaac "docker run -d --name wandb-local -p 8080:8080 -v wandb-data:/vol wandb/local:latest"

# W&B is accessible via SSH port forward (no public access needed):
#   ssh -f -N -L 8080:localhost:8080 dcv-isaac
#   Then open http://localhost:8080 in your browser
# Create account and generate API key from Settings -> API Keys

# Sync offline runs
ssh dcv-isaac "bash -l -c '
  export WANDB_BASE_URL=http://localhost:8080
  export WANDB_API_KEY=<your-local-api-key>
  source /home/ubuntu/.venv/bin/activate
  wandb sync /mnt/efs/gr00t/checkpoints/$JOB_ID/wandb/offline-run-*
'"
```

View loss curves at `http://localhost:8080` (after starting the SSH port forward above).
Stop/restart the server anytime — data persists in the `wandb-data` Docker volume.

## Phase 7d: Prepare Dataset on EFS for Evaluation

The training job uses the dataset baked into the container image during CodeBuild, but
open-loop evaluation runs on the DCV instance and needs the dataset on EFS. Copy the
sample dataset from your local machine, then generate the metadata files the eval
scripts require.

```bash
# Copy sample dataset to EFS
REPO_ROOT=$(git rev-parse --show-toplevel)
rsync -avz "$REPO_ROOT/training/sample_dataset/" dcv-isaac:/mnt/efs/gr00t/sample_dataset/

# Create modality.json (maps dataset columns to model modality keys)
ssh dcv-isaac "cat > /mnt/efs/gr00t/sample_dataset/meta/modality.json << 'EOF'
{
  \"action\": {
    \"single_arm\": {\"start\": 0, \"end\": 5},
    \"gripper\": {\"start\": 5, \"end\": 6}
  },
  \"state\": {
    \"single_arm\": {\"start\": 0, \"end\": 5},
    \"gripper\": {\"start\": 5, \"end\": 6}
  },
  \"annotation\": {
    \"human.action.task_description\": {
      \"original_key\": \"task_index\"
    }
  },
  \"video\": {
    \"front\": {
      \"original_key\": \"observation.images.front\",
      \"shape\": [480, 640, 3]
    },
    \"wrist\": {
      \"original_key\": \"observation.images.wrist\",
      \"shape\": [480, 640, 3]
    }
  }
}
EOF"

# Generate stats.json (normalization statistics used by the eval pipeline)
ssh dcv-isaac "docker run --gpus all --rm \
  -v /mnt/efs:/mnt/efs \
  --entrypoint /bin/sh \
  $EcrImageUri \
  -c 'cd /workspace/gr00t-repo && python3 -m gr00t.data.stats \
    --dataset-path /mnt/efs/gr00t/sample_dataset \
    --embodiment-tag NEW_EMBODIMENT'"
```

> `modality.json` maps the dataset's column names to the model's expected modality keys
> (video cameras, state joints, action joints, language annotations). `stats.json`
> contains per-feature normalization statistics. Both are required by `open_loop_eval`.

## Phase 8: Model Evaluation

### Open-loop evaluation

Computes MSE and cosine similarity against a held-out dataset:

```bash
CHECKPOINT=/mnt/efs/gr00t/checkpoints/$JOB_ID/checkpoint-6000

ssh dcv-isaac "aws ecr get-login-password --region $AWS_DEFAULT_REGION | \
  docker login --username AWS --password-stdin ${EcrImageUri%%/*}"

ssh dcv-isaac "docker pull $EcrImageUri"

ssh dcv-isaac "docker run --gpus all --rm \
  -v /mnt/efs:/mnt/efs:ro \
  --shm-size=8g \
  --entrypoint /bin/sh \
  $EcrImageUri \
  -c 'cd /workspace/gr00t-repo && python3 -m gr00t.eval.open_loop_eval \
    --model-path $CHECKPOINT \
    --embodiment-tag NEW_EMBODIMENT \
    --dataset-path /mnt/efs/gr00t/sample_dataset \
    --modality-config-path /workspace/scripts/so101_modality_config.py \
    --modality-keys single_arm gripper \
    --video-backend torchvision_av'"
```

> **Note:** The module is `gr00t.eval.open_loop_eval` (not `robot_eval`). Use
> `--entrypoint /bin/sh` because the container's default entrypoint invokes a uv-managed
> Python that can't be exec'd directly. The `--embodiment-tag` is case-sensitive and must
> be `NEW_EMBODIMENT` (uppercase).

### Closed-loop evaluation (policy server)

Serves a trained checkpoint as a ZMQ policy server for real-time robot control or
simulation testing on TCP port 5555. Do **not** use `--use-sim-policy-wrapper` — the
`Gr00t16ServicePolicyClient` in LeIsaac sends observations in the nested format the
server expects natively, and the wrapper would cause a key mismatch error.

```bash
CHECKPOINT=/mnt/efs/gr00t/checkpoints/$JOB_ID/checkpoint-6000

ssh dcv-isaac "docker run --gpus all -d \
  --name gr00t-policy-server \
  --shm-size=8g \
  --network host \
  --entrypoint /bin/sh \
  -v /mnt/efs:/mnt/efs \
  $EcrImageUri \
  -c 'cd /workspace/gr00t-repo && python3 -m gr00t.eval.run_gr00t_server \
    --model-path $CHECKPOINT \
    --embodiment-tag NEW_EMBODIMENT \
    --port 5555'"
```

> Use `--entrypoint /bin/sh` because the container's default entrypoint invokes a
> uv-managed Python that can't be exec'd directly. The server module is
> `gr00t.eval.run_gr00t_server` which wraps `Gr00tPolicy` in a `PolicyServer` (ZMQ).
> Allow ~60 seconds for model loading before the server begins accepting connections.

Verify the server is listening:
```bash
ssh dcv-isaac "docker logs gr00t-policy-server 2>&1 | tail -5"
ssh dcv-isaac "ss -tlnp | grep 5555"
```

For a direct inference test (without IsaacSim), see [../references/policy-server-test.md](../references/policy-server-test.md).

For observation/response format details, see [../references/eval-format.md](../references/eval-format.md).

## Phase 8a: LeIsaac Closed-Loop Evaluation

Test trained N1.6 policies in simulation using [LeIsaac](https://github.com/LightwheelAI/leisaac).
Requires the policy server running (Phase 8).

> **Important:** Do NOT use `run-isaaclab.sh` for automated evaluation — it launches an
> interactive bash shell and routes arguments through `runheadless.sh` (the Kit launcher),
> which does not execute the Python eval script. Use the direct `docker run` command below
> with `--entrypoint` set to `python.sh`.

`run-isaaclab.sh` handles leisaac package install, scene asset download, and repo clone
on first launch. Run it once interactively to set up prerequisites if they haven't been
installed yet:
```bash
ssh dcv-isaac "run-isaaclab.sh -c 'echo prerequisites installed'"
```

**Run the closed-loop evaluation via SSH** (fully autonomous, no DCV desktop needed):
```bash
ssh dcv-isaac "docker run --gpus all --rm --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace/isaaclab-pkgs \
  -e PYTHONUNBUFFERED=1 \
  -e DISPLAY=:1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v /home/ubuntu/isaaclab-pkgs:/workspace/isaaclab-pkgs:rw \
  -v /home/ubuntu/leisaac-repo/scripts:/workspace/scripts:ro \
  -v /home/ubuntu/leisaac-assets:/assets:ro \
  -v /mnt/efs:/mnt/efs \
  -e LEISAAC_ASSETS_ROOT=/assets \
  --shm-size=8g \
  --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
  nvcr.io/nvidia/isaac-lab:2.3.0 \
  /workspace/scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 \
    --policy_type=gr00tn1.6 \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_action_horizon=16 \
    --policy_language_instruction='Pick up the orange and place it on the plate' \
    --device=cuda \
    --enable_cameras 2>&1 | tee /mnt/efs/gr00t/eval-results.log"
```

> `DISPLAY=:1` and the X11 socket mount are required because `--enable_cameras` uses
> Vulkan rendering for camera frames, which needs an active display. DCV runs on
> display `:1` (not `:0`). Without these flags, the render loop blocks silently with
> 0% GPU utilization despite allocating VRAM.
>
> `PYTHONUNBUFFERED=1` ensures eval output streams in real time over SSH.
> Results are persisted to `/mnt/efs/gr00t/eval-results.log` so the final success rate
> is available even after the container exits (`--rm`).
> First run takes ~5 minutes for IsaacSim shader compilation before episodes begin.

For manual visual inspection, start an SSH port forward to the DCV desktop:
```bash
ssh -f -N -L 8443:localhost:8443 dcv-isaac
```
Then open `https://localhost:8443` in your browser and run `run-isaaclab.sh` from the DCV terminal.

**Expected:** Per-episode success/failure and a final success rate (e.g. `Final success rate: 0.700 [7/10]`).

For eval parameters and format details, see [../references/eval-format.md](../references/eval-format.md).

## Cleanup: Tearing Down Stacks

### Step 1: Destroy stacks (DCV first, then Batch)

```bash
aws ec2 modify-instance-attribute --instance-id $InstanceId --no-disable-api-termination

cd training/gr00t/infra
npx cdk destroy IsaacLabDcvStack --force
npx cdk destroy IsaacGr00tBatchStack --force
```

### Step 2: Clean up retained resources

Three resources have `RemovalPolicy.RETAIN` and survive stack deletion:

```bash
# S3 checkpoint bucket
BUCKET=<bucket-name-from-stack-outputs>
aws s3 rb s3://$BUCKET --force

# ECR repository
aws ecr delete-repository --repository-name gr00t-finetune --force --region $AWS_DEFAULT_REGION

# EFS file system
aws efs delete-file-system --file-system-id $EFSFileSystemId
```

> If EFS deletion fails with "FileSystemInUse", delete lingering mount targets first:
> `aws efs describe-mount-targets --file-system-id $EFSFileSystemId --query 'MountTargets[].MountTargetId' --output text | xargs -n1 aws efs delete-mount-target --mount-target-id`

See [../references/troubleshooting.md](../references/troubleshooting.md) for common issues and fixes.
