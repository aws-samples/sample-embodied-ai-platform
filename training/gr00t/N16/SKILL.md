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

This skill walks through deploying the two CDK stacks that make up the platform's AWS
infrastructure, setting up SSH access to the GPU workstation, submitting training jobs,
visualizing metrics, and running model evaluations.

The two stacks are:
- **IsaacGr00tBatchStack** — VPC, EFS, ECR, CodeBuild, AWS Batch compute environment and job queue
- **IsaacLabDcvStack** — GPU-accelerated DCV workstation (EC2 instance with Elastic IP)

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

## Phase 3: Deploy Stacks

No context parameters needed — the stacks auto-create VPC, EFS, and ECR.

```bash
cd training/gr00t/infra

# Step 1: Batch stack (VPC, EFS, ECR, Batch — ~3 min)
npx cdk deploy IsaacGr00tBatchStack --require-approval=never

# Step 2: DCV stack (GPU instance — ~3 min for CFN, then ~15 min bootstrap)
npx cdk deploy IsaacLabDcvStack --require-approval=never
```

> **GPU capacity:** `g6.2xlarge` and `g6.4xlarge` can be exhausted in popular AZs.
> If you hit `InsufficientInstanceCapacity`, edit `training/gr00t/infra/app.py`:
> 1. Try a different `availability_zone` (`us-west-2b`, `us-west-2c`, etc.)
> 2. Fall back to `g5.2xlarge` or `g6e.xlarge` (At least 32GiB memory is required
>    to run GR00T server and simulation at the same time)
> 3. Redeploy with `npx cdk deploy IsaacLabDcvStack` (which also updates BatchStack
>    cross-stack exports for the new AZ's subnet)

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
before marking CREATE_COMPLETE. Monitor via SSM (no SSH needed yet):

```bash
# Check progress (repeat every 2-3 minutes)
CMD_ID=$(aws ssm send-command \
  --instance-ids $InstanceId \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["cat /var/log/dcv-bootstrap.summary 2>/dev/null || echo BOOTSTRAP_NOT_STARTED"]' \
  --output text --query 'Command.CommandId') \
&& sleep 5 \
&& aws ssm get-command-invocation \
  --command-id $CMD_ID \
  --instance-id $InstanceId \
  --query 'StandardOutputContent' --output text
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

## Phase 5e: Reboot for NVIDIA Driver

The NVIDIA kernel module can't load on the same boot that installs the driver package.
Reboot the instance once before proceeding — GPU containers won't work without this.

```bash
ssh dcv-isaac "sudo reboot"
# Wait ~60 seconds, then verify:
ssh dcv-isaac "nvidia-smi --query-gpu=name --format=csv,noheader"
# Expected: NVIDIA L4 (or similar, depending on instance type)
```

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

# 5. NVIDIA GPU detected (should work after Phase 5e reboot)
ssh dcv-isaac "nvidia-smi --query-gpu=name --format=csv,noheader"
```

The DCV web console is also available at `https://<elastic-ip>:8443` (accept the
self-signed certificate warning).

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

# LeIsaac N1.6 client is present
grep "class Gr00t16ServicePolicyClient" /workspace/isaaclab-pkgs/leisaac/policy/service_policy_clients.py

# GPU accessible
nvidia-smi
```

Exit the container with `exit` or Ctrl-D.

## Phase 7: Submit Training Job

### 7a. Verify container image is ready

The Batch stack triggers a CodeBuild project that builds and pushes the training container
to ECR. This takes ~15 minutes for N1.6 (PyTorch3D compilation). **Note:** Redeploying
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

# Create account at http://<elastic-ip>:8080, generate API key from Settings -> API Keys

# Sync offline runs
ssh dcv-isaac "bash -l -c '
  export WANDB_BASE_URL=http://localhost:8080
  export WANDB_API_KEY=<your-local-api-key>
  source /home/ubuntu/.venv/bin/activate
  wandb sync /mnt/efs/gr00t/checkpoints/$JOB_ID/wandb/offline-run-*
'"
```

View loss curves at `http://<elastic-ip>:8080`. Stop/restart the server anytime —
data persists in the `wandb-data` Docker volume.

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
  $EcrImageUri \
  python -m gr00t.eval.robot_eval \
    --model-path $CHECKPOINT \
    --embodiment-tag new_embodiment \
    --dataset-path /path/to/eval-dataset \
    --modality-config-path /workspace/scripts/so101_modality_config.py"
```

### Closed-loop evaluation (policy server)

Serves a trained checkpoint as a ZMQ policy server for real-time robot control or
simulation testing on TCP port 5555.

```bash
CHECKPOINT=/mnt/efs/gr00t/checkpoints/$JOB_ID/checkpoint-6000

ssh dcv-isaac "docker run --gpus all -d \
  --name gr00t-policy-server \
  --shm-size=8g \
  --network host \
  --entrypoint /bin/sh \
  -v $CHECKPOINT:/workspace/checkpoint \
  $EcrImageUri \
  -c '/workspace/gr00t-repo/.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model_path /workspace/checkpoint \
    --embodiment_tag NEW_EMBODIMENT \
    --host 0.0.0.0'"
```

> Use `--entrypoint /bin/sh` (NGC `/usr/bin/bash` is broken). The server CLI expects
> uppercase `NEW_EMBODIMENT` (tyro parses enum names). Pass `--host 0.0.0.0` to allow
> remote clients.

Verify: `ssh dcv-isaac "docker logs gr00t-policy-server 2>&1 | tail -5"`

For a direct inference test (without IsaacSim), see [references/policy-server-test.md](references/policy-server-test.md).

For observation/response format details, see [references/eval-format.md](references/eval-format.md).

## Phase 8a: LeIsaac Closed-Loop Evaluation

Test trained N1.6 policies in simulation using [LeIsaac](https://github.com/LightwheelAI/leisaac).
Requires the policy server running (Phase 8) and a DCV desktop session for the IsaacSim GUI.

`run-isaaclab.sh` handles all prerequisites automatically on first launch: leisaac package
install, scene asset download, language key patch, repo clone, and script mount. No manual
setup is needed.

**Launch the container from a DCV terminal** (`https://<elastic-ip>:8443`):
```bash
run-isaaclab.sh
```

**Inside the container, run the evaluation:**
```bash
/workspace/isaaclab/_isaac_sim/python.sh /workspace/scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 \
    --policy_type=gr00tn1.6 \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --device=cuda \
    --enable_cameras
```

**Expected:** Per-episode success/failure and a final success rate (e.g. `Final success rate: 0.700 [7/10]`).

For eval parameters and format details, see [references/eval-format.md](references/eval-format.md).

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

See [references/troubleshooting.md](references/troubleshooting.md) for common issues and fixes.
