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
ls -la /path/to/repo/.venv/bin/python

# If venv is missing, create it:
cd /path/to/repo && python3 -m venv .venv
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

# Step 2: DCV stack (GPU instance — ~3 min)
npx cdk deploy IsaacLabDcvStack --require-approval=never
```

After each deploy, capture the stack outputs — you'll need them for SSH setup and later
operations. Key outputs include instance ID, Elastic IP, EFS ID, ECR URI, checkpoint S3
path, and DCV credentials.

## Phase 4: Monitor Bootstrap Completion

The DCV instance runs a ~15-minute bootstrap script that installs NVIDIA drivers, DCV,
conda, IsaacSim, IsaacLab, and mounts EFS. Monitor it via SSM (no SSH needed yet).

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

Steps appear one by one as `STEP_OK` entries. The full sequence (19 steps):
`baseline-update` → `set-ubuntu-password` → `disable-nouveau` → `install-nvidia-driver` →
`install-desktop` → `disable-gnome-initial-setup` → `install-nice-dcv` → `configure-dcv` →
`install-auto-dcv-service` → `install-aws-cli-v2` → `install-efs-utils` →
`install-docker-nvidia-toolkit` → `install-miniforge` → `create-conda-env-isaac` →
`install-pytorch-isaacsim` (slowest, ~5 min) → `install-isaaclab` → `install-leisaac` →
`install-wandb` → `install-firefox`

After the last step, the instance **reboots automatically**. Wait ~60 seconds before
attempting SSH. If any step shows `STEP_FAIL`, check the detailed log:

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
# 1. Bootstrap summary — all steps should be STEP_OK
ssh dcv-isaac "cat /var/log/dcv-bootstrap.summary"

# 2. EFS mounted
ssh dcv-isaac "mount | grep efs"
# Expected: 127.0.0.1:/ on /mnt/efs type nfs4 ...

# 3. Conda env exists with correct Python
ssh dcv-isaac "conda run -n isaac python --version"
# Expected: Python 3.11.x

# 4. NVIDIA GPU detected
ssh dcv-isaac "nvidia-smi --query-gpu=name --format=csv,noheader"
# Expected: NVIDIA L4 (or similar)
```

The DCV web console is also available at `https://<elastic-ip>:8443` (accept the
self-signed certificate warning).

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
  conda activate isaac
  export WANDB_BASE_URL=http://localhost:8080
  export WANDB_API_KEY=<your-local-api-key>
  wandb sync /mnt/efs/gr00t/checkpoints/<JOB_ID>/wandb/offline-run-*
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

```bash
CHECKPOINT=/mnt/efs/gr00t/checkpoints/<JOB_ID>/checkpoint-6000

ssh dcv-isaac "docker run --gpus all --rm -d \
  -v $CHECKPOINT:$CHECKPOINT:ro \
  -p 5555:5555 \
  --shm-size=8g \
  --name gr00t-policy-server \
  --entrypoint /workspace/gr00t-repo/.venv/bin/python \
  $ECR_URI \
  gr00t/eval/run_gr00t_server.py \
    --embodiment-tag new_embodiment \
    --model-path $CHECKPOINT \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 5555"
```

To connect a client (robot or sim), send observations as msgpack-serialized numpy arrays
over ZMQ REQ/REP on `tcp://<elastic-ip>:5555`. The server returns 16 action steps;
clients typically use an action horizon of 8 for responsiveness.

Ensure the DCV security group allows inbound TCP 5555 from the client IP:
```bash
SG_ID=<dcv-security-group-id>
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 5555 \
  --cidr <client-ip>/32
```

Stop the server: `ssh dcv-isaac "docker stop gr00t-policy-server"`

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
SSH in and run `sudo bash /tmp/dcv_bootstrap.sh`. If `/tmp` was cleared after reboot,
check `sudo cat /var/lib/cloud/instance/scripts/part-001` for the S3 asset URL to
re-download the script.

**SSM "TargetNotConnected":** The instance is booting or rebooting. Wait 60-90 seconds.

**CDK destroy "Cannot delete export":** Happens when destroying Batch before DCV.
Always destroy DCV first.

**`--shm-size=8g` required:** Both open-loop eval and policy server need this flag or
DataLoader workers crash with a bus error.
