# Blog Update Guide — Embodied AI Blog Series Part 1

This document tracks the changes required to update the published blog post
([Embodied AI Blog Series Part 1](https://aws.amazon.com/blogs/spatial/embodied-ai-blog-series-part-1/))
to reflect the containerized DCV workstation setup.

**Scope:** Minimal changes — keep the same structure, section order, and CDK deploy commands.
Only the DCV setup, reboot step, policy server, TensorBoard, and LeIsaac eval steps changed.

---

## Summary of Changes

| Section | Change Type | Reason |
|---|---|---|
| DCV Workstation Setup | Rewrite | IsaacSim/Lab now runs in NGC containers, not host-installed |
| Post-deploy reboot | Automated | Bootstrap auto-reboots after cfn-signal; no manual step needed |
| Policy Server Command | Update | Entrypoint, shm-size, and language key changed |
| TensorBoard | Update | Now runs in host venv, not inside a container |
| LeIsaac Simulation Eval | Update | Launch via `run-isaaclab.sh`; language key bug fixed |
| CDK deploy commands | None | Unchanged |
| Batch job submission | None | Unchanged |
| Monitoring / cleanup | None | Unchanged |
| Architecture diagram | None | Unchanged |

---

## Section-by-Section Edits

### 1. DCV Workstation Setup

**What changed:** IsaacSim 4.5.0 and IsaacLab v2.2.0 are no longer host-installed via
conda/pip. They are pulled from NGC as a pre-built container (`nvcr.io/nvidia/isaac-lab:2.2.0`)
during the EC2 bootstrap. The `run-isaaclab.sh` helper script (installed at `/usr/local/bin/`)
wraps `docker run` with all required flags.

**Remove or replace any steps that:**
- Install Miniforge / conda
- Run `pip install isaacsim` from `pypi.nvidia.com`
- Clone IsaacLab via `git clone`
- Set up a conda environment for IsaacSim

**Replace with:**

> The DCV workstation bootstrap automatically pulls the official NVIDIA IsaacLab container
> (`nvcr.io/nvidia/isaac-lab:2.2.0`) from NGC during instance startup. A helper script
> `run-isaaclab.sh` is installed at `/usr/local/bin/` and handles GPU access, X11 forwarding,
> persistent cache volumes, and leisaac package installation automatically on first launch.
>
> To start IsaacLab, open a terminal in the DCV desktop session and run:
> ```bash
> run-isaaclab.sh
> ```
>
> On first launch, leisaac and its scene assets are downloaded automatically (~60 seconds).
> Subsequent launches start immediately from cache.

---

### 2. Post-Deploy Reboot (AUTOMATED)

**What changed:** The bootstrap now automatically reboots the instance after sending the
CloudFormation signal. This loads the NVIDIA kernel module (which can't load on the same
boot that installs the driver). No manual reboot is needed.

The CDK deploy returns `CREATE_COMPLETE` as soon as it receives the cfn-signal. The instance
then reboots ~1 minute later. After the reboot (~60 seconds), the GPU is ready.

**No blog step needed.** Just note that there may be a ~2 minute window after `cdk deploy`
returns during which the instance is rebooting. If you SSH in immediately and `nvidia-smi`
fails, wait a minute and retry.

---

### 3. Policy Server Command

**What changed:**
- Use `--entrypoint python` (the N1.5 `gr00t-finetune:latest` container has `ENTRYPOINT ["/bin/bash"]` — you need to override it to run a Python script directly)
- Add `--shm-size=8g` (required — without it, DataLoader workers crash with a bus error)
- The `--embodiment_tag` argument takes the enum **name** in uppercase (`NEW_EMBODIMENT`), not the value (`new_embodiment`)
- Language key is `annotation.human.task_description` (the `.action.` variant is N1.6 only)

**Old command (approximate):**
```bash
docker run --gpus all -d \
  --name gr00t-policy-server \
  --network host \
  -v $CHECKPOINT:/workspace/checkpoint \
  $EcrImageUri \
  python gr00t/eval/run_gr00t_server.py \
    --model_path /workspace/checkpoint \
    --embodiment_tag new_embodiment
```

**New command:**
```bash
CHECKPOINT=/mnt/efs/gr00t/checkpoints/$JOB_ID/checkpoint-6000

ssh dcv-isaac "docker run --gpus all -d \
  --name gr00t-policy-server \
  --shm-size=8g \
  --network host \
  --entrypoint python \
  -v $CHECKPOINT:/workspace/checkpoint \
  $EcrImageUri \
  gr00t/eval/run_gr00t_server.py \
    --model_path /workspace/checkpoint \
    --embodiment_tag NEW_EMBODIMENT \
    --host 0.0.0.0"
```

Verify server is ready:
```bash
ssh dcv-isaac "docker logs gr00t-policy-server 2>&1 | tail -5"
```

---

### 4. TensorBoard

**What changed:** TensorBoard is now installed in a host venv (`/home/ubuntu/.venv`) during
the EC2 bootstrap. It does **not** need to run inside a container.

**Old approach (approximate):**
```bash
# Inside the training container or manual pip install
tensorboard --logdir ...
```

**New approach:**
```bash
# Start TensorBoard on the DCV instance (SSH port-forward for local browser access)
ssh -L 6006:localhost:6006 dcv-isaac \
  "bash -l -c 'tensorboard --logdir /mnt/efs/gr00t/checkpoints/$JOB_ID --host 0.0.0.0 --port 6006'"
```

Open `http://localhost:6006` in your local browser.

> TensorBoard data persists on EFS after the Batch job completes — view it anytime without
> the training container running.

---

### 5. LeIsaac Simulation Evaluation

**What changed:**
- IsaacLab environment is now inside the `run-isaaclab.sh` container, not a host conda env
- Language key bug fixed: use `annotation.human.task_description` (not the `.action.` variant)
- Leisaac is installed automatically by `run-isaaclab.sh` on first launch
- Use `--policy_type=gr00tn1.5` for N1.5 checkpoints

**Old approach (approximate):**
```bash
# Activate conda env, then run evaluation script
conda activate isaaclab
python scripts/evaluation/policy_inference.py --task=... --policy_type=gr00t ...
```

**New approach:**

Step 1 — Ensure the policy server is running (Section 3 above).

Step 2 — Open a terminal in the DCV desktop session (`https://<elastic-ip>:8443`) and launch the container:
```bash
run-isaaclab.sh
```

Step 3 — Inside the container, run the evaluation:
```bash
/workspace/isaaclab/_isaac_sim/python.sh /workspace/scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 \
    --policy_type=gr00tn1.5 \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --device=cuda \
    --enable_cameras
```

**Expected output:** Per-episode success/failure and a final success rate, e.g.:
```
Final success rate: 0.700 [7/10]
```

---

## Sections With No Changes

The following sections are correct as-is and require no edits:

- CDK prerequisite setup and `cdk bootstrap`
- `cdk deploy IsaacGr00tBatchStack IsaacLabDcvStack` command
- AWS Batch job submission (`aws batch submit-job`)
- Job monitoring (`aws batch describe-jobs`, CloudWatch Logs)
- Architecture diagram and data flow description
- Stack teardown (`cdk destroy`) and retained resource cleanup
- IAM roles and security group explanations
- EFS shared storage explanation
