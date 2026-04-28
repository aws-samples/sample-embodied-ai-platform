# N1.7 End-to-End Test: Changes & Fixes Log

This file tracks all changes made during the N1.7 E2E deployment test,
explaining what was changed and why. The N16 module runs E2E successfully —
all changes here are N1.7-specific adjustments.

## Baseline

- N17 module cloned from N16 (commit `533e895`)
- Model path: `nvidia/GR00T-N1.7-3B`
- Stable commit: `e8e625f4f21898c506a1d8f7d20a289c97a52acf` (carried from N1.6)
- Base image: `nvcr.io/nvidia/pytorch:25.04-py3` (carried from N1.6)

## Changes

### 1. Update STABLE_COMMIT to N1.7 release tag

**File:** `Dockerfile`
**Error:** `The checkpoint you are trying to load has model type 'Gr00tN1d7' but Transformers does not recognize this architecture.`
**Cause:** The N1.6 stable commit (`e8e625f`) doesn't include the `Gr00tN1d7` model class. The N1.7 model architecture is defined in the Isaac-GR00T repo itself (not in the `transformers` library).
**Fix:** Changed `STABLE_COMMIT` from `e8e625f4f21898c506a1d8f7d20a289c97a52acf` (N1.6) to `23ace64f17aa5015259b8609d371eb61a357c776` (N1.7 release tag `n1.7-release`, dated 2026-04-18).

### 2. HuggingFace gated model authentication required for Cosmos-Reason2-2B

**Error:** `You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/nvidia/Cosmos-Reason2-2B. Access to model nvidia/Cosmos-Reason2-2B is restricted.`
**Cause:** N1.7's VLM backbone is `nvidia/Cosmos-Reason2-2B` (Qwen3-VL architecture), which is a gated model on HuggingFace. N1.6 used Eagle which was not gated. The training job needs an HF_TOKEN with accepted licenses for both `nvidia/GR00T-N1.7-3B` and `nvidia/Cosmos-Reason2-2B`.
**Fix:** User must accept model licenses on HuggingFace, then pass HF_TOKEN as a container override when submitting the Batch job. Updated env.example to document this requirement.

### 3. Remove hard-coded Eagle backbone config from finetune_gr00t.py

**File:** `finetune_gr00t.py`
**Error:** `nvidia/Eagle-Block2A-2B-v2 is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'`
**Cause:** The N1.6 training script hard-coded `config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"` and `config.model.eagle_collator = True`. N1.7 uses Cosmos-Reason2-2B as its VLM backbone (not Eagle), and the Eagle model no longer exists on HuggingFace. The N1.7 model correctly loads its own backbone from the checkpoint, but the script then tries to override it with Eagle.
**Fix:** Removed `config.model.eagle_collator = True` and `config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"` lines. Let N1.7's `get_default_config()` handle backbone config automatically.

## E2E Test Results (2026-04-28)

- **Training:** 6000 steps, ~94 min on g6e.4xlarge, final loss 0.079
- **Open-loop eval:** MSE 11.65, MAE 2.11 (1 trajectory, 159 steps)
- **Closed-loop eval:** 0/10 success (all episodes timed out)
- **Job ID:** `a8cf4bbe-b7f5-4345-81c3-3febadb0f074`
- **Checkpoints:** checkpoint-2000, checkpoint-4000, checkpoint-6000 on EFS

### Closed-loop eval notes
- Used `--policy_type=gr00tn1.6` since LeIsaac does not yet have native N1.7 support
- All 10 episodes timed out — the robot arm did not complete the pick-and-place task
- Likely causes: observation format mismatch via gr00tn1.6 client, small sample dataset, and N1.7 early access backbone differences
- Infrastructure worked correctly (policy server responded to all requests) — this is a model performance issue, not an infra issue

### Notes for deployment
- HF_TOKEN is **required** for N1.7 (unlike N1.6). Must be passed as env var to both training jobs and the policy server.
- User must accept licenses for both `nvidia/Cosmos-Reason2-2B` and `nvidia/GR00T-N1.7-3B` on HuggingFace.
