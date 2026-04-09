# N16 Branch Deployment Test Report

**Date:** April 9, 2026
**Branch:** `feature/add-N16`
**Tested by:** Claude Code (Opus 4.6 via Bedrock) + manual verification
**Region:** us-west-2
**Instance:** g6.2xlarge (NVIDIA L4)

---

## Summary

End-to-end deployment test of the N1.6 GR00T fine-tuning pipeline using Claude Code
to autonomously follow `training/gr00t/N16/SKILL.md`. The full pipeline (deploy → train →
evaluate) completed successfully. Training loss converged to 0.0077. Closed-loop evaluation
ran 10 episodes with 0/10 success rate (model quality, not infrastructure failure).

Total time: ~4 hours (deploy ~30min, training ~2hrs, evaluation ~1.5hrs including debugging).

---

## Findings

### 1. Wrong Language Modality Key in N1.6 Config

| | |
|---|---|
| **Severity** | Critical — blocks training |
| **File** | `training/gr00t/N16/so101_modality_config.py` |
| **Error** | `AssertionError: Key human.task_description not found in language modality` |
| **Cause** | The config used the N1.5 key `annotation.human.task_description` instead of the N1.6 key `annotation.human.action.task_description`. The training script (`finetune_gr00t.py`) patches parquet files with the N1.6 key, but the modality config still referenced the N1.5 key. |
| **Status** | ✅ Fixed |
| **Fix** | Changed `modality_keys` from `["annotation.human.task_description"]` to `["annotation.human.action.task_description"]` on line 48. |
| **Files changed** | `training/gr00t/N16/so101_modality_config.py` |

---

### 2. Public Security Group Exposes DCV and W&B to Internet

| | |
|---|---|
| **Severity** | Critical — triggers enterprise security isolation |
| **File** | `dcv/dcv_construct.py` |
| **Error** | DyePack scanner detected publicly accessible web endpoint without strong authentication. Instance `i-0f0612f29e00d866b` was auto-isolated. |
| **Cause** | Security group opened ports 8443 (DCV) and 8080 (W&B) to `0.0.0.0/0` via `ec2.Peer.any_ipv4()`. DCV uses a simple password, W&B has zero authentication. |
| **Status** | ✅ Fixed |
| **Fix** | Removed both `add_ingress_rule` calls. All access now goes through SSH port forwarding via SSM (already configured in Phase 5). Added comment explaining the SSM-only approach. |
| **Files changed** | `dcv/dcv_construct.py` |

---

### 3. SKILL.md References Public IP for DCV and W&B Access

| | |
|---|---|
| **Severity** | High — instructions broken after security group fix |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | Multiple references to `https://<elastic-ip>:8443` and `http://<elastic-ip>:8080` which no longer work without public ingress rules. |
| **Cause** | SKILL.md was written assuming public port access. After removing public ingress (finding #2), these instructions become invalid. |
| **Status** | ✅ Fixed |
| **Fix** | Updated Phase 6, Phase 7c, and Phase 8a to use SSH port forwarding: `ssh -f -N -L 8443:localhost:8443 -L 8080:localhost:8080 dcv-isaac`. Added note that Claude Code doesn't need port forwards — it accesses everything via SSH commands. |
| **Files changed** | `training/gr00t/N16/SKILL.md` |

---

### 4. Phase 8a Eval Uses Wrong Container Entrypoint

| | |
|---|---|
| **Severity** | Critical — eval never executes |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | IsaacSim loaded but evaluation script never ran. Robot arm never moved. Stuck for 40+ minutes with no output. |
| **Cause** | SKILL.md instructed to use `run-isaaclab.sh` which routes arguments through `runheadless.sh` (Kit launcher), not `python.sh`. The eval script path was passed as Kit arguments, not executed as Python. |
| **Status** | ✅ Fixed |
| **Fix** | Rewrote Phase 8a with a direct `docker run` command using `--entrypoint /workspace/isaaclab/_isaac_sim/python.sh`. Added `PYTHONUNBUFFERED=1` for real-time output. Added warning not to use `run-isaaclab.sh` for automated evaluation. |
| **Files changed** | `training/gr00t/N16/SKILL.md` |

---

### 5. LeIsaac v0.3.0 Doesn't Support gr00tn1.6 Policy Type

| | |
|---|---|
| **Severity** | Critical — eval fails with unrecognized policy type |
| **File** | `scripts/evaluation/policy_inference.py` (LeIsaac repo) |
| **Error** | `policy_inference.py` only handles `gr00tn1.5`, `lerobot`, and `openpi` in both `preprocess_obs_dict` and policy creation. `gr00tn1.6` is not recognized. |
| **Cause** | LeIsaac v0.3.0 was released before N1.6 support was added. The `Gr00t16ServicePolicyClient` class exists in the package but the eval script hasn't been wired up to use it. |
| **Status** | ✅ Patched locally |
| **Fix** | Added a post-clone `sed` patch in `dcv_construct.py` inside the `run-isaaclab.sh` helper script. After cloning LeIsaac, it patches `policy_inference.py` to accept `gr00tn1.6` in both the preprocessing and policy creation blocks. |
| **Files changed** | `dcv/dcv_construct.py` |
| **Note** | LeIsaac upstream (`LightwheelAI/leisaac`) needs native N1.6 support added. Once upstream adds `gr00tn1.6` handling, the local `sed` patch can be removed. Consider raising a PR or flagging to colleague. |

---

### 6. Hardcoded Python Path in cdk.json

| | |
|---|---|
| **Severity** | High — breaks for every new user |
| **File** | `training/gr00t/infra/cdk.json` |
| **Error** | `cdk.json` contains `/home/aaron/Projects/sample-embodied-ai-platform/.venv/bin/python app.py` — a path specific to the original developer's machine. |
| **Cause** | Path was committed with an absolute path instead of a relative one. |
| **Status** | ⚠️ Needs fix |
| **Suggested fix** | Change to a relative path, e.g. `"app": "python3 app.py"` or use a path relative to the repo root. Claude Code fixed this at runtime by replacing with the local user's path, but it will break again for every new user. |

---

### 7. Claude Code Stops After Phase 6

| | |
|---|---|
| **Severity** | Medium — requires manual intervention to continue |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | Claude Code completed Phase 6 verification and stopped, saying "ready for training jobs whenever you are" instead of continuing to Phase 7. |
| **Cause** | SKILL.md has no explicit instruction to execute all phases end-to-end without pausing. Claude Code interpreted the phase boundary as a checkpoint for user input. |
| **Status** | ✅ Fixed |
| **Suggested fix** | Added instruction at the top of SKILL.md: "Execute all phases sequentially from Phase 1 through Phase 8a without pausing. Only stop if a phase fails or requires information not available in this document. When a phase depends on a long-running process, poll until completion then proceed automatically." |
| **Files changed** | `training/gr00t/N16/SKILL.md` |

---

### 8. Phase 8 Open-Loop Eval Missing Parameters

| | |
|---|---|
| **Severity** | High — eval fails with wrong modality keys and video codec |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | Eval script defaults to `right_arm/left_arm` modality keys (humanoid), but SO-100/101 uses `single_arm/gripper`. Also, sample dataset uses AV1 video codec which `torchcodec` (default) doesn't support. |
| **Cause** | Phase 8 eval command doesn't specify `--modality-keys` or `--video-backend` parameters for the SO-100/101 embodiment. |
| **Status** | ✅ Fixed |
| **Fix** | Updated Phase 8 open-loop eval command: added `--modality-keys single_arm gripper` for SO-100/101 embodiment, `--video-backend torchvision_av` for AV1 codec support, and changed `--dataset-path` from placeholder `/path/to/eval-dataset` to `/mnt/efs/gr00t/sample_dataset`. |
| **Files changed** | `training/gr00t/N16/SKILL.md` |

---

### 11. Phase 8 Policy Server Command Doesn't Match N1.6 Container

| | |
|---|---|
| **Severity** | High — policy server fails to start with documented command |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | SKILL.md references `run_gr00t_server.py` but the N1.6 container has a different server structure. The ZMQ server is in `gr00t/eval/service.py` via `RobotInferenceServer`, not a standalone script. |
| **Cause** | The documented command was written for a different container layout. Claude Code had to explore the container to find the correct server interface. |
| **Status** | ✅ Fixed |
| **Fix** | Rewrote Phase 8 policy server command to use `RobotInferenceServer` from `gr00t.eval.robot` with an inline Python script, instead of the non-existent `run_gr00t_server.py`. Uses `Gr00tPolicy` to load the checkpoint and `RobotInferenceServer` to serve on port 5555. Added `ss -tlnp | grep 5555` verification step. |
| **Files changed** | `training/gr00t/N16/SKILL.md` |

---

### 12. Eval Container Uses --rm, Losing Final Output

| | |
|---|---|
| **Severity** | Low — final success rate not captured |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | The eval container was started with `--rm` flag. When it completed, the container and its logs were automatically deleted. The final success rate output was lost. |
| **Cause** | `--rm` is convenient for cleanup but incompatible with post-hoc log inspection. |
| **Status** | ✅ Fixed |
| **Fix** | Added `2>&1 | tee /mnt/efs/gr00t/eval-results.log` to the Phase 8a eval command and mounted EFS (`-v /mnt/efs:/mnt/efs`) so results persist to `/mnt/efs/gr00t/eval-results.log` even after the `--rm` container exits. |
| **Files changed** | `training/gr00t/N16/SKILL.md` |

---

### 13. Claude Code Improvises Instead of Following SKILL.md

| | |
|---|---|
| **Severity** | Low — works but wastes time |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | Claude Code inspected parquet files, explored container internals, and ran diagnostic commands not in the SKILL.md. While this helped debug issues, it added ~30 minutes of unnecessary exploration. |
| **Cause** | SKILL.md is not prescriptive enough — it describes what to do but not what NOT to do. Claude Code fills gaps with its own judgment. |
| **Status** | 📋 For colleague — make SKILL.md more explicit |
| **Suggested fix** | Add guardrails to SKILL.md: "Do not inspect dataset contents or container internals unless a command fails. Follow the commands exactly as written." |

---

### 15. No Resume/Recovery Guidance

| | |
|---|---|
| **Severity** | Medium — interrupted deployments require improvisation |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | When the DCV instance was isolated by security, Claude Code had no documented path to recover. It improvised by fixing the security group, restarting the instance, and manually transferring data. |
| **Cause** | SKILL.md assumes a clean, uninterrupted deployment. No guidance for resuming from a specific phase or recovering from failures. |
| **Status** | 📋 For colleague — add recovery section |
| **Suggested fix** | Add a "Resuming from Interruption" section that documents how to check current state and resume from any phase. Include common failure scenarios (instance stopped, security group changed, training job failed). |

---

## Files Changed (Local)

| File | Change |
|------|--------|
| `training/gr00t/N16/so101_modality_config.py` | Fixed language modality key for N1.6 |
| `dcv/dcv_construct.py` | Removed public ingress rules; added LeIsaac N1.6 patch |
| `training/gr00t/N16/SKILL.md` | SSM port forwarding; Phase 8a eval rewrite |
| `training/gr00t/infra/cdk.json` | Fixed hardcoded path (Claude Code runtime fix) |
| `training/gr00t/infra/app.py` | Set availability_zone and instance_type (Claude Code runtime fix) |

---

## Test Results

| Phase | Result |
|-------|--------|
| Phase 1-3: Deploy stacks | ✅ Both stacks CREATE_COMPLETE |
| Phase 4-6: Bootstrap + verify | ✅ All STEP_OK, GPU detected, EFS mounted |
| Phase 7: Training | ✅ 6000 steps, loss 0.0077, checkpoint saved |
| Phase 8: Open-loop eval | ✅ MSE 13.836 |
| Phase 8a: Closed-loop eval | ✅ Pipeline works, 0/10 success (model quality) |
