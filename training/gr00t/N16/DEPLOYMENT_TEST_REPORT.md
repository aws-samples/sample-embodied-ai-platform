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

### 9. Phase 8 Policy Server Command Doesn't Match N1.6 Container

| | |
|---|---|
| **Severity** | High — policy server fails to start with documented command |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | SKILL.md references `run_gr00t_server.py` but the N1.6 container has a different server structure. The ZMQ server is in `gr00t/eval/service.py` via `RobotInferenceServer`, not a standalone script. |
| **Cause** | The documented command was written for a different container layout. Claude Code had to explore the container to find the correct server interface. |
| **Status** | ✅ Fixed |
| **Fix** | Replaced the inline `RobotInferenceServer` script with `python3 -m gr00t.eval.run_gr00t_server` and added `--use-sim-policy-wrapper` to the Phase 8 policy server command in SKILL.md. Added `ss -tlnp | grep 5555` verification step. |
| **Files changed** | `training/gr00t/N16/SKILL.md` |

---

### 10. Eval Container Uses --rm, Losing Final Output

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

### 11. Non-ASCII Em Dash in Security Group Description Breaks CDK Synth

| | |
|---|---|
| **Severity** | Medium - blocks deployment on some environments |
| **File** | `dcv/dcv_construct.py` |
| **Error** | CDK synth failed due to a Unicode em dash character in the security group description string. Some environments and CI pipelines choke on non-ASCII characters in CloudFormation template strings. |
| **Cause** | The description `"DCV workstation - access via SSM port forwarding only"` used an em dash (U+2014) instead of a regular hyphen (U+002D). |
| **Status** | ✅ Fixed |
| **Fix** | Replaced the em dash with a regular dash: `"DCV workstation - access via SSM port forwarding only"`. |
| **Files changed** | `dcv/dcv_construct.py` |

---

### 12. UserData Exceeds 16KB EC2 Limit on Fresh Deploy

| | |
|---|---|
| **Severity** | Critical - DCV stack deployment fails with CREATE_FAILED |
| **File** | `dcv/dcv_construct.py` |
| **Error** | `IsaacLabDcvStack` deployment failed because the EC2 UserData exceeded the 16KB hard limit. CloudFormation returned a creation error for the EC2 instance resource. |
| **Cause** | The `run-isaaclab.sh` helper script (~3,500 bytes) was inlined as a heredoc directly in UserData alongside the bootstrap script (~10,000 bytes). With the LeIsaac N1.6 patches (finding #5) and other fixes added in the previous test run, the combined UserData grew past 16KB. Previous deployments were at ~99% of the limit and the additional content tipped it over. |
| **Status** | ✅ Fixed |
| **Fix** | Extracted the `run-isaaclab.sh` helper script into a standalone template file (`dcv/run-isaaclab.sh.tpl`) with `__CONTAINER_IMAGE__` and `__LEISAAC_COMMIT__` placeholders. At CDK synth time, placeholders are substituted and the rendered script is uploaded as an S3 asset via `aws_cdk.aws_s3_assets.Asset`. The UserData now downloads the script with `aws s3 cp` instead of inlining it, freeing ~3,500 bytes of UserData space. The instance IAM role is granted read access to the S3 asset automatically. |
| **Files changed** | `dcv/dcv_construct.py`, `dcv/run-isaaclab.sh.tpl` (new file) |
| **Note** | This is a permanent architectural improvement. As the project grows, more scripts can be externalized to S3 assets using the same pattern, avoiding future 16KB limit issues. |

---

### 13. Sample Dataset Not Available on EFS for Open-Loop Evaluation

| | |
|---|---|
| **Severity** | High - Phase 8 open-loop eval fails with missing dataset |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | Phase 8 open-loop eval command references `--dataset-path /mnt/efs/gr00t/sample_dataset` but no prior step copies the dataset to EFS. The training job (Phase 7) uses the dataset baked into the container image during `docker build`, not from EFS. |
| **Cause** | The sample dataset is included in the git repo at `training/sample_dataset/` and gets cloned into the container during the CodeBuild image build. The Batch training job accesses it at `/workspace/sample-embodied-ai-platform/training/sample_dataset` inside the container. However, the DCV instance has no copy of the dataset on EFS, so the eval container can't find it. |
| **Status** | ✅ Fixed |
| **Fix** | Added Phase 7d to SKILL.md with three steps: rsync sample dataset to EFS, create `modality.json` with correct column mappings, and generate `stats.json` via the training container. |
| **Files changed** | `training/gr00t/N16/SKILL.md` |

---

### 14. LeIsaac Eval Hangs When --enable_cameras Used Without DCV Display Context

| | |
|---|---|
| **Severity** | High - eval container hangs indefinitely, wastes over an hour |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | The LeIsaac closed-loop eval container ran for over an hour with 119% CPU but 0% GPU utilization. No episode output was produced. Docker logs stuck at 414 lines with no growth. |
| **Cause** | The Phase 8a `docker run` command did not pass the display environment (`DISPLAY=:1`) or mount the X11 socket (`/tmp/.X11-unix`). IsaacSim requires a rendering context for `--enable_cameras`. The DCV auto-session runs on display `:1` (not `:0`), and without passing this to the container, IsaacSim hangs during rendering initialization. The `run-isaaclab.sh` helper script handles this correctly (it sets `-e DISPLAY` and mounts Xauth), but the direct `docker run` in SKILL.md Phase 8a did not. |
| **Status** | ✅ Fixed |
| **Fix** | Added `-e DISPLAY=:1 -v /tmp/.X11-unix:/tmp/.X11-unix:ro` to the Phase 8a eval `docker run` command in SKILL.md. Added a keyboard null-check patch in `run-isaaclab.sh.tpl` to handle missing keyboard in headless fallback. |
| **Files changed** | `training/gr00t/N16/SKILL.md`, `dcv/run-isaaclab.sh.tpl` |
| **Note** | The display number (`:1`) is set by DCV's auto-console-session and may vary. This fix should be baked into the SKILL.md eval command permanently. |

---

### 15. Policy Server Missing Sim Wrapper - Action Key Mismatch

| | |
|---|---|
| **Severity** | High - eval crashes after first episode with KeyError |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | LeIsaac eval crashed with `KeyError: 'action.single_arm'` after completing episode 1. The policy server returns keys like `single_arm` and `gripper`, but the LeIsaac `Gr00tServicePolicyClient` expects `action.single_arm` and `action.gripper` (with the `action.` prefix). |
| **Cause** | The GR00T `Gr00tPolicy` returns raw action keys without the `action.` prefix. The `Gr00tSimPolicyWrapper` class adds this prefix, but the SKILL.md Phase 8 policy server command does not use the wrapper. The policy server needs `--use-sim-policy-wrapper` to produce keys compatible with LeIsaac. |
| **Status** | ✅ Fixed |
| **Fix** | Replaced the inline `RobotInferenceServer` script with `python3 -m gr00t.eval.run_gr00t_server` and added `--use-sim-policy-wrapper` to the Phase 8 policy server command in SKILL.md. |
| **Files changed** | `training/gr00t/N16/SKILL.md` |

---

### 17. Claude Code Improvises Instead of Following SKILL.md

| | |
|---|---|
| **Severity** | Low - works but wastes time |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | Claude Code inspected parquet files, explored container internals, and ran diagnostic commands not in the SKILL.md. While this helped debug issues, it added ~30 minutes of unnecessary exploration. |
| **Cause** | SKILL.md is not prescriptive enough - it describes what to do but not what NOT to do. Claude Code fills gaps with its own judgment. |
| **Status** | For colleague - make SKILL.md more explicit |
| **Suggested fix** | Add guardrails to SKILL.md: "Do not inspect dataset contents or container internals unless a command fails. Follow the commands exactly as written." |

---

### 18. No Resume/Recovery Guidance

| | |
|---|---|
| **Severity** | Medium - interrupted deployments require improvisation |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | When the DCV instance was isolated by security, Claude Code had no documented path to recover. It improvised by fixing the security group, restarting the instance, and manually transferring data. |
| **Cause** | SKILL.md assumes a clean, uninterrupted deployment. No guidance for resuming from a specific phase or recovering from failures. |
| **Status** | For colleague - add recovery section |
| **Suggested fix** | Add a "Resuming from Interruption" section that documents how to check current state and resume from any phase. Include common failure scenarios (instance stopped, security group changed, training job failed). |

---

### 19. run-isaaclab.sh Multiline Sed Missing Filename Argument

| | |
|---|---|
| **Severity** | High — keyboard subscription patch silently fails, policy client patch never applies |
| **File** | `dcv/run-isaaclab.sh.tpl` |
| **Error** | `sed: no input files` during `run-isaaclab.sh -c 'echo prerequisites installed'`. Script exits with code 4 due to `set -euo pipefail`. The multiline `sed` command that wraps the keyboard subscription in `if self._keyboard:` guard is missing the `"$EVAL_SCRIPT"` filename argument. The subsequent Python heredoc patch for the policy client also fails to execute because the script aborts at the sed error. |
| **Cause** | The multiline sed on line 50-55 of `run-isaaclab.sh.tpl` closes with `}'` but has no `"$EVAL_SCRIPT"` filename after the closing quote. |
| **Status** | Fixed |
| **Fix** | Changed `}'` to `}' "$EVAL_SCRIPT"` on line 55 of `dcv/run-isaaclab.sh.tpl`. Also manually applied both patches (keyboard subscription guard and policy client N1.6 compatibility) on the remote instance via SSH. |
| **Files changed** | `dcv/run-isaaclab.sh.tpl` |

---

### 20. Phase 6a.2 References Non-Existent Gr00t16ServicePolicyClient Class

| | |
|---|---|
| **Severity** | Low — verification step fails but not a blocker |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | Phase 6a.2 instructs `grep "class Gr00t16ServicePolicyClient"` but LeIsaac v0.3.0 only has `Gr00tServicePolicyClient`. The N1.6 support is patched into the existing class, not a separate one. |
| **Cause** | SKILL.md assumes a class naming convention that doesn't match the actual LeIsaac codebase. |
| **Status** | Not a blocker — the existing `Gr00tServicePolicyClient` class is patched for N1.6 compatibility. |
| **Suggested fix** | Update SKILL.md Phase 6a.2 to check for `class Gr00tServicePolicyClient` instead, or check for the N1.6 patch marker string `GR00T N1.6 SimPolicyWrapper`. |

---

### 21. Open-Loop Eval in SKILL.md Uses Flags Not Accepted by N1.6 Eval Script

| | |
|---|---|
| **Severity** | Medium — open-loop eval fails with unrecognized options |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | Phase 8 open-loop eval command includes `--modality-config-path` and `--video-backend` flags which are not accepted by the N1.6 `gr00t.eval.open_loop_eval` module. The N1.6 eval also requires a running policy server (`--host`/`--port`) rather than loading the model directly. |
| **Cause** | SKILL.md Phase 8 eval command was written for the N1.5 eval interface. The N1.6 eval module has a different CLI interface. |
| **Status** | Worked around — ran eval without the invalid flags and connected to the policy server. |
| **Suggested fix** | Update Phase 8 to start the policy server first, then run open-loop eval with `--host 127.0.0.1 --port 5555` instead of `--modality-config-path` and `--video-backend`. |

---

### 22. EFS Permission Denied When Copying Dataset

| | |
|---|---|
| **Severity** | Medium — Phase 7d dataset copy fails |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | `mkdir: cannot create directory '/mnt/efs/gr00t/sample_dataset': Permission denied` when copying dataset via scp/rsync. |
| **Cause** | The EFS mount root is owned by root, and the ubuntu user doesn't have write permission to create `/mnt/efs/gr00t/`. The training Batch job runs as root inside the container so it doesn't hit this issue, but direct SSH commands as ubuntu fail. |
| **Status** | Worked around — ran `sudo mkdir -p /mnt/efs/gr00t/sample_dataset && sudo chown -R ubuntu:ubuntu /mnt/efs/gr00t` before copying. |
| **Suggested fix** | Add `sudo chown -R ubuntu:ubuntu /mnt/efs` to the bootstrap script, or add a `sudo mkdir -p && sudo chown` step in Phase 7d of SKILL.md before the rsync command. |

---

### 23. LeIsaac ef16f98 Requires `lerobot` Dependency Not in isaac-lab:2.3.0

| | |
|---|---|
| **Severity** | High — eval import fails with `ModuleNotFoundError: No module named 'lerobot'` |
| **File** | `dcv/run-isaaclab.sh.tpl` |
| **Error** | After updating LeIsaac from `v0.3.0` to `ef16f985e3bb`, the `leisaac.tasks` import chain pulls in `leisaac.enhance.datasets.lerobot_dataset_handler` which requires the `lerobot` package. The `isaac-lab:2.3.0` container doesn't include this. |
| **Cause** | The newer LeIsaac commit added `lerobot` as an implicit dependency through its task registration code, but `pip install 'leisaac[gr00t]'` doesn't pull it in automatically. |
| **Status** | Worked around — manually ran `pip install --target /workspace/isaaclab-pkgs lerobot` in the container. |
| **Suggested fix** | Add `lerobot` to `run-isaaclab.sh.tpl` pip install, or pin the LeIsaac dependency to include it: `'leisaac[gr00t,lerobot]'`. Alternatively, the LeIsaac `pyproject.toml` should declare `lerobot` as a required dependency for the `gr00t` extra. |

---

### 24. Newer LeIsaac Removes `Isaac-Gr00t-Franka-Cabinet-Direct-v0` Task

| | |
|---|---|
| **Severity** | Critical — closed-loop eval cannot run against trained Franka model |
| **File** | `dcv/versions.py`, LeIsaac task registry |
| **Error** | `gymnasium.error.NameNotFound: Environment 'Isaac-Gr00t-Franka-Cabinet-Direct' doesn't exist`. Using built-in `Isaac-Franka-Cabinet-Direct-v0` fails with `AttributeError: 'FrankaCabinetEnvCfg' has no attribute 'use_teleop_device'`. |
| **Cause** | LeIsaac `ef16f985e3bb` removed all `Isaac-Gr00t-*` tasks and replaced them with `LeIsaac-SO101-*` and `LeIsaac-LeKiwi-*` tasks for SO101/LeKiwi robots. The built-in IsaacLab Franka task lacks the `use_teleop_device()` method that the new `policy_inference.py` requires. Our trained model is Franka-based and incompatible with the SO101 tasks. |
| **Status** | ⚠️ Blocking — cannot validate closed-loop eval with newer LeIsaac + Franka model |
| **Suggested fix** | Either: (1) pin LeIsaac to a commit that still has the Franka task, (2) retrain on an SO101 task, or (3) write a thin task wrapper that makes `Isaac-Franka-Cabinet-Direct-v0` compatible with the new `policy_inference.py` by adding `use_teleop_device()`. |

---

### 25. All Three `run-isaaclab.sh.tpl` Patches Now Unnecessary

| | |
|---|---|
| **Severity** | Improvement |
| **File** | `dcv/run-isaaclab.sh.tpl` |
| **Error** | N/A — positive finding |
| **Cause** | LeIsaac `ef16f985e3bb` has native support for: (1) `gr00tn1.6` policy type in `policy_inference.py`, (2) `Gr00t16ServicePolicyClient` class, (3) keyboard headless mode (`self._appwindow ... if self._appwindow else None` + `if self._keyboard:` guard). All three patches from `run-isaaclab.sh.tpl` are now redundant. |
| **Status** | ✅ Fixed — all patches removed from `dcv/run-isaaclab.sh.tpl` |
| **Files changed** | `dcv/run-isaaclab.sh.tpl` (reduced from 202 lines to 72 lines) |

---

## Files Changed (Local)

| File | Change |
|------|--------|
| `training/gr00t/N16/so101_modality_config.py` | Fixed language modality key for N1.6 |
| `dcv/dcv_construct.py` | Removed public ingress rules; added LeIsaac N1.6 patch; extracted helper script to S3 asset; fixed em dash encoding |
| `dcv/run-isaaclab.sh.tpl` | New file: externalized run-isaaclab.sh helper script template; all N1.6 patches removed (native in ef16f98) |
| `dcv/versions.py` | Updated LeIsaac commit from `v0.3.0` to `ef16f985e3bb2bf6f3012d0a40c2ca5c17c31cb6` |
| `training/gr00t/N16/SKILL.md` | SSM port forwarding; Phase 8a eval rewrite |
| `training/gr00t/infra/cdk.json` | Fixed hardcoded path (Claude Code runtime fix) |
| `training/gr00t/infra/app.py` | Set availability_zone and instance_type (Claude Code runtime fix) |