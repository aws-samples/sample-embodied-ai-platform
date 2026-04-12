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
| **Cause** | The modality config and training script must use the same language key. The correct key for LeIsaac compatibility is `annotation.human.task_description` (not `annotation.human.action.task_description`). The training script, modality config, and LeIsaac client all need to agree on this key. |
| **Status** | ✅ Fixed |
| **Fix** | Aligned all files to use `annotation.human.task_description`: modality config (`so101_modality_config.py` line 48), training script annotation key (`finetune_gr00t.py` line 116), and parquet patching function (`finetune_gr00t.py` line 157). |
| **Files changed** | `training/gr00t/N16/so101_modality_config.py`, `training/gr00t/N16/finetune_gr00t.py` |

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
| **Status** | ✅ Fixed (updated by #25) |
| **Fix** | Replaced the inline `RobotInferenceServer` script with `python3 -m gr00t.eval.run_gr00t_server`. Originally added `--use-sim-policy-wrapper` but this was later removed — see #25. Added `ss -tlnp | grep 5555` verification step. |
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
| **Status** | ✅ Fixed (keyboard patch superseded by #24) |
| **Fix** | Added `-e DISPLAY=:1 -v /tmp/.X11-unix:/tmp/.X11-unix:ro` to the Phase 8a eval `docker run` command in SKILL.md. Originally added a keyboard null-check patch in `run-isaaclab.sh.tpl`, but this is now native in LeIsaac `ef16f98` and the patch was removed — see #24. |
| **Files changed** | `training/gr00t/N16/SKILL.md` |
| **Note** | The display number (`:1`) is set by DCV's auto-console-session and may vary. This fix should be baked into the SKILL.md eval command permanently. |


---

### 15. Open-Loop Eval in SKILL.md Uses Flags Not Accepted by N1.6 Eval Script

| | |
|---|---|
| **Severity** | Medium — open-loop eval fails with unrecognized options |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | Phase 8 open-loop eval command includes `--modality-config-path` and `--video-backend` flags which are not accepted by the N1.6 `gr00t.eval.open_loop_eval` module. The N1.6 eval also requires a running policy server (`--host`/`--port`) rather than loading the model directly. |
| **Cause** | SKILL.md Phase 8 eval command was written for the N1.5 eval interface. The N1.6 eval module has a different CLI interface. |
| **Status** | Worked around — ran eval without the invalid flags and connected to the policy server. |
| **Suggested fix** | Update Phase 8 to start the policy server first, then run open-loop eval with `--host 127.0.0.1 --port 5555` instead of `--modality-config-path` and `--video-backend`. |

---

### 16. EFS Permission Denied When Copying Dataset

| | |
|---|---|
| **Severity** | Medium — Phase 7d dataset copy fails |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | `mkdir: cannot create directory '/mnt/efs/gr00t/sample_dataset': Permission denied` when copying dataset via scp/rsync. |
| **Cause** | The EFS mount root is owned by root, and the ubuntu user doesn't have write permission to create `/mnt/efs/gr00t/`. The training Batch job runs as root inside the container so it doesn't hit this issue, but direct SSH commands as ubuntu fail. |
| **Status** | Worked around — ran `sudo mkdir -p /mnt/efs/gr00t/sample_dataset && sudo chown -R ubuntu:ubuntu /mnt/efs/gr00t` before copying. |
| **Suggested fix** | Add `sudo chown -R ubuntu:ubuntu /mnt/efs` to the bootstrap script, or add a `sudo mkdir -p && sudo chown` step in Phase 7d of SKILL.md before the rsync command. |

---

### 17. LeIsaac ef16f98 Requires `lerobot` Dependency Not in isaac-lab:2.3.0

| | |
|---|---|
| **Severity** | High — eval import fails with `ModuleNotFoundError: No module named 'lerobot'` |
| **File** | `dcv/run-isaaclab.sh.tpl` |
| **Error** | After updating LeIsaac from `v0.3.0` to `ef16f985e3bb`, the `leisaac.tasks` import chain pulls in `leisaac.enhance.datasets.lerobot_dataset_handler` which requires the `lerobot` package. The `isaac-lab:2.3.0` container doesn't include this. |
| **Cause** | The newer LeIsaac commit added `lerobot` as an implicit dependency through its task registration code, but `pip install 'leisaac[gr00t]'` doesn't pull it in automatically. |
| **Status** | ✅ Fixed |
| **Fix** | Added `lerobot` to the pip install line in `dcv/run-isaaclab.sh.tpl`. |
| **Files changed** | `dcv/run-isaaclab.sh.tpl` |

---

### 18. All Three `run-isaaclab.sh.tpl` Patches Now Unnecessary

| | |
|---|---|
| **Severity** | Improvement |
| **File** | `dcv/run-isaaclab.sh.tpl` |
| **Error** | N/A — positive finding |
| **Cause** | LeIsaac `ef16f985e3bb` has native support for: (1) `gr00tn1.6` policy type in `policy_inference.py`, (2) `Gr00t16ServicePolicyClient` class, (3) keyboard headless mode (`self._appwindow ... if self._appwindow else None` + `if self._keyboard:` guard). All three patches from `run-isaaclab.sh.tpl` are now redundant. |
| **Status** | ✅ Fixed — all patches removed from `dcv/run-isaaclab.sh.tpl` |
| **Files changed** | `dcv/run-isaaclab.sh.tpl` (reduced from 202 lines to 72 lines) |

---

### 19. `--use-sim-policy-wrapper` Incompatible with `Gr00t16ServicePolicyClient`

| | |
|---|---|
| **Severity** | Critical — eval crashes with `"Video key 'video.front' must be in observation"` |
| **File** | `training/gr00t/N16/SKILL.md` |
| **Error** | Policy server returns `{'error': "Video key 'video.front' must be in observation"}` when called from `Gr00t16ServicePolicyClient`. The client then hits `KeyError: 0` trying to index the error dict as an action. |
| **Cause** | The `Gr00t16ServicePolicyClient` sends observations in a nested format (`{"observation": {"video": {"front": ...}, "state": {...}, "language": {...}}}`). The `--use-sim-policy-wrapper` flag expects flat-keyed observations (`video.front`, `state.single_arm`). These two formats are incompatible. The old `Gr00tServicePolicyClient` (N1.5) used flat keys and needed the wrapper, but the new N1.6 client does not. |
| **Status** | ✅ Fixed |
| **Fix** | Removed `--use-sim-policy-wrapper` from the Phase 8 policy server command in SKILL.md. Updated description to explain why the flag should not be used with the N1.6 client. |
| **Files changed** | `training/gr00t/N16/SKILL.md` |

---

### 20. Language Key Mismatch in `Gr00t16ServicePolicyClient` (Upstream Issue #145)

| | |
|---|---|
| **Severity** | High — eval crashes with `KeyError: 0` on first action request |
| **File** | LeIsaac `leisaac/policy/service_policy_clients.py` line 136 (upstream) |
| **Error** | `Gr00t16ServicePolicyClient.get_action()` sends language key `annotation.human.task_description`, but the GR00T N1.6 policy server's observation validation rejects it. The server returns `{'error': '...'}` and the client crashes with `KeyError: 0` when trying `action_chunk[0]` on the error dict. |
| **Cause** | The language key used in LeIsaac's `Gr00t16ServicePolicyClient` must match the key the model was trained with. Our model was trained with `annotation.human.action.task_description` (N1.6 default from the main fine-tuning guide), but LeIsaac sends `annotation.human.task_description`. This is the mismatch documented in [LightwheelAI/leisaac#145](https://github.com/LightwheelAI/leisaac/issues/145). |
| **Status** | ⚠️ Upstream — needs alignment between training config and LeIsaac client |
| **Workaround** | Manually patched `annotation.human.task_description` to `annotation.human.action.task_description` in the installed `service_policy_clients.py` on the remote instance. |
| **Note** | Colleague is aware of this issue and says `annotation.human.task_description` is the correct key for LeIsaac compatibility. The fix may need to happen on the training config side (use a modality config that trains with `annotation.human.task_description`) rather than patching LeIsaac. See the EverNorif response on issue #145 for the recommended training config. |

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