# Troubleshooting

## Bootstrap Issues

**Bootstrap failed partway:** The script is idempotent — re-running skips completed steps.
SSH in and check `sudo cat /var/lib/cloud/instance/scripts/part-001` for the full script.
Re-run with `sudo bash /var/lib/cloud/instance/scripts/part-001`.

**SSM "TargetNotConnected":** The instance is booting or the SSM agent is still registering.
Wait 3-5 minutes after instance start and retry — the SSM agent retries credential acquisition
on boot, so `TargetNotConnected` is transient and not a sign of a broken instance.

**CDK destroy "Cannot delete export":** Happens when destroying Batch before DCV.
Always destroy DCV first.

## GPU and Driver Issues

**nvidia-smi fails after first deploy:** The NVIDIA kernel module can't load via `modprobe`
on the same boot that installs the driver. Run `sudo reboot` and reconnect after ~60
seconds. This only affects the first deploy.

## Container Issues

**`--shm-size=8g` required:** Both open-loop eval and policy server need this flag or
DataLoader workers crash with a bus error.

**NGC base image bash broken (N1.6 only):** The `nvcr.io/nvidia/pytorch:25.04-py3` base
image ships a `/usr/bin/bash` that fails with "cannot execute binary file". Use
`--entrypoint /bin/sh` when running `docker run` commands against the N1.6 training
container (`gr00t-finetune:6`). The N1.5 container (`gr00t-finetune:latest`) uses
`nvidia/cuda` base and works fine with `--entrypoint python`.

**Helper script not found:** If `run-isaaclab.sh` doesn't exist, check the bootstrap
summary for `create-helper-script` status. If missing, the container setup step failed —
check detailed logs: `grep -A 50 "pull-isaaclab-container" /var/log/dcv-bootstrap.log`.

**IsaacLab container cache dirs owned by root:** `run-isaaclab.sh` creates cache dirs
(`~/docker/isaac-sim/cache/...`) on first run; if a prior `docker run` created them as root,
subsequent runs fail with "permission denied". Fix:
```bash
sudo chown -R ubuntu:ubuntu ~/docker/ ~/isaaclab-pkgs/
```

## Policy Server Issues

**Uppercase embodiment tag in server CLI:** The `run_gr00t_server.py` CLI (via `tyro`)
expects the `EmbodimentTag` enum **name** in uppercase (`NEW_EMBODIMENT`), not the enum
**value** in lowercase (`new_embodiment`). The training script uses the value form.

## LeIsaac / Eval Issues

**LeIsaac `KeyError: 0` during eval:** The leisaac `Gr00t16ServicePolicyClient` at commit
`d2cbfd2` has a bug: it sends `annotation.human.task_description` but N1.6 expects
`annotation.human.action.task_description` (note the `.action.` segment). The GR00T server's
strict validation fails, returns `{"error": "..."}` dict, and the client's `action_chunk[0]`
raises `KeyError: 0` on the dict. `run-isaaclab.sh` auto-patches this after install via `sed`.
If you installed leisaac manually, apply the fix:
```bash
sed -i 's/"annotation.human.task_description"/"annotation.human.action.task_description"/' \
  ~/isaaclab-pkgs/leisaac/policy/service_policy_clients.py
```
Upstream issue: https://github.com/LightwheelAI/leisaac/issues/145

**LeIsaac import fails inside container:** Verify the persistent package dir is mounted:
`ls /workspace/isaaclab-pkgs/.leisaac-installed`. If missing, the auto-install didn't run —
exit the container and run `run-isaaclab.sh` again (it retries on each launch).

## EFS and DCV Issues

**EFS not mounted after instance stop/start:** The `efs-ensure-mount.service` systemd unit
(installed by the bootstrap) retries the mount after `network-online.target`. If you deployed
before this fix was added, mount manually:
```bash
sudo mount -t nfs4 -o nfsvers=4.1 fs-<ID>.efs.us-west-2.amazonaws.com:/ /mnt/efs
```
New deployments use `nofail` in fstab (boot doesn't hang) plus `efs-ensure-mount.service`
(retries up to 10 x 10s after the network is ready).

**DCV "no session available" after stop/start:** For new deployments, `dcv_construct.py`
writes `create-session = true` under `[session-management/automatic-console-session]` in
`/etc/dcv/dcv.conf`, so DCV auto-creates a console session on every server start. For
existing instances, apply manually then restart:
```bash
sudo sed -i '/\[session-management\/automatic-console-session\]/,/\[/{s/^#create-session = true/create-session = true/}' /etc/dcv/dcv.conf
# If the section is absent entirely:
printf '\n[session-management/automatic-console-session]\ncreate-session = true\nowner = ubuntu\n' | sudo tee -a /etc/dcv/dcv.conf
sudo systemctl restart dcvserver
```

## CodeBuild Issues

**N1.6 CodeBuild fails: `cc1plus` killed (OOM):** PyTorch3D's C++ compilation exhausts
CodeBuild LARGE's 15 GB RAM when all 8 ninja workers run in parallel. The fix
(`MAX_JOBS=2` in `N16/Dockerfile`) was applied in commit `41277b3`. If you see
`fatal error: Killed signal terminated program cc1plus`, rebuild — the fix is already in.
