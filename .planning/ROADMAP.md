# Roadmap: DCV Workstation Containerization

**Project:** DCV Workstation Containerization
**Granularity:** Coarse (3 phases)
**Coverage:** 16/16 v1 requirements mapped

## Phases

- [ ] **Phase 1: Container Foundation** - Replace host-installed IsaacSim/Lab with NVIDIA container, inline bootstrap in UserData
- [ ] **Phase 2: Persistence & Tooling** - Add leisaac volume mount and host utilities (uv, tensorboard, wandb)
- [ ] **Phase 3: Bootstrap Orchestration** - Reorder bootstrap execution, add completion monitoring (cfn-signal + marker)

## Phase Details

### Phase 1: Container Foundation
**Goal**: DCV workstation launches IsaacLab from NVIDIA container instead of host-installed packages, with bootstrap script fitting in UserData

**Depends on**: Nothing (first phase)

**Requirements**: BOOT-01, BOOT-02, BOOT-03, BOOT-04, CNTR-01, CNTR-02, CNTR-03

**Success Criteria** (what must be TRUE):
1. Bootstrap script is under 16KB after parameter substitution and contains no IsaacSim/Lab pip/conda installation
2. `dcv_construct.py` passes bootstrap script directly to UserData without S3 asset upload/download
3. `versions.py` maps IsaacLab v2.1.1 and v2.3.0 to corresponding `nvcr.io/nvidia/isaac-lab` container tags
4. User can SSH to DCV instance and run helper script to launch IsaacLab container with GPU access and X11 forwarding to DCV session
5. User can open IsaacSim GUI applications in the DCV remote desktop session (X11 display working from container)

**Plans:** 2 plans

Plans:
- [x] 01-01-PLAN.md — Redesign versions.py with container image tags
- [x] 01-02-PLAN.md — Rewrite dcv_construct.py inline bootstrap + helper script

**UI hint**: yes

---

### Phase 2: Persistence & Tooling
**Goal**: leisaac packages persist across container restarts via host volume, and lightweight host utilities (uv, tensorboard, wandb) are available

**Depends on**: Phase 1

**Requirements**: PKGS-01, PKGS-02, PKGS-03, HOST-01, HOST-02

**Success Criteria** (what must be TRUE):
1. Directory `/home/ubuntu/isaaclab-pkgs/` exists on host and is mounted into IsaacLab container at a known path
2. User can install leisaac via `pip install --target /mounted/path leisaac[gr00t]` in first container launch
3. User can restart the container and import leisaac without reinstallation (PYTHONPATH includes persistent volume)
4. User can run `tensorboard --logdir /mnt/efs/...` from host (not container) to visualize training metrics
5. User can run `wandb login` and `wandb sync` from host for W&B CLI operations

**Plans:** 1 plan

Plans:
- [x] 02-01-PLAN.md — Add persistent volume mount, leisaac auto-install, and host utilities (uv, tensorboard, wandb)

---

### Phase 3: Bootstrap Orchestration
**Goal**: Bootstrap executes in optimal order (environment first, DCV last) and signals completion to CloudFormation and local marker

**Depends on**: Phase 2

**Requirements**: ORCH-01, ORCH-02, ORCH-03, ORCH-04

**Success Criteria** (what must be TRUE):
1. Bootstrap script executes steps in order: NVIDIA driver → Docker+toolkit → container pull → EFS mount → uv/tools → DCV desktop (last)
2. CloudFormation stack shows CREATE_COMPLETE only after receiving cfn-signal from bootstrap (success or failure)
3. File `/var/lib/dcv-bootstrap/ALL_DONE` exists after successful bootstrap completion
4. User can re-run bootstrap script (simulating instance restart) and all idempotent state markers prevent duplicate work
5. User who logs into DCV for the first time sees a fully configured environment with no reboot required

**Plans:** 1 plan

Plans:
- [x] 03-01-PLAN.md — Reorder bootstrap, add DCV to add_commands, cfn-signal, CreationPolicy, ALL_DONE marker

**UI hint**: yes

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Container Foundation | 0/2 | Planning complete | - |
| 2. Persistence & Tooling | 0/1 | Planning complete | - |
| 3. Bootstrap Orchestration | 0/1 | Planning complete | - |

---
*Roadmap created: 2026-03-23*
*Last updated: 2026-03-24 after Phase 3 planning*
