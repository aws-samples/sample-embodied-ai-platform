# Requirements: DCV Workstation Containerization

**Defined:** 2026-03-22
**Core Value:** DCV workstation boots to a fully-configured, ready-to-use IsaacLab environment with minimal setup time and zero manual intervention.

## v1 Requirements

### Bootstrap Simplification

- [ ] **BOOT-01**: Bootstrap script replaces host-installed IsaacSim/Lab (steps 11-15) with NVIDIA container image pull
- [ ] **BOOT-02**: Bootstrap script fits within EC2 UserData 16KB limit after parameter substitution (no S3 asset)
- [ ] **BOOT-03**: `dcv_construct.py` inlines the bootstrap script directly in UserData (removes S3 asset upload/download)
- [x] **BOOT-04**: `versions.py` matrix supports IsaacLab v2.1.1 and v2.3.0 container image tags as CDK parameter

### Container Runtime

- [ ] **CNTR-01**: Bootstrap pulls official `nvcr.io/nvidia/isaac-lab:{version}` image during instance setup
- [ ] **CNTR-02**: Container launches with GPU access (`--gpus all`) and X11 display forwarding to DCV session
- [ ] **CNTR-03**: Helper script on host launches IsaacLab container with correct flags (GPU, display, volumes, caches, EULA)

### Package Persistence

- [x] **PKGS-01**: Host directory `/home/ubuntu/isaaclab-pkgs/` is created at bootstrap and mounted into container
- [x] **PKGS-02**: leisaac is installed via `pip install --target` to the persistent volume on first container launch
- [x] **PKGS-03**: PYTHONPATH is configured in the container so persistent packages are importable without reinstall

### Host Utilities

- [x] **HOST-01**: uv is installed on the host for lightweight Python package management
- [x] **HOST-02**: tensorboard and wandb CLI are available on the host via uv-managed virtual environment

### Bootstrap Ordering & Monitoring

- [ ] **ORCH-01**: Bootstrap executes in order: NVIDIA driver → Docker+toolkit → container pull → EFS mount → uv/tools → DCV desktop (last)
- [ ] **ORCH-02**: Bootstrap sends cfn-signal to CloudFormation on completion (success or failure)
- [ ] **ORCH-03**: Bootstrap writes marker file `/var/lib/dcv-bootstrap/ALL_DONE` on successful completion
- [ ] **ORCH-04**: Existing idempotent state markers (`/var/lib/dcv-bootstrap/*.done`) are preserved for re-run safety

## v2 Requirements

### Extended Container Management

- **CNTR-04**: Docker Compose file for managing IsaacLab container lifecycle (start/stop/restart)
- **CNTR-05**: Multiple container profiles (headless training vs GUI development)

### Advanced Persistence

- **PKGS-04**: EFS-backed package volume for cross-instance persistence
- **PKGS-05**: Pre-cached container images on EFS to avoid re-pulling after instance replacement

## Out of Scope

| Feature | Reason |
|---------|--------|
| Custom Dockerfile / ECR build for IsaacLab | Using NVIDIA pre-built images directly, no custom pipeline needed |
| Runtime version switching between IsaacLab versions | CDK parameter at deploy time; switching requires redeploy |
| Changes to training pipeline (Batch, CodeBuild) | DCV workstation scope only |
| Multi-GPU container support | Single GPU workstation use case |
| Container orchestration (Kubernetes, ECS) | Single EC2 instance, Docker CLI sufficient |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BOOT-01 | Phase 1 | Pending |
| BOOT-02 | Phase 1 | Pending |
| BOOT-03 | Phase 1 | Pending |
| BOOT-04 | Phase 1 | Complete |
| CNTR-01 | Phase 1 | Pending |
| CNTR-02 | Phase 1 | Pending |
| CNTR-03 | Phase 1 | Pending |
| PKGS-01 | Phase 2 | Complete |
| PKGS-02 | Phase 2 | Complete |
| PKGS-03 | Phase 2 | Complete |
| HOST-01 | Phase 2 | Complete |
| HOST-02 | Phase 2 | Complete |
| ORCH-01 | Phase 3 | Pending |
| ORCH-02 | Phase 3 | Pending |
| ORCH-03 | Phase 3 | Pending |
| ORCH-04 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16/16 ✓
- Unmapped: 0

---
*Requirements defined: 2026-03-22*
*Last updated: 2026-03-23 after roadmap creation*
