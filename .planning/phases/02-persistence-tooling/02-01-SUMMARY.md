---
phase: 02-persistence-tooling
plan: 01
subsystem: infra
tags: [docker, volumes, pythonpath, leisaac, uv, tensorboard, wandb, bootstrap]

# Dependency graph
requires:
  - phase: 01-container-foundation
    provides: Inline UserData bootstrap with helper script and container pull
provides:
  - Persistent isaaclab-pkgs volume mount for container package survival
  - leisaac auto-install with marker-file idempotency guard
  - PYTHONPATH injection for persistent package imports
  - Host utilities (uv, tensorboard, wandb) via venv with PATH update
affects: [03-bootstrap-orchestration]

# Tech tracking
tech-stack:
  added: [uv, tensorboard, wandb]
  patterns: [marker-file idempotency, pip-install-target, host-venv-for-cli-tools]

key-files:
  created: []
  modified: [dcv/dcv_construct.py]

key-decisions:
  - "leisaac auto-install uses marker file (.leisaac-installed) in persistent volume for idempotency"
  - "PYTHONPATH set via docker run -e flag, appending to container default"
  - "uv installed as ubuntu user (not root) so PATH works without sudo"
  - "Host venv at /home/ubuntu/.venv with tensorboard+wandb; PATH added to .bashrc idempotently"

patterns-established:
  - "Persistent package volume: host dir mounted into container, pip --target installs survive restarts"
  - "Host CLI tools via uv venv: lightweight, no conda/miniforge needed"

requirements-completed: [PKGS-01, PKGS-02, PKGS-03, HOST-01, HOST-02]

# Metrics
duration: 2min
completed: 2026-03-24
---

# Phase 2 Plan 1: Persistence & Tooling Summary

**Persistent leisaac volume mount with auto-install guard, PYTHONPATH injection, and host uv/tensorboard/wandb bootstrap**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-24T04:26:56Z
- **Completed:** 2026-03-24T04:28:50Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Helper script now mounts `/home/ubuntu/isaaclab-pkgs` into container at `/workspace/isaaclab-pkgs` for persistent package storage
- leisaac auto-installs on first container launch via marker-file guard (`.leisaac-installed`), version pinned from `versions.py`
- PYTHONPATH set in container environment so persistent packages are importable without reinstall
- Host bootstrap installs uv, creates venv with tensorboard + wandb, and adds `.venv/bin` to PATH

## Task Commits

Each task was committed atomically:

1. **Task 1: Update helper script with persistent volume mount, PYTHONPATH, and leisaac auto-install** - `47ec7c3` (feat)
2. **Task 2: Add bootstrap commands for uv, host venv, tensorboard, wandb, and PATH** - `ba77063` (feat)

## Files Created/Modified
- `dcv/dcv_construct.py` - Updated helper script heredoc with volume mount, PYTHONPATH, leisaac auto-install; added bootstrap commands for host utilities

## Decisions Made
- Used marker file approach (`.leisaac-installed`) for leisaac idempotency guard -- faster than import check, aligns with existing bootstrap state marker pattern
- leisaac version injected via CDK f-string from `versions.py` (no hardcoded version)
- Removed old `props.leisaac_enabled` conditional install block; auto-install always runs on first launch (controlled by marker file)
- uv installed as ubuntu user via `su - ubuntu -c "..."` to ensure correct PATH ownership
- `.bashrc` PATH append uses `grep -q` guard for idempotency

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Persistent package volume and host utilities are in place
- Ready for Phase 3 (bootstrap orchestration) to finalize step ordering and cfn-signal

## Self-Check: PASSED

---
*Phase: 02-persistence-tooling*
*Completed: 2026-03-24*
