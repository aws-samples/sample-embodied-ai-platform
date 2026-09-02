#!/usr/bin/env bash
# Eval-sweep helper — mutates success_stage on the FSx-mounted task cfg
# in-place, with backup and restore. Runs INSIDE a k8s pod that has FSx mounted.
#
# Usage (from control host, via kubectl exec on a running pod):
#   kubectl exec -n training <pod> -- bash /path/on/fsx/step-a-patch-success-stage.sh <backup|set N|restore>
#
# Or SSH/SSM to the underlying node and run against the node's FSx mount.
#
# Assumes:
#   FSX_MOUNT=/mnt/fsx
#   TARGET=${FSX_MOUNT}/workflows/rheo/scripts/simulation/tasks/assemble_trocar/g1_assemble_trocar_env_cfg.py
#   BACKUP=${FSX_MOUNT}/scratch/step-a/g1_assemble_trocar_env_cfg.py.orig
set -euo pipefail

FSX_MOUNT="${FSX_MOUNT:-/mnt/fsx}"
TARGET="${FSX_MOUNT}/workflows/rheo/scripts/simulation/tasks/assemble_trocar/g1_assemble_trocar_env_cfg.py"
BACKUP_DIR="${FSX_MOUNT}/scratch/step-a"
BACKUP="${BACKUP_DIR}/g1_assemble_trocar_env_cfg.py.orig"

usage() {
    echo "Usage: $0 <backup|set N|restore|show>"
    echo ""
    echo "  backup   — copy TARGET to BACKUP (must not already exist; refuses to overwrite)"
    echo "  set N    — set success_stage=N in TARGET (N in 1..4); BACKUP must exist"
    echo "  restore  — copy BACKUP back to TARGET (leaves BACKUP intact)"
    echo "  show     — grep current success_stage value in TARGET and BACKUP"
    echo ""
    echo "TARGET=$TARGET"
    echo "BACKUP=$BACKUP"
    exit 1
}

require_target() {
    if [ ! -f "$TARGET" ]; then
        echo "ERROR: TARGET not found: $TARGET" >&2
        exit 2
    fi
}

case "${1:-}" in
    backup)
        require_target
        if [ -f "$BACKUP" ]; then
            echo "ERROR: BACKUP already exists at $BACKUP — refusing to overwrite. Restore or remove first." >&2
            exit 3
        fi
        mkdir -p "$BACKUP_DIR"
        cp -v "$TARGET" "$BACKUP"
        grep -n 'success_stage' "$BACKUP"
        ;;
    set)
        require_target
        if [ ! -f "$BACKUP" ]; then
            echo "ERROR: BACKUP not found at $BACKUP — run '$0 backup' first" >&2
            exit 4
        fi
        N="${2:-}"
        case "$N" in
            1|2|3|4) ;;
            *) echo "ERROR: invalid stage N=$N (must be 1..4)" >&2; exit 5 ;;
        esac
        # Restore-from-backup first, then patch. This makes 'set' idempotent
        # regardless of what state TARGET was in.
        cp "$BACKUP" "$TARGET"
        sed -i "s/\"success_stage\": [1-4]/\"success_stage\": ${N}/" "$TARGET"
        echo "Set success_stage=${N} in $TARGET"
        grep -n 'success_stage' "$TARGET"
        ;;
    restore)
        if [ ! -f "$BACKUP" ]; then
            echo "ERROR: BACKUP not found at $BACKUP" >&2
            exit 6
        fi
        cp -v "$BACKUP" "$TARGET"
        grep -n 'success_stage' "$TARGET"
        ;;
    show)
        echo "== TARGET: $TARGET =="
        [ -f "$TARGET" ] && grep -n 'success_stage' "$TARGET" || echo "(missing)"
        echo "== BACKUP: $BACKUP =="
        [ -f "$BACKUP" ] && grep -n 'success_stage' "$BACKUP" || echo "(missing)"
        ;;
    *)
        usage
        ;;
esac
