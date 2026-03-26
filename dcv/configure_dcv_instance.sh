#!/bin/bash
# Bootstrap script for Amazon DCV on Ubuntu 22.04 — static infrastructure steps.
# Installs: NVIDIA drivers, Docker+toolkit, AWS CLI, EFS utils, cfn-bootstrap.
# DCV desktop is installed by dcv_construct.py add_commands.
#
# TROUBLESHOOTING
# ============================================================
# Logs:
#   Summary (one line per step):  sudo cat /var/log/dcv-bootstrap.summary
#     Prefixes: STEP_OK (success), STEP_WARN (non-fatal), STEP_FAIL (critical)
#   Detailed log:                 sudo less +G /var/log/dcv-bootstrap.log
#   Auto-DCV session service:     sudo journalctl -u auto-dcv.service -e --no-pager
#
# Step-specific debugging:
#   View logs for a step:         sudo grep -A 50 "== START: <step-name> ==" /var/log/dcv-bootstrap.log
#   View failure context:         sudo grep -A 100 -B 10 "== FAIL: <step-name> ==" /var/log/dcv-bootstrap.log
#   Check step completion:        ls -la /var/lib/dcv-bootstrap/
#   Check specific step state:    test -f "/var/lib/dcv-bootstrap/<step-name>.done" && echo "done"
#
# Re-running failed steps (idempotent):
#   State markers live in /var/lib/dcv-bootstrap/<step>.done — completed steps are skipped.
#   To force re-run a step:
#     sudo rm "/var/lib/dcv-bootstrap/<step-name>.done"
#     sudo bash /var/lib/cloud/instance/scripts/part-001
#
# Common checks:
#   DCV server:    sudo systemctl status dcvserver --no-pager
#   DCV sessions:  sudo dcv list-sessions
#   EFS mount:     mount | grep ' /mnt/efs '
# ============================================================

set -Eeuo pipefail

LOG="/var/log/dcv-bootstrap.log"
SUMMARY="/var/log/dcv-bootstrap.summary"
STATE_DIR="/var/lib/dcv-bootstrap"
mkdir -p "$STATE_DIR"

# Timestamped logging to file and syslog
exec > >(awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0 }' | tee -a "$LOG" | logger -t user-data -s 2>/dev/null) 2>&1

CURRENT_STEP=""
FAILURES=0
export DEBIAN_FRONTEND=noninteractive

on_error() {
  local line="$1" cmd="$2" rc="$3"
  echo "ERROR: step='$CURRENT_STEP' line=$line rc=$rc cmd='$cmd'"
  echo "STEP_FAIL:${CURRENT_STEP}:line=${line}:rc=${rc}:cmd=${cmd}" >> "$SUMMARY"
}
trap 'on_error "$LINENO" "$BASH_COMMAND" "$?"' ERR

log() { echo "$*"; }
mark_done() { touch "${STATE_DIR}/$1.done"; }
is_done() { [[ -f "${STATE_DIR}/$1.done" ]]; }

retry() {
  local tries="${3:-5}" delay="${4:-5}"
  for ((i=1;i<=tries;i++)); do
    if eval "$1"; then return 0; fi
    echo "Retry $i/$tries for: $2"
    sleep "$delay"
  done
  return 1
}

must() {
  local desc="$1"; shift
  CURRENT_STEP="$desc"
  if is_done "$desc"; then
    log "SKIP (done): $desc"; return 0
  fi
  log "== START: $desc =="
  if eval "$@"; then
    log "== OK: $desc =="
    echo "STEP_OK:${desc}" >> "$SUMMARY"
    mark_done "$desc"
    return 0
  else
    FAILURES=$((FAILURES+1))
    log "== FAIL: $desc =="
    return 1
  fi
}

try_step() {
  local desc="$1"; shift
  CURRENT_STEP="$desc"
  if is_done "$desc"; then
    log "SKIP (done): $desc"; return 0
  fi
  log "== START (non-fatal): $desc =="
  set +e
  eval "$@"
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    log "== OK: $desc =="
    echo "STEP_OK:${desc}" >> "$SUMMARY"
    mark_done "$desc"
  else
    log "== WARN (ignored rc=${rc}): $desc =="
    echo "STEP_WARN:${desc}:rc=${rc}" >> "$SUMMARY"
  fi
  return 0
}

apt_ready() {
  ! fuser /var/lib/dpkg/lock >/dev/null 2>&1 && \
  ! fuser /var/lib/apt/lists/lock >/dev/null 2>&1 && \
  ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1
}
apt_update() {
  retry "apt-get update -yq" "apt update" 6 8
}
apt_install() {
  local pkgs="$*"
  retry "apt-get install -yq --no-install-recommends $pkgs" "apt install: $pkgs" 6 8
}

# Auto-create DCV virtual session service (function definition)
install_auto_dcv_service() {
  cat >/usr/local/bin/auto-create-virtual-dcv.sh <<'EOF'
#!/bin/bash
set -Eeuo pipefail
LOG="/var/log/auto-dcv.log"
exec > >(awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0 }' | tee -a "${LOG}") 2>&1

SESSION_ID="isaac-workspace"
owner="ubuntu"

attempts=0
until systemctl is-active --quiet dcvserver; do
  attempts=$((attempts+1))
  echo "Waiting for dcvserver (attempt ${attempts})..."
  sleep 3
done

# xhost comes from x11-xserver-utils; best-effort
command -v xhost >/dev/null 2>&1 || apt-get update -yq || true
command -v xhost >/dev/null 2>&1 || apt-get install -yq --no-install-recommends x11-xserver-utils || true

if ! dcv list-sessions | grep -q "^Session: ${SESSION_ID}"; then
  echo "Creating DCV virtual session ${SESSION_ID} ..."
  dcv create-session "${SESSION_ID}" --type virtual --owner "${owner}" --name "Isaac Sim"
else
  echo "DCV session ${SESSION_ID} already exists."
fi

# Optional: GUI tweaks (non-fatal)
sudo -u "${owner}" dbus-launch gsettings set org.gnome.desktop.lockdown disable-lock-screen true || true
sudo -u "${owner}" dbus-launch gsettings set org.gnome.desktop.interface gtk-theme Yaru-dark || true
sudo -u "${owner}" dbus-launch gsettings set org.gnome.desktop.interface color-scheme prefer-dark || true
EOF
  chmod +x /usr/local/bin/auto-create-virtual-dcv.sh

  cat >/etc/systemd/system/auto-dcv.service <<'EOF'
[Unit]
Description=Auto-create Amazon DCV virtual session
Wants=network-online.target
After=dcvserver.service network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/auto-create-virtual-dcv.sh
RemainAfterExit=yes
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable auto-dcv.service
  systemctl start auto-dcv.service
}

# 0) Baseline and password (early)
must "baseline-update" '
  while ! apt_ready; do sleep 3; done
  apt_update
  apt_install ca-certificates curl wget gnupg lsb-release jq
'
must "set-ubuntu-password" '
  test -n "__PASSWORD__"
  echo "ubuntu:__PASSWORD__" | chpasswd
  passwd -S ubuntu
'

# 1) Disable nouveau (non-fatal)
try_step "disable-nouveau" '
  cat >/etc/modprobe.d/blacklist-nouveau.conf <<EOF
blacklist nouveau
options nouveau modeset=0
EOF
  sed -i '\''s/GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="rdblacklist=nouveau /'\'' /etc/default/grub || true
  update-initramfs -u || true
  update-grub || true
'

# 2) NVIDIA driver (critical)
must "install-nvidia-driver" '
  apt_install ubuntu-drivers-common
  ubuntu-drivers autoinstall
'

# Load NVIDIA kernel module (nouveau already blacklisted above)
if ! is_done "nvidia-driver-loaded"; then
  modprobe nvidia 2>/dev/null || true
  if nvidia-smi > /dev/null 2>&1; then
    log "nvidia-smi OK -- driver loaded via modprobe"
    mark_done "nvidia-driver-loaded"
  else
    log "WARNING: nvidia-smi failed after modprobe -- driver will load after next reboot"
    echo "STEP_WARN:nvidia-driver-loaded:nvidia-smi-failed-will-retry-after-reboot" >> "$SUMMARY"
  fi
fi

# 3) Docker + NVIDIA Container Toolkit (critical for container-based workflow)
must "install-docker-nvidia-toolkit" '
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sh /tmp/get-docker.sh
  systemctl enable docker
  systemctl start docker
  usermod -aG docker ubuntu || true
  install -m 0755 -d /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  apt_update
  apt_install nvidia-container-toolkit
  systemctl restart docker
'

# 4) AWS CLI v2 (critical)
must "install-aws-cli-v2" '
  apt_update
  apt_install unzip
  TMP_DIR="$(mktemp -d)"
  cd "$TMP_DIR"
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  unzip -q awscliv2.zip
  ./aws/install --update
  aws --version
'

# 5) amazon-efs-utils (critical for EFS mount)
must "install-efs-utils" '
  apt_update
  apt_install git binutils rustc cargo pkg-config libssl-dev ca-certificates cmake golang-go build-essential
  rm -rf /tmp/efs-utils
  git clone --branch v2.4.0 --single-branch https://github.com/aws/efs-utils /tmp/efs-utils
  cd /tmp/efs-utils
  ./build-deb.sh
  apt-get install -yq /tmp/efs-utils/build/amazon-efs-utils*deb
'

# 6) Firefox browser (non-fatal)
try_step "install-firefox" '
  apt_update
  apt_install firefox
'

# 7) cfn-bootstrap (critical for CloudFormation signaling)
must "install-cfn-bootstrap" '
  apt_install python3-pip
  pip3 install https://s3.amazonaws.com/cloudformation-examples/aws-cfn-bootstrap-py3-latest.tar.gz
  test -x /usr/local/bin/cfn-signal
'

# DCV desktop, container pull, host tools, EFS mount, cfn-signal, and ALL_DONE
# are handled dynamically by dcv_construct.py add_commands.
