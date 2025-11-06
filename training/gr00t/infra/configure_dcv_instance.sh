#!/bin/bash
# Bootstrap script for Amazon DCV on Ubuntu 22.04 with IsaacLab and EFS
# - Robust logging to /var/log/dcv-bootstrap.log and concise step summary
# - Retries for apt and network operations
# - Non-fatal GUI tweaks; separation of critical vs optional steps
#
# USAGE GUIDE (read this if something failed)
# ============================================================
# Where to look:
#   - Summary (quick, one line per step): /var/log/dcv-bootstrap.summary
#       Entries are prefixed with one of: STEP_OK, STEP_WARN, STEP_FAIL
#         * STEP_OK  : step completed successfully
#         * STEP_WARN: step failed but was non-fatal and intentionally ignored
#         * STEP_FAIL: critical step failed; see detailed log
#   - Detailed log: /var/log/dcv-bootstrap.log
#   - Auto session service log: journalctl -u auto-dcv.service -e --no-pager
#
# How to interpret and fix:
#   1) Session Manager/EC2 Instance Connect/SSH into the instance and review the summary:
#        sudo cat /var/log/dcv-bootstrap.summary
#   2) For each STEP_FAIL, open the detailed log around the time it ran:
#        sudo less +G /var/log/dcv-bootstrap.log
#   3) Fix the underlying issue (e.g., networking, package mirror, permissions).
#
# Step-specific log viewing commands:
#   - View logs for a specific step:
#        sudo grep -A 50 "== START: <step-name> ==" /var/log/dcv-bootstrap.log
#        sudo grep -A 100 -B 10 "== FAIL: <step-name> ==" /var/log/dcv-bootstrap.log
#   - Examples:
#        sudo grep -A 50 "== START: install-nice-dcv ==" /var/log/dcv-bootstrap.log
#   - View step completion status:
#        ls -la /var/lib/dcv-bootstrap/
#   - Check specific step state:
#        test -f "/var/lib/dcv-bootstrap/<step-name>.done" && echo "Step completed" || echo "Step not done"
#
# Re-running only the failed steps (idempotent):
#   - This script creates state markers in: /var/lib/dcv-bootstrap/<step-name>.done
#   - Re-running the entire script will SKIP steps already marked done.
#   - To force re-run a specific step, delete its marker and re-run the script:
#        sudo rm "/var/lib/dcv-bootstrap/<step-name>.done"
#        sudo bash /var/lib/cloud/instance/scripts/part-001
#     Examples:
#        sudo rm "/var/lib/dcv-bootstrap/install-nice-dcv.done" && \
#          sudo bash /var/lib/cloud/instance/scripts/part-001
#        sudo rm "/var/lib/dcv-bootstrap/install-desktop.done" && \
#          sudo bash /var/lib/cloud/instance/scripts/part-001
#   - Note: The path /var/lib/cloud/instance/scripts/part-001 is the cloud-init
#           copy of this user-data. If unavailable, paste the script or
#           store it as /usr/local/sbin/dcv-bootstrap.sh and execute that.
#
# Common checks:
#   - DCV server status:    sudo systemctl status dcvserver --no-pager
#   - DCV sessions:         sudo dcv list-sessions
#   - Auto session service: sudo systemctl status auto-dcv.service --no-pager
#                           sudo journalctl -u auto-dcv.service -e --no-pager
#   - EFS mount:            mount | grep ' /mnt/efs '  (should show type efs,tls)
#                           sudo tail -n 200 /var/log/amazon/efs/mount.log
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

# 3) Desktop + GDM (Wayland off) (critical)
must "install-desktop" '
  apt_install ubuntu-desktop gdm3 dbus-x11
  sed -i "s/^#\\(WaylandEnable=false\\)/\\1/" /etc/gdm3/custom.conf || true
'

# 4) Remove GNOME first-run wizard (non-fatal)
try_step "disable-gnome-initial-setup" '
  apt-get remove --purge -yq gnome-initial-setup || true
  sed -i "s/^X-GNOME-Autostart-enabled=true/X-GNOME-Autostart-enabled=false/" /etc/xdg/autostart/gnome-initial-setup-first-login.desktop || true
  systemctl restart gdm3 || true
'

# 5) Amazon DCV server (critical)
must "install-nice-dcv" '
  DCV_URL="https://d1uj6qtbmh3dt5.cloudfront.net/2024.0/Servers/nice-dcv-2024.0-19030-ubuntu2204-x86_64.tgz"
  cd /tmp
  wget -q "$DCV_URL" -O /tmp/dcv.tgz
  tar -xzf /tmp/dcv.tgz -C /tmp
  cd /tmp/nice-dcv-2024.0-19030-ubuntu2204-x86_64
  apt_install libpulse-mainloop-glib0 libpulse0 libgstreamer-plugins-base1.0-0 libcrack2 libxcb-damage0 libxcb-xkb1 libxcb-xtest0 keyutils
  apt_install alsa-utils
  apt-get install -yq ./*.deb
  usermod -aG video dcv || true
  systemctl enable dcvserver
  systemctl restart dcvserver
'

# 6) DCV config (non-fatal)
try_step "configure-dcv" '
  sed -i "/^\\[display\\]/a max-head-resolution = \"(4096, 2160)\"\\nweb-client-max-head-resolution = \"(4096, 4096)\"" /etc/dcv/dcv.conf || true
  if ! grep -q "\\[display/linux\\]" /etc/dcv/dcv.conf; then
    cat <<EOF >>/etc/dcv/dcv.conf
[display/linux]
disable-local-console=false
EOF
  fi
  systemctl restart dcvserver || true
'

# 7) Auto-create DCV virtual session service (critical)
must "install-auto-dcv-service" install_auto_dcv_service

# 8) AWS CLI v2 (critical)
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

# 9) amazon-efs-utils (critical for EFS mount)
must "install-efs-utils" '
  apt_update
  apt_install git binutils rustc cargo pkg-config libssl-dev ca-certificates cmake golang-go
  rm -rf /tmp/efs-utils
  git clone --branch v2.4.0 --single-branch https://github.com/aws/efs-utils /tmp/efs-utils
  cd /tmp/efs-utils
  ./build-deb.sh
  apt-get install -yq /tmp/efs-utils/build/amazon-efs-utils*deb
'

# For using IsaacSim/Lab in a container
# 10) Docker + NVIDIA Container Toolkit (non-fatal)
try_step "install-docker-nvidia-toolkit" '
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

# Alternatively, steps 11-15 for using IsaacSim/Lab on host
# 11) Miniforge (critical for Python/pip)
must "install-miniforge" '
  if [[ ! -x /opt/conda/bin/conda ]]; then
    TMP_INSTALLER="/tmp/Miniforge3.sh"
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$TMP_INSTALLER"
    bash "$TMP_INSTALLER" -b -p /opt/conda
    ln -sf /opt/conda/bin/conda /usr/local/bin/conda
  fi
  chown -R ubuntu:ubuntu /opt/conda
'

# 12) Create conda env and configure default activation (critical for Python/pip)
must "create-conda-env-isaac" '
  /opt/conda/bin/conda info >/dev/null
  if ! /opt/conda/bin/conda env list | awk "{print $1}" | grep -q "^isaac$"; then
    su - ubuntu -c "/opt/conda/bin/conda create -y -n isaac python=3.10"
  fi
  /opt/conda/bin/conda config --system --set auto_activate_base false
  su - ubuntu -c "/opt/conda/bin/conda init bash"
  if ! su - ubuntu -c "grep -q 'conda activate isaac' ~/.bashrc"; then
    echo "conda activate isaac" >> /home/ubuntu/.bashrc
  fi
'

# 13) PyTorch + IsaacSim via pip (non-fatal)
try_step "install-pytorch-isaacsim" '
  su - ubuntu -c "/opt/conda/bin/conda install -y -n isaac -c 'nvidia/label/cuda-11.8.0' cuda-toolkit"
  su - ubuntu -c "/opt/conda/bin/conda run -n isaac python -V"
  su - ubuntu -c "/opt/conda/bin/conda run -n isaac python -m pip install --upgrade pip"
  su - ubuntu -c "/opt/conda/bin/conda run -n isaac pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118"
  su - ubuntu -c "/opt/conda/bin/conda run -n isaac pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com"
'

# 14) Isaac Lab (non-fatal)
try_step "install-isaaclab" '
  apt_install cmake build-essential
  cd /home/ubuntu
  if [[ ! -d IsaacLab ]]; then
    su - ubuntu -c "git clone https://github.com/isaac-sim/IsaacLab.git"
  fi
  su - ubuntu -c "cd /home/ubuntu/IsaacLab && git fetch --tags || true"
  su - ubuntu -c "cd /home/ubuntu/IsaacLab && git checkout v2.1.1 || true"
  su - ubuntu -c "cd /home/ubuntu/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES /opt/conda/bin/conda run -n isaac ./isaaclab.sh --install || true"
'

# 15) leisaac (install + assets, non-fatal)
# by default use a stable version as of 1 Sep 2025, remove the git checkout line to use the latest version
try_step "install-leisaac" '
  cd /home/ubuntu
  if [[ ! -d leisaac ]]; then
    su - ubuntu -c "git clone https://github.com/LightwheelAI/leisaac.git"
  fi 
  su - ubuntu -c "cd /home/ubuntu/leisaac && git fetch --tags || true"
  su - ubuntu -c "cd /home/ubuntu/leisaac && git checkout v0.2.0 || true"
  su - ubuntu -c "cd /home/ubuntu/leisaac && /opt/conda/bin/conda run -n isaac pip install -e \"source/leisaac[gr00t,lerobot-async]\" || true"
  su - ubuntu -c "cd /home/ubuntu/leisaac && /opt/conda/bin/conda run -n isaac pip install pynput pyserial deepdiff feetech-servo-sdk || true"

  mkdir -p /home/ubuntu/leisaac/assets/robots
  mkdir -p /home/ubuntu/leisaac/assets/scenes

  apt_update || true
  apt_install unzip wget || true

  ROBOT_URL="https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"
  SCENE_ZIP_URL="https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip"

  if [[ ! -f /home/ubuntu/leisaac/assets/robots/so101_follower.usd ]]; then
    su - ubuntu -c "wget -qO /home/ubuntu/leisaac/assets/robots/so101_follower.usd \"${ROBOT_URL}\""
  fi

  if [[ ! -d /home/ubuntu/leisaac/assets/scenes/kitchen_with_orange ]]; then
    TMP_ZIP="$(mktemp -u /tmp/kitchen_with_orange.XXXXXX.zip)"
    wget -qO "$TMP_ZIP" "${SCENE_ZIP_URL}"
    unzip -oq "$TMP_ZIP" -d /home/ubuntu/leisaac/assets/scenes
    rm -f "$TMP_ZIP"
  fi

  chown -R ubuntu:ubuntu /home/ubuntu/leisaac/assets
'

# 16) Firefox browser (non-fatal)
try_step "install-firefox" '
  apt_update
  apt_install firefox
'

# Final: summary and optional reboot
log "==== SUMMARY (also in $SUMMARY) ===="
cat "$SUMMARY" || true

if [[ $FAILURES -gt 0 ]]; then
  log "One or more critical steps failed ($FAILURES). Not rebooting automatically."
else
  log "All critical steps OK. Scheduling a reboot to finalize configuration."
  nohup shutdown -r +1 "Rebooting to finalize configuration" >/dev/null 2>&1 &
fi
