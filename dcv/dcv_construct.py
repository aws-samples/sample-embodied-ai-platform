"""L3 construct for GPU-accelerated DCV workstation with containerized IsaacLab.

Bootstrap flow (Phase 3):
- Static prerequisites (drivers, Docker, CLI, EFS, cfn-bootstrap) from configure_dcv_instance.sh
- Dynamic application steps (container, tools, EFS mount, DCV, cfn-signal) via add_commands
"""
import os
import re
from typing import Optional
from aws_cdk import aws_ec2 as ec2, aws_iam as iam, Stack, CfnOutput, CfnCreationPolicy, CfnResourceSignal
from constructs import Construct
try:
    from .versions import validate_version_config
except ImportError:
    from versions import validate_version_config


class DcvWorkstationProps:
    """Properties for DcvWorkstation construct.

    Attributes:
        vpc: Optional direct VPC object reference (for integrated mode)
        vpc_id: Optional VPC ID for lookup (for standalone mode)
        instance_type: EC2 instance type (default: g6.4xlarge). Must be a GPU instance type.
        isaac_sim_version: IsaacSim version (default: 5.1.0)
        isaac_lab_version: IsaacLab version (default: v2.3.0)
        python_version: Python version (kept for backwards compatibility, unused)
        efs_id: Optional EFS file system ID (for persistent storage mode)
        efs_sg_id: Optional EFS security group ID (required with efs_id)
        leisaac_enabled: Enable leisaac installation (default: False)
        availability_zone: Optional AZ to constrain subnet selection (e.g. "us-west-2b")
    """

    def __init__(
        self,
        vpc: Optional[ec2.IVpc] = None,
        vpc_id: Optional[str] = None,
        instance_type: str = "g6.4xlarge",
        isaac_sim_version: str = "5.1.0",
        isaac_lab_version: str = "v2.3.0",
        python_version: Optional[str] = None,
        efs_id: Optional[str] = None,
        efs_sg_id: Optional[str] = None,
        leisaac_enabled: bool = False,
        availability_zone: Optional[str] = None,
    ):
        self.vpc = vpc
        self.vpc_id = vpc_id
        self.instance_type = instance_type
        self.isaac_sim_version = isaac_sim_version
        self.isaac_lab_version = isaac_lab_version
        self.python_version = python_version
        self.efs_id = efs_id
        self.efs_sg_id = efs_sg_id
        self.leisaac_enabled = leisaac_enabled
        self.availability_zone = availability_zone


class DcvWorkstation(Construct):
    """L3 construct for GPU-accelerated DCV workstation.

    This construct provides a reusable DCV workstation that can be used in two modes:
    1. Standalone: Deploy independently with auto-created or looked-up VPC
    2. Integrated: Import into another CDK app and provide VPC reference

    Bootstrap flow:
    - Static prerequisites (drivers, Docker, CLI, EFS, cfn-bootstrap) from configure_dcv_instance.sh
    - Dynamic application steps (container, tools, EFS mount, DCV, cfn-signal) via add_commands
    - CloudFormation CreationPolicy waits for cfn-signal before marking stack CREATE_COMPLETE
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        props: DcvWorkstationProps,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # VALIDATE VERSION COMPATIBILITY (fail fast at synth time)
        try:
            version_config = validate_version_config(
                props.isaac_sim_version,
                props.isaac_lab_version
            )
        except ValueError as e:
            raise ValueError(
                f"Invalid version configuration for DcvWorkstation '{construct_id}': {e}"
            )

        # Validate EFS parameters (both required or both None)
        if (props.efs_id is not None) != (props.efs_sg_id is not None):
            raise ValueError(
                "Both efs_id and efs_sg_id must be provided together, or both must be None. "
                f"Got efs_id={props.efs_id}, efs_sg_id={props.efs_sg_id}"
            )

        # Extract version config fields
        container_image = version_config["container_image"]
        dcv_version_build = version_config["dcv"]
        dcv_version, dcv_build = dcv_version_build.split("-")
        leisaac_commit = version_config.get("leisaac", "d2cbfd2e33517f2094e1904ff817aa17de6e8939")

        # Import EFS file system if provided
        efs_fs = None
        if props.efs_id and props.efs_sg_id:
            from aws_cdk import aws_efs as efs
            efs_sg = ec2.SecurityGroup.from_security_group_id(
                self, "EFSSecurityGroup", props.efs_sg_id, mutable=True
            )
            efs_fs = efs.FileSystem.from_file_system_attributes(
                self,
                "ImportedEFS",
                file_system_id=props.efs_id,
                security_group=efs_sg,
            )

        # VPC resolution with three modes:
        # 1. Direct VPC provided - use it
        # 2. VPC ID provided - lookup via from_lookup()
        # 3. Neither provided - auto-create minimal VPC

        if props.vpc is not None:
            # Mode 1: Direct VPC reference (integrated mode)
            self._vpc = props.vpc
        elif props.vpc_id is not None:
            # Mode 2: VPC lookup by ID (standalone mode with existing VPC)
            self._vpc = ec2.Vpc.from_lookup(
                self,
                "LookupVpc",
                vpc_id=props.vpc_id,
            )
        else:
            # Mode 3: Auto-create minimal VPC (standalone mode, no prerequisites)
            self._vpc = ec2.Vpc(
                self,
                "AutoVpc",
                max_azs=2,
                nat_gateways=1,
                subnet_configuration=[
                    ec2.SubnetConfiguration(
                        name="Public",
                        subnet_type=ec2.SubnetType.PUBLIC,
                        cidr_mask=24,
                    ),
                    ec2.SubnetConfiguration(
                        name="Private",
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                        cidr_mask=24,
                    ),
                ],
            )

        # IAM Role for EC2 instance
        # Grants S3 read access for datasets, SSM for troubleshooting, ECR for container workflows
        self._instance_role = iam.Role(
            self, "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryPowerUser"),
            ],
        )

        # ================================================================
        # UserData: Static bootstrap from .sh file + dynamic add_commands
        # ================================================================

        # Read the static bootstrap script and inject parameters
        user_data_path = os.path.join(os.path.dirname(__file__), "configure_dcv_instance.sh")
        with open(user_data_path, "r") as f:
            bootstrap_script = f.read()

        password = f"dcv{Stack.of(self).account}"
        bootstrap_script = bootstrap_script.replace("__PASSWORD__", password)
        bootstrap_script = bootstrap_script.replace("__DCV_VERSION__", dcv_version)
        bootstrap_script = bootstrap_script.replace("__DCV_BUILD__", dcv_build)

        # Validate no unresolved placeholders
        unresolved = re.findall(r'__[A-Z_]+__', bootstrap_script)
        if unresolved:
            raise ValueError(
                f"Unresolved placeholders in UserData script: {unresolved}. "
                "This indicates a bug in version replacement logic."
            )

        # Inline the bootstrap script directly into UserData (no S3 asset needed —
        # the trimmed script fits within the 16KB UserData limit)
        self._user_data = ec2.UserData.for_linux()
        # Skip the shebang — CDK's for_linux() already adds #!/bin/bash
        lines = bootstrap_script.split('\n')
        if lines and lines[0].startswith('#!'):
            lines = lines[1:]
        self._user_data.add_commands(*lines)

        # ================================================================
        # Dynamic steps: container pull, helper script, leisaac
        # These use CDK token resolution (f-strings with props) so they
        # must be in add_commands, not in the static .sh file.
        # ================================================================

        # Pull the NVIDIA IsaacLab container from NGC
        self._user_data.add_commands(
            '# === Container Setup (dynamic, from CDK) ===',
            f'must "pull-isaaclab-container" "docker pull {container_image}"',
        )

        # Create /usr/local/bin/run-isaaclab.sh helper script
        # This script wraps `docker run` with GPU, X11, EULA, cache volume mounts,
        # persistent package volume, leisaac auto-install, and LeIsaac asset download.
        # NOTE: The heredoc is emitted directly (not inside must '...') because
        # nested single quotes break shell parsing and cause $VAR expansion
        # under set -u from the outer bootstrap script.
        self._user_data.add_commands(
            f"cat > /usr/local/bin/run-isaaclab.sh << 'HELPER_EOF'",
            '#!/bin/bash',
            'set -euo pipefail',
            f'CONTAINER_IMAGE="{container_image}"',
            'SESSION_NAME="isaac-lab"',
            f'LEISAAC_COMMIT="{leisaac_commit}"',
            'PKGS_DIR="/home/ubuntu/isaaclab-pkgs"',
            'MARKER="$PKGS_DIR/.leisaac-installed"',
            'ASSETS_DIR="/home/ubuntu/leisaac-assets"',
            'ASSETS_MARKER="$ASSETS_DIR/.assets-downloaded"',
            '',
            '# Create cache directories for persistent NVIDIA/Isaac caches',
            'mkdir -p ~/docker/isaac-sim/cache/kit',
            'mkdir -p ~/docker/isaac-sim/cache/ov',
            'mkdir -p ~/docker/isaac-sim/cache/pip',
            'mkdir -p ~/docker/isaac-sim/cache/glcache',
            'mkdir -p ~/docker/isaac-sim/cache/computecache',
            'mkdir -p ~/docker/isaac-sim/logs',
            'mkdir -p ~/docker/isaac-sim/data',
            'mkdir -p ~/docker/isaac-sim/documents',
            '',
            '# Create persistent package directory',
            'mkdir -p "$PKGS_DIR"',
            '',
            '# Auto-install leisaac on first launch (skips if already installed)',
            '# Uses git commit SHA (not PyPI tag) — Gr00t16ServicePolicyClient was added',
            '# after the v0.3.0 release tag. Uses the Isaac Sim python.sh pip wrapper.',
            'if [[ ! -f "$MARKER" ]]; then',
            '  echo "Installing leisaac @${LEISAAC_COMMIT} to persistent volume..."',
            '  docker run --rm --gpus all \\',
            '    -e ACCEPT_EULA=Y \\',
            '    -e PYTHONPATH=/workspace/isaaclab-pkgs \\',
            '    -v "$PKGS_DIR":/workspace/isaaclab-pkgs:rw \\',
            '    "$CONTAINER_IMAGE" \\',
            '    -c "/workspace/isaaclab/_isaac_sim/python.sh -m pip install --target /workspace/isaaclab-pkgs \'leisaac[gr00t] @ git+https://github.com/LightwheelAI/leisaac.git@${LEISAAC_COMMIT}#subdirectory=source/leisaac\' && touch /workspace/isaaclab-pkgs/.leisaac-installed"',
            'fi',
            '',
            '# Auto-download LeIsaac scene assets from GitHub releases (skips if already downloaded)',
            '# The pip package only ships empty .gitkeep placeholders — actual USD scenes and',
            '# robot models must be downloaded separately.',
            'if [[ ! -f "$ASSETS_MARKER" ]]; then',
            '  echo "Downloading LeIsaac scene assets from GitHub releases..."',
            '  mkdir -p "$ASSETS_DIR/scenes" "$ASSETS_DIR/robots"',
            '  curl -fsSL -o /tmp/kitchen_with_orange.zip \\',
            '    https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip',
            '  unzip -o /tmp/kitchen_with_orange.zip -d "$ASSETS_DIR/scenes/"',
            '  rm -f /tmp/kitchen_with_orange.zip',
            '  curl -fsSL -o "$ASSETS_DIR/robots/so101_follower.usd" \\',
            '    https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd',
            '  touch "$ASSETS_MARKER"',
            'fi',
            '',
            '# X11 forwarding (aligned with official IsaacLab Docker pattern: xhost + / -e DISPLAY / -v .Xauthority)',
            '# DCV stores xauth at /run/user/1000/dcv/console.xauth, not ~/.Xauthority.',
            '# xhost +local:docker allows local connections without xauth file dependency.',
            'xhost +local:docker 2>/dev/null || true',
            '',
            'XAUTH_FILE="/run/user/1000/dcv/console.xauth"',
            'if [[ ! -f "$XAUTH_FILE" ]]; then',
            '  XAUTH_FILE="$HOME/.Xauthority"',
            'fi',
            '',
            '# Build Xauthority mount only if the file exists (avoids Docker creating a directory)',
            'XAUTH_MOUNT=""',
            'if [[ -f "$XAUTH_FILE" ]]; then',
            '  XAUTH_MOUNT="-v $XAUTH_FILE:/root/.Xauthority:ro"',
            'fi',
            '',
            'docker run \\',
            '  --name "$SESSION_NAME" \\',
            '  --entrypoint bash \\',
            '  -it \\',
            '  --gpus all \\',
            '  -e "ACCEPT_EULA=Y" \\',
            '  -e "PRIVACY_CONSENT=Y" \\',
            '  -e DISPLAY \\',
            '  -e LEISAAC_ASSETS_ROOT=/assets \\',
            '  -e PYTHONPATH=/workspace/isaaclab-pkgs:${PYTHONPATH:-} \\',
            '  $XAUTH_MOUNT \\',
            '  -v "$ASSETS_DIR":/assets:ro \\',
            '  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \\',
            '  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \\',
            '  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \\',
            '  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \\',
            '  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \\',
            '  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \\',
            '  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \\',
            '  -v ~/docker/isaac-sim/documents:/root/Documents:rw \\',
            '  -v "$PKGS_DIR":/workspace/isaaclab-pkgs:rw \\',
            '  --rm \\',
            '  --network=host \\',
            '  -v $HOME/leisaac-repo/scripts:/workspace/scripts:ro \\',
            '  "$CONTAINER_IMAGE" \\',
            '  "$@"',
            'HELPER_EOF',
            'chmod +x /usr/local/bin/run-isaaclab.sh',
            'must "create-helper-script" "test -x /usr/local/bin/run-isaaclab.sh"',
        )

        # Create persistent package directory for container volume mount
        self._user_data.add_commands(
            '# === Persistent Package Volume (Phase 2) ===',
            'mkdir -p /home/ubuntu/isaaclab-pkgs',
            'chown ubuntu:ubuntu /home/ubuntu/isaaclab-pkgs',
        )

        # Install uv package manager for ubuntu user
        self._user_data.add_commands(
            '# === Host Utilities (Phase 2) ===',
            'su - ubuntu -c "curl -LsSf https://astral.sh/uv/install.sh | sh"',
        )

        # Create host venv with tensorboard and wandb
        self._user_data.add_commands(
            'su - ubuntu -c "/home/ubuntu/.local/bin/uv venv /home/ubuntu/.venv"',
            'su - ubuntu -c "/home/ubuntu/.local/bin/uv pip install --python /home/ubuntu/.venv/bin/python tensorboard wandb"',
        )

        # Add host venv to ubuntu user's PATH
        self._user_data.add_commands(
            "grep -q '/home/ubuntu/.venv/bin' /home/ubuntu/.bashrc || "
            "echo 'export PATH=\"/home/ubuntu/.venv/bin:$PATH\"' >> /home/ubuntu/.bashrc",
        )

        # EFS mount — lives in UserData so CloudFormation resolves the
        # cross-stack EFS ID token at deploy time.
        if props.efs_id:
            self._user_data.add_commands(
                "mkdir -p /mnt/efs",
                # nofail prevents boot hang if DNS is not ready at fstab mount time.
                # efs-ensure-mount.service (installed below) retries after network-online.target.
                f"grep -q '{props.efs_id}' /etc/fstab || echo '{props.efs_id}:/ /mnt/efs efs _netdev,tls,nofail 0 0' >> /etc/fstab",
                # Install the retry-mount service (function defined in configure_dcv_instance.sh)
                "install_efs_ensure_mount_service",
                "mount /mnt/efs || true",
                "chown ubuntu:ubuntu /mnt/efs || true",
            )

        # ================================================================
        # DCV Desktop Installation (Phase 3 — runs AFTER all other steps)
        # These call must/try_step/install_auto_dcv_service defined in the
        # static .sh file inlined above.
        # ================================================================

        self._user_data.add_commands(
            '# === DCV Desktop (Phase 3 — last application installed) ===',
            'must "install-desktop" \'',
            '  apt_install ubuntu-desktop gdm3 dbus-x11',
            '  sed -i "s/^#\\(WaylandEnable=false\\)/\\1/" /etc/gdm3/custom.conf || true',
            "'",
        )

        self._user_data.add_commands(
            'try_step "disable-gnome-initial-setup" \'',
            '  apt-get remove --purge -yq gnome-initial-setup || true',
            '  sed -i "s/^X-GNOME-Autostart-enabled=true/X-GNOME-Autostart-enabled=false/" /etc/xdg/autostart/gnome-initial-setup-first-login.desktop || true',
            '  systemctl restart gdm3 || true',
            "'",
        )

        self._user_data.add_commands(
            f'must "install-nice-dcv" \'',
            f'  DCV_URL="https://d1uj6qtbmh3dt5.cloudfront.net/{dcv_version}/Servers/nice-dcv-{dcv_version}-{dcv_build}-ubuntu2204-x86_64.tgz"',
            '  cd /tmp',
            '  wget -q "$DCV_URL" -O /tmp/dcv.tgz',
            '  tar -xzf /tmp/dcv.tgz -C /tmp',
            f'  cd /tmp/nice-dcv-{dcv_version}-{dcv_build}-ubuntu2204-x86_64',
            '  apt_install libpulse-mainloop-glib0 libpulse0 libgstreamer-plugins-base1.0-0 libcrack2 libxcb-damage0 libxcb-xkb1 libxcb-xtest0 keyutils',
            '  apt_install alsa-utils',
            '  apt-get install -yq ./*.deb',
            '  usermod -aG video dcv || true',
            '  systemctl enable dcvserver',
            '  systemctl restart dcvserver',
            "'",
        )

        self._user_data.add_commands(
            'try_step "configure-dcv" \'',
            '  sed -i "/^\\[display\\]/a max-head-resolution = \\"(4096, 2160)\\"\\nweb-client-max-head-resolution = \\"(4096, 4096)\\"" /etc/dcv/dcv.conf || true',
            '  if ! grep -q "\\[display/linux\\]" /etc/dcv/dcv.conf; then',
            '    cat <<DCVEOF >>/etc/dcv/dcv.conf',
            '[display/linux]',
            'disable-local-console=false',
            'DCVEOF',
            '  fi',
            # Auto-create a console session on every DCV server start (covers stop/start).
            # Console sessions attach to GDM3 which provides the GNOME desktop automatically.
            '  if ! grep -q "\\[session-management/automatic-console-session\\]" /etc/dcv/dcv.conf; then',
            '    cat <<DCVEOF >>/etc/dcv/dcv.conf',
            '',
            '[session-management/automatic-console-session]',
            'create-session = true',
            'owner = ubuntu',
            'DCVEOF',
            '  fi',
            '  systemctl restart dcvserver || true',
            "'",
        )

        self._user_data.add_commands(
            'must "install-auto-dcv-service" install_auto_dcv_service',
        )

        # Security Group
        # Controls network access to the instance
        self._security_group = ec2.SecurityGroup(
            self, "SecurityGroup",
            vpc=self._vpc,
            description="Allow DCV, TensorBoard, and W&B access",
            allow_all_outbound=True, # Do not allow all outbound traffic in production
        )
        self._security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(8443),
            "Allow Amazon DCV access",
        )
        self._security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(8080),
            "Allow W&B local server access",
        )

        # Allow EFS access from DCV instance (only if EFS provided)
        if efs_fs is not None:
            efs_fs.connections.allow_default_port_from(self._security_group)

        # EC2 Instance
        # GPU-accelerated instance for Isaac Sim and DCV visualization
        if props.availability_zone:
            subnet_selection = ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PUBLIC,
                availability_zones=[props.availability_zone],
            )
        else:
            subnet_selection = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)

        self._instance = ec2.Instance(
            self, "Instance",
            instance_type=ec2.InstanceType(props.instance_type),
            machine_image=ec2.MachineImage.lookup(
                name="ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-20251111",
                owners=["099720109477"],  # Canonical
            ),
            vpc=self._vpc,
            vpc_subnets=subnet_selection,
            role=self._instance_role,
            security_group=self._security_group,
            user_data=self._user_data,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/sda1",
                    volume=ec2.BlockDeviceVolume.ebs(150, delete_on_termination=True),
                )
            ],
        )

        # Elastic IP
        # Ensures stable public IP across instance stop/start cycles
        self._eip = ec2.CfnEIP(self, "EIP", domain="vpc")
        ec2.CfnEIPAssociation(
            self, "EIPAssociation",
            allocation_id=self._eip.attr_allocation_id,
            instance_id=self._instance.instance_id,
        )

        # ================================================================
        # CreationPolicy: CloudFormation waits for bootstrap signal (D-04)
        # ================================================================
        cfn_instance = self._instance.node.default_child
        cfn_instance.cfn_options.creation_policy = CfnCreationPolicy(
            resource_signal=CfnResourceSignal(
                count=1,
                timeout="PT60M"  # D-05: 60 min timeout (container pull + DCV desktop install)
            )
        )

        # cfn-signal command in UserData (must reference logical_id token)
        logical_id = cfn_instance.logical_id
        stack_name = Stack.of(self).stack_name
        region = Stack.of(self).region

        self._user_data.add_commands(
            '# === CloudFormation Signal (Phase 3) ===',
            f'/usr/local/bin/cfn-signal --stack {stack_name} --resource {logical_id} --region {region} -e 0 || true',
            'echo "STEP_OK:cfn-signal" >> "$SUMMARY"',
        )

        self._user_data.add_commands(
            '# === ALL_DONE Marker (Phase 3) ===',
            'date -Iseconds > /var/lib/dcv-bootstrap/ALL_DONE',
            'log "Bootstrap complete. ALL_DONE marker written."',
        )

        self._user_data.add_commands(
            '# === Final Summary ===',
            'log "==== SUMMARY (also in $SUMMARY) ===="',
            'cat "$SUMMARY" || true',
        )

        # CloudFormation Outputs
        password = f"dcv{Stack.of(self).account}"

        CfnOutput(
            self, "InstancePublicIP",
            value=self._eip.ref,
            description="Public IP address of the DCV instance",
        )
        CfnOutput(
            self, "DCVWebURL",
            value=f"https://{self._eip.ref}:8443",
            description="DCV web client URL",
        )
        CfnOutput(
            self, "DCVCredentials",
            value=f"Username: ubuntu, Password: {password}",
            description="DCV login credentials",
        )
        CfnOutput(
            self, "InstanceId",
            value=self._instance.instance_id,
            description="EC2 Instance ID for SSM access",
        )

    @property
    def vpc(self) -> ec2.IVpc:
        """VPC where the DCV workstation is deployed."""
        return self._vpc

    @property
    def instance_role(self) -> iam.Role:
        """IAM role attached to the DCV instance."""
        return self._instance_role

    @property
    def user_data(self) -> ec2.UserData:
        """UserData script for instance bootstrap."""
        return self._user_data

    @property
    def security_group(self) -> ec2.SecurityGroup:
        """Security group for the DCV workstation."""
        return self._security_group

    @property
    def instance(self) -> ec2.Instance:
        """EC2 instance for the DCV workstation."""
        return self._instance

    @property
    def elastic_ip(self) -> str:
        """Elastic IP address for the DCV workstation."""
        return self._eip.ref
