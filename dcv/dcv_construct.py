"""L3 construct for GPU-accelerated DCV workstation with containerized IsaacLab."""
import os
import re
from typing import Optional
from aws_cdk import aws_ec2 as ec2, aws_iam as iam, Stack, CfnOutput
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
    - Static steps (drivers, DCV, Docker, etc.) come from configure_dcv_instance.sh
    - Dynamic steps (container pull, helper script, leisaac) are added via add_commands
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
        leisaac_version = version_config.get("leisaac", "v0.3.0")

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
        # persistent package volume, and leisaac auto-install
        helper_script = f"""cat > /usr/local/bin/run-isaaclab.sh << 'HELPER_EOF'
#!/bin/bash
set -euo pipefail
CONTAINER_IMAGE="{container_image}"
SESSION_NAME="isaac-lab"
LEISAAC_VERSION="{leisaac_version}"
PKGS_DIR="/home/ubuntu/isaaclab-pkgs"
MARKER="$PKGS_DIR/.leisaac-installed"

# Create cache directories for persistent NVIDIA/Isaac caches
mkdir -p ~/docker/isaac-sim/cache/kit
mkdir -p ~/docker/isaac-sim/cache/ov
mkdir -p ~/docker/isaac-sim/cache/pip
mkdir -p ~/docker/isaac-sim/cache/glcache
mkdir -p ~/docker/isaac-sim/cache/computecache
mkdir -p ~/docker/isaac-sim/logs
mkdir -p ~/docker/isaac-sim/data

# Create persistent package directory
mkdir -p "$PKGS_DIR"

# Auto-install leisaac on first launch (skips if already installed)
if [[ ! -f "$MARKER" ]]; then
  echo "Installing leisaac $LEISAAC_VERSION to persistent volume..."
  docker run --rm --gpus all \\
    -e ACCEPT_EULA=Y \\
    -e PYTHONPATH=/workspace/isaaclab-pkgs \\
    -v "$PKGS_DIR":/workspace/isaaclab-pkgs:rw \\
    "$CONTAINER_IMAGE" \\
    -c "pip install --target /workspace/isaaclab-pkgs 'leisaac[gr00t]==$LEISAAC_VERSION' && touch /workspace/isaaclab-pkgs/.leisaac-installed"
fi

docker run \\
  --name "$SESSION_NAME" \\
  --entrypoint bash \\
  -it \\
  --gpus all \\
  -e "ACCEPT_EULA=Y" \\
  -e "PRIVACY_CONSENT=Y" \\
  -e DISPLAY \\
  -e PYTHONPATH=/workspace/isaaclab-pkgs:$PYTHONPATH \\
  -v $HOME/.Xauthority:/root/.Xauthority \\
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \\
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \\
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \\
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \\
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \\
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \\
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \\
  -v "$PKGS_DIR":/workspace/isaaclab-pkgs:rw \\
  --rm \\
  --network=host \\
  "$CONTAINER_IMAGE" \\
  "$@"
HELPER_EOF
chmod +x /usr/local/bin/run-isaaclab.sh"""
        self._user_data.add_commands(
            f'must "create-helper-script" \'{helper_script}\'',
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
                "apt-get install -y -qq amazon-efs-utils || true",
                "mkdir -p /mnt/efs",
                f"grep -q '{props.efs_id}' /etc/fstab || echo '{props.efs_id}:/ /mnt/efs efs _netdev,tls 0 0' >> /etc/fstab",
                "mount -a",
                "chown ubuntu:ubuntu /mnt/efs || true",
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
