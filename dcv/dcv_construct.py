"""L3 construct for GPU-accelerated DCV workstation. Phase 1: Skeleton with VPC resolution only."""
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
        isaac_lab_version: IsaacLab version (default: v2.3.2)
        python_version: Python version (default: auto-derived from IsaacSim version)
        efs_id: Optional EFS file system ID (for persistent storage mode)
        efs_sg_id: Optional EFS security group ID (required with efs_id)
        leisaac_enabled: Enable leisaac installation (default: False)
    """

    def __init__(
        self,
        vpc: Optional[ec2.IVpc] = None,
        vpc_id: Optional[str] = None,
        instance_type: str = "g6.4xlarge",
        isaac_sim_version: str = "5.1.0",
        isaac_lab_version: str = "v2.3.2",
        python_version: Optional[str] = None,
        efs_id: Optional[str] = None,
        efs_sg_id: Optional[str] = None,
        leisaac_enabled: bool = False,
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


class DcvWorkstation(Construct):
    """L3 construct for GPU-accelerated DCV workstation. Phase 1: Skeleton with VPC resolution only.

    This construct provides a reusable DCV workstation that can be used in two modes:
    1. Standalone: Deploy independently with auto-created or looked-up VPC
    2. Integrated: Import into another CDK app and provide VPC reference

    Phase 1 implements only VPC resolution to validate the architecture.
    Full DCV instance infrastructure will be added in Phase 2.
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

        # Resolve Python version (use explicit or derive from matrix)
        python_version = props.python_version or version_config["python"]

        # Extract derived versions from compatibility matrix
        pytorch_version = version_config["pytorch"]
        cuda_index = version_config["cuda_index"]
        dcv_version_build = version_config["dcv"]
        dcv_version, dcv_build = dcv_version_build.split("-")
        leisaac_version = version_config.get("leisaac", "v0.2.0")

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

        # UserData Script
        # Load bootstrap script and inject all version parameters
        user_data_path = os.path.join(os.path.dirname(__file__), "configure_dcv_instance.sh")
        with open(user_data_path, "r") as f:
            user_data_script = f.read()

        # Generate password using account ID (alphanumeric only, safe for bash)
        password = f"dcv{Stack.of(self).account}"

        # Inject all parameters using string replacement
        user_data_script = user_data_script.replace("__PASSWORD__", password)
        user_data_script = user_data_script.replace("__PYTHON_VERSION__", python_version)
        user_data_script = user_data_script.replace("__ISAAC_SIM_VERSION__", props.isaac_sim_version)
        user_data_script = user_data_script.replace("__ISAAC_LAB_VERSION__", props.isaac_lab_version)
        user_data_script = user_data_script.replace("__PYTORCH_VERSION__", pytorch_version)
        user_data_script = user_data_script.replace("__CUDA_INDEX__", cuda_index)
        user_data_script = user_data_script.replace("__DCV_VERSION__", dcv_version)
        user_data_script = user_data_script.replace("__DCV_BUILD__", dcv_build)
        user_data_script = user_data_script.replace(
            "__LEISAAC_ENABLED__",
            "true" if props.leisaac_enabled else "false"
        )
        user_data_script = user_data_script.replace("__LEISAAC_VERSION__", leisaac_version)

        # Validate no unresolved placeholders (catches bugs in replacement logic)
        unresolved = re.findall(r'__[A-Z_]+__', user_data_script)
        if unresolved:
            raise ValueError(
                f"Unresolved placeholders in UserData script: {unresolved}. "
                "This indicates a bug in version replacement logic."
            )

        # Create UserData object
        self._user_data = ec2.UserData.for_linux()
        self._user_data.add_commands(user_data_script)

        # Add EFS mounting commands (only if EFS is provided)
        if efs_fs is not None:
            self._user_data.add_commands(
                "# Mount EFS file system with TLS",
                "echo 'Setting up EFS mount with TLS...'",
                "echo 'STEP_INFO:EFS:Configuring fstab and mounting' >> /var/log/dcv-bootstrap.summary || true",
                "mkdir -p /mnt/efs",
                f"echo '{efs_fs.file_system_id}:/ /mnt/efs efs _netdev,tls 0 0' >> /etc/fstab",
                (
                    "if mount -a; then\n"
                    "  echo 'STEP_OK:EFS mount' >> /var/log/dcv-bootstrap.summary;\n"
                    "  echo 'EFS mounted at /mnt/efs' | tee -a /var/log/dcv-bootstrap.log;\n"
                    "else\n"
                    "  echo 'STEP_FAIL:EFS mount' >> /var/log/dcv-bootstrap.summary;\n"
                    "fi"
                ),
                "chown ubuntu:ubuntu /mnt/efs || true",
            )

        # Security Group
        # Controls network access to the instance
        # Allows DCV (8443) and TensorBoard (6006) from any IP for development
        # Production deployments should restrict source IPs
        self._security_group = ec2.SecurityGroup(
            self, "SecurityGroup",
            vpc=self._vpc,
            description="Allow DCV and TensorBoard access",
            allow_all_outbound=True,
        )
        self._security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(8443),
            "Allow Amazon DCV access",
        )
        self._security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(6006),
            "Allow TensorBoard access",
        )

        # Allow EFS access from DCV instance (only if EFS provided)
        if efs_fs is not None:
            efs_fs.connections.allow_default_port_from(self._security_group)

        # EC2 Instance
        # GPU-accelerated instance for Isaac Sim and DCV visualization
        self._instance = ec2.Instance(
            self, "Instance",
            instance_type=ec2.InstanceType(props.instance_type),
            machine_image=ec2.MachineImage.lookup(
                name="ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-20251111",
                owners=["099720109477"],  # Canonical
            ),
            vpc=self._vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            role=self._instance_role,
            security_group=self._security_group,
            user_data=self._user_data,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/sda1",
                    volume=ec2.BlockDeviceVolume.ebs(100, delete_on_termination=True),
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
        # Expose connection details to users
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
        """VPC where the DCV workstation is deployed.

        Returns:
            The resolved VPC (provided, looked-up, or auto-created)
        """
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
