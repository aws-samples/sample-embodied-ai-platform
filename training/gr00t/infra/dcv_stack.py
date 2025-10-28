import os
from aws_cdk import aws_ec2 as ec2, aws_iam as iam, aws_efs as efs, Stack, CfnOutput
from constructs import Construct


class DcvStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc: ec2.IVpc = None,
        vpc_id: str = None,
        efs_id: str = None,
        efs_sg_id: str = None,
        batch_stack=None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Resolve VPC - either use provided VPC, lookup by ID, or get from batch stack
        # The VPC must be in the same region as the stack and contain a public subnet
        if vpc is not None:
            resolved_vpc = vpc
        elif vpc_id is not None:
            resolved_vpc = ec2.Vpc.from_lookup(self, "BatchVPC", vpc_id=vpc_id)
        elif batch_stack is not None:
            resolved_vpc = batch_stack.vpc
        else:
            raise ValueError("Either vpc, vpc_id, or batch_stack must be provided")

        # 1. IAM Role for the EC2 Instance
        # This role allows the instance to access S3 for checkpoints and manage itself via SSM
        instance_role = iam.Role(
            self,
            "DcvInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonS3ReadOnlyAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMManagedInstanceCore"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonEC2ContainerRegistryPowerUser"
                ),
            ],
        )

        # 2. EFS File System Setup
        # Import EFS from batch stack if provided, otherwise skip EFS functionality
        efs_fs = None
        if efs_id and efs_sg_id:
            efs_fs = efs.FileSystem.from_file_system_attributes(
                self,
                "BatchEFS",
                file_system_id=efs_id,
                security_group=ec2.SecurityGroup.from_security_group_id(
                    self, "BatchEFSSecurityGroup", efs_sg_id, mutable=True
                ),
            )

        # 3. Security Group
        # This controls access to the instance. It allows Amazon DCV traffic by default.
        # Optional: Allow SSH access from your IP for debugging purposes.
        # IMPORTANT: For production, you should restrict the source IP for all ports.
        sg = ec2.SecurityGroup(
            self,
            "DcvSecurityGroup",
            vpc=resolved_vpc,
            description="Allow Amazon DCV and optional TensorBoard access",
            allow_all_outbound=True,
        )
        sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(8443),
            "Allow Amazon DCV access from anywhere",
        )
        # sg.add_ingress_rule(ec2.Peer.ipv4("<your_ip_address>"), ec2.Port.tcp(22), "Allow SSH access from your IP")
        sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(6006),
            "Allow TensorBoard access from anywhere",
        )

        # Allow EFS access from the DCV instance (only if EFS is provided)
        if efs_fs is not None:
            efs_fs.connections.allow_default_port_from(sg)

        # 4. User Data Script
        # This script runs on the first boot to set up the instance.
        # Read the user data script
        user_data_path = os.path.join(
            os.path.dirname(__file__), "configure_dcv_instance.sh"
        )
        with open(user_data_path, "r") as f:
            user_data_script = f.read()

        # Create the dynamic password and inject it into the script
        # Keep this consistent with the stack output below
        password = f"dcv{self.account}"
        user_data_script = user_data_script.replace("__PASSWORD__", password)

        # Add the modified script to user data
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(user_data_script)

        # Add EFS mounting commands (only if EFS is provided)
        if efs_fs is not None:
            user_data.add_commands(
                "# Mount EFS file system with TLS",
                "echo 'Setting up EFS mount with TLS...'",
                # EFS utils are installed in configure_dcv_instance.sh; only perform mount here
                # Log mount action into the bootstrap logs/summary for visibility
                "echo 'STEP_INFO:EFS:Configuring fstab and mounting' >> /var/log/dcv-bootstrap.summary || true",
                "mkdir -p /mnt/efs",
                f"echo '{efs_fs.file_system_id}:/ /mnt/efs efs _netdev,tls 0 0' >> /etc/fstab",
                # Attempt mount and record success/failure
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

        # 5. EC2 Instance
        # The actual virtual machine for visualization.
        instance = ec2.Instance(
            self,
            "DcvInstance",
            # G6 -> 24GB vRAM, G4dn -> 16GB vRAM
            # Recommended: 2xlarge for running simulation, 4xlarge if also running policy inference
            instance_type=ec2.InstanceType("g6.4xlarge"),
            machine_image=ec2.MachineImage.lookup(
                name="ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*",
                owners=["099720109477"],
            ),
            vpc=resolved_vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            role=instance_role,
            security_group=sg,
            user_data=user_data,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/sda1",
                    volume=ec2.BlockDeviceVolume.ebs(100, delete_on_termination=True),
                )
            ],
        )

        # Allocate and associate an Elastic IP to ensure a stable public IPv4
        eip = ec2.CfnEIP(self, "DcvEip", domain="vpc")
        ec2.CfnEIPAssociation(
            self,
            "DcvEipAssociation",
            allocation_id=eip.attr_allocation_id,
            instance_id=instance.instance_id,
        )

        # 6. Outputs
        CfnOutput(
            self,
            "InstancePublicIP",
            value=eip.ref,
            description="Public IP address of the Amazon DCV instance. Connect to this IP.",
        )
        CfnOutput(
            self,
            "DCVConnectionCommand",
            value=f"dcvviewer -hostname {eip.ref} -port 8443 -user ubuntu",
            description="Command to connect using the Amazon DCV client.",
        )
        CfnOutput(
            self,
            "DCVWebURL",
            value=f"https://{eip.ref}:8443",
            description="URL to connect using a web browser. You may need to accept a self-signed certificate warning.",
        )
        CfnOutput(
            self,
            "DCVCredentials",
            value=f"Username: ubuntu, Password: dcv{self.account}",
            description="Default credentials for the DCV session.",
        )

        # EFS outputs (only if EFS is provided)
        if efs_fs is not None:
            CfnOutput(
                self,
                "EFSFileSystemId",
                value=efs_fs.file_system_id,
                description="EFS File System ID mounted at /mnt/efs for shared storage.",
            )

            CfnOutput(
                self,
                "EFSMountPoint",
                value="/mnt/efs",
                description="EFS mount point on the DCV instance for shared storage with Batch jobs.",
            )
