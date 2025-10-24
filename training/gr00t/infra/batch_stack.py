import os
from aws_cdk import (
    aws_ec2 as ec2,
    aws_batch as batch,
    aws_ecr_assets as ecr_assets,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_efs as efs,
    aws_ecs as ecs,
    Stack,
    CfnOutput,
    Duration,
    Size,
)
from constructs import Construct


class BatchStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc_id: str = None,
        efs_id: str = None,
        efs_sg_id: str = None,
        **kwargs,
    ) -> None:
        """
        CDK stack for the AWS Batch resources used by the GR00T fine-tuning workflow.

        This file is intentionally structured to mirror the step-by-step flow in the
        blog (Draft Code Walkthrough) so it's easy to follow and customize:

        - 2.1 Create VPC and EFS
        - 2.2 Build the fine-tuning container and push to ECR
        - 2.3 Create Launch Template
        - 2.4 Create Compute Environment
        - 2.5 Create Job Queue and Job Definition

        Notes on container image (Step 2.2 in blog):
        - If you built and pushed the fine-tune image to ECR already, set env var
          ECR_IMAGE_URI to the ECR image URI (e.g. 123456789012.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest)
          and this stack will reference that image directly.
        - Otherwise, this stack will build an image from `training/gr00t/Dockerfile` at synth time using CDK assets.
        """
        super().__init__(scope, construct_id, **kwargs)

        # ==============================================================
        # region 2.1 Create VPC and EFS
        # ==============================================================
        # Create or reference VPC. If you already have a VPC, pass its ID; otherwise we
        # create a VPC with one NAT gateway (to match "VPC and more" in the console flow).
        if vpc_id:
            vpc = ec2.Vpc.from_lookup(self, "BatchVPC", vpc_id=vpc_id)
        else:
            vpc = ec2.Vpc(
                self,
                "BatchVPC",
                vpc_name="BatchVPC",
                max_azs=2,
                ip_addresses=ec2.IpAddresses.cidr("10.0.0.0/16"),
                nat_gateways=1,
                subnet_configuration=[
                    # We keep a small public subnet for NAT/Egress. Jobs run in private subnets.
                    ec2.SubnetConfiguration(
                        name="Public",
                        subnet_type=ec2.SubnetType.PUBLIC,
                    ),
                    ec2.SubnetConfiguration(
                        name="Private",
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    ),
                ],
            )

        self.vpc = vpc

        # Create or import an EFS file system. This is where checkpoints and logs will be stored
        # so they persist across jobs and can be visualized from a DCV instance.
        if efs_id:
            efs_sg = ec2.SecurityGroup.from_security_group_id(
                self, "BatchEFSSecurityGroup", efs_sg_id, mutable=True
            )
            efs_fs = efs.FileSystem.from_file_system_attributes(
                self,
                "BatchEFS",
                file_system_id=efs_id,
                security_group=efs_sg,
            )
            # Expose attributes for cross-stack use
            self.efs_id = efs_id
            self.efs_sg_id = efs_sg_id
        else:
            efs_sg = ec2.SecurityGroup(
                self,
                "BatchEFSSecurityGroup",
                vpc=vpc,
                description="Security group for Batch instances and EFS",
            )
            efs_fs = efs.FileSystem(
                self,
                "BatchEFS",
                file_system_name="BatchEFS",
                vpc=vpc,
                security_group=efs_sg,
                performance_mode=efs.PerformanceMode.GENERAL_PURPOSE,
                throughput_mode=efs.ThroughputMode.BURSTING,
            )
            # Expose attributes for cross-stack use
            self.efs_id = efs_fs.file_system_id
            self.efs_sg_id = efs_sg.security_group_id

        # Add a self-referencing NFS rule so instances and EFS within the same SG can communicate.
        efs_sg.add_ingress_rule(
            peer=efs_sg,
            connection=ec2.Port.tcp(2049),
            description="Allow NFS within Batch EFS SG",
        )
        # endregion

        # ==============================================================
        # region 2.2 Build the fine-tuning container and push to ECR
        # ==============================================================
        # Container image selection strategy:
        # - Prefer an existing ECR image in the same account via ECR_IMAGE_URI
        # - Else build via CDK asset from local Dockerfile
        image_uri = os.getenv("ECR_IMAGE_URI")
        if image_uri:
            # Prefer from_ecr_repository so execution role gets precise ECR permissions
            # Expected format: <account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
            repo_and_tag = image_uri.split("/")[-1]
            if ":" in repo_and_tag:
                repo_name, tag = repo_and_tag.split(":", 1)
            else:
                repo_name, tag = repo_and_tag, "latest"
            repo = ecr.Repository.from_repository_name(
                self, "IsaacGr00tEcrRepo", repository_name=repo_name
            )
            container_image = ecs.ContainerImage.from_ecr_repository(
                repository=repo, tag=tag
            )
        else:
            asset = ecr_assets.DockerImageAsset(
                self,
                "IsaacGr00tImage",
                directory=os.path.join(os.path.dirname(__file__), ".."),
                file="Dockerfile",
            )
            container_image = ecs.ContainerImage.from_docker_image_asset(asset)
        # endregion

        # ==============================================================
        # region 2.3 Create Launch Template
        # ==============================================================
        # Increase the Linux root volume size to 100 GiB for pulling docker containers.
        launch_template = ec2.LaunchTemplate(
            self,
            "BatchLaunchTemplate",
            launch_template_name="BatchLaunchTemplate",
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",  # standard root device for Amazon Linux 2
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_size=100,
                        delete_on_termination=True,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                    ),
                )
            ],
        )
        # endregion

        # ==============================================================
        # region 2.4 Create Compute Environment
        # ==============================================================
        # IAM role for Batch EC2 instances so they can pull images, mount EFS, access S3, etc.
        compute_env = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "ComputeEnvironment",
            compute_environment_name="IsaacGr00tComputeEnvironment",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),  # Jobs run in private subnets
            launch_template=launch_template,
            security_groups=[efs_sg],
            instance_types=[
                # Limit to g6e family for 48GB vRAM
                ec2.InstanceType("g6e.2xlarge"),
                ec2.InstanceType("g6e.4xlarge"),
                ec2.InstanceType("g6e.8xlarge"),
            ],
            minv_cpus=0,
            maxv_cpus=64,
            instance_role=iam.Role(
                self,
                "BatchInstanceRole",
                role_name="BatchInstanceRole",
                assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name(
                        "service-role/AmazonEC2ContainerServiceforEC2Role"
                    ),
                    iam.ManagedPolicy.from_aws_managed_policy_name(
                        "AmazonSSMManagedInstanceCore"
                    ),
                ],
            ),
            # Uncomment for cost-optimized runs on Spot
            # spot=True,
            # spot_bid_percentage=70,
        )

        # No explicit allow needed beyond the SG rule above since EFS and instances share the SG.
        # endregion

        # ==============================================================
        # region 2.5 Create Job Queue and Job Definition
        # ==============================================================
        job_queue = batch.JobQueue(
            self,
            "JobQueue",
            job_queue_name="IsaacGr00tJobQueue",
            compute_environments=[
                batch.OrderedComputeEnvironment(
                    compute_environment=compute_env, order=1
                )
            ],
            priority=1,
        )

        # Job role for the container tasks (access to S3 during training/upload)
        job_role = iam.Role(
            self,
            "JobRole",
            role_name="IsaacGr00tJobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[],
        )

        # Prefer least-privilege S3 access if TRAINING_S3_BUCKET_NAME is provided; otherwise allow S3 read-only access for dataset download.
        s3_bucket_name = os.getenv("TRAINING_S3_BUCKET_NAME")
        if s3_bucket_name:
            job_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:ListBucket"],
                    resources=[f"arn:aws:s3:::{s3_bucket_name}"],
                )
            )
            job_role.add_to_policy(
                iam.PolicyStatement(
                    actions=[
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:AbortMultipartUpload",
                        "s3:ListMultipartUploadParts",
                    ],
                    resources=[f"arn:aws:s3:::{s3_bucket_name}/*"],
                )
            )
        else:
            job_role.add_managed_policy(
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess")
            )

        # Mount EFS at /mnt/efs inside the container. The workflow script and training output
        # (checkpoints, tensorboard logs) are configured to write under this path.
        efs_volume = batch.EcsVolume.efs(
            name="BatchEFS", file_system=efs_fs, container_path="/mnt/efs"
        )

        job_def = batch.EcsJobDefinition(
            self,
            "IsaacGr00tJobDefinition",
            job_definition_name="IsaacGr00tJobDefinition",
            container=batch.EcsEc2ContainerDefinition(
                self,
                "IsaacGr00tContainer",
                image=container_image,
                memory=Size.gibibytes(64),
                cpu=8,
                gpu=1,
                job_role=job_role,
                environment={
                    # Optional default locations on EFS. You can override at submit time.
                    "OUTPUT_DIR": "/mnt/efs/gr00t/checkpoints"
                },
                volumes=[efs_volume],
                linux_parameters=batch.LinuxParameters(
                    self,
                    "IsaacGr00tLinuxParameters",
                    shared_memory_size=Size.gibibytes(64),
                ),
            ),
            timeout=Duration.hours(6),
        )
        # endregion

        # ==============================================================
        # region Outputs
        # ==============================================================
        CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
        CfnOutput(self, "EFSFileSystemId", value=self.efs_id)
        CfnOutput(self, "EFSSecurityGroupId", value=self.efs_sg_id)

        # Additional outputs for convenience when submitting jobs via CLI/Console
        CfnOutput(self, "ComputeEnvironmentName", value="IsaacGr00tComputeEnvironment")
        CfnOutput(self, "JobQueueName", value="IsaacGr00tJobQueue")
        CfnOutput(self, "JobDefinitionName", value="IsaacGr00tJobDefinition")
        # endregion
