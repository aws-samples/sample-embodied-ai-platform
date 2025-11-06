import os
from aws_cdk import (
    aws_ec2 as ec2,
    aws_batch as batch,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_efs as efs,
    aws_ecs as ecs,
    aws_s3 as s3,
    Stack,
    CfnOutput,
    Duration,
    Size,
    RemovalPolicy,
)
from constructs import Construct
from codebuild_stack import CodeBuildStack


class BatchStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc_id: str = None,
        efs_id: str = None,
        efs_sg_id: str = None,
        ecr_image_uri: str = None,
        dataset_bucket: str = None,
        s3_upload_uri: str = None,
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

        Args:
            vpc_id: Existing VPC ID to reuse (optional)
            efs_id: Existing EFS file system ID to reuse (optional)
            efs_sg_id: Existing EFS security group ID (required if efs_id is provided)
            ecr_image_uri: Existing ECR image URI (e.g. 123456789012.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest).
                          If not provided, builds from local Dockerfile.
            dataset_bucket: S3 bucket name for dataset read-only access (optional)
            s3_upload_uri: S3 URI for checkpoint uploads (e.g., s3://bucket/path). If not provided, creates a new bucket.
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
        # - Prefer an existing ECR image in the same account via ecr_image_uri
        # - Else automatically build via CodeBuild (works on any architecture)
        if ecr_image_uri:
            # Use provided ECR image
            # Expected format: <account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
            repo_and_tag = ecr_image_uri.split("/")[-1]
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
            codebuild_stack = None  # No CodeBuild needed
        else:
            # Automatically build container using CodeBuild
            # This works on any architecture (x86, ARM) since build happens in the cloud
            codebuild_stack = CodeBuildStack(
                self,
                "CodeBuild",
                ecr_repository_name="gr00t-finetune",
                use_stable=True,
            )
            # Use the built image
            container_image = ecs.ContainerImage.from_ecr_repository(
                repository=codebuild_stack.ecr_repository, tag="latest"
            )
            ecr_image_uri = codebuild_stack.image_uri

        # Store codebuild_stack for conditional outputs later
        self.codebuild_stack = codebuild_stack
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
                # By default, limit to g6e family for cost savings
                # Single-GPU instances
                ec2.InstanceType("g6e.2xlarge"),
                ec2.InstanceType("g6e.4xlarge"),
                ec2.InstanceType("g6e.8xlarge"),
                ec2.InstanceType("g6e.16xlarge"),
                # ec2.InstanceType("p5.4xlarge"),
                # Multi-GPU instances
                ec2.InstanceType("g6e.12xlarge"),  # 4 GPUs
                ec2.InstanceType("g6e.24xlarge"),  # 4 GPUs
                ec2.InstanceType("g6e.48xlarge"),  # 8 GPUs
                # ec2.InstanceType("p4d.24xlarge"),  # 8 GPUs
                # ec2.InstanceType("p5.48xlarge"),  # 8 GPUs
            ],
            minv_cpus=0,
            maxv_cpus=192,
            instance_role=iam.Role(
                self,
                "BatchInstanceRole",
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
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[],
        )

        # Separate dataset bucket (read-only) from checkpoint upload bucket (read/write).
        # 1) If dataset_bucket is provided, allow read-only on that bucket.
        # 2) If s3_upload_uri is provided (e.g., s3://bucket/path), allow read/write to its bucket/prefix.
        # 3) If s3_upload_uri is not provided, create a new checkpoint bucket and derive s3_upload_uri.
        # 4) If neither dataset nor upload buckets are specified, fall back to S3 read-only (useful for dataset downloads).
        if dataset_bucket:
            job_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:ListBucket"],
                    resources=[f"arn:aws:s3:::{dataset_bucket}"],
                )
            )
            job_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:GetObject"],
                    resources=[f"arn:aws:s3:::{dataset_bucket}/*"],
                )
            )

        # Resolve or create checkpoint upload bucket/URI
        checkpoint_bucket = None
        original_s3_upload_uri = s3_upload_uri  # Track original state
        if not s3_upload_uri:
            # Create a new S3 bucket for checkpoints
            checkpoint_bucket = s3.Bucket(
                self,
                "IsaacGr00tCheckpointBucket",
                block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                encryption=s3.BucketEncryption.S3_MANAGED,
                enforce_ssl=True,
                versioned=True,
                removal_policy=RemovalPolicy.RETAIN,
                auto_delete_objects=False,
            )
            # Derive s3_upload_uri from the created bucket
            s3_upload_uri = f"s3://{checkpoint_bucket.bucket_name}/gr00t/checkpoints"

        if checkpoint_bucket is not None:
            # Grant RW to the created checkpoint bucket
            checkpoint_bucket.grant_read_write(job_role)
        elif s3_upload_uri and s3_upload_uri.startswith("s3://"):
            remainder = s3_upload_uri[5:]
            if "/" in remainder:
                upload_bucket, upload_prefix = remainder.split("/", 1)
            else:
                upload_bucket, upload_prefix = remainder, ""
            job_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:ListBucket"],
                    resources=[f"arn:aws:s3:::{upload_bucket}"],
                )
            )
            object_resource = (
                f"arn:aws:s3:::{upload_bucket}/{upload_prefix}*"
                if upload_prefix
                else f"arn:aws:s3:::{upload_bucket}/*"
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
                    resources=[object_resource],
                )
            )

        # Only grant general S3 read-only access if no buckets were originally specified
        if (
            not dataset_bucket
            and not original_s3_upload_uri
            and checkpoint_bucket is None
        ):
            job_role.add_managed_policy(
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess")
            )

        # Mount EFS at /mnt/efs inside the container. The workflow script and training output
        # (checkpoints, tensorboard logs) are configured to write under this path.
        efs_volume = batch.EcsVolume.efs(
            name="BatchEFS", file_system=efs_fs, container_path="/mnt/efs"
        )

        # Prepare container environment with defaults and optional S3 settings
        container_environment = {
            # Optional default locations on EFS. You can override at submit time.
            "OUTPUT_DIR": "/mnt/efs/gr00t/checkpoints"
        }
        if s3_upload_uri:
            container_environment["UPLOAD_TARGET"] = "s3"
            container_environment["S3_UPLOAD_URI"] = s3_upload_uri

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
                environment=container_environment,
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
        CfnOutput(
            self, "ComputeEnvironmentName", value=compute_env.compute_environment_name
        )
        CfnOutput(self, "JobQueueName", value=job_queue.job_queue_name)
        CfnOutput(self, "JobDefinitionName", value=job_def.job_definition_name)
        CfnOutput(self, "EcrImageUri", value=ecr_image_uri)
        if s3_upload_uri:
            CfnOutput(self, "CheckpointS3UploadUri", value=s3_upload_uri)

        # CodeBuild outputs (only when CodeBuild is used)
        if codebuild_stack:
            CfnOutput(
                self,
                "CodeBuildProjectName",
                value=codebuild_stack.build_project.project_name,
                description="CodeBuild project name for building the container",
            )
        # endregion
