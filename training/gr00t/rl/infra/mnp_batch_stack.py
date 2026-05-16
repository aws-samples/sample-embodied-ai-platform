import os
from aws_cdk import (
    aws_ec2 as ec2,
    aws_batch as batch,
    aws_codebuild as codebuild,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_efs as efs,
    aws_ecs as ecs,
    aws_s3 as s3,
    aws_s3_assets as s3_assets,
    custom_resources as cr,
    Stack,
    CfnOutput,
    Duration,
    Size,
    RemovalPolicy,
)
from constructs import Construct


class RLBatchMNPStack(Stack):
    """AWS Batch Multi-Node Parallel stack for GR00T RL post-training via RLinf.

    Architecture:
      - Main node (g7e.48xlarge): Ray head + RLinf learner with intra-node FSDP
      - Child nodes (g6e.4xlarge): Ray workers + RLinf rollout workers with Isaac Sim

    The MNP job gang-schedules the entire cluster as a single job:
      - All nodes start together or not at all
      - AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS injected into all containers
      - Single job ID for observability (CloudWatch, logs)

    Three-tier artifact flow:
      - Ray object store: trajectories (in-memory, low-latency for on-policy PPO)
      - EFS: checkpoints, TensorBoard logs (shared, persistent)
      - S3: episode logs, eval videos, sim renders (durable, for Cosmos Transfer)
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc_id: str = None,
        efs_id: str = None,
        efs_sg_id: str = None,
        image_uri: str = None,
        num_rollout_nodes: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ==============================================================
        # region 1. VPC and EFS (reuse from Part 1 or create new)
        # ==============================================================
        if vpc_id:
            vpc = ec2.Vpc.from_lookup(self, "VPC", vpc_id=vpc_id)
        else:
            vpc = ec2.Vpc(
                self,
                "VPC",
                vpc_name="GR00T-RL-VPC",
                max_azs=2,
                ip_addresses=ec2.IpAddresses.cidr("10.1.0.0/16"),
                nat_gateways=1,
                subnet_configuration=[
                    ec2.SubnetConfiguration(
                        name="Public", subnet_type=ec2.SubnetType.PUBLIC
                    ),
                    ec2.SubnetConfiguration(
                        name="Private",
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    ),
                ],
            )

        self.vpc = vpc

        if efs_id:
            efs_sg = ec2.SecurityGroup.from_security_group_id(
                self, "EFSSecurityGroup", efs_sg_id, mutable=True
            )
            efs_fs = efs.FileSystem.from_file_system_attributes(
                self,
                "EFS",
                file_system_id=efs_id,
                security_group=efs_sg,
            )
        else:
            efs_sg = ec2.SecurityGroup(
                self,
                "EFSSecurityGroup",
                vpc=vpc,
                description="Security group for EFS mount targets",
            )
            efs_fs = efs.FileSystem(
                self,
                "EFS",
                file_system_name="GR00T-RL-EFS",
                vpc=vpc,
                security_group=efs_sg,
                performance_mode=efs.PerformanceMode.GENERAL_PURPOSE,
                throughput_mode=efs.ThroughputMode.ELASTIC,
                removal_policy=RemovalPolicy.RETAIN,
            )
            efs_sg.add_ingress_rule(
                peer=efs_sg,
                connection=ec2.Port.tcp(2049),
                description="NFS within EFS SG",
            )
        # endregion

        # ==============================================================
        # region 2. S3 bucket for episode artifacts
        # ==============================================================
        artifact_bucket = s3.Bucket(
            self,
            "RLArtifactBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
        )
        # endregion

        # ==============================================================
        # region 3. Security group for inter-node communication
        # ==============================================================
        batch_sg = ec2.SecurityGroup(
            self,
            "BatchSecurityGroup",
            vpc=vpc,
            description="SG for Batch MNP nodes - Ray, NCCL, RLinf inter-node traffic",
            allow_all_outbound=True,
        )
        # Ray requires arbitrary TCP ports for GCS, object store, worker communication
        batch_sg.add_ingress_rule(
            peer=batch_sg,
            connection=ec2.Port.all_tcp(),
            description="All TCP between MNP nodes (Ray GCS, object store, NCCL, FSDP)",
        )
        batch_sg.add_ingress_rule(
            peer=batch_sg,
            connection=ec2.Port.all_udp(),
            description="All UDP between MNP nodes (NCCL)",
        )
        # Allow EFS access from batch nodes
        efs_fs.connections.allow_default_port_from(batch_sg)
        # endregion

        # ==============================================================
        # region 4. Launch template (larger EBS for container images)
        # ==============================================================
        launch_template = ec2.LaunchTemplate(
            self,
            "RLLaunchTemplate",
            security_group=batch_sg,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_size=200,
                        delete_on_termination=True,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                    ),
                )
            ],
        )
        # endregion

        # ==============================================================
        # region 5. IAM roles
        # ==============================================================
        instance_role = iam.Role(
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
        )

        job_role = iam.Role(
            self,
            "RLJobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        artifact_bucket.grant_read_write(job_role)
        # endregion

        # ==============================================================
        # region 6. Compute environment (single pool, heterogeneous via MNP)
        # ==============================================================
        compute_env = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "RLComputeEnvironment",
            compute_environment_name="GR00T-RL-ComputeEnv",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            launch_template=launch_template,
            security_groups=[batch_sg],
            images=[
                batch.EcsMachineImage(
                    image_type=batch.EcsMachineImageType.ECS_AL2_NVIDIA,
                )
            ],
            instance_types=[
                # Learner node
                ec2.InstanceType("g6e.48xlarge"),  # 8× L40S, 192 vCPU, 1.5TB RAM
                # Rollout nodes
                ec2.InstanceType("g6e.4xlarge"),  # 1× L40S, 16 vCPU, 64GB RAM
                ec2.InstanceType("g6e.8xlarge"),  # 1× L40S, 32 vCPU, 128GB RAM
            ],
            minv_cpus=0,
            maxv_cpus=512,
            instance_role=instance_role,
        )
        # endregion

        # ==============================================================
        # region 7. Job queue
        # ==============================================================
        job_queue = batch.JobQueue(
            self,
            "RLJobQueue",
            job_queue_name="GR00T-RL-JobQueue",
            compute_environments=[
                batch.OrderedComputeEnvironment(
                    compute_environment=compute_env, order=1
                )
            ],
            priority=1,
        )
        # endregion

        # ==============================================================
        # region 8. Container images (CodeBuild or pre-built)
        # ==============================================================
        # Two deployment paths:
        #   1. Default: CodeBuild builds unified image automatically on deploy
        #   2. Pre-built: Pass image_uri to skip CodeBuild
        docker_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "docker")
        )

        unified_build = None

        if not image_uri:
            source_asset = s3_assets.Asset(
                self,
                "RLDockerSourceAsset",
                path=docker_dir,
                exclude=["*.pyc", "__pycache__"],
            )

            unified_ecr = ecr.Repository(
                self,
                "UnifiedECR",
                repository_name="gr00t-rl-unified",
                removal_policy=RemovalPolicy.RETAIN,
                empty_on_delete=False,
                image_scan_on_push=True,
                lifecycle_rules=[
                    ecr.LifecycleRule(max_image_count=10, rule_priority=1)
                ],
            )

            unified_build = codebuild.Project(
                self,
                "UnifiedBuild",
                project_name="GR00T-RL-Unified-Build",
                description="Build unified GR00T RL container (Isaac Sim + torch 2.8 + flash-attn + Ray + all deps)",
                source=codebuild.Source.s3(
                    bucket=source_asset.bucket,
                    path=source_asset.s3_object_key,
                ),
                environment=codebuild.BuildEnvironment(
                    build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                    compute_type=codebuild.ComputeType.X2_LARGE,  # 72 vCPU, 145 GB RAM, 824 GB disk — needed for Isaac Sim base image
                    privileged=True,
                ),
                build_spec=codebuild.BuildSpec.from_source_filename("buildspec.yml"),
                environment_variables={
                    "ECR_REPOSITORY_NAME": codebuild.BuildEnvironmentVariable(
                        value="gr00t-rl-unified"
                    ),
                    "DOCKERFILE": codebuild.BuildEnvironmentVariable(
                        value="Dockerfile.unified"
                    ),
                    "IMAGE_TAG": codebuild.BuildEnvironmentVariable(value="latest"),
                    "NGC_API_KEY": codebuild.BuildEnvironmentVariable(
                        value="/gr00t-rl/ngc-api-key",
                        type=codebuild.BuildEnvironmentVariableType.PARAMETER_STORE,
                    ),
                },
                timeout=Duration.hours(3),
                # Note: No local cache — not supported on X2_LARGE compute type
            )
            unified_ecr.grant_pull_push(unified_build.role)
            source_asset.grant_read(unified_build.role)
            unified_build.role.add_to_policy(
                iam.PolicyStatement(
                    actions=["ecr:GetAuthorizationToken"], resources=["*"]
                )
            )

            unified_trigger = cr.AwsCustomResource(
                self,
                "UnifiedBuildTrigger",
                policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                    resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE
                ),
                timeout=Duration.minutes(5),
                on_create=cr.AwsSdkCall(
                    service="CodeBuild",
                    action="startBuild",
                    parameters={"projectName": unified_build.project_name},
                    physical_resource_id=cr.PhysicalResourceId.of(
                        f"{unified_build.project_name}-{source_asset.s3_object_key}"
                    ),
                ),
                on_update=cr.AwsSdkCall(
                    service="CodeBuild",
                    action="batchGetProjects",
                    parameters={"names": [unified_build.project_name]},
                    physical_resource_id=cr.PhysicalResourceId.of(
                        f"{unified_build.project_name}-{source_asset.s3_object_key}"
                    ),
                ),
                install_latest_aws_sdk=True,
            )
            unified_trigger.node.add_dependency(unified_build)

            unified_image = ecs.ContainerImage.from_ecr_repository(
                unified_ecr, tag="latest"
            )
        else:
            # Pre-built image path: import existing ECR repository
            repo_tag = image_uri.split("/")[-1]
            repo_name = repo_tag.split(":")[0] if ":" in repo_tag else repo_tag
            tag = repo_tag.split(":")[1] if ":" in repo_tag else "latest"
            unified_ecr = ecr.Repository.from_repository_name(
                self, "UnifiedRepoImport", repository_name=repo_name
            )
            unified_image = ecs.ContainerImage.from_ecr_repository(unified_ecr, tag=tag)
        # endregion

        # ==============================================================
        # region 9. EFS volume
        # ==============================================================
        efs_volume = batch.EcsVolume.efs(
            name="rl-efs", file_system=efs_fs, container_path="/mnt/efs"
        )
        # endregion

        # ==============================================================
        # region 10. MNP Job Definition
        # ==============================================================
        # AWS Batch MNP requires homogeneous instance types across all nodes.
        # Two compute backends are supported via --context compute_backend:
        #   "batch-mnp" (default): All nodes use the same instance type (g6e.4xlarge).
        #     Learner uses 1 GPU; for production FSDP use compute_backend=sagemaker.
        #   "sagemaker": Batch queue submits to SageMaker Training with heterogeneous
        #     InstanceGroups (g6e.48xlarge learner + g6e.4xlarge rollouts).

        compute_backend = self.node.try_get_context("compute_backend") or "batch-mnp"

        shared_env = {
            "EFS_MOUNT": "/mnt/efs",
            "S3_BUCKET": artifact_bucket.bucket_name,
            "S3_PREFIX": "rl-training",
            "RAY_PORT": "6379",
            "RAY_DASHBOARD_PORT": "8265",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_IB_DISABLE": "1",
            "NCCL_DEBUG": "WARN",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
            "MODEL_PATH": "/mnt/efs/models/GR00T-N1.5-RL-Rheo-AssembleTrocar",
            "CONFIG_NAME": "isaaclab_ppo_gr00t_assemble_trocar",
            "NUM_ROLLOUT_ENVS": "64",
        }

        total_nodes = 1 + num_rollout_nodes

        # Resolve image URIs for SageMaker backend references
        unified_image_uri = image_uri or f"{unified_ecr.repository_uri}:latest"

        if compute_backend == "batch-mnp":
            # Homogeneous MNP: all nodes use the SAME unified image (g6e.4xlarge, 1 GPU each).
            # The unified image contains Isaac Sim + RLinf + GR00T + Ray + all deps.
            # Entrypoint differentiates learner vs rollout by AWS_BATCH_JOB_NODE_INDEX.
            # For production 8-GPU FSDP training, use compute_backend=sagemaker.

            job_def = batch.MultiNodeJobDefinition(
                self,
                "RLTrainingJobDef",
                main_node=0,
                instance_type=ec2.InstanceType("g6e.4xlarge"),
                containers=[
                    batch.MultiNodeContainer(
                        container=batch.EcsEc2ContainerDefinition(
                            self,
                            "LearnerContainer",
                            image=unified_image,
                            memory=Size.gibibytes(56),
                            cpu=14,
                            gpu=1,
                            job_role=job_role,
                            environment={
                                **shared_env,
                                "NODE_ROLE": "learner",
                                "CUDA_VISIBLE_DEVICES": "0",
                                "NUM_LEARNER_GPUS": "1",
                            },
                            volumes=[efs_volume],
                            linux_parameters=batch.LinuxParameters(
                                self,
                                "LearnerLinuxParams",
                                shared_memory_size=Size.gibibytes(32),
                            ),
                        ),
                        start_node=0,
                        end_node=0,
                    ),
                    batch.MultiNodeContainer(
                        container=batch.EcsEc2ContainerDefinition(
                            self,
                            "RolloutContainer",
                            image=unified_image,
                            memory=Size.gibibytes(56),
                            cpu=14,
                            gpu=1,
                            job_role=job_role,
                            environment={
                                **shared_env,
                                "NODE_ROLE": "rollout",
                                "CUDA_VISIBLE_DEVICES": "0",
                            },
                            volumes=[efs_volume],
                            linux_parameters=batch.LinuxParameters(
                                self,
                                "RolloutLinuxParams",
                                shared_memory_size=Size.gibibytes(32),
                            ),
                        ),
                        start_node=1,
                        end_node=total_nodes - 1,
                    ),
                ],
                retry_attempts=1,
                timeout=Duration.hours(24),
            )
            job_def_arn_output = job_def.job_definition_arn

        elif compute_backend == "sagemaker":
            # ==========================================================
            # Batch → SageMaker Training with heterogeneous InstanceGroups
            # ==========================================================
            # Uses AWS Batch's SageMaker integration (2025) to submit training
            # jobs with heterogeneous instance groups:
            #   - Learner group: 1× g6e.48xlarge (8× L40S) for FSDP training
            #   - Rollout group: N× g6e.4xlarge (1× L40S) for Isaac Sim workers
            # Batch handles scheduling/queuing; SageMaker handles gang scheduling,
            # service discovery, and instance lifecycle.
            # Reference: https://aws.amazon.com/blogs/machine-learning/introducing-aws-batch-support-for-amazon-sagemaker-training-jobs/

            # --- SageMaker Execution Role ---
            sagemaker_execution_role = iam.Role(
                self,
                "SageMakerExecutionRole",
                role_name="GR00T-RL-SageMaker-ExecutionRole",
                assumed_by=iam.CompositePrincipal(
                    iam.ServicePrincipal("sagemaker.amazonaws.com"),
                    iam.ServicePrincipal("batch.amazonaws.com"),
                ),
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name(
                        "AmazonSageMakerFullAccess"
                    ),
                ],
            )

            # Grant access to S3 artifacts bucket
            artifact_bucket.grant_read_write(sagemaker_execution_role)

            # Grant ECR pull for training images
            sagemaker_execution_role.add_to_policy(
                iam.PolicyStatement(
                    sid="ECRPullAccess",
                    actions=[
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetAuthorizationToken",
                    ],
                    resources=["*"],
                )
            )

            # Grant EFS access for model/code/checkpoint sharing
            sagemaker_execution_role.add_to_policy(
                iam.PolicyStatement(
                    sid="EFSAccess",
                    actions=[
                        "elasticfilesystem:ClientMount",
                        "elasticfilesystem:ClientWrite",
                        "elasticfilesystem:ClientRootAccess",
                        "elasticfilesystem:DescribeMountTargets",
                        "elasticfilesystem:DescribeFileSystems",
                    ],
                    resources=[
                        f"arn:aws:elasticfilesystem:{Stack.of(self).region}:{Stack.of(self).account}:file-system/{efs_fs.file_system_id}"
                    ],
                )
            )

            # Grant VPC networking permissions for SageMaker Training
            sagemaker_execution_role.add_to_policy(
                iam.PolicyStatement(
                    sid="VPCNetworking",
                    actions=[
                        "ec2:CreateNetworkInterface",
                        "ec2:CreateNetworkInterfacePermission",
                        "ec2:DeleteNetworkInterface",
                        "ec2:DeleteNetworkInterfacePermission",
                        "ec2:DescribeNetworkInterfaces",
                        "ec2:DescribeVpcs",
                        "ec2:DescribeDhcpOptions",
                        "ec2:DescribeSubnets",
                        "ec2:DescribeSecurityGroups",
                    ],
                    resources=["*"],
                )
            )

            # CloudWatch Logs for training job output
            sagemaker_execution_role.add_to_policy(
                iam.PolicyStatement(
                    sid="CloudWatchLogs",
                    actions=[
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:DescribeLogStreams",
                    ],
                    resources=[
                        f"arn:aws:logs:{Stack.of(self).region}:{Stack.of(self).account}:log-group:/aws/sagemaker/TrainingJobs*"
                    ],
                )
            )

            # --- Batch Service Role for SageMaker submissions ---
            batch_sagemaker_service_role = iam.Role(
                self,
                "BatchSageMakerServiceRole",
                role_name="GR00T-RL-Batch-SageMaker-ServiceRole",
                assumed_by=iam.ServicePrincipal("batch.amazonaws.com"),
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name(
                        "service-role/AWSBatchServiceRole"
                    ),
                ],
            )
            batch_sagemaker_service_role.add_to_policy(
                iam.PolicyStatement(
                    sid="SageMakerTrainingAccess",
                    actions=[
                        "sagemaker:CreateTrainingJob",
                        "sagemaker:DescribeTrainingJob",
                        "sagemaker:StopTrainingJob",
                        "sagemaker:ListTags",
                        "sagemaker:AddTags",
                    ],
                    resources=[
                        f"arn:aws:sagemaker:{Stack.of(self).region}:{Stack.of(self).account}:training-job/gr00t-rl-*"
                    ],
                )
            )
            batch_sagemaker_service_role.add_to_policy(
                iam.PolicyStatement(
                    sid="PassRoleToSageMaker",
                    actions=["iam:PassRole"],
                    resources=[sagemaker_execution_role.role_arn],
                    conditions={
                        "StringEquals": {
                            "iam:PassedToService": "sagemaker.amazonaws.com"
                        }
                    },
                )
            )

            # --- SageMaker-type Compute Environment (L1/CfnResource) ---
            # This is a "MANAGED" compute environment with serviceRole that targets
            # SageMaker Training instead of EC2/ECS.
            private_subnets = vpc.select_subnets(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            )

            sm_compute_env = batch.CfnComputeEnvironment(
                self,
                "SageMakerComputeEnv",
                compute_environment_name="GR00T-RL-SageMaker-ComputeEnv",
                type="MANAGED",
                state="ENABLED",
                service_role=batch_sagemaker_service_role.role_arn,
                eks_configuration=None,
                compute_resources=batch.CfnComputeEnvironment.ComputeResourcesProperty(
                    type="SAGEMAKER",
                    maxv_cpus=512,
                    subnets=private_subnets.subnet_ids,
                    security_group_ids=[batch_sg.security_group_id],
                ),
            )

            # --- Job Queue for SageMaker compute environment ---
            sm_job_queue = batch.CfnJobQueue(
                self,
                "SageMakerJobQueue",
                job_queue_name="GR00T-RL-SageMaker-JobQueue",
                state="ENABLED",
                priority=1,
                compute_environment_order=[
                    batch.CfnJobQueue.ComputeEnvironmentOrderProperty(
                        compute_environment=sm_compute_env.attr_compute_environment_arn,
                        order=1,
                    )
                ],
            )

            # --- Hyperparameters (passed as environment to SageMaker containers) ---
            hyperparameters = {
                **{k: str(v) for k, v in shared_env.items()},
                "NUM_ROLLOUT_NODES": str(num_rollout_nodes),
                "TOTAL_NODES": str(total_nodes),
            }

            # --- SageMaker Training Job Definition (Custom Resource for Batch API) ---
            # This defines the heterogeneous training job that Batch submits to SageMaker.
            # Uses AWS::Batch::JobDefinition with SageMaker-specific properties via override.
            # Since CDK L2 constructs don't yet support this, we use CfnResource.

            sm_training_job_template = cr.AwsCustomResource(
                self,
                "SageMakerTrainingJobTemplate",
                policy=cr.AwsCustomResourcePolicy.from_statements([
                    iam.PolicyStatement(
                        actions=[
                            "batch:RegisterJobDefinition",
                            "batch:DeregisterJobDefinition",
                            "batch:DescribeJobDefinitions",
                        ],
                        resources=["*"],
                    ),
                    iam.PolicyStatement(
                        actions=["iam:PassRole"],
                        resources=[
                            sagemaker_execution_role.role_arn,
                            job_role.role_arn,
                        ],
                    ),
                ]),
                timeout=Duration.minutes(5),
                on_create=cr.AwsSdkCall(
                    service="Batch",
                    action="registerJobDefinition",
                    parameters={
                        "jobDefinitionName": "GR00T-RL-SageMaker-HeterogeneousTraining",
                        "type": "container",
                        "platformCapabilities": ["EC2"],
                        "retryStrategy": {
                            "attempts": 2,
                            "evaluateOnExit": [
                                {
                                    "action": "RETRY",
                                    "onStatusReason": "Host EC2*",
                                },
                                {
                                    "action": "EXIT",
                                    "onStatusReason": "*",
                                },
                            ],
                        },
                        "timeout": {"attemptDurationSeconds": 86400},
                        "containerProperties": {
                            "image": unified_image_uri,
                            "command": ["python", "-m", "rlinf.train"],
                            "jobRoleArn": job_role.role_arn,
                            "executionRoleArn": sagemaker_execution_role.role_arn,
                            "resourceRequirements": [
                                {"type": "VCPU", "value": "1"},
                                {"type": "MEMORY", "value": "2048"},
                            ],
                            "environment": [
                                {"name": k, "value": v}
                                for k, v in hyperparameters.items()
                            ],
                        },
                        # SageMaker Training configuration for heterogeneous instance groups
                        "sageMakerConfig": {
                            "roleArn": sagemaker_execution_role.role_arn,
                            "outputDataConfig": {
                                "s3OutputPath": f"s3://{artifact_bucket.bucket_name}/sagemaker-output/",
                            },
                            "resourceConfig": {
                                "instanceGroups": [
                                    {
                                        "instanceGroupName": "learner",
                                        "instanceType": "ml.g6e.48xlarge",
                                        "instanceCount": 1,
                                        "instanceStorageConfigs": [
                                            {
                                                "ebsVolumeConfig": {
                                                    "volumeSizeInGb": 500,
                                                }
                                            }
                                        ],
                                    },
                                    {
                                        "instanceGroupName": "rollout",
                                        "instanceType": "ml.g6e.4xlarge",
                                        "instanceCount": num_rollout_nodes,
                                        "instanceStorageConfigs": [
                                            {
                                                "ebsVolumeConfig": {
                                                    "volumeSizeInGb": 200,
                                                }
                                            }
                                        ],
                                    },
                                ],
                            },
                            "vpcConfig": {
                                "securityGroupIds": [batch_sg.security_group_id],
                                "subnets": private_subnets.subnet_ids,
                            },
                            "stoppingCondition": {
                                "maxRuntimeInSeconds": 86400,  # 24 hours
                            },
                            "algorithmSpecification": {
                                "trainingInputMode": "File",
                                "trainingImage": unified_image_uri,
                                "containerEntrypoint": ["python", "-m", "rlinf.train"],
                                "containerArguments": [
                                    "--config", shared_env["CONFIG_NAME"],
                                    "--model-path", shared_env["MODEL_PATH"],
                                    "--num-rollout-nodes", str(num_rollout_nodes),
                                ],
                                "instanceGroupAlgorithmSpecifications": [
                                    {
                                        "instanceGroupName": "learner",
                                        "trainingImage": unified_image_uri,
                                        "containerEntrypoint": ["python", "-m", "rlinf.train"],
                                        "containerArguments": [
                                            "--node-role", "learner",
                                            "--config", shared_env["CONFIG_NAME"],
                                            "--model-path", shared_env["MODEL_PATH"],
                                            "--num-gpus", "8",
                                            "--fsdp",
                                        ],
                                    },
                                    {
                                        "instanceGroupName": "rollout",
                                        "trainingImage": unified_image_uri,
                                        "containerEntrypoint": ["python", "-m", "rlinf.rollout_worker"],
                                        "containerArguments": [
                                            "--node-role", "rollout",
                                            "--config", shared_env["CONFIG_NAME"],
                                            "--num-envs", shared_env["NUM_ROLLOUT_ENVS"],
                                        ],
                                    },
                                ],
                            },
                            "hyperParameters": hyperparameters,
                            "inputDataConfig": [
                                {
                                    "channelName": "model",
                                    "dataSource": {
                                        "fileSystemDataSource": {
                                            "fileSystemId": efs_fs.file_system_id,
                                            "fileSystemType": "EFS",
                                            "directoryPath": "/models/GR00T-N1.5-RL-Rheo-AssembleTrocar",
                                            "fileSystemAccessMode": "ro",
                                        }
                                    },
                                },
                                {
                                    "channelName": "code",
                                    "dataSource": {
                                        "fileSystemDataSource": {
                                            "fileSystemId": efs_fs.file_system_id,
                                            "fileSystemType": "EFS",
                                            "directoryPath": "/training-code",
                                            "fileSystemAccessMode": "ro",
                                        }
                                    },
                                },
                                {
                                    "channelName": "checkpoints",
                                    "dataSource": {
                                        "fileSystemDataSource": {
                                            "fileSystemId": efs_fs.file_system_id,
                                            "fileSystemType": "EFS",
                                            "directoryPath": "/checkpoints",
                                            "fileSystemAccessMode": "rw",
                                        }
                                    },
                                },
                            ],
                        },
                    },
                    physical_resource_id=cr.PhysicalResourceId.from_response(
                        "jobDefinitionArn"
                    ),
                ),
                on_update=cr.AwsSdkCall(
                    service="Batch",
                    action="registerJobDefinition",
                    parameters={
                        "jobDefinitionName": "GR00T-RL-SageMaker-HeterogeneousTraining",
                        "type": "container",
                        "platformCapabilities": ["EC2"],
                        "retryStrategy": {
                            "attempts": 2,
                            "evaluateOnExit": [
                                {
                                    "action": "RETRY",
                                    "onStatusReason": "Host EC2*",
                                },
                                {
                                    "action": "EXIT",
                                    "onStatusReason": "*",
                                },
                            ],
                        },
                        "timeout": {"attemptDurationSeconds": 86400},
                        "containerProperties": {
                            "image": unified_image_uri,
                            "command": ["python", "-m", "rlinf.train"],
                            "jobRoleArn": job_role.role_arn,
                            "executionRoleArn": sagemaker_execution_role.role_arn,
                            "resourceRequirements": [
                                {"type": "VCPU", "value": "1"},
                                {"type": "MEMORY", "value": "2048"},
                            ],
                            "environment": [
                                {"name": k, "value": v}
                                for k, v in hyperparameters.items()
                            ],
                        },
                        "sageMakerConfig": {
                            "roleArn": sagemaker_execution_role.role_arn,
                            "outputDataConfig": {
                                "s3OutputPath": f"s3://{artifact_bucket.bucket_name}/sagemaker-output/",
                            },
                            "resourceConfig": {
                                "instanceGroups": [
                                    {
                                        "instanceGroupName": "learner",
                                        "instanceType": "ml.g6e.48xlarge",
                                        "instanceCount": 1,
                                        "instanceStorageConfigs": [
                                            {
                                                "ebsVolumeConfig": {
                                                    "volumeSizeInGb": 500,
                                                }
                                            }
                                        ],
                                    },
                                    {
                                        "instanceGroupName": "rollout",
                                        "instanceType": "ml.g6e.4xlarge",
                                        "instanceCount": num_rollout_nodes,
                                        "instanceStorageConfigs": [
                                            {
                                                "ebsVolumeConfig": {
                                                    "volumeSizeInGb": 200,
                                                }
                                            }
                                        ],
                                    },
                                ],
                            },
                            "vpcConfig": {
                                "securityGroupIds": [batch_sg.security_group_id],
                                "subnets": private_subnets.subnet_ids,
                            },
                            "stoppingCondition": {
                                "maxRuntimeInSeconds": 86400,
                            },
                            "algorithmSpecification": {
                                "trainingInputMode": "File",
                                "trainingImage": unified_image_uri,
                                "containerEntrypoint": ["python", "-m", "rlinf.train"],
                                "containerArguments": [
                                    "--config", shared_env["CONFIG_NAME"],
                                    "--model-path", shared_env["MODEL_PATH"],
                                    "--num-rollout-nodes", str(num_rollout_nodes),
                                ],
                                "instanceGroupAlgorithmSpecifications": [
                                    {
                                        "instanceGroupName": "learner",
                                        "trainingImage": unified_image_uri,
                                        "containerEntrypoint": ["python", "-m", "rlinf.train"],
                                        "containerArguments": [
                                            "--node-role", "learner",
                                            "--config", shared_env["CONFIG_NAME"],
                                            "--model-path", shared_env["MODEL_PATH"],
                                            "--num-gpus", "8",
                                            "--fsdp",
                                        ],
                                    },
                                    {
                                        "instanceGroupName": "rollout",
                                        "trainingImage": unified_image_uri,
                                        "containerEntrypoint": ["python", "-m", "rlinf.rollout_worker"],
                                        "containerArguments": [
                                            "--node-role", "rollout",
                                            "--config", shared_env["CONFIG_NAME"],
                                            "--num-envs", shared_env["NUM_ROLLOUT_ENVS"],
                                        ],
                                    },
                                ],
                            },
                            "hyperParameters": hyperparameters,
                            "inputDataConfig": [
                                {
                                    "channelName": "model",
                                    "dataSource": {
                                        "fileSystemDataSource": {
                                            "fileSystemId": efs_fs.file_system_id,
                                            "fileSystemType": "EFS",
                                            "directoryPath": "/models/GR00T-N1.5-RL-Rheo-AssembleTrocar",
                                            "fileSystemAccessMode": "ro",
                                        }
                                    },
                                },
                                {
                                    "channelName": "code",
                                    "dataSource": {
                                        "fileSystemDataSource": {
                                            "fileSystemId": efs_fs.file_system_id,
                                            "fileSystemType": "EFS",
                                            "directoryPath": "/training-code",
                                            "fileSystemAccessMode": "ro",
                                        }
                                    },
                                },
                                {
                                    "channelName": "checkpoints",
                                    "dataSource": {
                                        "fileSystemDataSource": {
                                            "fileSystemId": efs_fs.file_system_id,
                                            "fileSystemType": "EFS",
                                            "directoryPath": "/checkpoints",
                                            "fileSystemAccessMode": "rw",
                                        }
                                    },
                                },
                            ],
                        },
                    },
                    physical_resource_id=cr.PhysicalResourceId.from_response(
                        "jobDefinitionArn"
                    ),
                ),
                on_delete=cr.AwsSdkCall(
                    service="Batch",
                    action="deregisterJobDefinition",
                    parameters={
                        "jobDefinition": cr.PhysicalResourceIdReference(),
                    },
                ),
                install_latest_aws_sdk=True,
            )
            sm_training_job_template.node.add_dependency(sm_compute_env)

            # Store ARN for outputs
            sm_job_def_arn = sm_training_job_template.get_response_field(
                "jobDefinitionArn"
            )

            # --- CDK Outputs for SageMaker backend ---
            CfnOutput(
                self,
                "SageMakerComputeEnvArn",
                value=sm_compute_env.attr_compute_environment_arn,
                description="SageMaker-type Batch compute environment ARN",
            )
            CfnOutput(
                self,
                "SageMakerJobQueueArn",
                value=sm_job_queue.attr_job_queue_arn,
                description="Job queue for SageMaker training submissions",
            )
            CfnOutput(
                self,
                "SageMakerExecutionRoleArn",
                value=sagemaker_execution_role.role_arn,
                description="SageMaker execution role for training jobs",
            )
            CfnOutput(
                self,
                "SageMakerJobDefinitionArn",
                value=sm_job_def_arn,
                description="Batch job definition ARN for heterogeneous SageMaker training",
            )
            CfnOutput(
                self,
                "SageMakerLearnerInstanceType",
                value="ml.g6e.48xlarge (8x L40S, FSDP)",
                description="Learner instance group type",
            )
            CfnOutput(
                self,
                "SageMakerRolloutInstanceType",
                value=f"ml.g6e.4xlarge (1x L40S) x {num_rollout_nodes}",
                description="Rollout instance group type and count",
            )
            CfnOutput(
                self,
                "SageMakerOutputPath",
                value=f"s3://{artifact_bucket.bucket_name}/sagemaker-output/",
                description="S3 output path for SageMaker training artifacts",
            )
        else:
            raise ValueError(
                f"Unknown compute_backend: {compute_backend}. "
                "Supported: 'batch-mnp', 'sagemaker'"
            )
        # endregion

        # ==============================================================
        # region 11. EFS staging via CodeBuild (model + code)
        # ==============================================================
        stage_source = s3_assets.Asset(
            self,
            "StageEFSSourceAsset",
            path=docker_dir,
            exclude=["*.pyc", "__pycache__", "Dockerfile.*"],
        )

        efs_stage_sg = ec2.SecurityGroup(
            self,
            "EFSStageSG",
            vpc=vpc,
            description="SG for CodeBuild EFS staging project",
            allow_all_outbound=True,
        )
        efs_fs.connections.allow_default_port_from(efs_stage_sg)

        efs_stage_build = codebuild.Project(
            self,
            "EFSStageBuild",
            project_name="GR00T-RL-Stage-EFS",
            description="Stage model checkpoint and training code onto EFS",
            source=codebuild.Source.s3(
                bucket=stage_source.bucket,
                path=stage_source.s3_object_key,
            ),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.LARGE,
                privileged=True,  # Required for EFS mount
            ),
            build_spec=codebuild.BuildSpec.from_source_filename(
                "buildspec-stage-efs.yml"
            ),
            vpc=vpc,
            subnet_selection=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            security_groups=[efs_stage_sg],
            file_system_locations=[
                codebuild.FileSystemLocation.efs(
                    identifier="efs_mount",
                    location=f"{efs_fs.file_system_id}.efs.{Stack.of(self).region}.amazonaws.com:/",
                    mount_point="/mnt/efs",
                    mount_options="nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2",
                ),
            ],
            timeout=Duration.hours(1),
        )
        stage_source.grant_read(efs_stage_build.role)
        efs_stage_build.role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "elasticfilesystem:ClientMount",
                    "elasticfilesystem:ClientWrite",
                    "elasticfilesystem:ClientRootAccess",
                ],
                resources=[
                    f"arn:aws:elasticfilesystem:{Stack.of(self).region}:{Stack.of(self).account}:file-system/{efs_fs.file_system_id}"
                ],
            )
        )

        # Auto-trigger EFS staging on first deploy
        efs_stage_trigger = cr.AwsCustomResource(
            self,
            "EFSStageTrigger",
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE
            ),
            timeout=Duration.minutes(5),
            on_create=cr.AwsSdkCall(
                service="CodeBuild",
                action="startBuild",
                parameters={"projectName": efs_stage_build.project_name},
                physical_resource_id=cr.PhysicalResourceId.of(
                    f"{efs_stage_build.project_name}-initial"
                ),
            ),
            on_update=cr.AwsSdkCall(
                service="CodeBuild",
                action="batchGetProjects",
                parameters={"names": [efs_stage_build.project_name]},
                physical_resource_id=cr.PhysicalResourceId.of(
                    f"{efs_stage_build.project_name}-initial"
                ),
            ),
            install_latest_aws_sdk=True,
        )
        efs_stage_trigger.node.add_dependency(efs_stage_build)
        # endregion

        # ==============================================================
        # region 12. Outputs
        # ==============================================================
        CfnOutput(self, "VpcId", value=vpc.vpc_id)
        CfnOutput(self, "EFSFileSystemId", value=efs_fs.file_system_id)
        CfnOutput(self, "ArtifactBucket", value=artifact_bucket.bucket_name)
        CfnOutput(self, "JobQueueName", value=job_queue.job_queue_name)
        CfnOutput(
            self, "ComputeEnvironmentName", value=compute_env.compute_environment_name
        )
        if unified_build:
            CfnOutput(self, "UnifiedECRUri", value=unified_ecr.repository_uri)
            CfnOutput(self, "UnifiedBuildProject", value=unified_build.project_name)
        CfnOutput(self, "EFSStageProject", value=efs_stage_build.project_name)
        CfnOutput(
            self,
            "JobDefinitionArn",
            value=job_def_arn_output if compute_backend == "batch-mnp" else "See SageMakerJobDefinitionArn output",
        )
        CfnOutput(self, "NumRolloutNodes", value=str(num_rollout_nodes))
        # endregion
