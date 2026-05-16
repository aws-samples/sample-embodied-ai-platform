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
        learner_image_uri: str = None,
        rollout_image_uri: str = None,
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
        #   1. Default: CodeBuild builds both images automatically on deploy
        #   2. Pre-built: Pass learner_image_uri / rollout_image_uri to skip CodeBuild
        docker_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "docker")
        )

        learner_build = None
        rollout_build = None

        if not learner_image_uri or not rollout_image_uri:
            source_asset = s3_assets.Asset(
                self,
                "RLDockerSourceAsset",
                path=docker_dir,
                exclude=["*.pyc", "__pycache__"],
            )

        # --- Learner image ---
        if learner_image_uri:
            repo_tag = learner_image_uri.split("/")[-1]
            repo_name = repo_tag.split(":")[0] if ":" in repo_tag else repo_tag
            tag = repo_tag.split(":")[1] if ":" in repo_tag else "latest"
            learner_repo = ecr.Repository.from_repository_name(
                self, "LearnerRepoImport", repository_name=repo_name
            )
            learner_image = ecs.ContainerImage.from_ecr_repository(learner_repo, tag=tag)
        else:
            learner_ecr = ecr.Repository(
                self,
                "LearnerECR",
                repository_name="gr00t-rl-learner",
                removal_policy=RemovalPolicy.RETAIN,
                empty_on_delete=False,
                image_scan_on_push=True,
                lifecycle_rules=[
                    ecr.LifecycleRule(max_image_count=10, rule_priority=1)
                ],
            )

            learner_build = codebuild.Project(
                self,
                "LearnerBuild",
                project_name="GR00T-RL-Learner-Build",
                description="Build GR00T RL learner container (RLinf + GR00T + Ray)",
                source=codebuild.Source.s3(
                    bucket=source_asset.bucket,
                    path=source_asset.s3_object_key,
                ),
                environment=codebuild.BuildEnvironment(
                    build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                    compute_type=codebuild.ComputeType.X_LARGE,  # 16 vCPU, 72 GB RAM, 368 GB disk
                    privileged=True,
                ),
                build_spec=codebuild.BuildSpec.from_source_filename("buildspec.yml"),
                environment_variables={
                    "ECR_REPOSITORY_NAME": codebuild.BuildEnvironmentVariable(
                        value="gr00t-rl-learner"
                    ),
                    "DOCKERFILE": codebuild.BuildEnvironmentVariable(
                        value="Dockerfile.learner"
                    ),
                    "IMAGE_TAG": codebuild.BuildEnvironmentVariable(value="latest"),
                },
                timeout=Duration.hours(2),
                cache=codebuild.Cache.local(
                    codebuild.LocalCacheMode.DOCKER_LAYER,
                ),
            )
            learner_ecr.grant_pull_push(learner_build.role)
            source_asset.grant_read(learner_build.role)
            learner_build.role.add_to_policy(
                iam.PolicyStatement(
                    actions=["ecr:GetAuthorizationToken"], resources=["*"]
                )
            )

            learner_trigger = cr.AwsCustomResource(
                self,
                "LearnerBuildTrigger",
                policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                    resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE
                ),
                timeout=Duration.minutes(5),
                on_create=cr.AwsSdkCall(
                    service="CodeBuild",
                    action="startBuild",
                    parameters={"projectName": learner_build.project_name},
                    physical_resource_id=cr.PhysicalResourceId.of(
                        f"{learner_build.project_name}-{source_asset.s3_object_key}"
                    ),
                ),
                on_update=cr.AwsSdkCall(
                    service="CodeBuild",
                    action="batchGetProjects",
                    parameters={"names": [learner_build.project_name]},
                    physical_resource_id=cr.PhysicalResourceId.of(
                        f"{learner_build.project_name}-{source_asset.s3_object_key}"
                    ),
                ),
                install_latest_aws_sdk=True,
            )
            learner_trigger.node.add_dependency(learner_build)

            learner_image = ecs.ContainerImage.from_ecr_repository(
                learner_ecr, tag="latest"
            )

        # --- Rollout image ---
        if rollout_image_uri:
            repo_tag = rollout_image_uri.split("/")[-1]
            repo_name = repo_tag.split(":")[0] if ":" in repo_tag else repo_tag
            tag = repo_tag.split(":")[1] if ":" in repo_tag else "latest"
            rollout_repo = ecr.Repository.from_repository_name(
                self, "RolloutRepoImport", repository_name=repo_name
            )
            rollout_image = ecs.ContainerImage.from_ecr_repository(rollout_repo, tag=tag)
        else:
            rollout_ecr = ecr.Repository(
                self,
                "RolloutECR",
                repository_name="gr00t-rl-rollout",
                removal_policy=RemovalPolicy.RETAIN,
                empty_on_delete=False,
                image_scan_on_push=True,
                lifecycle_rules=[
                    ecr.LifecycleRule(max_image_count=10, rule_priority=1)
                ],
            )

            rollout_build = codebuild.Project(
                self,
                "RolloutBuild",
                project_name="GR00T-RL-Rollout-Build",
                description="Build GR00T RL rollout container (Isaac Sim + RLinf + GR00T + Ray)",
                source=codebuild.Source.s3(
                    bucket=source_asset.bucket,
                    path=source_asset.s3_object_key,
                ),
                environment=codebuild.BuildEnvironment(
                    build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                    compute_type=codebuild.ComputeType.X2_LARGE,  # 72 vCPU, 145 GB RAM, 824 GB disk — needed for Isaac Sim
                    privileged=True,
                ),
                build_spec=codebuild.BuildSpec.from_source_filename("buildspec.yml"),
                environment_variables={
                    "ECR_REPOSITORY_NAME": codebuild.BuildEnvironmentVariable(
                        value="gr00t-rl-rollout"
                    ),
                    "DOCKERFILE": codebuild.BuildEnvironmentVariable(
                        value="Dockerfile.rollout"
                    ),
                    "IMAGE_TAG": codebuild.BuildEnvironmentVariable(value="latest"),
                },
                timeout=Duration.hours(3),
                # No local cache — not supported on X2_LARGE compute type
            )
            rollout_ecr.grant_pull_push(rollout_build.role)
            source_asset.grant_read(rollout_build.role)
            rollout_build.role.add_to_policy(
                iam.PolicyStatement(
                    actions=["ecr:GetAuthorizationToken"], resources=["*"]
                )
            )

            rollout_trigger = cr.AwsCustomResource(
                self,
                "RolloutBuildTrigger",
                policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                    resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE
                ),
                timeout=Duration.minutes(5),
                on_create=cr.AwsSdkCall(
                    service="CodeBuild",
                    action="startBuild",
                    parameters={"projectName": rollout_build.project_name},
                    physical_resource_id=cr.PhysicalResourceId.of(
                        f"{rollout_build.project_name}-{source_asset.s3_object_key}"
                    ),
                ),
                on_update=cr.AwsSdkCall(
                    service="CodeBuild",
                    action="batchGetProjects",
                    parameters={"names": [rollout_build.project_name]},
                    physical_resource_id=cr.PhysicalResourceId.of(
                        f"{rollout_build.project_name}-{source_asset.s3_object_key}"
                    ),
                ),
                install_latest_aws_sdk=True,
            )
            rollout_trigger.node.add_dependency(rollout_build)

            rollout_image = ecs.ContainerImage.from_ecr_repository(
                rollout_ecr, tag="latest"
            )
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

        learner_image_uri_resolved = learner_image_uri or f"{learner_ecr.repository_uri}:latest"
        rollout_image_uri_resolved = rollout_image_uri or f"{rollout_ecr.repository_uri}:latest"

        if compute_backend == "batch-mnp":
            # Homogeneous MNP: all nodes are g6e.4xlarge (1 GPU each)
            # Learner node uses 1 GPU — suitable for GR00T 3B in bf16 with smaller batch sizes.
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
                            image=ecs.ContainerImage.from_ecr_repository(
                                ecr.Repository.from_repository_name(
                                    self, "LearnerECRRef",
                                    repository_name="gr00t-rl-learner"
                                ), tag="latest"
                            ) if not learner_image_uri else ecs.ContainerImage.from_ecr_repository(
                                ecr.Repository.from_repository_name(
                                    self, "LearnerECRRef",
                                    repository_name=learner_image_uri.split("/")[-1].split(":")[0]
                                ), tag=learner_image_uri.split(":")[-1] if ":" in learner_image_uri else "latest"
                            ),
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
                            image=ecs.ContainerImage.from_ecr_repository(
                                ecr.Repository.from_repository_name(
                                    self, "RolloutECRRef",
                                    repository_name="gr00t-rl-rollout"
                                ), tag="latest"
                            ) if not rollout_image_uri else ecs.ContainerImage.from_ecr_repository(
                                ecr.Repository.from_repository_name(
                                    self, "RolloutECRRef",
                                    repository_name=rollout_image_uri.split("/")[-1].split(":")[0]
                                ), tag=rollout_image_uri.split(":")[-1] if ":" in rollout_image_uri else "latest"
                            ),
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
            # TODO: Implement Batch → SageMaker Training with heterogeneous InstanceGroups
            # - Batch ServiceEnvironment targeting SageMaker
            # - InstanceGroups: g6e.48xlarge (learner, 8 GPU FSDP) + g6e.4xlarge × N (rollouts)
            # - SageMaker handles gang scheduling and service discovery
            # Reference: https://aws.amazon.com/blogs/machine-learning/introducing-aws-batch-support-for-amazon-sagemaker-training-jobs/
            raise NotImplementedError(
                "SageMaker backend not yet implemented. "
                "Use --context compute_backend=batch-mnp for now."
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
        if learner_build:
            CfnOutput(self, "LearnerECRUri", value=learner_ecr.repository_uri)
            CfnOutput(self, "LearnerBuildProject", value=learner_build.project_name)
        if rollout_build:
            CfnOutput(self, "RolloutECRUri", value=rollout_ecr.repository_uri)
            CfnOutput(self, "RolloutBuildProject", value=rollout_build.project_name)
        CfnOutput(self, "EFSStageProject", value=efs_stage_build.project_name)
        CfnOutput(self, "JobDefinitionArn", value=job_def_arn_output if compute_backend == "batch-mnp" else "N/A")
        CfnOutput(self, "NumRolloutNodes", value=str(num_rollout_nodes))
        # endregion
