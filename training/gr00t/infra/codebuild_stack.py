import os
from aws_cdk import (
    aws_codebuild as codebuild,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_s3_assets as s3_assets,
    custom_resources as cr,
    CfnOutput,
    RemovalPolicy,
    Duration,
)
from constructs import Construct


class CodeBuildStack(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        ecr_repository_name: str = "gr00t-finetune",
        use_stable: bool = True,
    ) -> None:
        """
        CDK construct for AWS CodeBuild project to build GR00T fine-tuning container.

        This construct creates:
        - ECR repository for storing container images
        - CodeBuild project with x86 compute for building containers
        - IAM roles and permissions
        - S3 bucket for source code (if using local source)

        Args:
            ecr_repository_name: Name for the ECR repository (default: gr00t-finetune)
            use_stable: Use stable GR00T commit vs latest (default: True)
        """
        super().__init__(scope, construct_id)

        # ==============================================================
        # 1. ECR Repository
        # ==============================================================
        # Create ECR repository to store the built container images
        ecr_repo = ecr.Repository(
            self,
            "IsaacGr00tEcrRepository",
            repository_name=ecr_repository_name,
            removal_policy=RemovalPolicy.RETAIN,  # Keep images after stack deletion
            image_scan_on_push=True,  # Scan for vulnerabilities
            lifecycle_rules=[
                # Keep last 10 images, delete older ones
                ecr.LifecycleRule(
                    description="Keep last 10 images",
                    max_image_count=10,
                    rule_priority=1,
                )
            ],
        )

        # ==============================================================
        # 2. Source Code Asset
        # ==============================================================
        # Package the local source code and upload to a CDK managed S3 bucket
        # CodeBuild will download from S3 to build the container
        asset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )  # training/gr00t/
        source_asset = s3_assets.Asset(
            self,
            "IsaacGr00tSourceAsset",
            path=asset_path,
            exclude=[
                ".git",
                ".gitignore",
                "*.pyc",
                "__pycache__",
                ".venv",
                "venv",
                "*.egg-info",
                ".pytest_cache",
                ".mypy_cache",
                "infra/cdk.out",
                "infra/.cdk.staging",
                "infra/codebuild/cdk.out",
                "infra/codebuild/.cdk.staging",
            ],
        )

        # ==============================================================
        # 3. CodeBuild Project
        # ==============================================================
        # Create CodeBuild project to build the Docker image
        build_project = codebuild.Project(
            self,
            "IsaacGr00tContainerBuild",
            project_name="IsaacGr00tContainerBuild",
            description="Build GR00T fine-tuning container and push to ECR",
            # Source: Use the S3 asset created above
            # Customize the source to use your own Git repository
            source=codebuild.Source.s3(
                bucket=source_asset.bucket,
                path=source_asset.s3_object_key,
            ),
            # Build environment
            environment=codebuild.BuildEnvironment(
                # Use x86_64 architecture (required for EC2 G6e, P4 and P5 instances)
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,  # Ubuntu 22.04, Docker 24
                compute_type=codebuild.ComputeType.LARGE,  # 8 vCPU, 15 GB RAM
                privileged=True,  # Required for Docker builds
            ),
            # Build specification (located in infra/ directory)
            build_spec=codebuild.BuildSpec.from_source_filename("infra/buildspec.yml"),
            # Environment variables
            environment_variables={
                "ECR_REPOSITORY_NAME": codebuild.BuildEnvironmentVariable(
                    value=ecr_repository_name
                ),
                "USE_STABLE": codebuild.BuildEnvironmentVariable(
                    value="true" if use_stable else "false"
                ),
                "IMAGE_TAG": codebuild.BuildEnvironmentVariable(value="latest"),
            },
            # Timeout (building flash-attn takes time)
            timeout=Duration.hours(2),
            # Cache for faster rebuilds (optional)
            cache=codebuild.Cache.local(
                codebuild.LocalCacheMode.DOCKER_LAYER,
                codebuild.LocalCacheMode.CUSTOM,
            ),
        )

        # ==============================================================
        # 4. IAM Permissions
        # ==============================================================
        # Grant CodeBuild permissions to push to ECR
        ecr_repo.grant_pull_push(build_project.role)

        # Grant permissions to read source from S3
        source_asset.grant_read(build_project.role)

        # Add ECR authorization token permission (required for docker login)
        build_project.role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            )
        )

        # ==============================================================
        # 5. Auto-trigger Build on Stack Creation
        # ==============================================================
        # Automatically trigger a CodeBuild build when the stack is created
        # This ensures the container image is built immediately after deployment
        trigger_build = cr.AwsCustomResource(
            self,
            "AutoTriggerBuild",
            # Use from_sdk_calls for simpler policy management
            # This automatically grants the necessary permissions for the SDK call
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE
            ),
            # Lambda timeout should be minimal - it just triggers the build, doesn't wait for completion
            timeout=Duration.minutes(5),
            # Only trigger on CREATE (not UPDATE or DELETE)
            # This ensures builds only happen on initial stack creation
            on_create=cr.AwsSdkCall(
                service="CodeBuild",
                action="startBuild",
                parameters={
                    "projectName": build_project.project_name,
                },
                # Use a static physical resource ID to ensure idempotency
                # This prevents re-triggering on stack updates
                physical_resource_id=cr.PhysicalResourceId.of(
                    f"{build_project.project_name}-initial-build"
                ),
            ),
            # Install latest AWS SDK in Lambda runtime for latest API support
            install_latest_aws_sdk=True,
        )
        # Ensure the build project exists before triggering
        trigger_build.node.add_dependency(build_project)

        # ==============================================================
        # 6. Outputs
        # ==============================================================
        CfnOutput(
            self,
            "CodeBuildProjectName",
            value=build_project.project_name,
            description="CodeBuild project name for building the container",
        )

        CfnOutput(
            self,
            "BuildCommand",
            value=f"aws codebuild start-build --project-name {build_project.project_name}",
            description="Command to trigger a container build",
        )

        CfnOutput(
            self,
            "EcrRepositoryUri",
            value=ecr_repo.repository_uri,
            description="ECR repository URI for the GR00T container image",
        )

        CfnOutput(
            self,
            "ImageUri",
            value=f"{ecr_repo.repository_uri}:latest",
            description="Full ECR image URI (use this for Batch stack deployment)",
        )

        # Store attributes for cross-stack references
        self.ecr_repository = ecr_repo
        self.build_project = build_project
        self.image_uri = f"{ecr_repo.repository_uri}:latest"
