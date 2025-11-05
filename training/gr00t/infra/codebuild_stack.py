import os
from aws_cdk import (
    aws_codebuild as codebuild,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_assets as s3_assets,
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
            "Gr00tEcrRepository",
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
        # Package the local source code (Dockerfile, scripts, etc.) and upload to S3
        # CodeBuild will download from S3 to build the container
        source_asset = s3_assets.Asset(
            self,
            "Gr00tSourceAsset",
            path=os.path.join(os.path.dirname(__file__), ".."),  # training/gr00t/
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
            "Gr00tContainerBuild",
            project_name="Gr00tContainerBuild",
            description="Build GR00T fine-tuning container and push to ECR",
            # Source: Use the S3 asset created above
            source=codebuild.Source.s3(
                bucket=source_asset.bucket,
                path=source_asset.s3_object_key,
            ),
            # Build environment
            environment=codebuild.BuildEnvironment(
                # Use x86_64 architecture (required for GR00T)
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
        # 5. Outputs
        # ==============================================================
        CfnOutput(
            self,
            "EcrRepositoryUri",
            value=ecr_repo.repository_uri,
            description="ECR repository URI for the GR00T container image",
        )

        CfnOutput(
            self,
            "EcrRepositoryName",
            value=ecr_repo.repository_name,
            description="ECR repository name",
        )

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
            "ImageUri",
            value=f"{ecr_repo.repository_uri}:latest",
            description="Full ECR image URI (use this for Batch stack deployment)",
        )

        # Store attributes for cross-stack references
        self.ecr_repository = ecr_repo
        self.build_project = build_project
        self.image_uri = f"{ecr_repo.repository_uri}:latest"
