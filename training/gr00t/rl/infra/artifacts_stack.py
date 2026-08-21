"""Standalone artifacts stack for GR00T RL heterogeneous training.

Owns the SHARED, PERSISTENT build+stage artifacts that the EKS backend only
CONSUMES:
  - ECR repo `gr00t-rl-unified` (RETAIN) for the unified image
  - CodeBuild project `GR00T-RL-Pipeline` (mode-switched: build-image | stage-data
    | all) that builds+pushes the image and/or stages pinned third-party code +
    the RL model + workflows to the S3 data bucket the FSx-Lustre DRA imports from

Why a separate stack? These resources must outlive any single backend deploy.
Putting them INSIDE the EKS stack created a circular bootstrap (the EKS stack
needs an image to run, but built the image only when no image_uri was given) and
an ownership-flip bug (re-deploying the EKS stack WITH a resolved digest set
build_image=False, which then tried to REMOVE the ECR repo + pipeline out from
under a running system). This stack owns them once, persistently; the EKS stack
goes back to being a pure consumer of `image_uri` + the staged bucket.

Deploy this FIRST (before the backend stack), then drive it with
infra/prepare-artifacts.sh:
  cdk deploy GR00TRLArtifactsStack --context compute_backend=eks \\
    --context s3_data_bucket=<your-s3-bucket>

There is intentionally NO auto-trigger custom resource here — triggering /
gating / verification lives in infra/prepare-artifacts.sh.
"""

import os

from aws_cdk import (
    Stack,
    CfnOutput,
    Duration,
    RemovalPolicy,
    aws_s3 as s3,
    aws_s3_assets as s3_assets,
    aws_iam as iam,
    aws_ecr as ecr,
    aws_codebuild as codebuild,
)
from constructs import Construct


class GR00TRLArtifactsStack(Stack):
    """Persistent ECR repo + mode-switched CodeBuild pipeline for GR00T RL."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        s3_data_bucket: str,
        image_tag: str = "latest",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Import the existing S3 data bucket (DRA-linked to FSx by the EKS stack).
        data_bucket = s3.Bucket.from_bucket_name(self, "DataBucket", s3_data_bucket)

        # ECR repo for the unified image. RETAIN so re-deploys / stack churn never
        # delete a repo that a running EKS backend is pulling from.
        repo = ecr.Repository(
            self,
            "UnifiedECR",
            repository_name="gr00t-rl-unified",
            removal_policy=RemovalPolicy.RETAIN,
            image_scan_on_push=True,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=10, rule_priority=1)],
        )

        # Source asset is the whole rl/ dir (NOT just docker/) so the buildspec can
        # reach docker/Dockerfile.unified, infra/stage-s3-eks.sh, patches/, and
        # workflows/. Dockerfiles are NOT excluded (build-image needs them). cdk.out,
        # bytecode, and .git are excluded to keep the asset small.
        rl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        asset = s3_assets.Asset(
            self,
            "PipelineSourceAsset",
            path=rl_dir,
            exclude=[
                "*.pyc",
                "__pycache__",
                ".git",
                ".git/**",
                "infra/cdk.out",
                "infra/cdk.out/**",
                "infra/cdk.context.json",
            ],
        )

        # ONE mode-switched CodeBuild project (STAGE_MODE=build-image|stage-data|all,
        # default "all"). No auto-startBuild — prepare-artifacts.sh drives it.
        project = codebuild.Project(
            self,
            "PipelineBuild",
            project_name="GR00T-RL-Pipeline",
            description=(
                "Mode-switched pipeline (STAGE_MODE=build-image|stage-data|all): "
                "build+push the unified image and/or stage pinned third-party code "
                "(RLinf _broadcast patch applied), the RL model, and workflows to "
                "the S3 data bucket for the EKS/FSx DRA"
            ),
            source=codebuild.Source.s3(
                bucket=asset.bucket,
                path=asset.s3_object_key,
            ),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                # X2_LARGE (72 vCPU, 145 GB RAM, ~800 GB disk) + privileged: needed
                # for the Isaac Sim base image `docker build`, and roomy for the
                # local-then-sync staging clone in the same project.
                compute_type=codebuild.ComputeType.X2_LARGE,
                privileged=True,
            ),
            build_spec=codebuild.BuildSpec.from_source_filename(
                "docker/buildspec-pipeline.yml"
            ),
            environment_variables={
                "STAGE_MODE": codebuild.BuildEnvironmentVariable(value="all"),
                "S3_DATA_BUCKET": codebuild.BuildEnvironmentVariable(
                    value=data_bucket.bucket_name
                ),
                "AWS_REGION": codebuild.BuildEnvironmentVariable(
                    value=Stack.of(self).region
                ),
                "AWS_DEFAULT_REGION": codebuild.BuildEnvironmentVariable(
                    value=Stack.of(self).region
                ),
                "ECR_REPOSITORY_NAME": codebuild.BuildEnvironmentVariable(
                    value="gr00t-rl-unified"
                ),
                "IMAGE_TAG": codebuild.BuildEnvironmentVariable(value=image_tag),
            },
            timeout=Duration.hours(3),
            # Serialize builds — the shared ECR tag + S3 staging marker make
            # concurrent builds race-prone; one at a time.
            concurrent_build_limit=1,
        )

        # IAM (least-privilege).
        repo.grant_pull_push(project.role)
        project.role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken"], resources=["*"]
            )
        )
        data_bucket.grant_read_write(project.role)
        asset.grant_read(project.role)

        # ==============================================================
        # CfnOutputs (consumed by prepare-artifacts.sh + build_unified_and_push.sh)
        # ==============================================================
        CfnOutput(
            self,
            "PipelineProject",
            value=project.project_name,
            description=(
                "Mode-switched CodeBuild project (STAGE_MODE=build-image|stage-data|"
                "all). Run: aws codebuild start-build --project-name GR00T-RL-Pipeline "
                "[--environment-variables-override name=STAGE_MODE,value=stage-data]"
            ),
        )
        CfnOutput(
            self,
            "UnifiedECRUri",
            value=repo.repository_uri,
            description="ECR repo the pipeline builds the unified image into",
        )
        CfnOutput(
            self,
            "DataBucketName",
            value=data_bucket.bucket_name,
            description="S3 data bucket the pipeline stages into (DRA-linked to FSx)",
        )
