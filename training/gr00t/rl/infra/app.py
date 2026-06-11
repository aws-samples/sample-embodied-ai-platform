#!/usr/bin/env python3
"""CDK app for GR00T RL post-training infrastructure.

Deploy (Batch MNP - default):
  cd training/gr00t/rl/infra
  cdk deploy --context compute_backend=batch-mnp --context num_rollout_nodes=4

Deploy (SageMaker heterogeneous):
  cdk deploy --context compute_backend=sagemaker --context num_rollout_nodes=4

Deploy (EKS + KubeRay):
  cdk deploy --context compute_backend=eks \\
    --context vpc_id=vpc-00ce44fb57e6e740e \\
    --context efs_id=fs-05cc94bf7eeacab6c \\
    --context efs_sg_id=<EFS-MOUNT-TARGET-SG> \\
    --context image_uri=215143956078.dkr.ecr.us-east-2.amazonaws.com/gr00t-rl-unified:latest
  NOTE: efs_sg_id must be the security group on the EFS MOUNT TARGETS (not the Batch stack SG).
  Find it with: aws efs describe-mount-target-security-groups --mount-target-id <mt-id>

Context parameters:
  vpc_id              - Existing VPC ID (creates new if omitted for batch-mnp)
  efs_id              - Existing EFS file system ID (creates new if omitted for batch-mnp)
  efs_sg_id           - EFS security group ID (required if efs_id provided)
  image_uri           - Pre-built ECR URI for unified image (skips CodeBuild if provided)
  num_rollout_nodes   - Number of rollout child nodes for batch-mnp/sagemaker (default: 4)
  num_rollout_workers - Number of rollout worker pods for eks (default: 4)
  compute_backend     - "batch-mnp" (default), "sagemaker", or "eks"
"""
import os
import aws_cdk as cdk
from mnp_batch_stack import RLBatchMNPStack
from eks_kuberay_stack import EKSKubeRayStack

app = cdk.App()

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-west-2"),
)

compute_backend = app.node.try_get_context("compute_backend") or "batch-mnp"

if compute_backend in ("batch-mnp", "sagemaker"):
    stack = RLBatchMNPStack(
        app,
        "GR00TRLBatchStack",
        vpc_id=app.node.try_get_context("vpc_id"),
        efs_id=app.node.try_get_context("efs_id"),
        efs_sg_id=app.node.try_get_context("efs_sg_id"),
        image_uri=app.node.try_get_context("image_uri"),
        num_rollout_nodes=int(app.node.try_get_context("num_rollout_nodes") or 4),
        env=env,
    )
elif compute_backend == "eks":
    stack = EKSKubeRayStack(
        app,
        "GR00TRLEKSStack",
        vpc_id=app.node.try_get_context("vpc_id"),
        efs_id=app.node.try_get_context("efs_id"),
        efs_sg_id=app.node.try_get_context("efs_sg_id"),
        image_uri=app.node.try_get_context("image_uri"),
        num_rollout_workers=int(
            app.node.try_get_context("num_rollout_workers") or 4
        ),
        env=env,
    )
else:
    raise ValueError(
        f"Unknown compute_backend: {compute_backend}. "
        "Supported: 'batch-mnp', 'sagemaker', 'eks'"
    )

app.synth()
