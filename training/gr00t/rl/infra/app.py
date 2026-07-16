#!/usr/bin/env python3
"""CDK app for GR00T RL post-training infrastructure.

Deploy (Batch MNP - default):
  cd training/gr00t/rl/infra
  cdk deploy --context compute_backend=batch-mnp --context num_rollout_nodes=4

Deploy (SageMaker heterogeneous):
  cdk deploy --context compute_backend=sagemaker --context num_rollout_nodes=4

Deploy (EKS + KubeRay):
  cdk deploy --context compute_backend=eks \\
    --context vpc_id=<your-vpc-id> \\
    --context s3_data_bucket=<your-s3-bucket> \\
    --context image_uri=<your-account>.dkr.ecr.<region>.amazonaws.com/<your-repo>:<tag>

Deploy (EKS + KubeRay - eval on a saved checkpoint):
  cdk deploy --context compute_backend=eks \\
    --context vpc_id=<your-vpc-id> \\
    --context s3_data_bucket=<your-s3-bucket> \\
    --context image_uri=<your-account>.dkr.ecr.<region>.amazonaws.com/<your-repo>:<tag> \\
    --context mode=eval \\
    --context eval_ckpt=/mnt/fsx/rl-training/results/<run>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

Context parameters:
  vpc_id                - Existing VPC ID (creates new if omitted for batch-mnp)
  efs_id                - Existing EFS file system ID (batch-mnp/sagemaker only)
  efs_sg_id             - EFS security group ID (batch-mnp/sagemaker only)
  s3_data_bucket        - S3 bucket name with staged training data (EKS only, DRA-linked to FSx)
  fsx_capacity_gib      - FSx for Lustre capacity in GiB (EKS only, default: 1200)
  image_uri             - Pre-built ECR URI for unified image (skips CodeBuild if provided)
  num_rollout_nodes     - Number of rollout child nodes for batch-mnp/sagemaker (default: 4)
  num_rollout_workers   - Number of rollout worker pods for eks (default: 4)
  learner_instance_type - EC2 instance type for learner node group (default: g6e.48xlarge)
  rollout_instance_type - EC2 instance type for rollout node group (default: g6e.4xlarge)
  compute_backend       - "batch-mnp" (default), "sagemaker", or "eks"
  "mode"                - "train" (default) or "eval" — routes the EKS backend to training or standalone eval
  "eval_ckpt"           - Full path to actor checkpoint (.pt) for mode=eval; ignored for mode=train
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
        s3_data_bucket=app.node.try_get_context("s3_data_bucket"),
        image_uri=app.node.try_get_context("image_uri"),
        num_rollout_workers=int(
            app.node.try_get_context("num_rollout_workers") or 4
        ),
        fsx_capacity_gib=int(
            app.node.try_get_context("fsx_capacity_gib") or 1200
        ),
        learner_instance_type=(
            app.node.try_get_context("learner_instance_type") or "g6e.48xlarge"
        ),
        rollout_instance_type=(
            app.node.try_get_context("rollout_instance_type") or "g6e.4xlarge"
        ),
        capacity_reservation_id=app.node.try_get_context("capacity_reservation_id"),
        mode=app.node.try_get_context("mode") or "train",
        eval_ckpt=app.node.try_get_context("eval_ckpt"),
        env=env,
    )
else:
    raise ValueError(
        f"Unknown compute_backend: {compute_backend}. "
        "Supported: 'batch-mnp', 'sagemaker', 'eks'"
    )

app.synth()
