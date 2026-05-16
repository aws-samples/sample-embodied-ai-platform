#!/usr/bin/env python3
"""CDK app for GR00T RL post-training on AWS Batch MNP.

Deploy:
  cd training/gr00t/rl/infra
  cdk deploy --context num_rollout_nodes=4

Context parameters:
  vpc_id              - Existing VPC ID (creates new if omitted)
  efs_id              - Existing EFS file system ID (creates new if omitted)
  efs_sg_id           - EFS security group ID (required if efs_id provided)
  learner_image_uri   - ECR URI for learner container
  rollout_image_uri   - ECR URI for rollout container
  num_rollout_nodes   - Number of rollout child nodes (default: 4)
"""
import os
import aws_cdk as cdk
from mnp_batch_stack import RLBatchMNPStack

app = cdk.App()

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-west-2"),
)

stack = RLBatchMNPStack(
    app,
    "GR00TRLBatchStack",
    vpc_id=app.node.try_get_context("vpc_id"),
    efs_id=app.node.try_get_context("efs_id"),
    efs_sg_id=app.node.try_get_context("efs_sg_id"),
    learner_image_uri=app.node.try_get_context("learner_image_uri"),
    rollout_image_uri=app.node.try_get_context("rollout_image_uri"),
    num_rollout_nodes=int(app.node.try_get_context("num_rollout_nodes") or 4),
    env=env,
)

app.synth()
