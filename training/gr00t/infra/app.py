#!/usr/bin/env python3
import os
from aws_cdk import App, Environment
from batch_stack import BatchStack
from dcv_stack import DcvStack

app = App()

# Get environment variables with defaults
env = Environment(account=os.getenv("CDK_DEFAULT_ACCOUNT"), region=os.getenv("CDK_DEFAULT_REGION", "us-west-2"))

# If context values are provided, it will import the existing VPC/EFS directly.
ctx_vpc_id = app.node.try_get_context("vpc_id")
ctx_efs_id = app.node.try_get_context("efs_id")
ctx_efs_sg_id = app.node.try_get_context("efs_sg_id")

batch_stack = BatchStack(
    app,
    "IsaacGr00tBatchStack",
    env=env,
    vpc_id=ctx_vpc_id,  # Optional: existing VPC
    efs_id=ctx_efs_id,  # Optional: existing EFS
    efs_sg_id=ctx_efs_sg_id,  # Optional: existing EFS SG
)

# The DCV stack for visualization. By default it consumes cross-stack refs from the Batch stack.
# Pass the context values directly to the DCV stack and let it handle VPC lookup internally
dcv_stack = DcvStack(
    app,
    "IsaacLabDcvStack",
    env=env,
    # Share the same EFS created/managed by the Batch stack by default.
    # If context values are provided, they will override these.
    efs_id=ctx_efs_id or batch_stack.efs_id,
    efs_sg_id=ctx_efs_sg_id or batch_stack.efs_sg_id,
    # Consume the VPC from the Batch stack to ensure both stacks land in the same network
    # (avoids needing to pass vpc_id explicitly).
    batch_stack=batch_stack,
)

app.synth()
