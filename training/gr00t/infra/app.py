#!/usr/bin/env python3
import os
from aws_cdk import App, Environment
from batch_stack import BatchStack
from dcv_stack import DcvStack

app = App()

# Get environment variables with defaults
env = Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION", "us-west-2"),
)

# If context values are provided, it will import the existing VPC/EFS directly.
ctx_vpc_id = app.node.try_get_context("vpc_id") or os.getenv("VPC_ID")
ctx_efs_id = app.node.try_get_context("efs_id") or os.getenv("EFS_ID")
ctx_efs_sg_id = app.node.try_get_context("efs_sg_id") or os.getenv("EFS_SG_ID")
ctx_ecr_image_uri = app.node.try_get_context("ecr_image_uri") or os.getenv(
    "ECR_IMAGE_URI"
)
ctx_dataset_bucket = app.node.try_get_context("dataset_bucket") or os.getenv(
    "DATASET_BUCKET"
)
ctx_s3_upload_uri = app.node.try_get_context("s3_upload_uri") or os.getenv(
    "S3_UPLOAD_URI"
)

batch_stack = BatchStack(
    app,
    "IsaacGr00tBatchStack",
    env=env,
    vpc_id=ctx_vpc_id,  # Optional: existing VPC
    efs_id=ctx_efs_id,  # Optional: existing EFS
    efs_sg_id=ctx_efs_sg_id,  # Optional: existing EFS SG
    ecr_image_uri=ctx_ecr_image_uri,  # Optional: pre-built ECR image
    dataset_bucket=ctx_dataset_bucket,  # Optional: S3 bucket for dataset access
    s3_upload_uri=ctx_s3_upload_uri,  # Optional: S3 URI for checkpoint uploads
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
