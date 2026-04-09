#!/usr/bin/env python3
import sys
import os

# Add dcv module to Python path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from aws_cdk import App, Environment, Stack
from batch_stack import BatchStack
from dcv import DcvWorkstation, DcvWorkstationProps

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


class IsaacLabDcvStack(Stack):
    """DCV stack for gr00t visualization, integrated with BatchStack.

    Consumes the standalone DCV module from dcv/ and shares VPC/EFS
    with the gr00t BatchStack.
    """

    def __init__(
        self,
        scope,
        construct_id: str,
        batch_stack: BatchStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Configure DCV workstation with gr00t-specific settings
        # isaac_sim_version="5.1.0", isaac_lab_version="v2.3.0" for both N1.5 and N1.6
        props = DcvWorkstationProps(
            vpc=batch_stack.vpc,              # Share VPC with Batch
            efs_id=batch_stack.efs_id,        # Share EFS with Batch
            efs_sg_id=batch_stack.efs_sg_id,  # Share security group
            # Preferred: g6.4xlarge, g6.2xlarge, g5.2xlarge or larger.
            # If capacity errors occur, try changing availability_zone between us-west-2a/b/c/d.
            availability_zone="us-west-2a",
            instance_type="g6.2xlarge",       # Probed available in us-west-2a
            isaac_sim_version="5.1.0",        # Latest version with valid NGC container
            isaac_lab_version="v2.3.0",       # Matches isaac-lab:2.3.0 on NGC
            leisaac_enabled=True,             # Required for gr00t
        )

        self.dcv_workstation = DcvWorkstation(self, "DCV", props)


# DCV stack consumes shared infrastructure from BatchStack
dcv_stack = IsaacLabDcvStack(
    app,
    "IsaacLabDcvStack",  # ✅ Same stack ID as before
    env=env,
    batch_stack=batch_stack,
)

app.synth()
