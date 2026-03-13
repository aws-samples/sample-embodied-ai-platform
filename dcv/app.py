#!/usr/bin/env python3
"""Standalone CDK app entry point for DCV module."""
import os
from aws_cdk import App, Environment
from dcv_stack import DcvStack

app = App()

# Read account/region from environment variables
env = Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION", "us-west-2"),
)

# Create the DCV stack
DcvStack(app, "DcvStack", env=env)

app.synth()
