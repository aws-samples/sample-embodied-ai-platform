#!/usr/bin/env python3
"""
Standalone CDK app for deploying only the CodeBuild stack.

This allows deploying the container build pipeline independently
from the Batch and DCV stacks.

Usage:
    cd training/gr00t/infra/codebuild
    cdk deploy Gr00tCodeBuildStack
"""
import os
from aws_cdk import App, Environment
from codebuild_stack import CodeBuildStack

app = App()

# Get environment
env = Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION", "us-west-2"),
)

# Get configuration from context or environment variables
ecr_repository_name = app.node.try_get_context("ecr_repository_name") or os.getenv(
    "ECR_REPOSITORY_NAME", "gr00t-finetune"
)
use_stable = app.node.try_get_context("use_stable")
if use_stable is None:
    use_stable = os.getenv("USE_STABLE", "true").lower() == "true"

# Deploy CodeBuild stack
codebuild_stack = CodeBuildStack(
    app,
    "Gr00tCodeBuildStack",
    env=env,
    ecr_repository_name=ecr_repository_name,
    use_stable=use_stable,
    description="AWS CodeBuild project for building GR00T fine-tuning container",
)

app.synth()
