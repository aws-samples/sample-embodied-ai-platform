# RL Training Stack - TODOs

## CRITICAL: Fix SageMaker backend implementation

The current `sagemaker` backend in `mnp_batch_stack.py` is WRONG. It uses `CfnComputeEnvironment`
with type "SAGEMAKER" which doesn't exist. The correct API is:

- `CfnServiceEnvironment` (CDK L1 available) with `service_environment_type="SAGEMAKER_TRAINING"`
- Job queue with `job_queue_type="SAGEMAKER_TRAINING"` and `service_environment_order`
- Submit via `SubmitServiceJob` API (not `SubmitJob`)
- Training payload is a SageMaker `CreateTrainingJob` JSON (supports heterogeneous InstanceGroups)

Reference: https://docs.aws.amazon.com/batch/latest/userguide/getting-started-sagemaker.html

CDK constructs available:
- `aws_cdk.aws_batch.CfnServiceEnvironment`
- `aws_cdk.aws_batch.CfnServiceEnvironmentProps`
- Job queue likely needs `CfnJobQueue` with `job_queue_type` property

## Build unified image for batch-mnp path

The `Dockerfile.unified` is ready but hasn't been tested via CodeBuild yet.
Need to update the CodeBuild rollout project to build from `Dockerfile.unified`
instead of `Dockerfile.rollout`.

## Optimization: Bake i4h-workflows into container images

Currently the repo code is staged onto EFS via a separate CodeBuild project.
The containers should instead bake the code in at build time:

- Clone i4h-workflows (pinned to v0.5.0) into the Dockerfiles
- Clone RLinf (pinned to 649e757) into the Dockerfiles
- Clone Isaac-GR00T (pinned to 4af2b62) into the rollout Dockerfile
- Simplify EFS staging to model-only (5GB checkpoint download)
- Remove code symlinks from buildspec-stage-efs.yml

This makes containers self-contained and reproducible. EFS only needed
for model checkpoint and training outputs.

## Pinned versions (for reproducibility)

| Dependency | Version/Commit | Source |
|---|---|---|
| i4h-workflows | v0.5.0 | https://github.com/isaac-for-healthcare/i4h-workflows |
| RLinf | 649e7579775997ade74efff33a7c23e90c61e60a | https://github.com/RLinf/RLinf |
| Isaac-GR00T (N1.5) | 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96 | https://github.com/NVIDIA/Isaac-GR00T |
| Isaac Sim | 5.1.0 | nvcr.io/nvidia/isaac-sim:5.1.0 |
| GR00T model | nvidia/GR00T-N1.5-RL-Rheo-AssembleTrocar | HuggingFace |
| PyTorch (learner) | 2.6.0+cu124 | https://download.pytorch.org/whl/cu124 |
| PyTorch (rollout) | 2.7.0+cu128 (bundled with Isaac Sim 5.1.0) | nvcr.io |
| Ray | 2.44.1 | PyPI |
