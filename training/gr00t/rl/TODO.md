# RL Training Stack - TODOs

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
