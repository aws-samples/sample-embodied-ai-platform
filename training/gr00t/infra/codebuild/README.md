# Path 4: Automated Container Building with AWS CodeBuild

This path uses AWS CodeBuild to automatically build the GR00T fine-tuning container image, eliminating the need for local x86 machines or early DCV deployment.

## Why Path 4?

**Problem**: Path 3 requires deploying a DCV instance just to build containers on ARM/non-x86 machines.

**Solution**: Use AWS CodeBuild with x86 compute to build containers in the cloud automatically.

## Prerequisites

1. AWS CLI configured
2. AWS CDK CLI installed (`npm install -g aws-cdk`)
3. Python 3.8+ with dependencies (`pip install -r requirements.txt`)
4. Docker installed locally (for packaging source)

## Deployment Steps

### Step 1: Deploy CodeBuild Stack

```bash
cd training/gr00t/infra/codebuild

# Install dependencies
pip install -r requirements.txt

# Set AWS region
export AWS_REGION=us-west-2  # or your preferred region

# Bootstrap CDK (if not done)
cdk bootstrap

# Deploy CodeBuild stack
cdk deploy Gr00tCodeBuildStack
```

**Outputs**:
- `EcrRepositoryUri`: ECR repository for the container image
- `CodeBuildProjectName`: Name of the CodeBuild project
- `BuildCommand`: Command to trigger a build

### Step 2: Trigger Container Build

After deployment, trigger a build using the AWS CLI:

```bash
# Get the project name from stack outputs
export PROJECT_NAME=$(aws cloudformation describe-stacks \
  --stack-name Gr00tCodeBuildStack \
  --query 'Stacks[0].Outputs[?OutputKey==`CodeBuildProjectName`].OutputValue' \
  --output text)

# Start the build
aws codebuild start-build --project-name $PROJECT_NAME
```

Or use the AWS Console:
1. Go to AWS CodeBuild → Build projects
2. Select `Gr00tContainerBuild`
3. Click "Start build"

### Step 3: Monitor Build Progress

```bash
# Get the latest build ID
BUILD_ID=$(aws codebuild list-builds-for-project \
  --project-name $PROJECT_NAME \
  --max-items 1 \
  --query 'ids[0]' \
  --output text)

# Check build status
aws codebuild batch-get-builds --ids $BUILD_ID \
  --query 'builds[0].buildStatus' \
  --output text

# Stream logs (once build is running)
aws logs tail /aws/codebuild/$PROJECT_NAME --follow
```

**Build time**: Approximately 15-25 minutes (includes flash-attn compilation)

### Step 4: Get ECR Image URI

Once the build completes successfully:

```bash
# Get ECR repository URI from stack outputs
export ECR_REPO_URI=$(aws cloudformation describe-stacks \
  --stack-name Gr00tCodeBuildStack \
  --query 'Stacks[0].Outputs[?OutputKey==`EcrRepositoryUri`].OutputValue' \
  --output text)

# Full image URI with tag
export ECR_IMAGE_URI="${ECR_REPO_URI}:latest"

echo "Container image ready: $ECR_IMAGE_URI"
```

### Step 5: Deploy Batch Stack with Built Image

```bash
cd ../  # Back to training/gr00t/infra

# Deploy Batch stack using the built image
cdk deploy IsaacGr00tBatchStack \
  --context ecr_image_uri=$ECR_IMAGE_URI \
  --context dataset_bucket=my-dataset-bucket \
  --context s3_upload_uri=s3://my-checkpoint-bucket/gr00t/checkpoints
```

### Step 6: (Optional) Deploy DCV Stack

```bash
# Deploy DCV stack for visualization
cdk deploy IsaacLabDcvStack
```

## Configuration Options

### CodeBuild Environment

The CodeBuild project uses:
- **Compute**: `BUILD_GENERAL1_LARGE` (8 vCPU, 15 GB RAM)
- **Image**: `aws/codebuild/standard:7.0` (Ubuntu 22.04, Docker 24)
- **Architecture**: x86_64
- **Privileged mode**: Enabled (required for Docker builds)

### Build Arguments

You can customize the build by modifying environment variables in `codebuild_stack.py`:

```python
environment_variables={
    "USE_STABLE": codebuild.BuildEnvironmentVariable(value="true"),
    "IMAGE_TAG": codebuild.BuildEnvironmentVariable(value="latest"),
}
```

## Troubleshooting

### Build Fails with "flash-attn compilation error"

**Issue**: flash-attn requires significant memory to compile

**Solution**: Increase CodeBuild compute type to `BUILD_GENERAL1_2XLARGE` in `codebuild_stack.py`:

```python
environment=codebuild.BuildEnvironment(
    compute_type=codebuild.ComputeType.LARGE,  # Change to X2_LARGE
    ...
)
```

### Build Fails with "No space left on device"

**Issue**: Docker build requires more disk space

**Solution**: Already configured with 100 GB in buildspec. If still failing, increase in `codebuild_stack.py`.

### ECR Push Permission Denied

**Issue**: CodeBuild role lacks ECR permissions

**Solution**: Verify the role has `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`, `ecr:PutImage`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload` permissions. These are automatically granted in the stack.

### Build Takes Too Long

**Issue**: flash-attn compilation is slow

**Solution**: 
1. Use larger compute type (X2_LARGE)
2. Consider caching Docker layers (add cache configuration to buildspec)
3. Use pre-built base images with flash-attn

## Cleanup

```bash
# Delete CodeBuild stack
cdk destroy Gr00tCodeBuildStack --force

# Manually delete ECR images if needed
aws ecr batch-delete-image \
  --repository-name gr00t-finetune \
  --image-ids imageTag=latest
```

## Next Steps

After successful container build:
1. Deploy Batch stack with the ECR image URI
2. Submit training jobs to AWS Batch
3. (Optional) Deploy DCV stack for visualization
4. Monitor training progress via CloudWatch Logs

## Support

For issues or questions:
- Check CloudWatch Logs for build errors
- Review [main infra README](../README.md) for general troubleshooting
- Open an issue in the repository
