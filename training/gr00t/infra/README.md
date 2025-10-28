# AWS CDK Stacks for GR00T Fine-tuning

This directory contains AWS Cloud Development Kit (CDK) stacks to deploy infrastructure for fine-tuning and evaluating NVIDIA Isaac GR00T models on AWS.

## Architecture Overview

![Architecture](./architecture.drawio.png)

The infrastructure consists of two independent but complementary CDK stacks:

1. **BatchStack** (`batch_stack.py`) - Creates AWS Batch resources for scalable fine-tuning jobs
2. **DcvStack** (`dcv_stack.py`) - Deploys an Amazon EC2 instance with Amazon DCV for visualization and evaluation

Both stacks can share common resources (VPC, EFS, Security Groups) to enable seamless data flow between training and evaluation workflows.

## Stack Dependencies

### BatchStack
Creates the following resources:
- Amazon VPC with public and private subnets (optional, can import existing)
- Amazon EFS file system for shared storage (optional, can import existing)
- Security group for EFS access
- Amazon ECR repository and container image (or references existing)
- EC2 Launch Template with increased root volume
- AWS Batch Compute Environment (EC2 with GPU instances)
- AWS Batch Job Queue and Job Definition
- IAM roles for Batch instances and job execution

**Dependencies**: None (fully self-contained)

### DcvStack
Creates the following resources:
- Amazon EC2 instance (g6.4xlarge with GPU)
- Amazon DCV server for remote visualization
- Security group for DCV and TensorBoard access
- Elastic IP for stable connectivity
- IAM role for EC2 instance
- Mounts shared EFS file system (if provided)

**Dependencies**: 
- VPC (can import from BatchStack, provide VPC ID, or use existing VPC object)
- EFS and Security Group (optional, for shared storage with Batch jobs)

## Deployment Paths

Choose the path that best fits your environment and requirements:

### Path 1: Fully Automated Deployment (Recommended for Quick Start)

Deploy both stacks automatically with CDK. This is ideal if you have a local x86 machine or don't need to customize resources manually.

```bash
# Install dependencies
cd training/gr00t/infra
pip install -r requirements.txt

# Set AWS region for deployment
export AWS_REGION=us-west-2  # or your preferred region

# Bootstrap CDK (one-time per account/region)
cdk bootstrap

# Deploy Batch and DCV stacks (creates VPC, EFS, and Batch resources)
cdk deploy IsaacGr00tBatchStack IsaacLabDcvStack

# Or use existing resources via CDK context parameters (optional)
cdk deploy IsaacGr00tBatchStack IsaacLabDcvStack \
  --context vpc_id=vpc-12345 \
  --context efs_id=fs-12345 \
  --context efs_sg_id=sg-12345 \
  --context ecr_image_uri=123456789012.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest \
  --context dataset_bucket=my-dataset-bucket \
  --context s3_upload_uri=s3://my-checkpoint-bucket/gr00t/checkpoints
```

**Notes**:
- The Batch stack will build the container image from the Dockerfile, which can take 10-30 minutes
- If you want to use a pre-built container image, use the `ecr_image_uri` context parameter or `ECR_IMAGE_URI` environment variable

### Path 2: Manual Console + CDK for DCV

Create AWS Batch resources manually via AWS Console (following the blog walkthrough), then deploy only the DCV stack with CDK.

```bash
# After manually creating VPC, EFS, and Batch resources in the console,
# add their IDs to cdk.json:
cat > cdk.json << EOF
{
  "app": "python app.py",
  "context": {
    "vpc_id": "vpc-xxxxxxxx",
    "efs_id": "fs-xxxxxxxx",
    "efs_sg_id": "sg-xxxxxxxx"
  }
}
EOF

# Set AWS region for deployment
export AWS_REGION=us-west-2  # or your preferred region

# Deploy only the DCV stack
cdk deploy IsaacLabDcvStack
```

### Path 3: DCV First, Then Batch (For ARM/Non-x86 Local Machines)

If you cannot build x86 container images locally (e.g., using an ARM-based Mac), you can deploy the DCV stack first, then use that EC2 instance to build the container and deploy the Batch stack. To ensure both stacks share the same resources, you must manually create VPC, EFS, and Security Group first.

```bash
# Step 1: Manually create shared resources following blog section 2.1
# - Create VPC (BatchVPC) with public and private subnets
# - Create Security Group (BatchEFSSecurityGroup) with self-referencing NFS rule
# - Create EFS (BatchEFS) and attach the security group to all mount targets
# Note the VPC ID, EFS ID, and Security Group ID

# Step 2: Configure cdk.json with the resource IDs
cd training/gr00t/infra
cat > cdk.json << EOF
{
  "app": "python app.py",
  "context": {
    "vpc_id": "vpc-xxxxxxxx",
    "efs_id": "fs-xxxxxxxx",
    "efs_sg_id": "sg-xxxxxxxx",
    "dataset_bucket": "my-dataset-bucket",
    "s3_upload_uri": "s3://my-checkpoint-bucket/gr00t/checkpoints"
  }
}
EOF

# Step 3: Deploy DCV stack (imports VPC and EFS)
pip install -r requirements.txt

# Set AWS region for deployment
export AWS_REGION=us-west-2  # or your preferred region

cdk bootstrap  # if not done
cdk deploy IsaacLabDcvStack

# Step 4: Connect to the DCV instance via the web URL or DCV client
# (Check stack outputs for connection details)

# Step 5: In the DCV instance terminal, clone the repo and build the container
git clone https://github.com/aws-samples/sample-embodied-ai-platform.git
cd sample-embodied-ai-platform/training/gr00t

# Authenticate to ECR (create repository first in ECR console)
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-west-2  # or your preferred region
aws ecr get-login-password | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push the container
export DOCKER_REGISTRY=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
./build_container.sh --test --push

# Step 6: Deploy Batch stack from local or the DCV instance (imports same VPC and EFS)
cd infra
pip install -r requirements.txt
# Add ecr_image_uri to the cdk.json context or use --context ecr_image_uri flag
cdk deploy IsaacGr00tBatchStack \
  --context ecr_image_uri=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/gr00t-finetune:latest
```

**Important**: Both stacks will import the same VPC, EFS, and Security Group via the context in `cdk.json`, ensuring they share resources for seamless data flow.

## Configuration Options

### Infrastructure Deployment

When deploying the CDK stack, you can configure infrastructure resources using CDK context parameters or environment variables:

#### CDK Context Parameters (Recommended)

Pass configuration via CDK context using `--context` flags:

```bash
cdk deploy IsaacGr00tBatchStack IsaacLabDcvStack \
  --context vpc_id=vpc-12345 \
  --context efs_id=fs-12345 \
  --context efs_sg_id=sg-12345 \
  --context ecr_image_uri=123456789012.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest \
  --context dataset_bucket=my-dataset-bucket \
  --context s3_upload_uri=s3://my-checkpoint-bucket/gr00t/checkpoints
```

#### Configuration Options

| Context Parameter | Env Variable | Description | Default |
|------------------|--------------|-------------|---------|
| `vpc_id` | `VPC_ID` | Existing VPC ID to reuse | Creates new VPC |
| `efs_id` | `EFS_ID` | Existing EFS file system ID | Creates new EFS |
| `efs_sg_id` | `EFS_SG_ID` | EFS security group ID (required if `efs_id` is set) | Creates new SG |
| `ecr_image_uri` | `ECR_IMAGE_URI` | Pre-built ECR image URI (in the same region as the deployment) | Builds from local Dockerfile |
| `dataset_bucket` | `DATASET_BUCKET` | S3 bucket name for dataset read-only access | No dataset bucket access |
| `s3_upload_uri` | `S3_UPLOAD_URI` | S3 URI for checkpoint uploads | Creates new checkpoint bucket |

**Note**: CDK context parameters take precedence over environment variables. This allows for flexible deployment configurations while maintaining consistency through context values in your `cdk.json` or CLI commands.

### CDK Context (cdk.json)

You can also use context in [cdk.json](cdk.json) to provide existing resource IDs to import rather than create new ones:

```json
{
  "app": "python app.py",
  "context": {
    "vpc_id": "vpc-xxxxxxxx",
    "efs_id": "fs-xxxxxxxx",
    "efs_sg_id": "sg-xxxxxxxx",
    "ecr_image_uri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest",
    "dataset_bucket": "my-dataset-bucket",
    "s3_upload_uri": "s3://my-checkpoint-bucket/gr00t/checkpoints"
  }
}
```

## Resource Sharing Between Stacks

When both stacks are deployed together, they share:

1. **VPC**: DCV instance and Batch compute nodes run in the same network
2. **EFS**: Mounted at `/mnt/efs` on both DCV instance and Batch containers
3. **Security Group**: Allows both DCV and Batch instances to access EFS

This enables:
- Real-time monitoring of training jobs via TensorBoard on the DCV instance
- Direct access to model checkpoints for evaluation
- Seamless data flow between training and evaluation workflows

## Prerequisites

1. **AWS Account**: With appropriate permissions to create VPC, EC2, EFS, Batch, IAM resources
2. **AWS CLI**: Configured with credentials (`aws configure`)
3. **Node.js**: For AWS CDK CLI (`npm install -g aws-cdk`)
4. **Python 3.8+**: For CDK app and dependencies
5. **Docker**: For building container images (if not using pre-built images)
6. **Service Quotas**: At least 8 vCPUs for "Running On-Demand G and VT instances" in your target region

## Deployment Checklist

- [ ] Install AWS CDK CLI: `npm install -g aws-cdk`
- [ ] Install Python dependencies: `pip install -r requirements.txt`
- [ ] Set AWS region: `export AWS_REGION=us-west-2`
- [ ] Bootstrap CDK: `cdk bootstrap`
- [ ] Request GPU instance quota (g6e.2xlarge or larger)
- [ ] (Optional) Build and push container image to ECR
- [ ] (Optional) Configure cdk.json with existing resource IDs
- [ ] Deploy stacks: `cdk deploy <StackName>`
- [ ] Verify stack outputs for connection details

## Cleanup

To avoid ongoing charges, destroy the stacks in reverse order:

```bash
# Set AWS region for deployment
export AWS_REGION=us-west-2  # or your preferred region

# Destroy DCV stack first (terminates EC2 instance)
cdk destroy IsaacLabDcvStack --force

# Destroy Batch stack (removes Batch resources, EFS, VPC if created by CDK)
cdk destroy IsaacGr00tBatchStack --force
```

**Important**: If you manually created resources or imported existing ones via context, those resources will NOT be deleted by `cdk destroy`. Delete them manually if they were created specifically for this project.

## Troubleshooting

### Container Build Fails
- **Issue**: Building flash-attn takes too long or fails on ARM machines
- **Solution**: Use Path 3 (deploy DCV first, build on x86 EC2 instance)

### DCV Connection Issues
- **Issue**: "No session found" error when connecting to DCV
- **Solution**: Wait 10-15 minutes for user data script to complete. Check `/var/log/dcv-bootstrap.summary` for status

### EFS Mount Fails
- **Issue**: EFS not accessible from Batch jobs or DCV instance
- **Solution**: Verify security group allows NFS (port 2049) from itself. Check EFS mount targets are in the correct subnets

### Batch Job Fails to Start
- **Issue**: Job stuck in RUNNABLE state
- **Solution**: Check compute environment has available capacity. Verify launch template and instance types are correct

### Permission Denied Errors
- **Issue**: Job cannot access S3 or ECR
- **Solution**: Verify IAM roles have appropriate policies. For S3, set `DATASET_BUCKET` environment variable

## Support and Contributions

For issues, questions, or contributions, please refer to the main repository README and contribution guidelines.

## Additional Resources

- [AWS Batch User Guide](https://docs.aws.amazon.com/batch/latest/userguide/)
- [AWS CDK Developer Guide](https://docs.aws.amazon.com/cdk/latest/guide/)
- [Amazon DCV User Guide](https://docs.aws.amazon.com/dcv/)
- [Amazon EFS User Guide](https://docs.aws.amazon.com/efs/latest/ug/)
- [NVIDIA Isaac GR00T Documentation](https://github.com/NVIDIA/Isaac-GR00T)
