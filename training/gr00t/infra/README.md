# AWS CDK for GR00T Fine-tuning

This directory contains AWS Cloud Development Kit (CDK) stacks to deploy infrastructure for fine-tuning and evaluating NVIDIA Isaac GR00T models on AWS.

## Architecture Overview

![Architecture](./architecture.drawio.png)

The infrastructure consists of two independent but complementary CDK stacks:

1. **BatchStack** (`batch_stack.py`) - Creates AWS Batch resources for distributed fine-tuning jobs
2. **DcvStack** (`dcv_stack.py`) - Deploys an Amazon EC2 instance with NICE DCV for visualization and evaluation

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
- NICE DCV server for remote visualization
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

# Bootstrap CDK (one-time per account/region)
cdk bootstrap --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION

# Deploy Batch stack (creates VPC, EFS, and Batch resources)
cdk deploy IsaacGr00tBatchStack --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION

# Deploy DCV stack (automatically imports VPC and EFS from Batch stack)
cdk deploy IsaacLabDcvStack --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION
```

**Notes**:
- The Batch stack will build the container image from the Dockerfile, which can take 10-30 minutes
- If you want to use a pre-built container image, set the `ECR_IMAGE_URI` environment variable:
  ```bash
  ECR_IMAGE_URI=<YOUR_ECR_IMAGE_URI> cdk deploy IsaacGr00tBatchStack
  ```

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

# Deploy only the DCV stack
cdk deploy IsaacLabDcvStack --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION
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
    "vpc_id": "vpc-xxxxxxxx",      // Your BatchVPC ID
    "efs_id": "fs-xxxxxxxx",        // Your BatchEFS ID
    "efs_sg_id": "sg-xxxxxxxx"      // Your BatchEFSSecurityGroup ID
  }
}
EOF

# Step 3: Deploy DCV stack (imports VPC and EFS)
pip install -r requirements.txt
cdk bootstrap --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION  # if not done
cdk deploy IsaacLabDcvStack --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION

# Step 4: Connect to the DCV instance via the web URL or DCV client
# (Check stack outputs for connection details)

# Step 5: In the DCV instance terminal, clone the repo and build the container
git clone https://github.com/aws-samples/sample-embodied-ai-platform.git
cd sample-embodied-ai-platform/training/gr00t

# Authenticate to ECR (create repository first in ECR console)
aws ecr get-login-password --region YOUR_AWS_REGION | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.YOUR_AWS_REGION.amazonaws.com

# Build and push the container
export DOCKER_REGISTRY=YOUR_ACCOUNT.dkr.ecr.YOUR_AWS_REGION.amazonaws.com
./build_container.sh --test --push

# Step 6: Deploy Batch stack from local or the DCV instance (imports same VPC and EFS)
cd infra
pip install -r requirements.txt
# Use the same cdk.json context or set environment variables
ECR_IMAGE_URI=YOUR_ACCOUNT.dkr.ecr.YOUR_AWS_REGION.amazonaws.com/gr00t-finetune:latest \
  cdk deploy IsaacGr00tBatchStack --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION
```

**Important**: Both stacks will import the same VPC, EFS, and Security Group via the context in `cdk.json`, ensuring they share resources for seamless data flow.

## Configuration Options

### Environment Variables

- `ECR_IMAGE_URI`: Use an existing ECR image instead of building from Dockerfile
- `TRAINING_S3_BUCKET_NAME`: S3 bucket for datasets and checkpoints (enables least-privilege IAM policies)
- `AWS_DEFAULT_REGION`: Override the deployment region

### CDK Context (cdk.json)

You can provide existing resource IDs to import rather than create new ones:

```json
{
  "app": "python app.py",
  "context": {
    "vpc_id": "vpc-xxxxxxxx",      // Import existing VPC
    "efs_id": "fs-xxxxxxxx",        // Import existing EFS
    "efs_sg_id": "sg-xxxxxxxx"      // Import existing Security Group
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
- [ ] Bootstrap CDK: `cdk bootstrap --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION`
- [ ] Request GPU instance quota (g6e.2xlarge or larger)
- [ ] (Optional) Build and push container image to ECR
- [ ] (Optional) Configure cdk.json with existing resource IDs
- [ ] Deploy stacks: `cdk deploy <StackName>`
- [ ] Verify stack outputs for connection details

## Cleanup

To avoid ongoing charges, destroy the stacks in reverse order:

```bash
# Destroy DCV stack first (terminates EC2 instance)
cdk destroy IsaacLabDcvStack --force --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION

# Destroy Batch stack (removes Batch resources, EFS, VPC if created by CDK)
cdk destroy IsaacGr00tBatchStack --force --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION
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
- **Solution**: Verify IAM roles have appropriate policies. For S3, set `TRAINING_S3_BUCKET_NAME` environment variable

## Support and Contributions

For issues, questions, or contributions, please refer to the main repository README and contribution guidelines.

## Additional Resources

- [AWS Batch User Guide](https://docs.aws.amazon.com/batch/latest/userguide/)
- [AWS CDK Developer Guide](https://docs.aws.amazon.com/cdk/latest/guide/)
- [Amazon DCV User Guide](https://docs.aws.amazon.com/dcv/)
- [Amazon EFS User Guide](https://docs.aws.amazon.com/efs/latest/ug/)
- [NVIDIA Isaac GR00T Documentation](https://github.com/NVIDIA/Isaac-GR00T)
