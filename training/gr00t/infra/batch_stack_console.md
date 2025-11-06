# Manual Console Setup for AWS Batch (Path 2)

This guide walks you through manually creating AWS Batch infrastructure resources via the AWS Console. This is **Path 2: Manual Console + CDK for DCV** from the [main README](README.md).

## When to Use This Path

Choose this path if you:
- Prefer manual control over Batch resource creation
- Want to understand each resource and its configuration
- Need to customize Batch resources beyond what the CDK stack provides
- Are using existing AWS resources and want to configure Batch manually

## Overview

This guide will help you create:
1. **Amazon VPC, Security Group, and EFS** - Network and shared storage infrastructure
2. **ECR Repository and Container Image** - Container registry and fine-tuning image
3. **EC2 Launch Template** - Instance configuration for Batch compute nodes
4. **AWS Batch Compute Environment** - Compute resources for running jobs
5. **AWS Batch Job Queue and Job Definition** - Job execution configuration

After completing these steps, you'll deploy the DCV stack using CDK to enable remote visualization and evaluation. See the [main README](README.md) for the complete workflow.

## Prerequisites

- AWS Account with appropriate permissions

---

## Step-by-Step Instructions

#### 1. Create Amazon VPC, Security Group and EFS

An Amazon Virtual Private Cloud (VPC) is a virtual network that is used to isolate the resources in your AWS account. Security group controls the traffic that is allowed to reach and leave the resources that it is associated with. Amazon Elastic File System (EFS) is a fully managed, scalable file storage service that can be shared across multiple EC2 instances, making it ideal for distributed training jobs. We will use the VPC to create a private network for the EC2 instances that will run the fine-tuning jobs in AWS Batch, a self-referencing security group to allow the Batch instances to access EFS securely and an EFS to store the checkpoints and logs. To learn more about EFS for AWS Batch, refer to [EFS for AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/efs-volumes.html).

1. Open the [VPC console](https://us-west-2.console.aws.amazon.com/vpc/home?region=us-west-2) and choose **Create VPC**.
    1. Select the **VPC and more** option.
    2. For *Name tag auto-generation*, enter `BatchVPC` (the prefix for VPC resources), for **NAT gateways**, select **In 1 AZ** (to reduce cost)
    3. Leave the rest as default and click **Create VPC**. 
    4. Make a note of the *VPC ID* (e.g. `vpc-xxxxxxxx`).

2. On the left navigation pane of the VPC console, select **Security groups** and choose **Create security group**.
    1. For *Security group name* enter `BatchEFSSecurityGroup`, for *Description* enter `Security group for Batch instances and EFS`, for *VPC* select `BatchVPC-vpc`.
    2. Leave the rest as default and select **Create security group**.
    3. In the newly created security group, select **Edit inbound rules** and **Add rule**: *Type* `NFS`,  *Source* `Custom` and find `BatchEFSSecurityGroup` (self-referencing).
    4. Select **Save rules** to apply the changes.
    5. Make a note of the *Security group ID* (e.g. `sg-xxxxxxxx`).

3. Open the [EFS console](https://us-west-2.console.aws.amazon.com/efs/home?region=us-west-2) and choose **Create file system**.
    1. For *Name*, enter `BatchEFS`, for *VPC*, select the `BatchVPC` you created above.
    2. Leave the rest as default and choose **Create**.
    3. Make a note of the *File system ID* (e.g. `fs-xxxxxxxx`).
    4. Click into the file system, select **Network** tab and click **Manage**.
    5. Look for *Security groups*, unselect the default security group and select `BatchEFSSecurityGroup` for all mount targets and choose **Save**.

#### 2. Build the fine-tuning container and push to ECR

The fine-tuning container is a Docker image that contains the GR00T dependencies and the fine-tuning workflow script. We will build the container and push it to Amazon Elastic Container Registry (ECR) so it can be used by AWS Batch. You can also customize the Dockerfile and scripts to fine-tune different models or datasets.
> NOTE: As all G6e instance family are based on x86 as of 2025, you may need a x86 machine to build the fine-tuning container. If you encounter challenges building the container in your local machine, try cleaning up the docker cache and re-run the script, use AWS CodeBuild, or build the container on the DCV instance deployed in section 4.1 to build the container (run section 4.1 to deploy the DCV instance then come back here).

1. Go to [Amazon Elastic Container Registry console](https://us-west-2.console.aws.amazon.com/ecr/private-registry/repositories?region=us-west-2) and **Create repository**. Use `gr00t-finetune` as the repository name, leave the rest as default and click on **Create**. 

2. Click into the newly created repository and click on **View push commands** on the top right to view the command to authenticate to ECR. The command should look like:
    ```bash
    aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <YOUR_CONTAINER_REGISTRY_PREFIX>
    ```
    Run the command to authenticate to ECR and take a note of your container registry prefix, e.g. `<YOUR_CONTAINER_REGISTRY_PREFIX>`.

3. Replace the `<YOUR_CONTAINER_REGISTRY_PREFIX>` in the following command with your container registry prefix and run the script to build, test and push the GR00T fine-tuning image. This may take between 5 minutes to 3 hours depending on your machine (primarily for building the flash-attn package).
    ```bash
    cd training/gr00t
    chmod +x build_container.sh
    export DOCKER_REGISTRY=<YOUR_CONTAINER_REGISTRY_PREFIX>
    ./build_container.sh --test --push
    ```

    Examine the output of the script to see if the tests pass. If not, resolve the issues and run the script again.

#### 3. Create Launch Template

An EC2 Launch Template is a template that defines the configuration for an EC2 instance that is used to run a job. We will use this template to increase the Linux root volume size to 100 GiB for pulling large fine-tuning containers. To learn more about Launch Templates, refer to [Launch templates for AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/launch-templates.html).

1. Open the [EC2 console](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2).
2. In the navigation pane at the left, select **Launch templates** and choose **Create launch template**.
3. Under *Name* enter `BatchLaunchTemplate`.
4. For *Application and OS Images (Amazon Machine Image)* and *Instance type*, leave them as default (i.e. `Don't include in launch template`) as AWS Batch will automatically select the appropriate Amazon Linux AMI with ECS agent and NVIDIA driver preinstalled.
5. Scroll down to *Storage (volumes)* and select **Add new volume**. For *Volume type* select `gp3` and for *Size (GiB)* enter `100`. For *Device name* select **Specify a custom value...** and enter `/dev/xvda` (standard root device for Amazon Linux 2).
> NOTE: Without increasing the root volume size, the container may fail to pull the image with the error"CannotPullContainerError: context canceled" due to the default 8 GiB root volume size.
6. Leave the rest as default and choose **Create launch template**.

#### 4. Create Compute Environment

An AWS Batch Compute Environment is a collection of compute resources on which jobs are executed. We will use this compute environment to define the compute, network and security settings for the host machine that will run the containers for fine-tuning jobs. To learn more about Compute environments, refer to [Compute environments for AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/compute_environments.html).

1. Open the [AWS Batch console](https://us-west-2.console.aws.amazon.com/batch/home?region=us-west-2)
2. In the navigation pane at the left, select **Environments** and expand **Create environment** in the upper right and choose **Compute environment**.
3. Under *Compute environment configuration*, choose **Amazon Elastic Compute Cloud (EC2)** and **Confirm** your selection. 
4. For *Name* enter `IsaacGr00tComputeEnvironment`, select an existing *Instance role* or **Create IAM role** following the [user guide](https://docs.aws.amazon.com/batch/latest/userguide/batch-check-ecsinstancerole.html), then choose **Next**.
5. Under *Instance configuration*, for *Allowed instance types*, select **g6e family** and uncheck **optimal**. 
6. Under *Launch templates*, for *Default launch template*, select **BatchLaunchTemplate**, and choose **Next** at the bottom.
> NOTE: When a compute environment is created, AWS Batch will automatically create a snapshot of the selected launch template for infrastructure stability. If you update the original launch template directly, you will need to explicitly update the compute environment too to generate a new snapshot. See [Use Amazon EC2 launch templates with AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/launch-templates.html) for more details.
7. Under *Network configuration*, choose `BatchVPC-vpc` for *VPC*, select the private subnets for *Subnets* and the `BatchEFSSecurityGroup` for *Security groups*.
8. Review the configurations and choose **Create compute environment**.

#### 5. Create Job Queue and Job Definition

An AWS Batch Job Queue is a collection of jobs that are executed on a compute environment. An AWS Batch Job Definition is a template that defines the configuration for a job. We will use the job definition to specify container-level requirements and environment variables for individual fine-tuning jobs, then submit them to the job queue for execution. To learn more about Job Queues and Job Definitions, refer to [Job queues for AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/job_queues.html) and [Job definitions for AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/job_definitions.html).

1. In the [AWS Batch console](https://us-west-2.console.aws.amazon.com/batch/home?region=us-west-2), go to **Job queues** and choose **Create**.
2. For *Orchestration type* select **Amazon EC2**.
3. Under *Job queue configuration*, for *Job queue name* enter `IsaacGr00tJobQueue`, for *Connected compute environment* select `IsaacGr00tComputeEnvironment`.
4. Leave the rest as default and choose **Create job queue**.
5. On the left navigation pane, select **Job definitions** and choose **Create**.
6. For *Orchestration type* select **Amazon EC2** and **Confirm** your selection.
7. Under *General configuration*, for *Name* enter `IsaacGr00tJobDefinition`, for *Execution timeout* enter `21600` (i.e. allow 6 hours for the job to run) and select **Next**.
8. Under *Container configuration*, for *Image* enter the ECR image URI for your fine-tuning container (e.g. `<YOUR_CONTAINER_REGISTRY_PREFIX>/gr00t-finetune:latest`), for *Command* delete the existing command (so that it defaults to the command `["/workspace/scripts/run_finetune_workflow.sh"]` in the Dockerfile). (Optional) For *Job role configuration*, if you're downloading datasets or uploading models to a private S3 bucket, you will need to create or assign a IAM role that grants `AmazonS3FullAccess`, then in production, update this to least-privilege access to only the bucket/prefix where you plan to upload or read.
9. Under *Environment configuration*, for *vCPUs* enter `8`, for *Memory* enter `65536` MiB (64 GiB), and for *GPUs* enter `1`, for *Environment variables* add `OUTPUT_DIR`:`/mnt/efs/gr00t/checkpoints`, select **Next**.
10. Under *Filesystem configuration*, for *Shared memory size* enter `65536` MiB (64 GiB). Expand the *Additional configuration* section; select **Add volume** and **Enable EFS**, for *Name* enter `BatchEFS`, for *Filesystem ID* enter the EFS file system ID you noted in step 1.3 (Access is controlled by security groups); select *Add Mount points*, for *Source volume* select `BatchEFS`, for *Container path* enter `/mnt/efs`, select **Next**.
11. Review the configurations and **Create job definition**.

---

## Next Steps: Deploy DCV Stack with CDK

You've successfully created all the AWS Batch infrastructure manually! Now you can deploy the DCV (Amazon Desktop Cloud Visualization) stack using CDK to enable remote visualization and evaluation of your training jobs.

### Collect Resource IDs

Before deploying the DCV stack, make sure you have the following resource IDs noted from the steps above:

- **VPC ID**: From step 1.1 (e.g., `vpc-xxxxxxxx`)
- **EFS File System ID**: From step 1.3 (e.g., `fs-xxxxxxxx`)
- **Security Group ID**: From step 1.2 (e.g., `sg-xxxxxxxx`)

### Deploy DCV Stack

Follow the instructions in the [main README](README.md#path-2-manual-console--cdk-for-dcv) to:

1. Configure `cdk.json` with your resource IDs
2. Deploy the DCV stack using CDK

The DCV stack will:
- Create an EC2 instance with GPU support (g6.4xlarge) and Amazon DCV
- Mount the EFS file system you created above
- Enable remote desktop access for monitoring training jobs and running evaluations
- Provide access to TensorBoard and other visualization tools

### Submitting Training Jobs

Once both Batch and DCV infrastructure are set up, you can submit fine-tuning jobs to your Batch job queue. The jobs will:
- Use the container image you built and pushed to ECR
- Store checkpoints and logs in the shared EFS file system
- Be accessible from the DCV instance for monitoring and evaluation

Refer to the [GR00T training README](../README.md) for details on submitting jobs and working with the DCV instance.

---

## Troubleshooting

If you encounter issues during setup:

- **Container build fails**: See the note in step 2 about x86 requirements. Consider using AWS CodeBuild or building on the DCV instance.
- **EFS mount issues**: Verify the security group allows NFS (port 2049) with self-referencing rules.
- **Batch job fails to start**: Check compute environment capacity and verify launch template is correctly configured.
- **Permission errors**: Ensure IAM roles have appropriate permissions for ECR, EFS, and S3 access.

For more troubleshooting tips, see the [main README](README.md#troubleshooting).