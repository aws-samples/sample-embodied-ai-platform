# Embodied AI Blog Series Part 1: Launching Training Jobs with AWS Batch and SageMaker for Pick-and-Place

## Architecture
![Architecture](./architecture.drawio.png)


## Draft Code Walkthrough

### 0. Prerequisites

1.  Install AWS CDK
    
    We will use AWS CDK to deploy an Amazon EC2 instance with Amazon DCV and NVIDIA IsaacLab to evaluate the fine-tuned policy in simulation, and optionally deploy the AWS Batch resources to fine-tune GR00T (follow along the instructions for more details).
    ```bash
    npm install -g aws-cdk
    ```

2. Clone the repo
    ```bash
    git clone https://github.com/aws-samples/sample-embodied-ai-platform.git
    cd sample-embodied-ai-platform
    ```

3.  Install Python dependencies for the CDK app:
    ```bash
    cd training/gr00t/infra
    pip install -r requirements.txt
    ```

4.  Bootstrap CDK (can be skipped if you have already done so for this account/region)
    > Note: Replace `YOUR_AWS_PROFILE` and `YOUR_AWS_REGION` with your credentials profile and target region.
    ```bash
    cdk bootstrap --profile YOUR_AWS_PROFILE --region YOUR_AWS_REGION
    ```

### 1. Review the Lerobot dataset for imitation learning

**Option 1**: Download and review our simulation dataset with Git LFS. 
> To install Git LFS, check out this [website](https://git-lfs.com/) for instructions.
```bash
git lfs pull
```
Sample dataset will be available in the `training/gr00t/sample_dataset` directory.

**Option 2**: Review compatible or your own [Lerobot datasets](https://huggingface.co/lerobot/datasets) with [Lerobot dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset). Make sure the dataset has the `modality.json` file in the `meta` folder. Refer to the [Isaac GR00T example](https://github.com/NVIDIA/Isaac-GR00T/blob/4ed4d45d83378c94f30aa228a4aff883d5cf285f/examples/SO-100/so100_dualcam__modality.json) for SO-ARM with dual-camera setup (assumed for the rest of the blog).

### 2. Set up the fine-tuning pipeline
> This blog uses us-west-2 region for fine-tuning but any [region with G6e instance family](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-regions.html) is supported.

We’ll create a reusable pipeline to fine-tune GR00T using AWS Batch on EC2 with NVIDIA GPU, so future fine-tuning runs on new datasets or models are as simple as submitting a new job with different environment variables. 

While an one-off job is easy to start in a Jupyter notebook (e.g., you can leverage Amazon SageMaker [CodeEditor](https://docs.aws.amazon.com/sagemaker/latest/dg/code-editor.html)/[JupyterLab](https://docs.aws.amazon.com/sagemaker/latest/dg/jupyterai.html) and follow the [Hugging Face × NVIDIA guide](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)), ML engineering teams often demand reliable, repeatable and cost efficient pipeline due to frequent dataset or model updates. Training physical AI models also commonly involves simulations with a multi-container setup. AWS Batch provides a secure, scalable, structured way to do this.

First, ensure you have quota to launch a g6e.2xlarge (or larger) GPU instance. You can request at least 8 vCPUs for “Running On-Demand G and VT instances” in your chosen region, e.g., for [us-west-2](https://us-west-2.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA).

#### Choose Your Deployment Path

We provide multiple paths to set up the infrastructure based on your environment and preferences:

**Path 1: Manual Console Setup** - Create all resources step-by-step via AWS Console. Choose this if you want to understand each service in depth and get familiar with AWS console navigation. Follow all sections below.

**Path 2: Automated CDK Deployment** - Deploy everything automatically with AWS CDK from your local machine. Choose this for quick setup or programmatic customization. Skip to section 3 and follow the CDK deployment instructions.

**Path 3: DCV First (for ARM/Non-x86 Machines)** - If you cannot build x86 container images locally (e.g., ARM-based machine), manually create shared resources (VPC, EFS, security group) first following section 2.1, deploy the DCV stack, then use the deployed EC2 instance to build containers then deploy the Batch stack.

> For detailed instructions on all CDK deployment paths, including dependencies, configuration options, and troubleshooting, see the [GR00T CDK Documentation](https://github.com/aws-samples/sample-embodied-ai-platform/blob/main/training/gr00t/infra/README.md).

The following sections (2.1-2.5) provide step-by-step manual instructions for **Path 1**. If you choose automated deployment (**Path 2**), you can skip to section 3, but we recommend reviewing these sections to understand the resources being created and how they work together.

#### 2.1 Create Amazon VPC, Security Group and EFS

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

#### 2.2 Build the fine-tuning container and push to ECR

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

#### 2.3 Create Launch Template

An EC2 Launch Template is a template that defines the configuration for an EC2 instance that is used to run a job. We will use this template to increase the Linux root volume size to 100 GiB for pulling large fine-tuning containers. To learn more about Launch Templates, refer to [Launch templates for AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/launch-templates.html).

1. Open the [EC2 console](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2).
2. In the navigation pane at the left, select **Launch templates** and choose **Create launch template**.
3. Under *Name* enter `BatchLaunchTemplate`.
4. For *Application and OS Images (Amazon Machine Image)* and *Instance type*, leave them as default (i.e. `Don't include in launch template`) as AWS Batch will automatically select the appropriate Amazon Linux AMI with ECS agent and NVIDIA driver preinstalled.
5. Scroll down to *Storage (volumes)* and select **Add new volume**. For *Volume type* select `gp3` and for *Size (GiB)* enter `100`. For *Device name* select **Specify a custom value...** and enter `/dev/xvda` (standard root device for Amazon Linux 2).
> NOTE: Without increasing the root volume size, the container may fail to pull the image with the error"CannotPullContainerError: context canceled" due to the default 8 GiB root volume size.
6. Leave the rest as default and choose **Create launch template**.

#### 2.4 Create Compute Environment

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

#### 2.5 Create Job Queue and Job Definition

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
10. Under *Filesystem configuration*, for *Shared memory size* enter `65536` MiB (64 GiB). Expand the *Additional configuration* section; select **Add volume** and **Enable EFS**, for *Name* enter `BatchEFS`, for *Filesystem ID* enter the EFS file system ID you noted in step 2.1 (Access is controlled by security groups); select *Add Mount points*, for *Source volume* select `BatchEFS`, for *Container path* enter `/mnt/efs`, select **Next**.
11. Review the configurations and **Create job definition**.

### 3. Submit and monitor fine-tuning jobs

Finally, we have created the necessary AWS Batch resources to run fine-tuning jobs repeatably. Now every time you collect a new dataset (e.g. for a new embodiment or task), you can simply update the job environment variables and submit a new job to the job queue. AWS Batch will automatically start and stop the compute resources as needed.
> If you did not manually follow the steps in section 2 above and want to deploy a CDK stack to create the AWS Batch resources up to this point automatically, you can simply run the following command:
> ```bash
> # From the root directory of the repo
> cd training/gr00t/infra
> cdk deploy IsaacGr00tBatchStack
> ```
> If you want to override the region for deployment, you can run:
> ```bash
> AWS_DEFAULT_REGION=<OPTIONAL_REGION_OVERRIDE> cdk deploy IsaacGr00tBatchStack
> ```
> The stack will build the container image from the Dockerfile in the repo and push it to ECR, which can take more than 10 minutes for building the flash-attn package and pushing the resulting image (~15GB). If you want to build and push the container separately following section 2.2 or use your own container image, you can supply the ECR image URI as environment variable when deploying the stack:
> ```bash
> ECR_IMAGE_URI=<YOUR_ECR_IMAGE_URI> cdk deploy IsaacGr00tBatchStack
> ```

To submit a job, you can use the AWS Batch console or the AWS CLI. 

- AWS Batch console: On the left navigation pane, select **Jobs** and choose **Submit new job** on the top right. For *Name* enter `IsaacGr00tFinetuning`, for *Job definition* select `IsaacGr00tJobDefinition`, for *Job queue* select `IsaacGr00tJobQueue` and choose **Next**. You can leave the rest as default and choose **Next** again and **Submit job**.
> NOTE:By default, the job will fine-tune GR00T on the sample dataset provided in the repo. If you want to fine-tune on a specific dataset, you can update the *Environment variables* under *Container overrides*. For example, you can set `HF_DATASET_ID` to fine-tune on a custom Lerobot dataset. Check out the [fine-tuning workflow script](https://github.com/aws-samples/embodied-ai-platform-examples/blob/main/training/gr00t/run_finetune_workflow.sh) for the full list of environment variables.

- AWS CLI: Make sure you have the AWS CLI installed and configured with the correct profile and region. Then simply run the following command to submit a job:
    ```bash
    aws batch submit-job --job-name "IsaacGr00tFinetuning" --job-queue "IsaacGr00tJobQueue" --job-definition "IsaacGr00tJobDefinition"
    ```
    > Optionally add the following to override the region with `--region <REGION>` and environment variables with `--container-overrides "environment=[{name=HF_DATASET_ID,value=<YOUR_HF_DATASET_ID>}]"`

> By default, the job will fine-tune GR00T for 6000 steps and save the model checkpoints every 2000 steps, which usually takes up to 3 hours on a g6e.4xlarge instance. You can change the number of steps and save frequency by overriding the `MAX_STEPS` and `SAVE_STEPS` environment variables when submitting the job.

#### Monitor job progress
You can use the console or CLI to track status and stream logs.

- AWS Batch console
    1. Go to **Jobs**, select the `IsaacGr00tJobQueue` job queue and choose **Search**. You should see the job you submitted in the list and its status.
    2. Click into the job you submitted and select the *Logging* tab. You should see the logs in real time.

- AWS CLI

    Provide the `JOB_ID` (e.g. from the above `batch submit-job` output) and optionally set `REGION` and `PROFILE` to run the following commands.
    Check job status:
    ```bash
    REGION=<REGION> PROFILE=<PROFILE> JOB_ID=<JOB_ID>; aws batch describe-jobs --jobs "$JOB_ID" \
        --query 'jobs[0].{status:status,statusReason:statusReason,createdAt:createdAt,startedAt:startedAt,stoppedAt:stoppedAt}' \
        --output table --region "$REGION" --profile "$PROFILE"
    ```
    Once the job is in RUNNING status, you can stream logs in real time:
    ```bash
    REGION=<REGION> PROFILE=<PROFILE> JOB_ID=<JOB_ID>; aws logs tail /aws/batch/job \
        --log-stream-names "$(aws batch describe-jobs --jobs "$JOB_ID" --query 'jobs[0].container.logStreamName' --output text --region "$REGION" --profile "$PROFILE")" \
        --follow --region "$REGION" --profile "$PROFILE"
    ```

### 4. Evaluate the fine-tuned policy

We can monitor the training process and evaluate the fine-tuned policy with a simulated and optionally a physical SO-ARM101. We will start by deploying an Amazon DCV instance connected to the same EFS file system, visualize the tensorboard logs as the training progresses, and then upon completion of the fine-tuning job, start the GR00T policy server in the instance and set up IsaacLab to evaluate the fine-tuned policy in a simulated environment.

1. Deploy the DCV CDK stack
    If you have followed the steps in section 2.1 to create the VPC, security group and EFS manually, add the following resource IDs as context to the `training/gr00t/infra/cdk.json` file. If you have skipped the steps in section 2.1, and deploy the `IsaacGr00tBatchStack` in section 3 directly, the context will be automatically imported and you can skip this step:
    ```json
    {
        "app": "python app.py",
        "context": {
            "vpc_id": "vpc-xxxxxxxx", // BatchVPC-vpc ID
            "efs_id": "fs-xxxxxxxx", // BatchEFS ID
            "efs_sg_id": "sg-xxxxxxxx" // BatchEFSSecurityGroup ID
        }
    }
    ```

    Now you can run the following command to deploy the DCV stack:
    ```bash
    cd training/gr00t/infra
    cdk deploy IsaacLabDcvStack
    ```
    > If you want to override the region for deployment, you can run:
    > ```bash
    > AWS_DEFAULT_REGION=<OPTIONAL_REGION_OVERRIDE> cdk deploy IsaacLabDcvStack
    >```
    
    Once deployed, the EC2 instance may still take a few minutes to initialize and run the user data script. You can run `sudo cat /var/log/dcv-bootstrap.summary` in the terminal to check the status, if you see 19 `STEP_OK`s and `STEP_OK:EFS mount` as the last one, the instance is ready. This stack mounts the same EFS created by the Batch stack at `/mnt/efs` using TLS via `amazon-efs-utils`, so you can live-inspect tensorboard logs and model checkpoints while jobs run.
    > NOTE: If you are accessing DCV via the Web URL, there may be a warning of "Your connection is not private" in the browser. You can ignore it and proceed to the next step or use the [DCV Client](https://www.amazondcv.com/latest.html) to connect to the instance. If the DCV interface is loaded but you get an error of "no session found", try again in 10 minutes. See the [user data script](training/gr00t/infra/configure_dcv_instance.sh) for troubleshooting and customization options.

2. Log into the DCV instance and visualize TensorBoard and inspect checkpoints

    Check the output of the deployed `IsaacLabDcvStack` to get the DCV instance public IP address and credentials and log in to the DCV session. Once logged in, you should see the tensorboard logs in the `/mnt/efs/gr00t/checkpoints/runs` directory. Run the following command to start the tensorboard server:
    ```bash
    # If the conda environment is not activated, run:
    # conda activate isaac 
    tensorboard --logdir /mnt/efs/gr00t/checkpoints/runs --bind_all
    ```
    The tensorboard server should be running on port 6006. You can access it either directly in the DCV instance by "ctrl + clicking" on the auto-generated URL or on any client (e.g. your local laptop browser) by using the DCV instance public IP address, e.g. `http://<DCV_INSTANCE_PUBLIC_IP>:6006`.

    Once the fine-tuning job is completed, you can inspect the model checkpoints in the `/mnt/efs/gr00t/checkpoints` directory.

3. Run the Isaac GR00T container and start the model server

    Go to [Amazon Elastic Container Registry console](https://us-west-2.console.aws.amazon.com/ecr/private-registry/repositories?region=us-west-2) and select the fine-tuning container repository created in section 2. Click on **View push commands** on the top right to view the first command to authenticate to ECR. The command should look like:
    ```bash
    aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <YOUR_CONTAINER_REGISTRY_PREFIX>
    ```

    Set your `<ECR_IMAGE_URI>` and start an interactive shell with EFS mounted as read-only:
    ```bash
    docker run -it --rm --gpus all --network host -v /mnt/efs:/mnt/efs:ro --entrypoint /bin/bash <ECR_IMAGE_URI>
    ```

    **In the container**, pick a checkpoint by its `<STEP>` (e.g. `6000`) and start the server:
    ```bash
    MODEL_STEP=<STEP>  # e.g. 6000
    MODEL_DIR="/mnt/efs/gr00t/checkpoints/checkpoint-$MODEL_STEP"

    python scripts/inference_service.py --server \
      --model_path "$MODEL_DIR" \
      --embodiment_tag new_embodiment \
      --data_config so100_dualcam \
      --denoising_steps 4
    ```
    You should see the following output when the server is ready: `Server is ready and listening on tcp://0.0.0.0:5555`

4. Run the leisaac kitchen scene orange picking task

    **In another terminal on the host machine**, run the following script to launch the leisaac kitchen scene orange picking task. This will connect a simualted SO-ARM101 to the GR00T policy server running in the container:
    ```bash
    # If the conda environment is not activated, run:
    # conda activate isaac 
    cd /home/ubuntu/leisaac
    OMNI_KIT_ACCEPT_EULA=YES python scripts/evaluation/policy_inference.py \
        --task=LeIsaac-SO101-PickOrange-v0 \
        --policy_type=gr00tn1.5 \
        --policy_host=localhost \
        --policy_port=5555 \
        --policy_timeout_ms=5000 \
        --policy_action_horizon=16 \
        --policy_language_instruction="Pick up an orange and place it on the plate" \
        --device=cuda \
        --enable_cameras
    ```
    IsaacSim may take a few minutes to initialize for the first time. 
    > NOTE: Make sure the inference server is running in the container before running this script. It may take 3 - 5 minutes for the scene to load and show `[INFO]: Completed setting up the environment...` in the terminal, which indicates the scene is ready to play. You can ignore the `[Warning]` messages in yellow and `[Error]` messages in red.
    Once the scene is loaded, you should see the simulated SO-ARM101 picking up the orange and placing it on the plate. You can stop the simulation by pressing `Ctrl+C` in the terminal.

If you have a physical SO-ARM101 with wrist and front cameras, you can also evaluate policy with a local physical SO-ARM101 by connecting a local client to the remote GR00T policy server. Continue with the following steps:

5. Assemble and calibrate SO-ARM101 with dual cameras following the [Lerobot guide](https://huggingface.co/docs/lerobot/en/so101)
6. Follow [Isaac GR00T official guide](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#installation-guide) to install dependencies on your local machine and run Isaac GR00T example client to control your physical SO-ARM101: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/examples/eval_gr00t_so100.py

If you also have a local GPU machine, you can run both GR00T policy server and example client locally. Refer to the [Isaac-GR00T guide](https://github.com/NVIDIA/Isaac-GR00T) for more details.

### Clean up
To avoid ongoing charges, tear down the resources you created. 

First delete the `IsaacLabDcvStack` in section 4:
```bash
# From repo root
cd training/gr00t/infra

# Destroy DCV stack first (terminates EC2 instance and releases EIP)
cdk destroy IsaacLabDcvStack --force
```

Then delete the Batch resources in section 2 based on the path you chose:

1) If you used CDK

Destroy `IsaacGr00tBatchStack` stack (removes Batch CE/Queue/JobDef, Log Group, EFS, VPC if created by CDK)
```bash
# Optionally override region: AWS_DEFAULT_REGION=<REGION>
cdk destroy IsaacGr00tBatchStack --force
```

- If you set the context in `training/gr00t/infra/cdk.json` manually, those resources were imported and will NOT be deleted by `cdk destroy`. Delete them manually if you created them for this walkthrough.

2) If you created resources manually in the console

- AWS Batch
  - Cancel any running jobs
  - Delete Job definitions: `IsaacGr00tJobDefinition`
  - Delete Job queue: `IsaacGr00tJobQueue`
  - Delete Compute environment: `IsaacGr00tComputeEnvironment`
  - Delete Launch template: `BatchLaunchTemplate`
  - (Optional) Delete CloudWatch Logs group: `/aws/batch/job/IsaacGr00t`

- ECR
    Delete repository `<ECR_IMAGE_URI>` on the console or via CLI:
    ```bash
    aws ecr delete-repository --repository-name <ECR_IMAGE_URI> --force \
      --region <REGION>
    ```
- Shared storage and networking
  - Delete EFS file system `BatchEFS` and its mount targets
  - Delete security group `BatchEFSSecurityGroup`
  - Delete VPC `BatchVPC` (this will remove subnets, route tables, NAT gateway, IGW)