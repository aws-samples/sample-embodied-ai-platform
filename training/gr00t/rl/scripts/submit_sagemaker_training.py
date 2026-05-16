#!/usr/bin/env python3
"""Submit a heterogeneous SageMaker training job via the AWS Batch SubmitServiceJob API.

This script uses the 2025 AWS Batch + SageMaker Training integration where:
- A CfnServiceEnvironment with type SAGEMAKER_TRAINING provides the compute backend
- A CfnJobQueue with job_queue_type SAGEMAKER_TRAINING routes jobs
- SubmitServiceJob accepts a full CreateTrainingJob payload inline (no job definition)
- Heterogeneous InstanceGroups allow different instance types per node group

Architecture:
  Learner group: 1x ml.g6e.48xlarge (8x L40S GPUs) - FSDP training via RLinf
  Rollout group: Nx ml.g6e.4xlarge (1x L40S GPU)   - Isaac Sim rollout workers

Reference: https://docs.aws.amazon.com/batch/latest/userguide/getting-started-sagemaker.html
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import boto3


def get_stack_outputs(stack_name: str, region: str) -> dict:
    """Read CloudFormation stack outputs into a key-value dict.

    Args:
        stack_name: CloudFormation stack name to query.
        region: AWS region.

    Returns:
        Dict mapping output key to output value.
    """
    cfn = boto3.client("cloudformation", region_name=region)
    response = cfn.describe_stacks(StackName=stack_name)
    stacks = response.get("Stacks", [])
    if not stacks:
        raise RuntimeError(f"Stack '{stack_name}' not found in region {region}")
    outputs = stacks[0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in outputs}


def build_payload(args) -> dict:
    """Construct the SubmitServiceJob request payload.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Dict representing the full SubmitServiceJob API call kwargs.
    """
    job_name = args.job_name or f"gr00t-rl-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    payload = {
        "jobName": job_name,
        "jobQueue": args.job_queue,
        "serviceJobConfiguration": {
            "sageMakerTrainingJobConfiguration": {
                "trainingJobName": job_name,
                "roleArn": args.execution_role_arn,
                "algorithmSpecification": {
                    "trainingInputMode": "File",
                    "trainingImage": args.image_uri,
                    "containerEntrypoint": ["python", "-m", "rlinf.train"],
                    "containerArguments": [
                        "--config", args.config_name,
                        "--model-path", args.model_path,
                        "--num-rollout-nodes", str(args.num_rollout_nodes),
                    ],
                    "instanceGroupAlgorithmSpecifications": [
                        {
                            "instanceGroupName": "learner",
                            "trainingImage": args.image_uri,
                            "containerEntrypoint": ["python", "-m", "rlinf.train"],
                            "containerArguments": [
                                "--node-role", "learner",
                                "--config", args.config_name,
                                "--model-path", args.model_path,
                                "--num-gpus", "8",
                                "--fsdp",
                            ],
                        },
                        {
                            "instanceGroupName": "rollout",
                            "trainingImage": args.image_uri,
                            "containerEntrypoint": ["python", "-m", "rlinf.rollout_worker"],
                            "containerArguments": [
                                "--node-role", "rollout",
                                "--config", args.config_name,
                                "--num-envs", str(args.num_envs),
                            ],
                        },
                    ],
                },
                "resourceConfig": {
                    "instanceGroups": [
                        {
                            "instanceGroupName": "learner",
                            "instanceType": "ml.g6e.48xlarge",
                            "instanceCount": 1,
                            "instanceStorageConfigs": [
                                {"ebsVolumeConfig": {"volumeSizeInGb": 500}}
                            ],
                        },
                        {
                            "instanceGroupName": "rollout",
                            "instanceType": "ml.g6e.4xlarge",
                            "instanceCount": args.num_rollout_nodes,
                            "instanceStorageConfigs": [
                                {"ebsVolumeConfig": {"volumeSizeInGb": 200}}
                            ],
                        },
                    ],
                },
                "vpcConfig": {
                    "securityGroupIds": args.security_group_ids.split(","),
                    "subnets": args.subnet_ids.split(","),
                },
                "stoppingCondition": {
                    "maxRuntimeInSeconds": 86400,  # 24 hours
                },
                "outputDataConfig": {
                    "s3OutputPath": args.s3_output,
                },
                "hyperParameters": {
                    "NUM_ROLLOUT_NODES": str(args.num_rollout_nodes),
                    "TOTAL_NODES": str(1 + args.num_rollout_nodes),
                    "CONFIG_NAME": args.config_name,
                    "NUM_ROLLOUT_ENVS": str(args.num_envs),
                    "NCCL_SOCKET_IFNAME": "eth0",
                    "NCCL_IB_DISABLE": "1",
                    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
                },
                "inputDataConfig": [
                    {
                        "channelName": "model",
                        "dataSource": {
                            "fileSystemDataSource": {
                                "fileSystemId": args.efs_id,
                                "fileSystemType": "EFS",
                                "directoryPath": "/models/GR00T-N1.5-RL-Rheo-AssembleTrocar",
                                "fileSystemAccessMode": "ro",
                            }
                        },
                    },
                    {
                        "channelName": "code",
                        "dataSource": {
                            "fileSystemDataSource": {
                                "fileSystemId": args.efs_id,
                                "fileSystemType": "EFS",
                                "directoryPath": "/training-code",
                                "fileSystemAccessMode": "ro",
                            }
                        },
                    },
                    {
                        "channelName": "checkpoints",
                        "dataSource": {
                            "fileSystemDataSource": {
                                "fileSystemId": args.efs_id,
                                "fileSystemType": "EFS",
                                "directoryPath": "/checkpoints",
                                "fileSystemAccessMode": "rw",
                            }
                        },
                    },
                ],
            }
        },
    }

    return payload


def print_monitoring_commands(job_id: str, job_name: str) -> None:
    """Print useful commands for monitoring the submitted training job.

    Args:
        job_id: The service job ID returned by SubmitServiceJob.
        job_name: The training job name (used by SageMaker).
    """
    print("\n" + "=" * 70)
    print("MONITORING COMMANDS")
    print("=" * 70)
    print(f"\n# Batch-side (job scheduling, queue status):")
    print(f"  aws batch describe-service-jobs --service-jobs {job_id}")
    print(f"\n# SageMaker-side (training progress, metrics):")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name}")
    print(f"\n# Live logs:")
    print(f"  aws logs tail /aws/sagemaker/TrainingJobs --follow")
    print(f"\n# Cancel job:")
    print(f'  aws batch cancel-service-job --service-job-id {job_id} --reason "Manual cancel"')
    print("=" * 70 + "\n")


def resolve_from_stack(args) -> argparse.Namespace:
    """Populate missing args from CloudFormation stack outputs.

    When --from-stack is provided, reads stack outputs to auto-populate:
    --image-uri, --execution-role-arn, --efs-id, --s3-output,
    --subnet-ids, --security-group-ids.

    Args:
        args: Parsed argparse namespace (may have None values).

    Returns:
        Updated namespace with values filled from stack outputs.
    """
    if not args.from_stack:
        return args

    print(f"Reading outputs from stack: {args.from_stack} (region: {args.region})")
    outputs = get_stack_outputs(args.from_stack, args.region)

    # Map stack output keys to CLI args
    if not args.image_uri and "SageMakerTrainingImage" in outputs:
        args.image_uri = outputs["SageMakerTrainingImage"]
        print(f"  --image-uri = {args.image_uri}")
    elif not args.image_uri and "UnifiedECRUri" in outputs:
        args.image_uri = f"{outputs['UnifiedECRUri']}:latest"
        print(f"  --image-uri = {args.image_uri}")

    if not args.execution_role_arn and "SageMakerExecutionRoleArn" in outputs:
        args.execution_role_arn = outputs["SageMakerExecutionRoleArn"]
        print(f"  --execution-role-arn = {args.execution_role_arn}")

    if not args.efs_id and "EFSFileSystemId" in outputs:
        args.efs_id = outputs["EFSFileSystemId"]
        print(f"  --efs-id = {args.efs_id}")

    if not args.s3_output and "SageMakerOutputPath" in outputs:
        args.s3_output = outputs["SageMakerOutputPath"]
        print(f"  --s3-output = {args.s3_output}")
    elif not args.s3_output and "ArtifactBucket" in outputs:
        args.s3_output = f"s3://{outputs['ArtifactBucket']}/sagemaker-output/"
        print(f"  --s3-output = {args.s3_output}")

    # Subnet and SG IDs require VPC lookup if not in outputs
    if not args.subnet_ids and "VpcId" in outputs:
        ec2_client = boto3.client("ec2", region_name=args.region)
        vpc_id = outputs["VpcId"]
        subnets = ec2_client.describe_subnets(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "map-public-ip-on-launch", "Values": ["false"]},
            ]
        )["Subnets"]
        if subnets:
            args.subnet_ids = ",".join(s["SubnetId"] for s in subnets)
            print(f"  --subnet-ids = {args.subnet_ids}")

    if not args.security_group_ids and "VpcId" in outputs:
        ec2_client = boto3.client("ec2", region_name=args.region)
        vpc_id = outputs["VpcId"]
        sgs = ec2_client.describe_security_groups(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "description", "Values": ["*Batch*MNP*"]},
            ]
        )["SecurityGroups"]
        if sgs:
            args.security_group_ids = ",".join(sg["GroupId"] for sg in sgs)
            print(f"  --security-group-ids = {args.security_group_ids}")

    if not args.job_queue and "SageMakerJobQueueArn" in outputs:
        args.job_queue = outputs["SageMakerJobQueueArn"]
        print(f"  --job-queue = {args.job_queue}")

    print()
    return args


def validate_args(args) -> None:
    """Validate that all required arguments are present after stack resolution.

    Args:
        args: Parsed and resolved argparse namespace.

    Raises:
        SystemExit: If required arguments are missing.
    """
    missing = []
    if not args.image_uri:
        missing.append("--image-uri")
    if not args.execution_role_arn:
        missing.append("--execution-role-arn")
    if not args.efs_id:
        missing.append("--efs-id")
    if not args.s3_output:
        missing.append("--s3-output")
    if not args.subnet_ids:
        missing.append("--subnet-ids")
    if not args.security_group_ids:
        missing.append("--security-group-ids")

    if missing:
        print(f"ERROR: Missing required arguments: {', '.join(missing)}", file=sys.stderr)
        print("Provide them directly or use --from-stack to auto-resolve from CloudFormation.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Submit heterogeneous SageMaker training job via AWS Batch SubmitServiceJob API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-resolve all params from deployed CDK stack:
  python submit_sagemaker_training.py --from-stack GR00TRLBatchStack --dry-run

  # Explicit params:
  python submit_sagemaker_training.py \\
    --image-uri 215143956078.dkr.ecr.us-west-2.amazonaws.com/gr00t-rl-unified:latest \\
    --execution-role-arn arn:aws:iam::215143956078:role/GR00T-RL-SageMaker-ExecutionRole \\
    --efs-id fs-12345 \\
    --s3-output s3://my-bucket/sagemaker-output/ \\
    --subnet-ids subnet-aaa,subnet-bbb \\
    --security-group-ids sg-123
        """,
    )

    # Convenience mode: resolve params from CloudFormation stack
    parser.add_argument(
        "--from-stack",
        default=None,
        help="CloudFormation stack name to auto-resolve params from (e.g., GR00TRLBatchStack)",
    )

    # Job identification
    parser.add_argument(
        "--job-name",
        default=None,
        help="Job name (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--job-queue",
        default="GR00T-RL-SageMaker-JobQueue",
        help="SageMaker Training job queue name or ARN (default: GR00T-RL-SageMaker-JobQueue)",
    )

    # Required params (can be resolved from --from-stack)
    parser.add_argument(
        "--image-uri",
        default=None,
        help="ECR URI for the unified training container image",
    )
    parser.add_argument(
        "--execution-role-arn",
        default=None,
        help="SageMaker execution role ARN",
    )
    parser.add_argument(
        "--efs-id",
        default=None,
        help="EFS filesystem ID for input channels (model, code, checkpoints)",
    )
    parser.add_argument(
        "--s3-output",
        default=None,
        help="S3 path for training output artifacts",
    )
    parser.add_argument(
        "--subnet-ids",
        default=None,
        help="Comma-separated private subnet IDs for VPC config",
    )
    parser.add_argument(
        "--security-group-ids",
        default=None,
        help="Comma-separated security group IDs for VPC config",
    )

    # Training configuration
    parser.add_argument(
        "--num-rollout-nodes",
        type=int,
        default=4,
        help="Number of rollout worker instances (default: 4)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Parallel environments per rollout node (default: 64)",
    )
    parser.add_argument(
        "--model-path",
        default="/opt/ml/input/data/model",
        help="Model path inside container (default: /opt/ml/input/data/model)",
    )
    parser.add_argument(
        "--config-name",
        default="isaaclab_ppo_gr00t_assemble_trocar",
        help="RLinf training config name (default: isaaclab_ppo_gr00t_assemble_trocar)",
    )

    # Execution options
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region (default: us-west-2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full JSON payload without submitting",
    )

    args = parser.parse_args()

    # Resolve from stack if requested
    args = resolve_from_stack(args)

    # Validate all required params are present
    validate_args(args)

    # Build the SubmitServiceJob payload
    payload = build_payload(args)

    if args.dry_run:
        print("DRY RUN - SubmitServiceJob payload:")
        print("-" * 70)
        print(json.dumps(payload, indent=2))
        print("-" * 70)
        print(f"\nWould submit to: {args.job_queue}")
        print(f"Learner: 1x ml.g6e.48xlarge (8 GPUs, FSDP)")
        print(f"Rollout: {args.num_rollout_nodes}x ml.g6e.4xlarge (1 GPU each, Isaac Sim)")
        print(f"Total nodes: {1 + args.num_rollout_nodes}")
        return

    # Submit the job
    batch_client = boto3.client("batch", region_name=args.region)
    print(f"Submitting training job to queue: {args.job_queue}")
    print(f"  Learner: 1x ml.g6e.48xlarge (8 GPUs, FSDP)")
    print(f"  Rollout: {args.num_rollout_nodes}x ml.g6e.4xlarge (1 GPU each)")
    print(f"  Config: {args.config_name}")
    print()

    response = batch_client.submit_service_job(**payload)

    job_id = response.get("serviceJobId", response.get("jobId", "UNKNOWN"))
    job_name = payload["serviceJobConfiguration"]["sageMakerTrainingJobConfiguration"]["trainingJobName"]

    print(f"Job submitted successfully!")
    print(f"  Service Job ID: {job_id}")
    print(f"  Training Job Name: {job_name}")

    print_monitoring_commands(job_id, job_name)


if __name__ == "__main__":
    main()
