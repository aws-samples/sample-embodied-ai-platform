"""EKS + KubeRay CDK stack for GR00T RL heterogeneous training.

Deploys an EKS cluster with:
  - Learner node group: 1x g6e.48xlarge (8x L40S, 768 GB RAM) for FSDP training
  - Rollout node group: Nx g6e.4xlarge (1x L40S each) for Isaac Sim rollout workers
  - KubeRay operator (Helm chart v1.1.0) managing RayCluster lifecycle
  - NVIDIA device plugin (Helm chart) exposing nvidia.com/gpu resources
  - EFS CSI driver (EKS addon) mounting existing filesystem for shared storage
  - RayCluster CR with heterogeneous head (8 GPU) + workers (1 GPU each)

Deploy:
  cdk deploy --context compute_backend=eks \\
    --context vpc_id=vpc-00ce44fb57e6e740e \\
    --context efs_id=fs-05cc94bf7eeacab6c \\
    --context efs_sg_id=<sg-id> \\
    --context image_uri=215143956078.dkr.ecr.us-east-2.amazonaws.com/gr00t-rl-unified:latest
"""

import pathlib

from aws_cdk import (
    Stack,
    CfnOutput,
    aws_eks as eks,
    aws_ec2 as ec2,
    aws_efs as efs,
    aws_iam as iam,
)
from aws_cdk.lambda_layer_kubectl_v31 import KubectlV31Layer
from constructs import Construct


class EKSKubeRayStack(Stack):
    """EKS cluster with KubeRay for heterogeneous GPU training.

    Architecture:
      - Head pod (g6e.48xlarge): Ray head + FSDP learner actors on 8 GPUs
      - Worker pods (g6e.4xlarge x N): Ray workers + Isaac Sim rollout on 1 GPU each
      - KubeRay operator manages Ray cluster formation (no manual ray start)
      - EFS shared storage for code, models, checkpoints, TensorBoard logs
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc_id: str,
        efs_id: str,
        efs_sg_id: str,
        image_uri: str,
        num_rollout_workers: int = 4,
        learner_instance_type: str = "g6e.48xlarge",
        rollout_instance_type: str = "g6e.4xlarge",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ==============================================================
        # region 1. Import existing VPC and EFS
        # ==============================================================
        vpc = ec2.Vpc.from_lookup(self, "VPC", vpc_id=vpc_id)
        efs_sg = ec2.SecurityGroup.from_security_group_id(
            self, "EFSSG", efs_sg_id, mutable=True
        )
        efs_fs = efs.FileSystem.from_file_system_attributes(
            self, "EFS", file_system_id=efs_id, security_group=efs_sg
        )
        # endregion

        # ==============================================================
        # region 2. EKS Cluster
        # ==============================================================
        # Masters role: allows the deployer (or any assumed role) kubectl access.
        # CDK's EKS construct only grants access to its own creation role by default.
        masters_role = iam.Role(
            self,
            "ClusterAdminRole",
            assumed_by=iam.AccountRootPrincipal(),
            role_name=f"gr00t-rl-eks-admin-{Stack.of(self).region}",
        )

        cluster = eks.Cluster(
            self,
            "TrainingCluster",
            cluster_name="gr00t-rl-eks",
            version=eks.KubernetesVersion.V1_31,
            kubectl_layer=KubectlV31Layer(self, "KubectlLayer"),
            default_capacity=0,
            vpc=vpc,
            vpc_subnets=[
                ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)
            ],
            masters_role=masters_role,
        )
        # endregion

        # ==============================================================
        # region 3. EFS security group ingress for EKS nodes
        # ==============================================================
        # Allow NFS (port 2049) from EKS cluster security group to EFS.
        # T-06-02 mitigation: scoped to cluster SG only, not 0.0.0.0/0.
        efs_sg.add_ingress_rule(
            peer=ec2.Peer.security_group_id(cluster.cluster_security_group_id),
            connection=ec2.Port.tcp(2049),
            description="EFS access from EKS cluster nodes",
        )
        # endregion

        # ==============================================================
        # region 4. Learner node group (D-01: g6e.48xlarge, 8x L40S)
        # ==============================================================
        learner_ng = cluster.add_nodegroup_capacity(
            "LearnerNodes",
            instance_types=[ec2.InstanceType(learner_instance_type)],
            ami_type=eks.NodegroupAmiType.AL2023_X86_64_NVIDIA,
            min_size=1,
            max_size=1,
            desired_size=1,
            disk_size=200,
            labels={"node-role": "learner"},
        )
        # endregion

        # ==============================================================
        # region 5. Rollout node group (D-02: g6e.4xlarge, 1x L40S each)
        # ==============================================================
        rollout_ng = cluster.add_nodegroup_capacity(
            "RolloutNodes",
            instance_types=[ec2.InstanceType(rollout_instance_type)],
            ami_type=eks.NodegroupAmiType.AL2023_X86_64_NVIDIA,
            min_size=num_rollout_workers,
            max_size=num_rollout_workers,
            desired_size=num_rollout_workers,
            disk_size=200,
            labels={"node-role": "rollout"},
        )
        # endregion

        # ==============================================================
        # region 6. EFS CSI driver addon
        # ==============================================================
        # Use the CDK L2 Addon (no service_account_role_arn needed —
        # the addon creates its own SA; node instance role has EFS policy)
        cluster.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AmazonEFSCSIDriverPolicy"
            )
        )
        efs_addon = eks.CfnAddon(
            self,
            "EFSCSIAddon",
            addon_name="aws-efs-csi-driver",
            cluster_name=cluster.cluster_name,
            addon_version="v3.2.0-eksbuild.1",
            resolve_conflicts="OVERWRITE",
        )
        # endregion

        # ==============================================================
        # region 7. NVIDIA device plugin Helm chart
        # ==============================================================
        nvidia_chart = cluster.add_helm_chart(
            "NvdpChart",
            chart="nvidia-device-plugin",
            release="nvidia-device-plugin",
            repository="https://nvidia.github.io/k8s-device-plugin",
            namespace="nvidia",
            create_namespace=True,
            values={
                "gfd": {"enabled": True},
                "mofedEnabled": False,
            },
        )
        # Device plugin must be ready before workloads schedule on GPU nodes.
        nvidia_chart.node.add_dependency(learner_ng)
        nvidia_chart.node.add_dependency(rollout_ng)
        # endregion

        # ==============================================================
        # region 8. KubeRay operator Helm chart
        # ==============================================================
        kuberay_chart = cluster.add_helm_chart(
            "KubeRayChart",
            chart="kuberay-operator",
            release="kuberay-operator",
            repository="https://ray-project.github.io/kuberay-helm/",
            namespace="kuberay-system",
            create_namespace=True,
            version="1.1.0",
            values={},
        )
        # endregion

        # ==============================================================
        # region 9. Training namespace
        # ==============================================================
        training_ns = cluster.add_manifest(
            "TrainingNamespace",
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": "training"},
            },
        )
        # endregion

        # ==============================================================
        # region 10. EFS StorageClass + PV + PVC (static provisioning)
        # ==============================================================
        efs_sc = cluster.add_manifest(
            "EFSStorageClass",
            {
                "apiVersion": "storage.k8s.io/v1",
                "kind": "StorageClass",
                "metadata": {"name": "efs-sc"},
                "provisioner": "efs.csi.aws.com",
            },
        )

        efs_pv = cluster.add_manifest(
            "EFSPersistentVolume",
            {
                "apiVersion": "v1",
                "kind": "PersistentVolume",
                "metadata": {"name": "efs-training-pv"},
                "spec": {
                    "capacity": {"storage": "1Ti"},
                    "volumeMode": "Filesystem",
                    "accessModes": ["ReadWriteMany"],
                    "persistentVolumeReclaimPolicy": "Retain",
                    "storageClassName": "efs-sc",
                    "csi": {
                        "driver": "efs.csi.aws.com",
                        "volumeHandle": efs_id,
                    },
                },
            },
        )
        efs_pv.node.add_dependency(efs_sc)
        efs_pv.node.add_dependency(efs_addon)

        efs_pvc = cluster.add_manifest(
            "EFSPersistentVolumeClaim",
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": "efs-training-pvc",
                    "namespace": "training",
                },
                "spec": {
                    "accessModes": ["ReadWriteMany"],
                    "storageClassName": "efs-sc",
                    "resources": {"requests": {"storage": "1Ti"}},
                },
            },
        )
        efs_pvc.node.add_dependency(training_ns)
        efs_pvc.node.add_dependency(efs_pv)
        # endregion

        # ==============================================================
        # region 10b. Entrypoint ConfigMap (mounted into head pod)
        # ==============================================================
        entrypoint_path = (
            pathlib.Path(__file__).parent.parent / "docker" / "entrypoint-eks.sh"
        )
        entrypoint_content = entrypoint_path.read_text()

        entrypoint_cm = cluster.add_manifest(
            "EntrypointConfigMap",
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "entrypoint-eks",
                    "namespace": "training",
                },
                "data": {"entrypoint-eks.sh": entrypoint_content},
            },
        )
        entrypoint_cm.node.add_dependency(training_ns)
        # endregion

        # ==============================================================
        # region 11. RayCluster CR (D-03: heterogeneous head + workers)
        # ==============================================================
        raycluster = cluster.add_manifest(
            "RayClusterTraining",
            {
                "apiVersion": "ray.io/v1",
                "kind": "RayCluster",
                "metadata": {
                    "name": "gr00t-rl-training",
                    "namespace": "training",
                    "annotations": {
                        "ray.io/overwrite-container-cmd": "true",
                    },
                },
                "spec": {
                    "rayVersion": "2.9.0",
                    "headGroupSpec": {
                        "rayStartParams": {
                            "dashboard-host": "0.0.0.0",
                            "num-gpus": "8",
                        },
                        "template": {
                            "metadata": {"labels": {"ray-role": "head"}},
                            "spec": {
                                "nodeSelector": {"node-role": "learner"},
                                "containers": [
                                    {
                                        "name": "ray-head",
                                        "image": image_uri,
                                        "command": ["/bin/bash", "-lc", "--"],
                                        "args": [
                                            "ulimit -n 65536; RAY_START_CMD=$(echo $KUBERAY_GEN_RAY_START_CMD | sed 's/--block//g'); eval $RAY_START_CMD; /opt/entrypoint-eks.sh"
                                        ],
                                        "resources": {
                                            "requests": {
                                                "cpu": "90",
                                                "memory": "600Gi",
                                                "nvidia.com/gpu": "8",
                                            },
                                            "limits": {
                                                "cpu": "192",
                                                "nvidia.com/gpu": "8",
                                            },
                                        },
                                        "env": [
                                            {
                                                "name": "PATH",
                                                "value": "/isaac-sim/kit/python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                                            },
                                            {
                                                "name": "RLINF_EXT_MODULE",
                                                "value": "rlinf_ext",
                                            },
                                            {
                                                "name": "TORCHDYNAMO_DISABLE",
                                                "value": "1",
                                            },
                                            {
                                                "name": "NCCL_IB_DISABLE",
                                                "value": "1",
                                            },
                                            {
                                                "name": "NCCL_SOCKET_IFNAME",
                                                "value": "eth0",
                                            },
                                            {
                                                "name": "NODE_ROLE",
                                                "value": "learner",
                                            },
                                            {
                                                "name": "RAY_memory_usage_threshold",
                                                "value": "0.99",
                                            },
                                            {
                                                "name": "NUM_ROLLOUT_WORKERS",
                                                "value": str(num_rollout_workers),
                                            },
                                            {
                                                "name": "PYTHONPATH",
                                                "value": "/mnt/efs/third_party/RLinf:/mnt/efs/third_party/Isaac-GR00T:/mnt/efs/third_party/embodied-ai-platform:/mnt/efs/third_party/IsaacLab/source:/mnt/efs/workflows/rheo/scripts:/mnt/efs/workflows/rheo/scripts/simulation/rl",
                                            },
                                        ],
                                        "volumeMounts": [
                                            {
                                                "name": "efs-storage",
                                                "mountPath": "/mnt/efs",
                                            },
                                            {
                                                "name": "dshm",
                                                "mountPath": "/dev/shm",
                                            },
                                            {
                                                "name": "entrypoint",
                                                "mountPath": "/opt/entrypoint-eks.sh",
                                                "subPath": "entrypoint-eks.sh",
                                            },
                                        ],
                                    }
                                ],
                                "volumes": [
                                    {
                                        "name": "efs-storage",
                                        "persistentVolumeClaim": {
                                            "claimName": "efs-training-pvc"
                                        },
                                    },
                                    {
                                        "name": "dshm",
                                        "emptyDir": {
                                            "medium": "Memory",
                                            "sizeLimit": "128Gi",
                                        },
                                    },
                                    {
                                        "name": "entrypoint",
                                        "configMap": {
                                            "name": "entrypoint-eks",
                                            "defaultMode": 0o755,
                                        },
                                    },
                                ],
                            },
                        },
                    },
                    "workerGroupSpecs": [
                        {
                            "groupName": "rollout-workers",
                            "replicas": num_rollout_workers,
                            "minReplicas": num_rollout_workers,
                            "maxReplicas": num_rollout_workers,
                            "rayStartParams": {"num-gpus": "1"},
                            "template": {
                                "metadata": {"labels": {"ray-role": "worker"}},
                                "spec": {
                                    "nodeSelector": {"node-role": "rollout"},
                                    "containers": [
                                        {
                                            "name": "ray-worker",
                                            "image": image_uri,
                                            "command": [
                                                "/bin/bash",
                                                "-lc",
                                                "--",
                                            ],
                                            "args": [
                                                "ulimit -n 65536; RAY_START_CMD=$(echo $KUBERAY_GEN_RAY_START_CMD | sed 's/--block//g'); eval $RAY_START_CMD; sleep infinity"
                                            ],
                                            "resources": {
                                                "requests": {
                                                    "cpu": "12",
                                                    "memory": "100Gi",
                                                    "nvidia.com/gpu": "1",
                                                },
                                                "limits": {
                                                    "cpu": "16",
                                                    "memory": "120Gi",
                                                    "nvidia.com/gpu": "1",
                                                },
                                            },
                                            "env": [
                                                {
                                                    "name": "PATH",
                                                    "value": "/isaac-sim/kit/python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                                                },
                                                {
                                                    "name": "RLINF_EXT_MODULE",
                                                    "value": "rlinf_ext",
                                                },
                                                {
                                                    "name": "TORCHDYNAMO_DISABLE",
                                                    "value": "1",
                                                },
                                                {
                                                    "name": "NCCL_IB_DISABLE",
                                                    "value": "1",
                                                },
                                                {
                                                    "name": "NCCL_SOCKET_IFNAME",
                                                    "value": "eth0",
                                                },
                                                {
                                                    "name": "NODE_ROLE",
                                                    "value": "rollout",
                                                },
                                                {
                                                    "name": "PYTHONPATH",
                                                    "value": "/mnt/efs/third_party/RLinf:/mnt/efs/third_party/Isaac-GR00T:/mnt/efs/third_party/embodied-ai-platform:/mnt/efs/third_party/IsaacLab/source:/mnt/efs/workflows/rheo/scripts:/mnt/efs/workflows/rheo/scripts/simulation/rl",
                                                },
                                            ],
                                            "volumeMounts": [
                                                {
                                                    "name": "efs-storage",
                                                    "mountPath": "/mnt/efs",
                                                },
                                                {
                                                    "name": "dshm",
                                                    "mountPath": "/dev/shm",
                                                },
                                            ],
                                        }
                                    ],
                                    "volumes": [
                                        {
                                            "name": "efs-storage",
                                            "persistentVolumeClaim": {
                                                "claimName": "efs-training-pvc"
                                            },
                                        },
                                        {
                                            "name": "dshm",
                                            "emptyDir": {
                                                "medium": "Memory",
                                                "sizeLimit": "16Gi",
                                            },
                                        },
                                    ],
                                },
                            },
                        }
                    ],
                },
            },
        )
        # endregion

        # ==============================================================
        # region 12. CDK dependency ordering
        # ==============================================================
        # RayCluster depends on KubeRay CRDs, NVIDIA device plugin, EFS PVC, namespace, entrypoint
        raycluster.node.add_dependency(kuberay_chart)
        raycluster.node.add_dependency(nvidia_chart)
        raycluster.node.add_dependency(efs_pvc)
        raycluster.node.add_dependency(training_ns)
        raycluster.node.add_dependency(entrypoint_cm)
        # endregion

        # ==============================================================
        # region 13. CfnOutputs
        # ==============================================================
        CfnOutput(
            self,
            "ClusterName",
            value=cluster.cluster_name,
            description="EKS cluster name",
        )
        CfnOutput(
            self,
            "ClusterEndpoint",
            value=cluster.cluster_endpoint,
            description="EKS cluster API endpoint",
        )
        CfnOutput(
            self,
            "LearnerNodeGroupArn",
            value=learner_ng.nodegroup_arn,
            description=f"Learner node group ARN ({learner_instance_type})",
        )
        CfnOutput(
            self,
            "RolloutNodeGroupArn",
            value=rollout_ng.nodegroup_arn,
            description=f"Rollout node group ARN ({rollout_instance_type})",
        )
        CfnOutput(
            self,
            "KubeconfigCommand",
            value=f"aws eks update-kubeconfig --name {cluster.cluster_name} --region {Stack.of(self).region} --role-arn {masters_role.role_arn}",
            description="Command to configure kubectl (assumes admin role)",
        )
        # endregion
