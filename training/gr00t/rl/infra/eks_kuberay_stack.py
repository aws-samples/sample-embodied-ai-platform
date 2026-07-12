"""EKS + KubeRay CDK stack for GR00T RL heterogeneous training.

Deploys an EKS cluster with:
  - Learner node group: 1x g6e.48xlarge (8x L40S, 768 GB RAM) for FSDP training
  - Rollout node group: Nx g6e.4xlarge (1x L40S each) for Isaac Sim rollout workers
  - KubeRay operator (Helm chart v1.1.0) managing RayCluster lifecycle
  - NVIDIA device plugin (Helm chart) exposing nvidia.com/gpu resources
  - FSx for Lustre (SCRATCH_2) backed by S3 via Data Repository Association
  - RayCluster CR with heterogeneous head (8 GPU) + workers (1 GPU each)

Deploy:
  cdk deploy --context compute_backend=eks \\
    --context vpc_id=vpc-00ce44fb57e6e740e \\
    --context s3_data_bucket=gr00t-rl-training-data \\
    --context image_uri=215143956078.dkr.ecr.us-east-2.amazonaws.com/gr00t-rl-unified:latest
"""

import pathlib

from aws_cdk import (
    Stack,
    CfnOutput,
    CfnJson,
    RemovalPolicy,
    aws_eks as eks,
    aws_ec2 as ec2,
    aws_fsx as fsx,
    aws_s3 as s3,
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
      - FSx for Lustre shared storage for code, models, checkpoints, TensorBoard logs
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc_id: str,
        s3_data_bucket: str,
        image_uri: str,
        num_rollout_workers: int = 4,
        fsx_capacity_gib: int = 1200,
        learner_instance_type: str = "g6e.48xlarge",
        rollout_instance_type: str = "g6e.4xlarge",
        capacity_reservation_id: str = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ==============================================================
        # region 1. Import existing VPC and S3 data bucket
        # ==============================================================
        vpc = ec2.Vpc.from_lookup(self, "VPC", vpc_id=vpc_id)
        data_bucket = s3.Bucket.from_bucket_name(
            self, "DataBucket", s3_data_bucket
        )
        # endregion

        # ==============================================================
        # region 2. EKS Cluster
        # ==============================================================
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
        # region 3. FSx for Lustre security group
        # ==============================================================
        fsx_sg = ec2.CfnSecurityGroup(
            self,
            "FsxLustreSG",
            vpc_id=vpc.vpc_id,
            group_description="FSx for Lustre - ports 988, 1018-1023",
            security_group_ingress=[
                ec2.CfnSecurityGroup.IngressProperty(
                    ip_protocol="tcp", from_port=988, to_port=988,
                    source_security_group_id=cluster.cluster_security_group_id,
                    description="Lustre from EKS nodes",
                ),
                ec2.CfnSecurityGroup.IngressProperty(
                    ip_protocol="tcp", from_port=1018, to_port=1023,
                    source_security_group_id=cluster.cluster_security_group_id,
                    description="Lustre from EKS nodes",
                ),
            ],
        )
        # Self-referencing rules must be added after SG creation
        ec2.CfnSecurityGroupIngress(
            self, "FsxSelfIngress988",
            group_id=fsx_sg.attr_group_id,
            ip_protocol="tcp", from_port=988, to_port=988,
            source_security_group_id=fsx_sg.attr_group_id,
            description="Lustre MGS/MGC (self)",
        )
        ec2.CfnSecurityGroupIngress(
            self, "FsxSelfIngress1018",
            group_id=fsx_sg.attr_group_id,
            ip_protocol="tcp", from_port=1018, to_port=1023,
            source_security_group_id=fsx_sg.attr_group_id,
            description="Lustre OST/MDT (self)",
        )
        fsx_sg_id = fsx_sg.attr_group_id
        # endregion

        # ==============================================================
        # region 4. FSx for Lustre filesystem + Data Repository Association
        # ==============================================================
        # FSx is single-AZ — pick one private subnet and pin node groups to it
        fsx_subnet = vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
        ).subnets[0]

        fsx_security_group = ec2.SecurityGroup.from_security_group_id(
            self, "FsxSGImport", fsx_sg_id
        )
        lustre_fs = fsx.LustreFileSystem(
            self,
            "TrainingFsx",
            vpc=vpc,
            vpc_subnet=fsx_subnet,
            storage_capacity_gib=fsx_capacity_gib,
            lustre_configuration=fsx.LustreConfiguration(
                deployment_type=fsx.LustreDeploymentType.PERSISTENT_2,
                per_unit_storage_throughput=250,
            ),
            security_group=fsx_security_group,
            removal_policy=RemovalPolicy.DESTROY,
        )

        fsx_dra = fsx.CfnDataRepositoryAssociation(
            self,
            "DataDRA",
            file_system_id=lustre_fs.file_system_id,
            file_system_path="/",
            data_repository_path=f"s3://{data_bucket.bucket_name}/",
            s3=fsx.CfnDataRepositoryAssociation.S3Property(
                auto_import_policy=fsx.CfnDataRepositoryAssociation.AutoImportPolicyProperty(
                    events=["NEW", "CHANGED", "DELETED"]
                ),
                auto_export_policy=fsx.CfnDataRepositoryAssociation.AutoExportPolicyProperty(
                    events=["NEW", "CHANGED", "DELETED"]
                ),
            ),
            batch_import_meta_data_on_create=True,
        )
        # endregion

        # ==============================================================
        # region 5. Node groups (pinned to FSx subnet for single-AZ alignment)
        # ==============================================================
        learner_ng_kwargs = dict(
            instance_types=[ec2.InstanceType(learner_instance_type)],
            ami_type=eks.NodegroupAmiType.AL2023_X86_64_NVIDIA,
            min_size=1,
            max_size=1,
            desired_size=1,
            disk_size=200,
            labels={"node-role": "learner"},
            subnets=ec2.SubnetSelection(subnets=[fsx_subnet]),
        )

        if capacity_reservation_id:
            learner_lt = ec2.CfnLaunchTemplate(
                self,
                "LearnerLaunchTemplate",
                launch_template_data=ec2.CfnLaunchTemplate.LaunchTemplateDataProperty(
                    instance_type=learner_instance_type,
                    instance_market_options=ec2.CfnLaunchTemplate.InstanceMarketOptionsProperty(
                        market_type="capacity-block",
                    ),
                    capacity_reservation_specification=ec2.CfnLaunchTemplate.CapacityReservationSpecificationProperty(
                        capacity_reservation_target=ec2.CfnLaunchTemplate.CapacityReservationTargetProperty(
                            capacity_reservation_id=capacity_reservation_id,
                        ),
                    ),
                    block_device_mappings=[
                        ec2.CfnLaunchTemplate.BlockDeviceMappingProperty(
                            device_name="/dev/xvda",
                            ebs=ec2.CfnLaunchTemplate.EbsProperty(
                                volume_size=200,
                                volume_type="gp3",
                            ),
                        )
                    ],
                ),
            )
            learner_ng_kwargs.pop("disk_size", None)
            learner_ng_kwargs.pop("instance_types", None)
            learner_ng_kwargs["capacity_type"] = eks.CapacityType.CAPACITY_BLOCK
            learner_ng_kwargs["launch_template_spec"] = eks.LaunchTemplateSpec(
                id=learner_lt.ref,
                version=learner_lt.attr_latest_version_number,
            )

        learner_ng = cluster.add_nodegroup_capacity("LearnerNodes", **learner_ng_kwargs)

        rollout_ng = cluster.add_nodegroup_capacity(
            "RolloutNodes",
            instance_types=[ec2.InstanceType(rollout_instance_type)],
            ami_type=eks.NodegroupAmiType.AL2023_X86_64_NVIDIA,
            min_size=num_rollout_workers,
            max_size=num_rollout_workers,
            desired_size=num_rollout_workers,
            disk_size=200,
            labels={"node-role": "rollout"},
            subnets=ec2.SubnetSelection(subnets=[fsx_subnet]),
        )
        # endregion

        # ==============================================================
        # region 6. FSx CSI driver addon (IRSA)
        # ==============================================================
        oidc_issuer = cluster.open_id_connect_provider.open_id_connect_provider_issuer
        fsx_csi_conditions = CfnJson(
            self,
            "FsxCsiOidcCondition",
            value={
                f"{oidc_issuer}:sub": "system:serviceaccount:kube-system:fsx-csi-controller-sa",
                f"{oidc_issuer}:aud": "sts.amazonaws.com",
            },
        )
        fsx_csi_sa_role = iam.Role(
            self,
            "FsxCsiDriverRole",
            assumed_by=iam.FederatedPrincipal(
                cluster.open_id_connect_provider.open_id_connect_provider_arn,
                conditions={"StringEquals": fsx_csi_conditions},
                assume_role_action="sts:AssumeRoleWithWebIdentity",
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonFSxFullAccess"),
            ],
        )

        fsx_addon = eks.CfnAddon(
            self,
            "FSxCSIAddon",
            addon_name="aws-fsx-csi-driver",
            cluster_name=cluster.cluster_name,
            resolve_conflicts="OVERWRITE",
            service_account_role_arn=fsx_csi_sa_role.role_arn,
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
        # region 10. FSx StorageClass + PV + PVC (static provisioning)
        # ==============================================================
        fsx_sc = cluster.add_manifest(
            "FsxStorageClass",
            {
                "apiVersion": "storage.k8s.io/v1",
                "kind": "StorageClass",
                "metadata": {"name": "fsx-sc"},
                "provisioner": "fsx.csi.aws.com",
            },
        )

        fsx_pv = cluster.add_manifest(
            "FsxPersistentVolume",
            {
                "apiVersion": "v1",
                "kind": "PersistentVolume",
                "metadata": {"name": "fsx-training-pv"},
                "spec": {
                    "capacity": {"storage": f"{fsx_capacity_gib}Gi"},
                    "volumeMode": "Filesystem",
                    "accessModes": ["ReadWriteMany"],
                    "persistentVolumeReclaimPolicy": "Retain",
                    "storageClassName": "fsx-sc",
                    "csi": {
                        "driver": "fsx.csi.aws.com",
                        "volumeHandle": lustre_fs.file_system_id,
                        "volumeAttributes": {
                            "dnsname": lustre_fs.dns_name,
                            "mountname": lustre_fs.mount_name,
                        },
                    },
                },
            },
        )
        fsx_pv.node.add_dependency(fsx_sc)
        fsx_pv.node.add_dependency(fsx_addon)

        fsx_pvc = cluster.add_manifest(
            "FsxPersistentVolumeClaim",
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": "fsx-training-pvc",
                    "namespace": "training",
                },
                "spec": {
                    "accessModes": ["ReadWriteMany"],
                    "storageClassName": "fsx-sc",
                    "resources": {"requests": {"storage": f"{fsx_capacity_gib}Gi"}},
                },
            },
        )
        fsx_pvc.node.add_dependency(training_ns)
        fsx_pvc.node.add_dependency(fsx_pv)
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
        # region 11. RayCluster CR (heterogeneous head + workers)
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
                                                "value": "/mnt/fsx/third_party/RLinf:/mnt/fsx/third_party/Isaac-GR00T:/mnt/fsx/third_party/embodied-ai-platform:/mnt/fsx/third_party/IsaacLab/source:/mnt/fsx/workflows/rheo/scripts:/mnt/fsx/workflows/rheo/scripts/simulation/rl",
                                            },
                                        ],
                                        "volumeMounts": [
                                            {
                                                "name": "fsx-storage",
                                                "mountPath": "/mnt/fsx",
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
                                        "name": "fsx-storage",
                                        "persistentVolumeClaim": {
                                            "claimName": "fsx-training-pvc"
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
                                                    "cpu": "24",
                                                    "memory": "200Gi",
                                                    "nvidia.com/gpu": "1",
                                                },
                                                "limits": {
                                                    "cpu": "32",
                                                    "memory": "240Gi",
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
                                                    "value": "/mnt/fsx/third_party/RLinf:/mnt/fsx/third_party/Isaac-GR00T:/mnt/fsx/third_party/embodied-ai-platform:/mnt/fsx/third_party/IsaacLab/source:/mnt/fsx/workflows/rheo/scripts:/mnt/fsx/workflows/rheo/scripts/simulation/rl",
                                                },
                                            ],
                                            "volumeMounts": [
                                                {
                                                    "name": "fsx-storage",
                                                    "mountPath": "/mnt/fsx",
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
                                            "name": "fsx-storage",
                                            "persistentVolumeClaim": {
                                                "claimName": "fsx-training-pvc"
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
        raycluster.node.add_dependency(kuberay_chart)
        raycluster.node.add_dependency(nvidia_chart)
        raycluster.node.add_dependency(fsx_pvc)
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
        CfnOutput(
            self,
            "FsxFileSystemId",
            value=lustre_fs.file_system_id,
            description="FSx for Lustre filesystem ID",
        )
        CfnOutput(
            self,
            "FsxMountName",
            value=lustre_fs.mount_name,
            description="FSx mount name",
        )
        CfnOutput(
            self,
            "DataBucketName",
            value=data_bucket.bucket_name,
            description="S3 data bucket (DRA-linked to FSx)",
        )
        # endregion
