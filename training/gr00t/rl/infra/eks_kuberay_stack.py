"""EKS + KubeRay CDK stack for GR00T RL heterogeneous training.

Deploys an EKS cluster with:
  - Learner node group: 1x g6e.48xlarge (8x L40S, 768 GB RAM) for FSDP training
  - Rollout node group: Nx g6e.4xlarge (1x L40S each) for Isaac Sim rollout workers
  - KubeRay operator (Helm chart v1.1.0) managing RayCluster lifecycle
  - NVIDIA device plugin (Helm chart) exposing nvidia.com/gpu resources
  - FSx for Lustre (PERSISTENT_2) backed by S3 via Data Repository Association
  - RayCluster CR with heterogeneous head (8 GPU) + workers (1 GPU each)

This stack is a pure CONSUMER of the unified image + staged data bucket. The
image is built and the bucket is staged by the standalone GR00TRLArtifactsStack
(infra/artifacts_stack.py) — deploy that first, then drive it with
infra/prepare-artifacts.sh, then deploy this stack with the resolved image_uri.

Deploy:
  cdk deploy --context compute_backend=eks \\
    --context vpc_id=<your-vpc-id> \\
    --context s3_data_bucket=<your-s3-bucket> \\
    --context image_uri=<your-account>.dkr.ecr.<region>.amazonaws.com/<your-repo>:<tag>
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
    aws_logs as logs,
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
        mode: str = "train",
        eval_ckpt: str = None,
        model_path: str = None,
        val_check_interval: str = None,
        envs_per_worker: str = None,
        max_epochs: str = None,
        eval_total_envs: str = None,
        eval_actor_gbs: str = None,
        task_description: str = None,
        eval_inject_noise: str = None,
        noise_level: str = None,
        kuberay_version: str = "1.1.0",
        rollout_subnet_ids: str = None,
        eval_learner_subnet_ids: str = None,
        fsx_subnet_id: str = None,
        enable_cloudwatch_logs: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Validate mode at synth time — the entrypoint's fail-fast handles the
        # mode=eval / eval_ckpt coupling at pod startup (07-CONTEXT.md decision 3),
        # so we only guard against typos in the mode string here.
        if mode not in ("train", "eval"):
            raise ValueError(f"Unknown mode: {mode}. Expected 'train' or 'eval'.")
        is_eval = mode == "eval"

        # Effective container image. This stack only CONSUMES an image; it never
        # builds one (that is GR00TRLArtifactsStack's job). image_uri is REQUIRED and
        # must be an explicit, verified reference — ideally a digest.
        #
        # We deliberately do NOT fall back to a floating ':latest' tag. A mutable tag
        # can resolve to an unverified or half-built image, and can even resolve to a
        # DIFFERENT digest between synth and the pod's pull — quietly landing the wrong
        # image on a GPU cluster. Fail closed at synth instead. The normal path is
        # infra/prepare-artifacts.sh, which builds+verifies the image and passes the
        # pinned gr00t-rl-unified@sha256:<digest> on the deploy line; app.py also skips
        # instantiating this stack when no image_uri is in context, so the phase-1
        # artifacts-stack deploy (before any digest exists) is unaffected.
        if not image_uri:
            raise ValueError(
                "GR00TRLEKSStack requires an explicit image_uri (no ':latest' "
                "fallback). Prefer a digest: "
                "<acct>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-unified@sha256:<digest>. "
                "Deploy GR00TRLArtifactsStack, then run infra/prepare-artifacts.sh to "
                "build+verify the image and deploy with the resolved digest — or pass "
                "--context image_uri=... explicitly (e.g. from "
                "scripts/build_unified_and_push.sh)."
            )

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
        # FSx is single-AZ — pick one private subnet and pin node groups to it.
        # Default (fsx_subnet_id unset): first PRIVATE_WITH_EGRESS subnet, so the
        # synthesized template is byte-identical to the historical single-AZ default.
        # Set fsx_subnet_id (--context) to pin FSx + the CB/on-demand learner NG to a
        # SPECIFIC subnet (e.g. the AZ that actually has g6e/H100 capacity) instead of
        # whichever subnet lands at index 0.
        if fsx_subnet_id:
            fsx_subnet = ec2.Subnet.from_subnet_id(self, "FsxSubnet", fsx_subnet_id)
        else:
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
        # Training learner NG: capacity-block-backed H100 fleet.
        # Gated on capacity_reservation_id: no CR → no NG created. Once a CR
        # expires or is not supplied, the whole CB-bound NG (and its launch
        # template) drops out of synth so CFN can cleanly remove them without
        # trying to validate an expired CapacityReservationSpecification.
        # This unblocks on-demand eval-mode deploys after a training CR ends
        # without booking a fresh block. Training deploys still require a CR
        # anyway, so this is functionally equivalent for that path.
        learner_ng = None
        if capacity_reservation_id:
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
        elif not is_eval:
            # Train mode with NO capacity reservation: still create a schedulable
            # learner NG, otherwise the head pod (nodeSelector node-role=learner)
            # has nowhere to land and the RayCluster never forms. This is a plain
            # ON_DEMAND nodegroup (no capacity-block launch template) with the same
            # node-role=learner label + desired size 1. Eval mode is unaffected — it
            # routes the head pod to the eval-learner NG below.
            learner_ng = cluster.add_nodegroup_capacity(
                "LearnerNodes",
                instance_types=[ec2.InstanceType(learner_instance_type)],
                ami_type=eks.NodegroupAmiType.AL2023_X86_64_NVIDIA,
                min_size=1,
                max_size=1,
                desired_size=1,
                disk_size=200,
                labels={"node-role": "learner"},
                subnets=ec2.SubnetSelection(subnets=[fsx_subnet]),
            )

        # Eval-learner NG subnet selection (capacity-resilient eval knob, mirrors the
        # rollout_subnet_ids pattern). Default-off: when eval_learner_subnet_ids is unset,
        # this is exactly [fsx_subnet] so the synthesized template is byte-identical to
        # the single-AZ default. Set it (comma-separated private subnet IDs at deploy via
        # --context) to place the eval-learner — which runs the eval head pod — in another
        # AZ (e.g. us-east-2b) when the FSx AZ (2a) is g6e-capacity-dry. FSx stays in 2a and
        # is read cross-AZ (the static CSI PV has no topology/nodeAffinity, so a pod in
        # another AZ binds + mounts it over the VPC; cross-AZ data-transfer cost applies).
        # Pair with rollout_subnet_ids=<same subnet> so head + workers co-locate intra-AZ.
        if eval_learner_subnet_ids:
            eval_learner_subnets = [
                ec2.Subnet.from_subnet_id(self, f"EvalLearnerSubnet{i}", sid.strip())
                for i, sid in enumerate(eval_learner_subnet_ids.split(","))
            ]
        else:
            eval_learner_subnets = [fsx_subnet]

        # Eval learner NG: ON_DEMAND, no CR, no LT. Independent of the training
        # NG's capacity-block launch template — so `cdk deploy --context mode=eval`
        # can succeed even after the training NG's CR has expired. Both NGs coexist
        # in the cluster; scaling is orthogonal. Head pod nodeSelector routes by
        # is_eval (region 11).
        eval_learner_ng = cluster.add_nodegroup_capacity(
            "EvalLearnerNodes",
            instance_types=[ec2.InstanceType(rollout_instance_type)],
            ami_type=eks.NodegroupAmiType.AL2023_X86_64_NVIDIA,
            min_size=0,
            max_size=1,
            desired_size=1 if is_eval else 0,
            disk_size=200,
            labels={"node-role": "eval-learner"},
            subnets=ec2.SubnetSelection(subnets=eval_learner_subnets),
        )

        # Rollout NG subnet selection (Phase 12 capacity-resilient knob).
        # Default-off: when rollout_subnet_ids is unset, this list is exactly
        # [fsx_subnet] so the synthesized template is byte-identical to the
        # single-AZ default. When set (comma-separated private subnet IDs passed
        # at deploy via --context), it overrides ONLY the rollout NG subnet(s) —
        # enabling a cross-AZ rollout fleet (e.g. us-east-2b) while the learner
        # (2a Capacity-Block launch template) and FSx stay in us-east-2a.
        # Reversible by omitting the flag. IDs never committed here.
        if rollout_subnet_ids:
            rollout_subnets = [
                ec2.Subnet.from_subnet_id(self, f"RolloutSubnet{i}", sid.strip())
                for i, sid in enumerate(rollout_subnet_ids.split(","))
            ]
        else:
            rollout_subnets = [fsx_subnet]

        rollout_ng = cluster.add_nodegroup_capacity(
            "RolloutNodes",
            instance_types=[ec2.InstanceType(rollout_instance_type)],
            ami_type=eks.NodegroupAmiType.AL2023_X86_64_NVIDIA,
            min_size=num_rollout_workers,
            max_size=num_rollout_workers,
            desired_size=num_rollout_workers,
            disk_size=200,
            labels={"node-role": "rollout"},
            subnets=ec2.SubnetSelection(subnets=rollout_subnets),
        )
        # endregion

        # ==============================================================
        # region 5.5. CloudWatch Container Insights — durable pod/application logs
        # ==============================================================
        # On by default; disable with --context enable_cloudwatch_logs=false. Installs the
        # amazon-cloudwatch-observability EKS add-on (CloudWatch agent + Fluent Bit), which
        # ships ALL container stdout/stderr to CloudWatch Logs at
        # /aws/containerinsights/<cluster>/application (+ /host, /dataplane) and enables
        # Container Insights metrics. AWS's recommended EKS pod-logging path (control-plane
        # logging captures only api/audit/scheduler, NOT pod stdout, so it would miss the
        # training/eval logs).
        #
        # Fluent Bit creates those log groups at RUNTIME (not via CloudFormation), so they
        # PERSIST after `cdk destroy` — the training/eval logs (incl. eval/success_once)
        # survive teardown for post-hoc review, which raw `kubectl logs` do not. LogRetention
        # bounds them to 30 days WITHOUT CDK owning the group (create-if-absent + set-retention
        # via a custom resource; no re-deploy collision, not deleted on teardown). The agent
        # authenticates via the node instance role (CloudWatchAgentServerPolicy below) — the
        # simplest of the two documented auth options (vs IRSA).
        if enable_cloudwatch_logs:
            _cw_node_groups = [eval_learner_ng, rollout_ng]
            if not is_eval:
                _cw_node_groups.append(learner_ng)  # train mode always creates it
            _cw_policy = iam.ManagedPolicy.from_aws_managed_policy_name(
                "CloudWatchAgentServerPolicy"
            )
            for _ng in _cw_node_groups:
                _ng.role.add_managed_policy(_cw_policy)

            cw_addon = eks.CfnAddon(
                self,
                "CloudWatchObservabilityAddon",
                addon_name="amazon-cloudwatch-observability",
                cluster_name=cluster.cluster_name,
                resolve_conflicts="OVERWRITE",
            )
            # The agent/Fluent Bit DaemonSet needs nodes present to schedule onto.
            cw_addon.node.add_dependency(rollout_ng)
            cw_addon.node.add_dependency(eval_learner_ng)

            # Bound retention on the (runtime-created, non-CFN) Container Insights log
            # groups. LogRetention creates the group if absent + sets retention via a
            # custom resource; it does NOT delete the group on teardown, so logs persist.
            for _suffix in ("application", "host", "dataplane"):
                logs.LogRetention(
                    self,
                    f"CILogRetention{_suffix.capitalize()}",
                    log_group_name=f"/aws/containerinsights/gr00t-rl-eks/{_suffix}",
                    retention=logs.RetentionDays.ONE_MONTH,
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
        if learner_ng is not None:
            nvidia_chart.node.add_dependency(learner_ng)
        nvidia_chart.node.add_dependency(eval_learner_ng)
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
            version=kuberay_version,
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
        # W5-revision: the mode branch is consolidated to a single dict-selection.
        # Every downstream field that varies between train and eval reads from
        # head_pod_shape["key"] — no scattered per-field ternaries inside the
        # manifest literal. A future third mode (e.g., train-with-eval) is a
        # one-branch addition here, not six sprinkled ternaries.
        train_head_pod = {
            "num_gpus": "8",
            "node_role": "learner",
            "requests": {
                "cpu": "90",
                "memory": "600Gi",
                "nvidia.com/gpu": "8",
            },
            # No memory limit for training — matches HEAD's head-pod resources.
            "limits": {
                "cpu": "192",
                "nvidia.com/gpu": "8",
            },
            "num_rollout_workers_env_value": str(num_rollout_workers),
            # Additive + reversible (mirrors the eval_head_pod train
            # pattern): when model_path is set, train FROM that checkpoint (e.g.
            # the SFT base) instead of the entrypoint's RL-checkpoint default;
            # when val_check_interval is set, stream in-flight aggregate eval to
            # TensorBoard every N global_steps; when envs_per_worker is set, it
            # overrides the entrypoint default (32) that drives
            # env.train.total_num_envs = num_rollout_workers * ENVS_PER_WORKER —
            # lowering it shrinks the co-located rollout L40S GPU footprint (the
            # Eagle/Qwen3 lm_head logits AND the Isaac Sim allocation both scale
            # with envs/GPU; 32/GPU OOM'd a 46 GiB L40S — see Phase 12 RUN-LOG).
            # When max_epochs is set, it bounds the run to N global_steps
            # (overrides the entrypoint default of 1000 -> runner.max_epochs=N),
            # giving a cost-bounded, deterministic stop for short/plumbing runs.
            # All unset => extra_env is empty and the train head env is
            # byte-identical to historical behavior.
            "extra_env": [
                *([{"name": "MODEL_PATH", "value": model_path}] if model_path else []),
                *([{"name": "VAL_CHECK_INTERVAL", "value": str(val_check_interval)}] if val_check_interval else []),
                *([{"name": "ENVS_PER_WORKER", "value": str(envs_per_worker)}] if envs_per_worker else []),
                *([{"name": "MAX_EPOCHS", "value": str(max_epochs)}] if max_epochs else []),
            ],
            "worker_replicas": num_rollout_workers,
        }
        eval_head_pod = {
            "num_gpus": "1",
            "node_role": "eval-learner",
            "requests": {
                "cpu": "24",
                "memory": "100Gi",
                "nvidia.com/gpu": "1",
            },
            "limits": {
                "cpu": "32",
                "memory": "200Gi",
                "nvidia.com/gpu": "1",
            },
            # Eval-mode worker count follows the top-level `num_rollout_workers`
            # context param. Phase 7's default was 1 (2-pod head+worker fleet);
            # Phase 07.1 Step A bumps this to 3 (4-pod fleet at 16 envs/GPU for
            # NVIDIA's `total_num_envs=64` benchmark topology). The entrypoint's
            # Ray-wait loop keys off NUM_ROLLOUT_WORKERS to expect the right count.
            "num_rollout_workers_env_value": str(num_rollout_workers),
            "extra_env": [
                {"name": "MODE", "value": "eval"},
                {"name": "EVAL_CKPT", "value": eval_ckpt or ""},
                *([{"name": "MODEL_PATH", "value": model_path}] if model_path else []),
                *([{"name": "EVAL_TOTAL_ENVS", "value": str(eval_total_envs)}] if eval_total_envs else []),
                *([{"name": "EVAL_ACTOR_GBS", "value": str(eval_actor_gbs)}] if eval_actor_gbs else []),
                *([{"name": "TASK_DESCRIPTION", "value": task_description}] if task_description else []),
                *([{"name": "EVAL_INJECT_NOISE", "value": str(eval_inject_noise)}] if eval_inject_noise else []),
                *([{"name": "NOISE_LEVEL", "value": str(noise_level)}] if noise_level else []),
            ],
            "worker_replicas": num_rollout_workers,
        }
        head_pod_shape = eval_head_pod if is_eval else train_head_pod

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
                    # Match the Ray shipped in the unified image (2.47.0). A stale
                    # rayVersion here trips KubeRay's version-compat checks / GCS
                    # handshake against the real ray binary in the container.
                    "rayVersion": "2.47.0",
                    "headGroupSpec": {
                        "rayStartParams": {
                            "dashboard-host": "0.0.0.0",
                            "num-gpus": head_pod_shape["num_gpus"],
                        },
                        "template": {
                            "metadata": {"labels": {"ray-role": "head"}},
                            "spec": {
                                "nodeSelector": {
                                    "node-role": head_pod_shape["node_role"]
                                },
                                "containers": [
                                    {
                                        "name": "ray-head",
                                        "image": image_uri,
                                        "command": ["/bin/bash", "-lc", "--"],
                                        "args": [
                                            "ulimit -n 65536; RAY_START_CMD=$(echo $KUBERAY_GEN_RAY_START_CMD | sed 's/--block//g'); eval $RAY_START_CMD; /opt/entrypoint-eks.sh"
                                        ],
                                        "resources": {
                                            "requests": head_pod_shape["requests"],
                                            "limits": head_pod_shape["limits"],
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
                                                "value": head_pod_shape[
                                                    "num_rollout_workers_env_value"
                                                ],
                                            },
                                            {
                                                "name": "PYTHONPATH",
                                                "value": "/mnt/fsx/third_party/RLinf:/mnt/fsx/third_party/Isaac-GR00T:/mnt/fsx/third_party/embodied-ai-platform:/mnt/fsx/third_party/IsaacLab/source:/mnt/fsx/workflows/rheo/scripts:/mnt/fsx/workflows/rheo/scripts/simulation/rl",
                                            },
                                            *head_pod_shape["extra_env"],
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
                            "replicas": head_pod_shape["worker_replicas"],
                            "minReplicas": head_pod_shape["worker_replicas"],
                            "maxReplicas": head_pod_shape["worker_replicas"],
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
                                                # Run KubeRay's generated ray-start command WITH its --block
                                                # (do NOT strip --block + sleep infinity like the head does).
                                                # --block keeps `ray start` in the foreground so it EXITS when
                                                # the head restarts and re-keys the Ray GCS; KubeRay then
                                                # recreates this worker and it reconnects to the CURRENT head.
                                                # Stripping --block + `sleep infinity` defeats that self-healing:
                                                # the worker's Ray gets orphaned on the old GCS id while the
                                                # container sleeps on, requiring a manual `kubectl delete pod`
                                                # (the head then loops on its "waiting for N nodes" timeout).
                                                # The worker has no post-ray-start work, so just block on it.
                                                "ulimit -n 65536; eval $KUBERAY_GEN_RAY_START_CMD"
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
                                                    # Rollout L40S (46 GiB) hosts co-located Isaac Sim
                                                    # (~22 GiB) + GR00T Eagle/Qwen3 policy inference on
                                                    # one GPU. The first predict_action_batch OOM'd at
                                                    # the Eagle-VLM lm_head (Phase 12 Wave 2). The
                                                    # expandable-segments allocator reduces fragmentation
                                                    # so the transient logits allocation can fit.
                                                    "name": "PYTORCH_CUDA_ALLOC_CONF",
                                                    "value": "expandable_segments:True",
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
        # The FSx Data Repository Association must exist before the RayCluster pods
        # start, or the first pod to read /mnt/fsx mounts an un-linked filesystem and
        # sees none of the S3-staged code/model (the DRA is what lazily imports it).
        raycluster.node.add_dependency(fsx_dra)
        # endregion

        # NOTE: the image build + S3 staging pipeline (ECR repo gr00t-rl-unified +
        # the GR00T-RL-Pipeline CodeBuild project) formerly lived here. It has been
        # extracted into the standalone GR00TRLArtifactsStack (infra/artifacts_stack.py)
        # so it OWNS those resources persistently — this stack only consumes the
        # resulting image_uri + the DRA-linked bucket. See that stack's docstring.

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
        if learner_ng is not None:
            CfnOutput(
                self,
                "LearnerNodeGroupArn",
                value=learner_ng.nodegroup_arn,
                description=f"Learner node group ARN ({learner_instance_type})",
            )
        CfnOutput(
            self,
            "EvalLearnerNodeGroupArn",
            value=eval_learner_ng.nodegroup_arn,
            description=f"Eval-mode learner node group ARN ({rollout_instance_type}, ON_DEMAND, no CR)",
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
        CfnOutput(
            self,
            "Mode",
            value=mode,
            description="Deployment mode (train or eval)",
        )
        # endregion
