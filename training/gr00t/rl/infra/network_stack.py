"""Optional cold-start network bootstrap for the GR00T RL EKS path.

Owns a SHARED, PERSISTENT, EKS-ready VPC that the ephemeral GR00TRLEKSStack only
CONSUMES via `--context vpc_id=<VpcId>`. This is a convenience OPT-IN for a fresh
AWS account that has no pre-provisioned VPC — it is NOT the default. Real orgs
deploy into their own VPC by passing `vpc_id` (the primary/enterprise path); the
EKS stack's `from_lookup(vpc_id=...)` behavior is untouched.

Why a SEPARATE stack (not baked into GR00TRLEKSStack)?
  - The EKS stack is ephemeral (torn down / redeployed between runs). Creating the
    VPC inside it would destroy+recreate the VPC + NAT gateway on every
    `cdk destroy`, and would defeat the cross-AZ capacity-chasing knobs
    (fsx_subnet_id / rollout_subnet_ids / eval_learner_subnet_ids) that pin FSx +
    node groups to specific subnets in a stable VPC.
  - This stack is deployed ONCE and left up. The VPC additionally carries
    RemovalPolicy.RETAIN so tearing down anything else never deletes it (delete it
    by hand when you are fully done — see the docs).

Topology (EKS-ready):
  - >=2 AZs, each with a public + a private (PRIVATE_WITH_EGRESS) subnet.
  - >=1 NAT gateway for private-subnet egress to the EKS API / ECR / PyPI / GitHub /
    Omniverse CDN. NOTE: a NAT gateway is a RUNNING COST the operator opts into.
  - EKS subnet discovery tags: `kubernetes.io/role/internal-elb=1` on private
    subnets, `kubernetes.io/role/elb=1` on public subnets.

Deploy this FIRST (once), read the outputs, then deploy the EKS backend with the
VpcId (and optionally the per-AZ subnet IDs for the capacity knobs):
  cdk deploy GR00TRLNetworkStack --context compute_backend=network \\
    --context s3_data_bucket=<your-s3-bucket>
  # read VpcId / PrivateSubnetIds / PrivateSubnetId0 / PrivateSubnetAz0 ... outputs
  cdk deploy --context compute_backend=eks --context vpc_id=<VpcId> ...
"""

from aws_cdk import (
    Stack,
    CfnOutput,
    RemovalPolicy,
    Tags,
    aws_ec2 as ec2,
)
from constructs import Construct


class GR00TRLNetworkStack(Stack):
    """Persistent, EKS-ready VPC for a cold-start (no pre-existing VPC) deploy."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc_cidr: str = "10.73.0.0/16",
        max_azs: int = 2,
        nat_gateways: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # EKS-ready VPC: >=2 AZs, each with a public + private (egress) subnet,
        # >=1 NAT gateway. CIDR is a generic RFC1918 default overridable with
        # --context vpc_cidr (no advanced/IPv6/peering topology — out of scope).
        vpc = ec2.Vpc(
            self,
            "VPC",
            vpc_name="GR00T-RL-Network",
            ip_addresses=ec2.IpAddresses.cidr(vpc_cidr),
            max_azs=max(2, max_azs),
            nat_gateways=max(1, nat_gateways),
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=20,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=19,
                ),
            ],
        )
        self.vpc = vpc

        # Persistence: RETAIN the VPC so a teardown of any OTHER stack never deletes
        # it. This stack is deployed once and left up; the operator deletes the VPC
        # by hand when fully done.
        vpc.apply_removal_policy(RemovalPolicy.RETAIN)

        # EKS subnet-discovery tags: the AWS Load Balancer Controller / EKS use these
        # to pick subnets for internal (private) vs internet-facing (public) LBs.
        for subnet in vpc.private_subnets:
            Tags.of(subnet).add("kubernetes.io/role/internal-elb", "1")
        for subnet in vpc.public_subnets:
            Tags.of(subnet).add("kubernetes.io/role/elb", "1")

        # ==============================================================
        # CfnOutputs — feed the EKS deploy's vpc_id path + the capacity knobs
        # (fsx_subnet_id / rollout_subnet_ids / eval_learner_subnet_ids).
        # ==============================================================
        CfnOutput(
            self,
            "VpcId",
            value=vpc.vpc_id,
            description=(
                "Pass to the EKS deploy: --context vpc_id=<this>. Primary handoff "
                "from this bootstrap stack to GR00TRLEKSStack."
            ),
        )
        CfnOutput(
            self,
            "PrivateSubnetIds",
            value=",".join(s.subnet_id for s in vpc.private_subnets),
            description=(
                "Comma-joined private (PRIVATE_WITH_EGRESS) subnet IDs. Use these "
                "for fsx_subnet_id / rollout_subnet_ids / eval_learner_subnet_ids — "
                "pick the AZ that actually has g6e/H100 capacity."
            ),
        )
        CfnOutput(
            self,
            "PublicSubnetIds",
            value=",".join(s.subnet_id for s in vpc.public_subnets),
            description="Comma-joined public subnet IDs (internet-facing LBs).",
        )
        # Per-AZ private subnet IDs so the operator can chase capacity by AZ: pick
        # the PrivateSubnetIdN whose PrivateSubnetAzN holds the g6e/H100 capacity.
        for i, subnet in enumerate(vpc.private_subnets):
            CfnOutput(
                self,
                f"PrivateSubnetId{i}",
                value=subnet.subnet_id,
                description=(
                    f"Private subnet #{i} (AZ in PrivateSubnetAz{i}). Feed to "
                    "fsx_subnet_id / rollout_subnet_ids / eval_learner_subnet_ids."
                ),
            )
            CfnOutput(
                self,
                f"PrivateSubnetAz{i}",
                value=subnet.availability_zone,
                description=f"Availability zone of PrivateSubnetId{i}.",
            )
