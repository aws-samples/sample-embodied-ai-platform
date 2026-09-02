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
  - This stack is deployed ONCE and left up. It is INDEPENDENT of the ephemeral
    EKS stack, so an EKS teardown never touches it — persistence across EKS
    redeploys is inherent, no per-resource RETAIN needed. It additionally carries
    stack-level termination_protection so an accidental `cdk destroy` is BLOCKED;
    an intentional teardown (disable protection, then destroy) removes the WHOLE
    topology cleanly — see the teardown note below.

Topology (EKS-ready):
  - >=2 AZs, each with a public + a private (PRIVATE_WITH_EGRESS) subnet.
  - AZ selection: default auto-picks the first 2 AZs (max_azs=2). Pass
    `--context network_azs=us-west-2a,us-west-2c` to pin SPECIFIC AZs — the
    auto-pick can land on AZs that EKS forbids or where FSx-Lustre / g6e aren't
    available, so choose AZs with EKS eligibility + FSx support + g6e capacity
    (probe first).
  - NAT gateways for private-subnet egress to the EKS API / ECR / PyPI / GitHub /
    Omniverse CDN. NOTE: a NAT gateway is a RUNNING COST the operator opts into.
    Default is 1 NAT gateway — both AZs egress through it (cross-AZ data-processing
    charges + a single-AZ egress dependency). Set `--context network_nat_gateways=2`
    for one-NAT-per-AZ resilience.
  - EKS subnet discovery tags: `kubernetes.io/role/internal-elb=1` on private
    subnets, `kubernetes.io/role/elb=1` on public subnets (scoped to the subnet
    resources only, not the route tables / NAT / EIP under them).

Deploy this FIRST (once), read the outputs, then deploy the EKS backend with the
VpcId (and optionally the per-AZ subnet IDs for the capacity knobs):
  cdk deploy GR00TRLNetworkStack --context compute_backend=network
  # optional: --context vpc_cidr=10.73.0.0/16
  #           --context network_azs=us-west-2a,us-west-2c
  #           --context network_nat_gateways=2   (one NAT per AZ; default 1)
  # read VpcId / PrivateSubnetIds / PrivateSubnetId0 / PrivateSubnetAz0 ... outputs
  cdk deploy --context compute_backend=eks --context vpc_id=<VpcId> ...

Teardown: this stack sets termination_protection=True, so an accidental
`cdk destroy` is blocked. To intentionally remove the WHOLE topology (VPC + NAT +
EIP + IGW + subnets + route tables) cleanly, first disable protection, then destroy:
  aws cloudformation update-termination-protection \\
    --stack-name GR00TRLNetworkStack --no-enable-termination-protection
  cdk destroy GR00TRLNetworkStack
(or toggle termination protection off in the CloudFormation console, then destroy).
"""

from aws_cdk import (
    Stack,
    CfnOutput,
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
        availability_zones=None,
        nat_gateways: int = 1,
        **kwargs,
    ) -> None:
        # Persistence is inherent (this stack is independent of the ephemeral EKS
        # stack, so an EKS teardown never touches it). Instead of a per-resource
        # RETAIN — which would leave a useless retained VPC shell if the stack were
        # destroyed while NAT/EIP/IGW/subnets/routes got deleted — protect the WHOLE
        # stack from an accidental `cdk destroy`. An intentional teardown disables
        # protection first (see the module docstring), then destroys the whole
        # topology cleanly.
        super().__init__(
            scope, construct_id, termination_protection=True, **kwargs
        )

        # EKS-ready VPC: >=2 AZs, each with a public + private (egress) subnet,
        # >=1 NAT gateway. CIDR is a generic RFC1918 default overridable with
        # --context vpc_cidr (no advanced/IPv6/peering topology — out of scope).
        vpc_kwargs = dict(
            vpc_name="GR00T-RL-Network",
            ip_addresses=ec2.IpAddresses.cidr(vpc_cidr),
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
        # Optional explicit AZ pinning: --context network_azs=us-west-2a,us-west-2c.
        # When set, hand the VPC an explicit AZ list (EKS-eligible + FSx + g6e);
        # otherwise fall back to auto-selecting the first max(2, max_azs) AZs.
        if availability_zones:
            vpc_kwargs["availability_zones"] = list(availability_zones)
        else:
            vpc_kwargs["max_azs"] = max(2, max_azs)
        vpc = ec2.Vpc(self, "VPC", **vpc_kwargs)
        self.vpc = vpc

        # EKS subnet-discovery tags: the AWS Load Balancer Controller / EKS use these
        # to pick subnets for internal (private) vs internet-facing (public) LBs.
        # Scope each tag to the SUBNET resource only so it does not propagate to the
        # route tables / NAT / EIP that CDK nests under the subnet construct.
        for subnet in vpc.private_subnets:
            Tags.of(subnet).add(
                "kubernetes.io/role/internal-elb",
                "1",
                include_resource_types=["AWS::EC2::Subnet"],
            )
        for subnet in vpc.public_subnets:
            Tags.of(subnet).add(
                "kubernetes.io/role/elb",
                "1",
                include_resource_types=["AWS::EC2::Subnet"],
            )

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
