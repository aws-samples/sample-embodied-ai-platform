"""Stack wrapper for standalone DCV deployment."""
from aws_cdk import Stack, CfnOutput
from constructs import Construct
from dcv_construct import DcvWorkstation, DcvWorkstationProps


class DcvStack(Stack):
    """Thin stack wrapper for standalone DCV deployment.

    This stack enables standalone deployment of the DCV workstation module
    via 'cdk deploy' from the dcv/ directory. It wraps the DcvWorkstation
    construct and provides CloudFormation outputs.

    The stack reads optional context values to configure the deployment:
    - vpc_id: Use existing VPC (if not provided, auto-creates VPC)
    - instance_type: EC2 instance type (if not provided, defaults to g6.4xlarge)
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Read optional context values
        vpc_id = self.node.try_get_context("vpc_id")
        instance_type = self.node.try_get_context("instance_type") or "g6.4xlarge"
        # Read optional version context values
        isaac_sim_version = self.node.try_get_context("isaac_sim_version") or "5.1.0"
        isaac_lab_version = self.node.try_get_context("isaac_lab_version") or "v2.3.2"
        python_version = self.node.try_get_context("python_version")  # None = auto-derive
        # Read optional EFS context values
        efs_id = self.node.try_get_context("efs_id")
        efs_sg_id = self.node.try_get_context("efs_sg_id")
        # Read optional leisaac context value
        leisaac_enabled = self.node.try_get_context("leisaac_enabled") or False

        # Create props with optional vpc_id, instance_type, and version parameters
        props = DcvWorkstationProps(
            vpc_id=vpc_id,
            instance_type=instance_type,
            isaac_sim_version=isaac_sim_version,
            isaac_lab_version=isaac_lab_version,
            python_version=python_version,
            efs_id=efs_id,
            efs_sg_id=efs_sg_id,
            leisaac_enabled=leisaac_enabled,
        )

        # Instantiate the DcvWorkstation construct
        self.dcv_workstation = DcvWorkstation(self, "DcvWorkstation", props)

        # Output the VPC ID for reference
        CfnOutput(
            self,
            "VpcId",
            value=self.dcv_workstation.vpc.vpc_id,
            description="VPC ID where DCV workstation is deployed",
        )
