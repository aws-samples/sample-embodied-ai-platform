# Standalone DCV Workstation Module

A reusable AWS CDK module for deploying GPU-accelerated EC2 instances with Amazon DCV (remote desktop), NVIDIA drivers, and configurable IsaacSim/IsaacLab installations.

## Overview

This module can be used in two ways:

1. **Standalone** — Deploy independently from `dcv/` with its own VPC
2. **Integrated** — Import the `DcvWorkstation` construct into another CDK app (e.g., gr00t training pipeline)

## Quick Start (Standalone)

```bash
cd dcv
pip install -r requirements.txt

# Bootstrap CDK (one-time per account/region)
AWS_DEFAULT_REGION=us-west-2 cdk bootstrap --profile <your-profile>

# Deploy with defaults (IsaacSim 5.1.0, IsaacLab v2.3.2, g6.4xlarge)
AWS_DEFAULT_REGION=us-west-2 cdk deploy --profile <your-profile>
```

After deployment (~5 min for infrastructure + ~15 min for bootstrap), the stack outputs will show:

| Output | Description |
|--------|-------------|
| `InstancePublicIP` | Elastic IP address |
| `DCVWebURL` | `https://<ip>:8443` — open in browser |
| `DCVCredentials` | `ubuntu` / `dcv<account_id>` |
| `InstanceId` | For SSM Session Manager access |

## Configuration

All parameters are optional and can be set via CDK context:

```bash
AWS_DEFAULT_REGION=us-west-2 cdk deploy --profile <your-profile> \
  --context instance_type=g6.2xlarge \
  --context isaac_sim_version=4.5.0 \
  --context isaac_lab_version=v2.1.1 \
  --context vpc_id=vpc-12345 \
  --context efs_id=fs-12345 \
  --context efs_sg_id=sg-12345 \
  --context leisaac_enabled=true
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `instance_type` | `g6.4xlarge` | EC2 instance type (must be GPU: g4dn, g5, g6, p-series) |
| `isaac_sim_version` | `5.1.0` | NVIDIA IsaacSim version |
| `isaac_lab_version` | `v2.3.2` | NVIDIA IsaacLab version |
| `python_version` | Auto-derived | Python version (derived from IsaacSim compatibility matrix) |
| `vpc_id` | Auto-create | Existing VPC ID (if omitted, creates a new VPC) |
| `efs_id` | None | Existing EFS file system ID for persistent storage |
| `efs_sg_id` | None | EFS security group ID (required if `efs_id` is set) |
| `leisaac_enabled` | `false` | Enable leisaac (LightwheelAI sim toolkit) installation |

### Supported Version Combinations

The module validates version compatibility at CDK synthesis time. Unsupported combinations will fail with a clear error message.

| IsaacSim | IsaacLab | Python | PyTorch | CUDA |
|----------|----------|--------|---------|------|
| 5.1.0 | v2.3.0 — v2.3.2 | 3.11 | 2.10.0 | 12.8 |
| 5.0.0 | v2.2.0 — v2.2.2 | 3.10 | 2.7.0 | 12.8 |
| 4.5.0 | v2.1.0 — v2.1.1 | 3.10 | 2.5.1 | 11.8 |

## Integrated Usage (Importing into Another CDK App)

To use the DCV workstation in another CDK app (e.g., alongside a Batch training stack):

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dcv import DcvWorkstation, DcvWorkstationProps

class MyTrainingDcvStack(Stack):
    def __init__(self, scope, id, batch_stack, **kwargs):
        super().__init__(scope, id, **kwargs)

        props = DcvWorkstationProps(
            vpc=batch_stack.vpc,              # Share VPC
            efs_id=batch_stack.efs_id,        # Share EFS
            efs_sg_id=batch_stack.efs_sg_id,  # Share security group
            instance_type="g6.4xlarge",
            isaac_sim_version="4.5.0",
            isaac_lab_version="v2.1.1",
            leisaac_enabled=True,
        )

        self.dcv = DcvWorkstation(self, "DCV", props)
```

### Exposed Properties

The `DcvWorkstation` construct exposes these properties for further customization:

| Property | Type | Description |
|----------|------|-------------|
| `vpc` | `ec2.IVpc` | Resolved VPC |
| `instance` | `ec2.Instance` | EC2 instance |
| `security_group` | `ec2.SecurityGroup` | Security group (add custom rules) |
| `instance_role` | `iam.Role` | IAM role (attach custom policies) |
| `elastic_ip` | `str` | Elastic IP address |

## What Gets Installed

The bootstrap script (`configure_dcv_instance.sh`) installs the following on Ubuntu 22.04:

1. NVIDIA GPU drivers (via `ubuntu-drivers autoinstall`)
2. Ubuntu Desktop + GDM (Wayland disabled for DCV)
3. Amazon DCV server with auto-session creation
4. AWS CLI v2
5. Amazon EFS utilities (for TLS-encrypted mounting)
6. Docker + NVIDIA Container Toolkit
7. Miniforge (conda) with Isaac environment
8. PyTorch (version matched to IsaacSim)
9. NVIDIA IsaacSim (configurable version)
10. NVIDIA IsaacLab (configurable version)
11. Leisaac (optional, flag-controlled)
12. Firefox browser

### Bootstrap Features

- **Idempotent**: State markers in `/var/lib/dcv-bootstrap/` — safe to re-run after failures
- **Retry logic**: Network operations (apt, downloads) retry up to 5 times
- **Structured logging**: Summary at `/var/log/dcv-bootstrap.summary`, detailed log at `/var/log/dcv-bootstrap.log`
- **Graceful degradation**: Critical steps (GPU driver, DCV) fail deployment; non-critical steps (leisaac, Firefox) log warnings and continue

## Monitoring Bootstrap Progress

After deployment, check bootstrap status via SSM:

```bash
aws ssm start-session --target <instance-id> --profile <your-profile> --region us-west-2
# Then:
sudo cat /var/log/dcv-bootstrap.summary
```

Each step shows `STEP_OK`, `STEP_WARN`, or `STEP_FAIL`. See the script header for detailed troubleshooting commands.

## Cleanup

```bash
AWS_DEFAULT_REGION=us-west-2 cdk destroy --profile <your-profile> --force
```

## Module Structure

```
dcv/
├── __init__.py              # Exports DcvWorkstation, DcvWorkstationProps
├── dcv_construct.py         # L3 CDK Construct (core infrastructure logic)
├── dcv_stack.py             # Thin Stack wrapper for standalone deployment
├── versions.py              # Version compatibility matrix + validation
├── configure_dcv_instance.sh # Parameterized bootstrap script (416 lines)
├── app.py                   # Standalone CDK app entry point
├── cdk.json                 # CDK app configuration
├── requirements.txt         # Pinned Python dependencies
└── README.md                # This file
```

## Network Access

The security group allows inbound traffic on:

| Port | Protocol | Purpose |
|------|----------|---------|
| 8443 | TCP | Amazon DCV (remote desktop) |
| 6006 | TCP | TensorBoard |

For production, restrict source IPs by adding custom security group rules via the exposed `security_group` property.
