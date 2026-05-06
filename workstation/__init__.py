"""Standalone EC2 DCV CDK Module.

Provides reusable L3 construct for GPU-accelerated DCV workstations
with NVIDIA drivers, IsaacSim, and IsaacLab.

Usage (standalone):
    cd workstation
    cdk deploy

Usage (imported):
    from workstation import DcvWorkstation, DcvWorkstationProps
    DcvWorkstation(self, "DCV", props=DcvWorkstationProps(...))
"""
from .dcv_construct import DcvWorkstation, DcvWorkstationProps

__all__ = ["DcvWorkstation", "DcvWorkstationProps"]
