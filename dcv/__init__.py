"""Standalone EC2 DCV CDK Module.

Provides reusable L3 construct for GPU-accelerated DCV workstations
with NVIDIA drivers, IsaacSim, and IsaacLab.

Usage (standalone):
    cd dcv
    cdk deploy

Usage (imported):
    from dcv import DcvWorkstation, DcvWorkstationProps
    DcvWorkstation(self, "DCV", props=DcvWorkstationProps(...))
"""
from .dcv_construct import DcvWorkstation, DcvWorkstationProps

__all__ = ["DcvWorkstation", "DcvWorkstationProps"]
