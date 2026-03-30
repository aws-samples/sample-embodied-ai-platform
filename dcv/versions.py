"""Version compatibility matrix and validation for DCV workstation.

This module provides a single source of truth for supported version combinations
of IsaacSim and IsaacLab, mapping each to the corresponding NVIDIA container image tag.

Currently, only isaac-lab:2.3.0 (IsaacSim 5.1) is tested. The eval container only needs
IsaacSim for simulation — the policy server runs separately via ZMQ — so the sim
container version is independent of the GR00T training version (N1.5 vs N1.6).
"""
from typing import Dict, Any


# Future versions can be added here.
SUPPORTED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "5.1.0": {
        "container_image": "nvcr.io/nvidia/isaac-lab:2.3.0",
        "dcv": "2025.0-20103",
        "leisaac": "v0.3.0",
    },
}


def validate_version_config(isaac_sim_version: str, isaac_lab_version: str) -> Dict[str, Any]:
    """Validate version combination and return the compatible container image and DCV version.

    The isaac_lab_version parameter is accepted for backwards compatibility but is not
    validated — the container image is determined solely by isaac_lab_version.

    Args:
        isaac_sim_version: IsaacSim version (informational only, depends on IsaacLab version)
        isaac_lab_version: IsaacLab version (e.g. "v2.3.0")

    Returns:
        Dict with keys: container_image, dcv, leisaac

    Raises:
        ValueError: If isaac_lab_version is unsupported

    Examples:
        >>> config = validate_version_config("5.1.0", "v2.3.0")
        >>> config["container_image"]
        'nvcr.io/nvidia/isaac-lab:2.3.0'
        >>> config["dcv"]
        '2025.0-20103'
    """
    if isaac_lab_version not in SUPPORTED_CONFIGS:
        supported_versions = ", ".join(sorted(SUPPORTED_CONFIGS.keys()))
        raise ValueError(
            f"Unsupported IsaacLab version: {isaac_lab_version}. "
            f"Supported versions: {supported_versions}"
        )

    config = SUPPORTED_CONFIGS[isaac_lab_version]
    return {
        "container_image": config["container_image"],
        "dcv": config["dcv"],
        "leisaac": config["leisaac"],
    }
