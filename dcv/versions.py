"""Version compatibility matrix and validation for DCV workstation.

This module provides a single source of truth for supported version combinations
of IsaacSim, IsaacLab, Python, PyTorch, and CUDA.
"""
from typing import Dict, List, Any


SUPPORTED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "5.1.0": {
        "python": "3.11",
        "pytorch": "2.7.0",
        "cuda_index": "cu128",
        "cuda_toolkit": "12.8",
        "compatible_isaaclab": ["v2.3.0", "v2.3.1", "v2.3.2"],
        "dcv": "2025.0-20103",
        "leisaac": "v0.3.0",
    },
    "4.5.0": {
        "python": "3.10",
        "pytorch": "2.5.1",
        "cuda_index": "cu118",
        "cuda_toolkit": "11.8",
        "compatible_isaaclab": ["v2.1.0", "v2.1.1"],
        "dcv": "2024.0-19030",
        "leisaac": "v0.2.0",
    },
}


def validate_version_config(isaac_sim_version: str, isaac_lab_version: str) -> Dict[str, Any]:
    """Validate version combination and return compatible dependencies.

    Args:
        isaac_sim_version: IsaacSim version (e.g., "5.1.0")
        isaac_lab_version: IsaacLab version (e.g., "v2.3.2")

    Returns:
        Dict with all compatible versions (python, pytorch, cuda_index, etc.)

    Raises:
        ValueError: If version combination is unsupported

    Examples:
        >>> config = validate_version_config("5.1.0", "v2.3.2")
        >>> config["python"]
        '3.11'
        >>> config["pytorch"]
        '2.7.0'
    """
    # Check if IsaacSim version is supported
    if isaac_sim_version not in SUPPORTED_CONFIGS:
        supported_versions = ", ".join(sorted(SUPPORTED_CONFIGS.keys()))
        raise ValueError(
            f"Unsupported IsaacSim version: {isaac_sim_version}. "
            f"Supported versions: {supported_versions}"
        )

    config = SUPPORTED_CONFIGS[isaac_sim_version]

    # Check if IsaacLab version is compatible with this IsaacSim version
    if isaac_lab_version not in config["compatible_isaaclab"]:
        compatible_versions = ", ".join(config["compatible_isaaclab"])
        raise ValueError(
            f"IsaacLab {isaac_lab_version} is not compatible with IsaacSim {isaac_sim_version}. "
            f"Compatible IsaacLab versions for IsaacSim {isaac_sim_version}: {compatible_versions}"
        )

    # Return the complete configuration
    return {
        "python": config["python"],
        "pytorch": config["pytorch"],
        "cuda_index": config["cuda_index"],
        "cuda_toolkit": config["cuda_toolkit"],
        "dcv": config["dcv"],
    }
