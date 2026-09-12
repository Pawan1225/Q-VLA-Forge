from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class HardwareInfo:
    """Describe the runtime hardware and Python environment."""

    python_version: str
    platform: str
    processor: str
    machine: str
    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None


def get_hardware_info() -> HardwareInfo:
    """Collect reproducible runtime hardware metadata."""
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0

    cuda_device_name: str | None = None

    if cuda_available and cuda_device_count > 0:
        cuda_device_name = torch.cuda.get_device_name(0)

    return HardwareInfo(
        python_version=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor(),
        machine=platform.machine(),
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
    )


def hardware_info_dict() -> dict[str, object]:
    """Return hardware information as a serializable dictionary."""
    return asdict(get_hardware_info())
