from q_vla_forge.utils.hardware import (
    HardwareInfo,
    get_hardware_info,
    hardware_info_dict,
)


def test_get_hardware_info() -> None:
    info = get_hardware_info()

    assert isinstance(info, HardwareInfo)
    assert info.python_version
    assert info.platform
    assert info.machine
    assert info.torch_version
    assert isinstance(info.cuda_available, bool)
    assert info.cuda_device_count >= 0


def test_hardware_info_dict() -> None:
    info = hardware_info_dict()

    assert isinstance(info, dict)
    assert "python_version" in info
    assert "platform" in info
    assert "processor" in info
    assert "machine" in info
    assert "torch_version" in info
    assert "cuda_available" in info
    assert "cuda_device_count" in info
    assert "cuda_device_name" in info
