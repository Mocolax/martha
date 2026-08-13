"""Runtime-light tests for hardware launch validation helpers."""

import importlib.util
from pathlib import Path

import pytest


try:
    from launch import LaunchContext, LaunchDescription  # noqa: F401
except ImportError:
    pytest.skip("ROS 2 launch is unavailable", allow_module_level=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = PROJECT_ROOT / "launch" / "hardware.launch.py"
SPEC = importlib.util.spec_from_file_location(
    "martha_hardware_launch",
    LAUNCH_PATH,
)
assert SPEC is not None and SPEC.loader is not None
HARDWARE_LAUNCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARDWARE_LAUNCH)


def test_serial_port_conflict_detects_equal_paths():
    assert HARDWARE_LAUNCH.serial_ports_conflict(
        "/dev/ttyUSB0",
        "/dev/ttyUSB0",
    )
    assert not HARDWARE_LAUNCH.serial_ports_conflict(
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
    )


def test_serial_port_conflict_resolves_symlinks(tmp_path):
    device = tmp_path / "ttyUSB0"
    device.touch()
    alias = tmp_path / "rplidar"
    alias.symlink_to(device)

    assert HARDWARE_LAUNCH.serial_ports_conflict(str(device), str(alias))


def test_port_validation_only_applies_when_lidar_starts():
    context = LaunchContext()
    context.launch_configurations.update({
        "port": "/dev/ttyUSB0",
        "lidar_port": "/dev/ttyUSB0",
        "start_lidar": "false",
    })
    assert HARDWARE_LAUNCH.validate_serial_ports(context) == []

    context.launch_configurations["start_lidar"] = "true"
    with pytest.raises(RuntimeError, match="mismo dispositivo serial"):
        HARDWARE_LAUNCH.validate_serial_ports(context)
