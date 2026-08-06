"""Runtime safety tests for the ROS-to-ESP32 bridge."""

from types import SimpleNamespace

import pytest


rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import Twist  # noqa: E402

from martha import cmd_vel_serial_bridge as bridge_module  # noqa: E402


class FakeSerial:
    """Small in-memory serial port used by the bridge safety test."""

    def __init__(self, *_args, **_kwargs):
        self.rx = bytearray()
        self.writes = []
        self.fail_writes = False
        self.is_open = True

    @property
    def in_waiting(self):
        """Return the number of queued receive bytes."""
        return len(self.rx)

    def read(self, size):
        """Consume up to ``size`` receive bytes."""
        result = bytes(self.rx[:size])
        del self.rx[:size]
        return result

    def write(self, payload):
        """Record a write or emulate a serial write failure."""
        if self.fail_writes:
            raise bridge_module.serial.SerialException("fake write failure")
        self.writes.append(bytes(payload))
        return len(payload)

    def flush(self):
        """Match the pyserial flush API."""

    def close(self):
        """Mark this fake port closed."""
        self.is_open = False


def _twist(vx=0.0, vy=0.0, wz=0.0):
    message = Twist()
    message.linear.x = float(vx)
    message.linear.y = float(vy)
    message.angular.z = float(wz)
    return message


def _feed_line(node, line):
    node.serial_port.rx.extend((line + "\n").encode("ascii"))
    node.drain_serial()


def test_bridge_bounds_commands_and_requires_safe_rearm(monkeypatch):
    monkeypatch.setattr(bridge_module.serial, "Serial", FakeSerial)
    rclpy.init()
    node = None
    try:
        node = bridge_module.CmdVelSerialBridge()

        node.cmd_vel_callback(_twist(vx=0.2))
        assert node.serial_port.writes[-1] == (
            b"cmd_vel,0.000000,0.000000,0.000000\n"
        )

        _feed_line(node, "MOTOR_READY")
        assert node.motor_fault is False
        node.cmd_vel_callback(_twist(9.0, -9.0, 9.0))
        assert node.serial_port.writes[-1] == (
            b"cmd_vel,0.350000,-0.350000,0.800000\n"
        )

        node.cmd_vel_callback(_twist(float("nan"), 0.0, 0.0))
        assert node.motor_fault is True
        assert node.fault_reason == "invalid_cmd_vel"
        assert node.serial_port.writes[-1] == (
            b"cmd_vel,0.000000,0.000000,0.000000\n"
        )

        response = SimpleNamespace(success=False, message="")
        node.reset_motor_protection(None, response)
        assert response.success is True
        _feed_line(node, "motor_protection_reset")
        assert node.reset_ack_received is True
        node.cmd_vel_callback(_twist(vx=0.1))
        assert node.motor_fault is True

        node.last_valid_serial_time -= node.serial_timeout + 1.0
        node.check_serial_watchdog()
        assert node.motor_fault is True
        assert node.fault_reason == "serial_timeout"
        assert node.reset_ack_received is False
        node.cmd_vel_callback(_twist())
        assert node.motor_fault is True

        _feed_line(node, "motor_protection_reset")
        node.cmd_vel_callback(_twist())
        assert node.motor_fault is False
        assert node.reset_ack_received is False

        node.serial_port.fail_writes = True
        response = SimpleNamespace(success=True, message="")
        node.reset_odometry(None, response)
        assert response.success is False
        assert node.fault_reason == "serial_write_error"
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
