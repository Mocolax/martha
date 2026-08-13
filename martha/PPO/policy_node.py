"""
Safe ROS 2 inference node for Martha PPO checkpoints.

The node deliberately reuses the exact observation builder and action helpers
used by :mod:`martha.PPO.martha_env`.  It never owns Gazebo or hardware reset;
it only consumes the common ROS contract and publishes bounded ``Twist``
commands.  Missing/stale sensors, close obstacles, invalid network output or a
hardware motor fault always produce an immediate zero command.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math
import time
from typing import Any

import numpy as np

from .actions import (
    ActionLimits,
    limit_action_rate,
    sanitize_action,
    scale_action,
)
from .checkpoint import choose_device, load_policy
from .martha_env import (
    ACTION_SIZE,
    LASER_SECTORS,
    OBSERVATION_SIZE,
    POLICY_CONTRACT_VERSION,
    RosObservationNode,
    _require_ros,
    scan_hits_footprint,
)
from .observations import build_observation, goal_features


# Normal inference values. The launch file may override any of these as ROS
# parameters, but editing this block changes the defaults in one place.
POLICY_DEFAULTS = {
    "checkpoint": "",
    "scan_topic": "/scan",
    "odometry_topic": "/odometry/filtered",
    "goal_topic": "/goal_pose",
    "cmd_vel_topic": "/cmd_vel",
    "fault_topic": "/hardware/motor_fault",
    "scan_range_max": 8.0,
    "odom_frame": "odom",
    "robot_name": "robot",
    "control_rate": 10.0,
    "sensor_timeout": 0.75,
    "goal_tolerance": 0.25,
    "collision_distance": 0.08,
    "footprint_safety_margin": 0.05,
    "max_goal_distance": 12.0,
    "max_vx": 0.35,
    "max_vy": 0.35,
    "max_wz": 0.80,
    "max_action_delta": 0.35,
    "device": "auto",
}


try:
    import rclpy
    from std_srvs.srv import Trigger

    _RCLPY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on host environment.
    rclpy = None
    Trigger = Any  # type: ignore
    _RCLPY_IMPORT_ERROR = exc


STATE_IDLE = "IDLE"
STATE_ACTIVE = "ACTIVE"
STATE_GOAL = "GOAL"
STATE_FAULT = "FAULT"


class PolicyNode(RosObservationNode):
    """Load one checkpoint and drive the common ROS navigation interface."""

    def __init__(self, *, context: Any | None = None) -> None:
        """Load parameters/checkpoint and start the safe control timer."""
        _require_ros()

        # These fields must exist before subscriptions are created in super().
        self.policy_state = STATE_IDLE
        self._fault_latched = False
        self._activation_wall_time = time.monotonic()
        self._previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._closed = False

        # Topic names are bootstrap parameters because the subscriptions are
        # created by RosObservationNode itself.
        bootstrap_name = "ppo_policy"
        super().__init__(
            node_name=bootstrap_name,
            scan_topic=POLICY_DEFAULTS["scan_topic"],
            odometry_topic=POLICY_DEFAULTS["odometry_topic"],
            goal_topic=POLICY_DEFAULTS["goal_topic"],
            cmd_vel_topic=POLICY_DEFAULTS["cmd_vel_topic"],
            fault_topic=POLICY_DEFAULTS["fault_topic"],
            scan_range_max=POLICY_DEFAULTS["scan_range_max"],
            odom_frame=POLICY_DEFAULTS["odom_frame"],
            robot_name=POLICY_DEFAULTS["robot_name"],
            context=context,
            enable_gazebo_services=False,
            declare_topic_parameters=True,
        )

        self.declare_parameter("checkpoint", POLICY_DEFAULTS["checkpoint"])
        for name in (
            "control_rate",
            "sensor_timeout",
            "goal_tolerance",
            "collision_distance",
            "footprint_safety_margin",
            "max_goal_distance",
            "max_vx",
            "max_vy",
            "max_wz",
            "max_action_delta",
            "device",
        ):
            self.declare_parameter(name, POLICY_DEFAULTS[name])
        checkpoint = str(self.get_parameter("checkpoint").value).strip()
        if not checkpoint:
            raise RuntimeError(
                "Parameter 'checkpoint' is required, for example: "
                "--ros-args -p checkpoint:=/path/to/best_model.pt"
            )

        self.control_rate = float(self.get_parameter("control_rate").value)
        self.sensor_timeout = float(self.get_parameter("sensor_timeout").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.collision_distance = float(
            self.get_parameter("collision_distance").value
        )
        self.footprint_safety_margin = float(
            self.get_parameter("footprint_safety_margin").value
        )
        self.max_goal_distance = float(
            self.get_parameter("max_goal_distance").value
        )
        self.action_limits = ActionLimits(
            max_vx=float(self.get_parameter("max_vx").value),
            max_vy=float(self.get_parameter("max_vy").value),
            max_wz=float(self.get_parameter("max_wz").value),
            max_action_delta=float(
                self.get_parameter("max_action_delta").value
            ),
        )
        numeric_parameters = np.asarray(
            (
                self.control_rate,
                self.sensor_timeout,
                self.goal_tolerance,
                self.collision_distance,
                self.footprint_safety_margin,
                self.max_goal_distance,
                self.action_limits.max_vx,
                self.action_limits.max_vy,
                self.action_limits.max_wz,
                self.action_limits.max_action_delta,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(numeric_parameters).all():
            raise ValueError("policy numeric parameters must be finite")
        if min(
            self.control_rate,
            self.sensor_timeout,
            self.goal_tolerance,
            self.max_goal_distance,
            self.action_limits.max_vx,
            self.action_limits.max_vy,
            self.action_limits.max_wz,
            self.action_limits.max_action_delta,
        ) <= 0.0:
            raise ValueError("policy rates, distances and limits must be positive")
        if self.collision_distance < 0.0:
            raise ValueError("collision_distance cannot be negative")
        if self.footprint_safety_margin < 0.0:
            raise ValueError("footprint_safety_margin cannot be negative")
        self.device = choose_device(str(self.get_parameter("device").value))
        expected_contract = {
            "version": POLICY_CONTRACT_VERSION,
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "laser_sectors": LASER_SECTORS,
            "scan_range_max": self.scan_range_max,
            "max_goal_distance": self.max_goal_distance,
            "action_limits": asdict(self.action_limits),
        }
        self.network, self.checkpoint, _ = load_policy(
            Path(checkpoint),
            self.device,
            expected_contract=expected_contract,
        )
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self._control_tick,
        )
        self.rearm_service = self.create_service(
            Trigger,
            "~/rearm",
            self._rearm_callback,
        )
        self.publish_stop()
        self.get_logger().info(
            f"PPO policy loaded on {self.device}; waiting for /goal_pose"
        )

    def _set_state(self, state: str, reason: str | None = None) -> None:
        if self.policy_state == state:
            return
        previous = self.policy_state
        self.policy_state = state
        suffix = "" if reason is None else f": {reason}"
        self.get_logger().info(f"Policy {previous} -> {state}{suffix}")

    def _enter_fault(self, reason: str) -> None:
        self._fault_latched = True
        self._previous_action.fill(0.0)
        self.publish_stop()
        self._set_state(STATE_FAULT, reason)

    def _goal_callback(self, message: Any) -> None:
        if self.motor_fault or self._fault_latched:
            self.publish_stop()
            self.get_logger().warning(
                "Ignoring goal while faulted; clear the cause and call ~/rearm"
            )
            return
        if not super()._goal_callback(message):
            return
        self._activation_wall_time = time.monotonic()
        self._previous_action.fill(0.0)
        self._set_state(STATE_ACTIVE, "new goal")

    def _fault_callback(self, message: Any) -> None:
        was_faulted = self._fault_latched
        super()._fault_callback(message)
        if bool(message.data):
            self._enter_fault("hardware motor fault")
        elif was_faulted:
            self._previous_action.fill(0.0)
            self.publish_stop()
            self.get_logger().warning(
                "Motor fault signal cleared; policy remains FAULT until ~/rearm"
            )

    def _rearm_callback(self, _request: Any, response: Any) -> Any:
        """Safely clear a latched policy fault after explicit operator action."""
        self.publish_stop()
        if self.motor_fault:
            response.success = False
            response.message = "hardware motor fault is still active"
            return response
        scan_age, odometry_age = self.data_age()
        if (
            not math.isfinite(scan_age)
            or not math.isfinite(odometry_age)
            or max(scan_age, odometry_age) > self.sensor_timeout
        ):
            response.success = False
            response.message = "fresh scan and odometry are required"
            return response
        sectors = self.latest_laser_sectors()
        obstacle_unsafe = sectors is None
        if sectors is not None:
            obstacle_unsafe = bool(
                scan_hits_footprint(
                    sectors,
                    self.scan_range_max,
                    safety_margin=self.footprint_safety_margin,
                )
                or float(np.min(sectors)) * self.scan_range_max
                <= self.collision_distance
            )
        if obstacle_unsafe:
            response.success = False
            response.message = "an obstacle still intersects the safety footprint"
            return response
        self._fault_latched = False
        self._previous_action.fill(0.0)
        self.clear_goal()
        self._set_state(STATE_IDLE, "operator rearmed; new goal required")
        response.success = True
        response.message = "policy rearmed; publish a new goal"
        return response

    def _observation(self, snapshot: Any) -> tuple[np.ndarray, float]:
        goal, distance, _ = goal_features(
            snapshot.x,
            snapshot.y,
            snapshot.yaw,
            snapshot.goal_x,
            snapshot.goal_y,
            self.max_goal_distance,
        )
        observation = build_observation(
            snapshot.laser_sectors,
            goal,
            snapshot.velocity,
            self._previous_action,
            max(self.action_limits.max_vx, self.action_limits.max_vy),
            self.action_limits.max_wz,
        )
        if observation.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(
                f"expected observation shape ({OBSERVATION_SIZE},), "
                f"got {observation.shape}"
            )
        return observation, distance

    def _control_tick(self) -> None:
        if self._closed:
            return
        if self.policy_state != STATE_ACTIVE:
            self.publish_stop()
            return
        if self.motor_fault:
            self._enter_fault("hardware motor fault")
            return

        scan_age, odometry_age = self.data_age()
        if max(scan_age, odometry_age) > self.sensor_timeout:
            # Allow one short acquisition window immediately after a new goal.
            if time.monotonic() - self._activation_wall_time <= self.sensor_timeout:
                self.publish_stop()
                return
            self._enter_fault(
                f"stale sensor data (scan={scan_age:.2f}s, odom={odometry_age:.2f}s)"
            )
            return

        snapshot = self.snapshot(use_ground_truth=False)
        if snapshot is None:
            if time.monotonic() - self._activation_wall_time <= self.sensor_timeout:
                self.publish_stop()
                return
            self._enter_fault("goal transform or sensor state unavailable")
            return
        footprint_collision = scan_hits_footprint(
            snapshot.laser_sectors,
            self.scan_range_max,
            safety_margin=self.footprint_safety_margin,
        )
        if (
            footprint_collision
            or snapshot.minimum_scan <= self.collision_distance
        ):
            self._enter_fault(
                "obstacle intersects the inflated footprint"
            )
            return

        try:
            observation, distance = self._observation(snapshot)
            if distance <= self.goal_tolerance:
                self._previous_action.fill(0.0)
                self.publish_stop()
                self._set_state(STATE_GOAL, f"goal reached at {distance:.3f} m")
                return
            action, _, _ = self.network.get_action(
                observation,
                deterministic=True,
            )
            if hasattr(action, "detach"):
                action = action.detach().cpu().numpy()
            requested = sanitize_action(action)
            limited = limit_action_rate(
                requested,
                self._previous_action,
                self.action_limits.max_action_delta,
            )
            command = scale_action(limited, self.action_limits)
        except Exception as exc:
            self._enter_fault(f"invalid inference output: {exc}")
            return

        self.publish_command(command)
        self._previous_action = limited

    def destroy_node(self) -> bool:
        """Publish a final stop before destroying ROS entities."""
        if not self._closed:
            self._closed = True
            try:
                self.publish_stop()
            except Exception:
                pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """ROS console-script entry point."""
    _require_ros()
    rclpy.init(args=args)
    node: PolicyNode | None = None
    try:
        node = PolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


__all__ = [
    "PolicyNode",
    "STATE_ACTIVE",
    "STATE_FAULT",
    "STATE_GOAL",
    "STATE_IDLE",
    "main",
]


if __name__ == "__main__":
    main()
