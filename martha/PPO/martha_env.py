"""
Gymnasium environment and ROS I/O shared by simulation and hardware.

The policy-facing contract is intentionally independent from the actuator
backend: four LiDAR, relative-goal and odometry frames span one second of ROS
time. Both Gazebo and the physical robot receive the same normalized action
through ``/cmd_vel``.

ROS, Gymnasium and Gazebo message packages are optional at import time.  This
keeps geometry/reward unit tests usable on development hosts without ROS; a
clear runtime error is raised when :class:`MarthaEnv` is instantiated without
the required dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np

from .actions import (
    ACTION_SIZE,
    ActionLimits,
    limit_action_rate,
    sanitize_action,
    scale_action,
)
from .observations import (
    DEFAULT_GOAL_DISTANCE_SCALE,
    GOAL_DISTANCE_ENCODING,
    GOAL_GUIDANCE_MODE,
    LASER_SECTORS,
    OBSERVATION_FRAME_SIZE,
    OBSERVATION_HISTORY_FRAMES,
    OBSERVATION_HISTORY_SECONDS,
    OBSERVATION_SIZE,
    ObservationHistory,
    build_observation_frame,
    goal_features,
    reduce_laser_scan,
)
from .reward import (
    RewardConfig,
    RewardState,
    calculate_reward,
    empty_reward_components,
    update_stagnation,
    validate_reward_config,
)
from .world_map import EpisodeSample, WorldMap, discover_worlds


try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_IMPORT_ERROR: Exception | None = None
    _GymEnvBase = gym.Env
except Exception as exc:  # pragma: no cover - depends on the host environment.
    gym = None
    spaces = None
    _GYM_IMPORT_ERROR = exc

    class _GymEnvBase:  # Minimal base so this module remains import-safe.
        pass


try:
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from rclpy.time import Time as RosTime

    from gazebo_msgs.msg import ContactsState, EntityState, ModelStates
    from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
    from geometry_msgs.msg import Pose, PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from robot_localization.srv import SetPose
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool
    from std_srvs.srv import Empty
    from tf2_ros import Buffer, TransformListener

    _ROS_IMPORT_ERROR: Exception | None = None
    _NodeBase = Node
except Exception as exc:  # pragma: no cover - depends on the host environment.
    rclpy = None
    Context = Any  # type: ignore
    MultiThreadedExecutor = Any  # type: ignore
    Node = None
    Parameter = Any  # type: ignore
    qos_profile_sensor_data = None
    DurabilityPolicy = QoSProfile = ReliabilityPolicy = Any  # type: ignore
    RosTime = None
    ContactsState = EntityState = ModelStates = Any  # type: ignore
    DeleteEntity = SetEntityState = SpawnEntity = SetPose = Any  # type: ignore
    Pose = PoseStamped = Twist = Any  # type: ignore
    Odometry = LaserScan = Bool = Empty = Any  # type: ignore
    Buffer = TransformListener = Any  # type: ignore
    _ROS_IMPORT_ERROR = exc
    _NodeBase = object


POLICY_CONTRACT_VERSION = 8
PPO_SCENARIO_ENTITY_PREFIX = "martha_ppo_s"
PPO_GOAL_ENTITY_PREFIX = "martha_ppo_goal_"
PPO_GOAL_ENTITY_NAME = "martha_ppo_goal_current"
MAX_RANDOM_RESET_ATTEMPTS = 20
GAZEBO_CONTROL_SERVICE_ATTEMPTS = 3


def build_policy_contract(
    scan_range_max: float,
    goal_distance_scale: float,
    action_limits: ActionLimits,
) -> dict[str, Any]:
    """Return the complete policy architecture and scaling contract."""
    # Imported lazily so geometry and reward tests can still import this module
    # on hosts that do not have PyTorch installed.
    from .network import (
        POLICY_ARCHITECTURE,
        RECURRENT_HIDDEN_SIZE,
        RECURRENT_NUM_LAYERS,
    )

    return {
        "version": POLICY_CONTRACT_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "laser_sectors": LASER_SECTORS,
        "architecture": POLICY_ARCHITECTURE,
        "recurrent_type": "lstm",
        "recurrent_hidden_size": RECURRENT_HIDDEN_SIZE,
        "recurrent_num_layers": RECURRENT_NUM_LAYERS,
        "observation_layout": "frame_major",
        "observation_frame_size": OBSERVATION_FRAME_SIZE,
        "observation_history_frames": OBSERVATION_HISTORY_FRAMES,
        "observation_history_seconds": OBSERVATION_HISTORY_SECONDS,
        "scan_range_max": float(scan_range_max),
        "goal_distance_encoding": GOAL_DISTANCE_ENCODING,
        "goal_guidance_mode": GOAL_GUIDANCE_MODE,
        "goal_distance_scale": float(goal_distance_scale),
        "action_limits": {
            "max_vx": float(action_limits.max_vx),
            "max_vy": float(action_limits.max_vy),
            "max_wz": float(action_limits.max_wz),
            "max_action_delta": float(action_limits.max_action_delta),
        },
    }


def scan_hits_footprint(
    normalized_sectors: Iterable[float],
    scan_range_max: float,
    *,
    lidar_offset_x: float = 0.2325,
    footprint_half_x: float = 0.29,
    footprint_half_y: float = 0.255,
    safety_margin: float = 0.05,
) -> bool:
    """
    Return whether a canonical scan enters Martha's inflated footprint.

    The LiDAR sits near the front of the chassis, so a single radial threshold
    is unsafe behind the sensor and unnecessarily restrictive in front.  This
    ray/rectangle test accounts for that offset and samples both edges of each
    reduced angular sector conservatively.
    """
    return footprint_scan_hit(
        normalized_sectors,
        scan_range_max,
        lidar_offset_x=lidar_offset_x,
        footprint_half_x=footprint_half_x,
        footprint_half_y=footprint_half_y,
        safety_margin=safety_margin,
    ) is not None


@dataclass(frozen=True)
class FootprintScanHit:
    """One reduced LiDAR sector entering Martha's inflated footprint."""

    sector_index: int
    angle_radians: float
    measured_range: float
    threshold: float


def footprint_scan_hit(
    normalized_sectors: Iterable[float],
    scan_range_max: float,
    *,
    lidar_offset_x: float = 0.2325,
    footprint_half_x: float = 0.29,
    footprint_half_y: float = 0.255,
    safety_margin: float = 0.05,
) -> FootprintScanHit | None:
    """Return the first sector entering the footprint, with diagnostics."""
    sectors = np.asarray(tuple(normalized_sectors), dtype=np.float32)
    if sectors.shape != (LASER_SECTORS,) or not np.isfinite(sectors).all():
        raise ValueError("canonical laser sectors must be a finite 36-vector")
    scalar_parameters = (
        scan_range_max,
        lidar_offset_x,
        footprint_half_x,
        footprint_half_y,
        safety_margin,
    )
    if not np.isfinite(scalar_parameters).all() or scan_range_max <= 0.0:
        raise ValueError("footprint scan parameters must be finite")
    if min(footprint_half_x, footprint_half_y, safety_margin) < 0.0:
        raise ValueError("footprint dimensions and margin cannot be negative")

    x_min = -footprint_half_x - lidar_offset_x
    x_max = footprint_half_x - lidar_offset_x
    y_min = -footprint_half_y
    y_max = footprint_half_y
    sector_width = 2.0 * math.pi / LASER_SECTORS

    def boundary_distance(angle: float) -> float:
        direction_x = math.cos(angle)
        direction_y = math.sin(angle)
        candidates = []
        if abs(direction_x) > 1e-9:
            boundary_x = x_max if direction_x > 0.0 else x_min
            candidate = boundary_x / direction_x
            if candidate >= 0.0:
                candidates.append(candidate)
        if abs(direction_y) > 1e-9:
            boundary_y = y_max if direction_y > 0.0 else y_min
            candidate = boundary_y / direction_y
            if candidate >= 0.0:
                candidates.append(candidate)
        return min(candidates) if candidates else math.inf

    ranges = np.clip(sectors, 0.0, 1.0) * float(scan_range_max)
    for index, measured_range in enumerate(ranges):
        center = -math.pi + (index + 0.5) * sector_width
        threshold = max(
            boundary_distance(center - 0.5 * sector_width),
            boundary_distance(center),
            boundary_distance(center + 0.5 * sector_width),
        ) + safety_margin
        if float(measured_range) <= threshold:
            return FootprintScanHit(
                sector_index=index,
                angle_radians=center,
                measured_range=float(measured_range),
                threshold=float(threshold),
            )
    return None


def external_contact_models(
    contact_pairs: Iterable[tuple[str, str]],
    robot_name: str,
) -> set[str]:
    """Return external model names touching ``robot_name``'s contact shell."""
    own_prefix = f"{robot_name}::"
    external: set[str] = set()
    for collision1, collision2 in contact_pairs:
        first_is_own = collision1.startswith(own_prefix)
        second_is_own = collision2.startswith(own_prefix)
        if first_is_own == second_is_own:
            continue
        other = collision2 if first_is_own else collision1
        other_model = other.split("::", 1)[0]
        if other_model and other_model not in {robot_name, "ground_plane"}:
            external.add(other_model)
    return external


def _require_gymnasium() -> None:
    if _GYM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "MarthaEnv requires Gymnasium. Install the project's RL "
            "dependencies in the ROS 2 Python environment."
        ) from _GYM_IMPORT_ERROR


def _require_ros() -> None:
    if _ROS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "MarthaEnv requires ROS 2 Python packages (rclpy, gazebo_msgs, "
            "geometry_msgs, nav_msgs, robot_localization, sensor_msgs, "
            "std_srvs and tf2_ros)."
        ) from _ROS_IMPORT_ERROR


def fault_topic_for_backend(backend: str) -> str:
    """Keep hardware-only protection state out of Gazebo sessions."""
    if backend == "gazebo":
        return ""
    if backend == "hardware":
        return "/hardware/motor_fault"
    raise ValueError("backend must be 'gazebo' or 'hardware'")


def _validate_action_limits(limits: ActionLimits) -> None:
    """Reject unsafe or non-finite velocity and slew limits."""
    values = np.asarray(
        (
            limits.max_vx,
            limits.max_vy,
            limits.max_wz,
            limits.max_action_delta,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("action velocity and slew limits must be finite and positive")


def _yaw_from_quaternion(quaternion: Any) -> float:
    sin_yaw = 2.0 * (
        float(quaternion.w) * float(quaternion.z)
        + float(quaternion.x) * float(quaternion.y)
    )
    cos_yaw = 1.0 - 2.0 * (
        float(quaternion.y) * float(quaternion.y)
        + float(quaternion.z) * float(quaternion.z)
    )
    return math.atan2(sin_yaw, cos_yaw)


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _rotate_vector_by_quaternion(
    x: float,
    y: float,
    z: float,
    quaternion: Any,
) -> tuple[float, float, float]:
    """Rotate one vector without requiring tf2_geometry_msgs."""
    qx = float(quaternion.x)
    qy = float(quaternion.y)
    qz = float(quaternion.z)
    qw = float(quaternion.w)
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


@dataclass(frozen=True)
class RosStreamActivity:
    """Read-only counters for the ROS streams required by Gazebo training."""

    scan_sequence: int
    odometry_sequence: int
    ground_truth_sequence: int
    contact_sequence: int
    scan_stamp_ns: int
    odometry_stamp_ns: int

    def missing_after(
        self,
        baseline: "RosStreamActivity",
        *,
        require_contact: bool,
    ) -> tuple[str, ...]:
        """Describe streams that have not advanced beyond ``baseline``."""
        missing = []
        if self.scan_sequence <= baseline.scan_sequence:
            missing.append(
                "scan"
                f"(sequence={self.scan_sequence}, need>{baseline.scan_sequence}, "
                f"stamp_ns={self.scan_stamp_ns})"
            )
        if self.odometry_sequence <= baseline.odometry_sequence:
            missing.append(
                "odometry"
                f"(sequence={self.odometry_sequence}, "
                f"need>{baseline.odometry_sequence}, "
                f"stamp_ns={self.odometry_stamp_ns})"
            )
        if self.ground_truth_sequence <= baseline.ground_truth_sequence:
            missing.append(
                "model_state"
                f"(sequence={self.ground_truth_sequence}, "
                f"need>{baseline.ground_truth_sequence})"
            )
        if require_contact and self.contact_sequence <= baseline.contact_sequence:
            missing.append(
                "contact"
                f"(sequence={self.contact_sequence}, "
                f"need>{baseline.contact_sequence})"
            )
        return tuple(missing)

    def missing_snapshot_data(
        self,
        after_sequences: tuple[int, int, int],
        *,
        use_ground_truth: bool,
        after_stamp_ns: int | None,
    ) -> tuple[str, ...]:
        """Describe unmet freshness conditions for a policy snapshot."""
        missing = []
        if self.scan_sequence <= after_sequences[0]:
            missing.append(
                "scan"
                f"(sequence={self.scan_sequence}, need>{after_sequences[0]})"
            )
        if self.odometry_sequence <= after_sequences[1]:
            missing.append(
                "odometry"
                f"(sequence={self.odometry_sequence}, "
                f"need>{after_sequences[1]})"
            )
        if use_ground_truth and self.ground_truth_sequence <= after_sequences[2]:
            missing.append(
                "model_state"
                f"(sequence={self.ground_truth_sequence}, "
                f"need>{after_sequences[2]})"
            )
        if after_stamp_ns is not None:
            if self.scan_stamp_ns <= after_stamp_ns:
                missing.append(
                    "scan_stamp"
                    f"(stamp_ns={self.scan_stamp_ns}, need>{after_stamp_ns})"
                )
            if self.odometry_stamp_ns <= after_stamp_ns:
                missing.append(
                    "odometry_stamp"
                    f"(stamp_ns={self.odometry_stamp_ns}, "
                    f"need>{after_stamp_ns})"
                )
        return tuple(missing)


@dataclass(frozen=True)
class SensorSnapshot:
    """Immutable, coherent view of the latest policy-relevant ROS data."""

    laser_sectors: np.ndarray
    minimum_scan: float
    x: float
    y: float
    yaw: float
    velocity: tuple[float, float, float]
    goal_x: float
    goal_y: float
    scan_sequence: int
    odometry_sequence: int
    ground_truth_sequence: int
    scan_wall_time: float
    odometry_wall_time: float
    scan_stamp_ns: int
    odometry_stamp_ns: int
    motor_fault: bool
    ground_truth_x: float | None
    ground_truth_y: float | None
    ground_truth_yaw: float | None


@dataclass(frozen=True)
class PendingStep:
    """Action and freshness barrier prepared while Gazebo is paused."""

    requested_action: np.ndarray
    limited_action: np.ndarray
    applied_action: np.ndarray
    command: np.ndarray
    command_inhibited: bool
    command_stamp_ns: int
    sequences: tuple[int, int, int]
    contact_sequence: int


@dataclass(frozen=True)
class PendingGazeboReset:
    """Episode placement and freshness barriers prepared while paused."""

    world_map: WorldMap
    sample: EpisodeSample
    reset_stamp_ns: int
    sequences: tuple[int, int, int]
    contact_sequence: int


@dataclass(frozen=True)
class ResetSafetyCheck:
    """Pure safety result for one post-teleport Gazebo observation."""

    map_free: bool
    footprint_clear: bool
    odom_xy_error: float
    odom_yaw_error: float
    connected: bool
    shortest_path: float
    truth_x: float
    truth_y: float
    truth_yaw: float
    minimum_scan: float
    scan_hit: FootprintScanHit | None

    @property
    def safe(self) -> bool:
        """Return whether every reset invariant is satisfied."""
        return bool(
            self.map_free
            and self.footprint_clear
            and self.odom_xy_error <= 0.08
            and self.odom_yaw_error <= 0.12
            and self.connected
        )

    def diagnostic(self) -> str:
        """Format a complete single-line reset diagnostic."""
        fields = [
            f"map_free={self.map_free}",
            f"footprint_clear={self.footprint_clear}",
            f"odom_xy_error={self.odom_xy_error:.6f}",
            f"odom_yaw_error={self.odom_yaw_error:.6f}",
            f"minimum_scan={self.minimum_scan:.6f}",
            f"connected={self.connected}",
            (
                "truth_pose=("
                f"{self.truth_x:.6f},{self.truth_y:.6f},{self.truth_yaw:.6f})"
            ),
        ]
        if self.scan_hit is not None:
            fields.extend(
                (
                    f"lidar_sector={self.scan_hit.sector_index}",
                    (
                        "lidar_angle_deg="
                        f"{math.degrees(self.scan_hit.angle_radians):.3f}"
                    ),
                    f"lidar_range={self.scan_hit.measured_range:.6f}",
                    f"lidar_threshold={self.scan_hit.threshold:.6f}",
                )
            )
        else:
            fields.append("lidar_sector=none")
        return ", ".join(fields)


class RosObservationNode(_NodeBase):
    """Own ROS subscriptions, commands and optional Gazebo service clients."""

    _SERVICE_CANDIDATES = {
        "pause": ("/gazebo/pause_physics", "/pause_physics"),
        "unpause": ("/gazebo/unpause_physics", "/unpause_physics"),
        "reset_world": ("/gazebo/reset_world", "/reset_world"),
        "set_entity_state": (
            "/gazebo/set_entity_state",
            "/set_entity_state",
        ),
        "spawn_entity": ("/spawn_entity", "/gazebo/spawn_entity"),
        "delete_entity": ("/delete_entity", "/gazebo/delete_entity"),
        "set_pose": ("/set_pose", "/ekf_filter_node/set_pose"),
    }

    def __init__(
        self,
        *,
        node_name: str,
        scan_topic: str,
        odometry_topic: str,
        goal_topic: str,
        cmd_vel_topic: str,
        contact_topic: str,
        fault_topic: str,
        scan_range_max: float = 8.0,
        odom_frame: str = "odom",
        robot_name: str = "robot",
        service_namespace: str = "",
        context: Any | None = None,
        enable_gazebo_services: bool = False,
        declare_topic_parameters: bool = False,
        accept_external_goal: bool = True,
        min_scan_coverage: float = 0.90,
        use_sim_time: bool | None = None,
    ) -> None:
        """Create subscriptions and optional Gazebo service clients."""
        _require_ros()
        super().__init__(node_name, context=context)
        if use_sim_time is not None:
            results = self.set_parameters(
                [Parameter("use_sim_time", value=bool(use_sim_time))]
            )
            if not all(result.successful for result in results):
                raise RuntimeError("Could not configure the ROS clock")
        if declare_topic_parameters:
            self.declare_parameter("scan_topic", scan_topic)
            self.declare_parameter("odometry_topic", odometry_topic)
            self.declare_parameter("goal_topic", goal_topic)
            self.declare_parameter("cmd_vel_topic", cmd_vel_topic)
            self.declare_parameter("fault_topic", fault_topic)
            self.declare_parameter("scan_range_max", scan_range_max)
            self.declare_parameter("min_scan_coverage", min_scan_coverage)
            self.declare_parameter("odom_frame", odom_frame)
            self.declare_parameter("robot_name", robot_name)
            scan_topic = str(self.get_parameter("scan_topic").value)
            odometry_topic = str(self.get_parameter("odometry_topic").value)
            goal_topic = str(self.get_parameter("goal_topic").value)
            cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
            fault_topic = str(self.get_parameter("fault_topic").value)
            scan_range_max = float(self.get_parameter("scan_range_max").value)
            min_scan_coverage = float(
                self.get_parameter("min_scan_coverage").value
            )
            odom_frame = str(self.get_parameter("odom_frame").value)
            robot_name = str(self.get_parameter("robot_name").value)
        self.odom_frame = odom_frame
        self.robot_name = robot_name
        self.accept_external_goal = bool(accept_external_goal)
        self.scan_range_max = float(scan_range_max)
        self.min_scan_coverage = float(min_scan_coverage)
        if not math.isfinite(self.scan_range_max) or self.scan_range_max <= 0.0:
            raise ValueError("scan_range_max must be positive")
        if (
            not math.isfinite(self.min_scan_coverage)
            or not 0.0 < self.min_scan_coverage <= 1.0
        ):
            raise ValueError("min_scan_coverage must be in (0, 1]")
        self._condition = threading.Condition()
        self._laser_sectors: np.ndarray | None = None
        self._minimum_scan = math.inf
        self._scan_sequence = 0
        self._odometry_sequence = 0
        self._ground_truth_sequence = 0
        self._scan_wall_time = -math.inf
        self._odometry_wall_time = -math.inf
        self._scan_stamp_ns = -1
        self._odometry_stamp_ns = -1
        self._odom_pose: tuple[float, float, float] | None = None
        self._ground_truth_pose: tuple[float, float, float] | None = None
        self._gazebo_model_names: set[str] = set()
        self._model_states_sequence = 0
        self._velocity = (0.0, 0.0, 0.0)
        self._goal: tuple[float, float, float, str] | None = None
        self._goal_sequence = 0
        self._motor_fault = False
        self._contact_sequence = 0
        self._contact_events: list[tuple[int, tuple[str, str]]] = []

        self.command_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.goal_publisher = self.create_publisher(PoseStamped, goal_topic, 10)
        self.scan_subscription = self.create_subscription(
            LaserScan,
            scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.odometry_subscription = self.create_subscription(
            Odometry,
            odometry_topic,
            self._odometry_callback,
            20,
        )
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            goal_topic,
            self._goal_callback,
            10,
        )
        self.model_states_subscription = self.create_subscription(
            ModelStates,
            "/gazebo/model_states",
            self._model_states_callback,
            qos_profile_sensor_data,
        )
        self.contact_subscription = None
        if contact_topic:
            self.contact_subscription = self.create_subscription(
                ContactsState,
                contact_topic,
                self._contact_callback,
                qos_profile_sensor_data,
            )
        self.fault_subscription = None
        if fault_topic:
            fault_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.fault_subscription = self.create_subscription(
                Bool,
                fault_topic,
                self._fault_callback,
                fault_qos,
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        self._service_clients: dict[str, list[Any]] = {}
        if enable_gazebo_services:
            service_types = {
                "pause": Empty,
                "unpause": Empty,
                "reset_world": Empty,
                "set_entity_state": SetEntityState,
                "spawn_entity": SpawnEntity,
                "delete_entity": DeleteEntity,
                "set_pose": SetPose,
            }
            for key, candidates in self._SERVICE_CANDIDATES.items():
                if key == "set_pose" and service_namespace:
                    namespace = service_namespace.strip("/")
                    candidates = (
                        f"/{namespace}/set_pose",
                        f"/{namespace}/ekf_filter_node/set_pose",
                    )
                self._service_clients[key] = [
                    (name, self.create_client(service_types[key], name))
                    for name in candidates
                ]

    def _contact_callback(self, message: Any) -> None:
        pairs = {
            (str(state.collision1_name), str(state.collision2_name))
            for state in message.states
            if state.collision1_name and state.collision2_name
        }
        with self._condition:
            self._contact_sequence += 1
            sequence = self._contact_sequence
            self._contact_events.extend((sequence, pair) for pair in pairs)
            self._condition.notify_all()

    def clear_contacts(self) -> int:
        """Discard old events and return the next transition's barrier."""
        with self._condition:
            self._contact_events.clear()
            return self._contact_sequence

    def consume_contact_pairs(
        self,
        *,
        after_sequence: int | None = None,
    ) -> set[tuple[str, str]]:
        """Return contact events newer than a captured callback barrier."""
        with self._condition:
            barrier = -1 if after_sequence is None else int(after_sequence)
            pairs = {
                pair
                for sequence, pair in self._contact_events
                if sequence > barrier
            }
            self._contact_events.clear()
            return pairs

    def wait_for_fresh_contact_message(
        self,
        after_sequence: int,
        timeout: float,
    ) -> bool:
        """Wait until the bumper publishes after a robot placement."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._contact_sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def _scan_callback(self, message: Any) -> None:
        try:
            raw_ranges = np.asarray(message.ranges, dtype=np.float32)
            if raw_ranges.size == 0:
                raise ValueError("laser scan has no usable samples")
            usable = np.isfinite(raw_ranges) | np.isposinf(raw_ranges)
            usable_fraction = float(np.count_nonzero(usable)) / raw_ranges.size
            if usable_fraction < self.min_scan_coverage:
                raise ValueError(
                    "laser scan usable fraction is only "
                    f"{usable_fraction:.3f}"
                )
            angular_coverage = abs(float(message.angle_increment)) * raw_ranges.size
            if angular_coverage < self.min_scan_coverage * 2.0 * math.pi:
                raise ValueError(
                    f"laser scan covers only {angular_coverage:.3f} rad"
                )
            sectors, minimum = reduce_laser_scan(
                raw_ranges,
                message.range_min,
                self.scan_range_max,
                sectors=LASER_SECTORS,
                angle_min=message.angle_min,
                angle_increment=message.angle_increment,
            )
            stamp_ns = self._message_stamp_ns(message)
        except (ValueError, FloatingPointError) as exc:
            self.get_logger().warning(f"Ignoring invalid laser scan: {exc}")
            return
        now = time.monotonic()
        with self._condition:
            self._laser_sectors = sectors
            self._minimum_scan = minimum
            self._scan_sequence += 1
            self._scan_wall_time = now
            self._scan_stamp_ns = stamp_ns
            self._condition.notify_all()

    def _odometry_callback(self, message: Any) -> None:
        pose = message.pose.pose
        twist = message.twist.twist
        values = (
            pose.position.x,
            pose.position.y,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
        )
        if not np.isfinite(values).all():
            self.get_logger().warning("Ignoring non-finite odometry")
            return
        try:
            stamp_ns = self._message_stamp_ns(message)
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring invalid odometry stamp: {exc}")
            return
        now = time.monotonic()
        with self._condition:
            self._odom_pose = (
                float(pose.position.x),
                float(pose.position.y),
                _yaw_from_quaternion(pose.orientation),
            )
            self._velocity = (
                float(twist.linear.x),
                float(twist.linear.y),
                float(twist.angular.z),
            )
            self._odometry_sequence += 1
            self._odometry_wall_time = now
            self._odometry_stamp_ns = stamp_ns
            self._condition.notify_all()

    def _goal_callback(self, message: Any) -> bool:
        if not self.accept_external_goal:
            return False
        frame_id = message.header.frame_id or self.odom_frame
        values = (
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        )
        if not np.isfinite(values).all():
            self.get_logger().warning("Ignoring non-finite navigation goal")
            return False
        with self._condition:
            self._goal = (
                float(message.pose.position.x),
                float(message.pose.position.y),
                float(message.pose.position.z),
                frame_id,
            )
            self._goal_sequence += 1
            self._condition.notify_all()
        return True

    def _model_states_callback(self, message: Any) -> None:
        model_names = {str(name) for name in message.name}
        with self._condition:
            self._gazebo_model_names = model_names
            self._model_states_sequence += 1
            self._condition.notify_all()
        try:
            index = message.name.index(self.robot_name)
            pose = message.pose[index]
        except (ValueError, IndexError):
            return
        values = (
            pose.position.x,
            pose.position.y,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not np.isfinite(values).all():
            self.get_logger().warning("Ignoring non-finite Gazebo model state")
            return
        with self._condition:
            self._ground_truth_pose = (
                float(pose.position.x),
                float(pose.position.y),
                _yaw_from_quaternion(pose.orientation),
            )
            self._ground_truth_sequence += 1
            self._condition.notify_all()

    def gazebo_model_names(self) -> set[str]:
        """Return the latest complete Gazebo model-name inventory."""
        with self._condition:
            return set(self._gazebo_model_names)

    def gazebo_model_inventory(self) -> tuple[int, set[str]]:
        """Return the model-state sequence and its complete name inventory."""
        with self._condition:
            return self._model_states_sequence, set(self._gazebo_model_names)

    def wait_for_gazebo_entities_absent(
        self,
        names: set[str],
        after_sequence: int,
        timeout: float,
    ) -> set[str]:
        """Wait for a fresh inventory and return any entities still present."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                remaining_names = names & self._gazebo_model_names
                if (
                    self._model_states_sequence > after_sequence
                    and not remaining_names
                ):
                    return set()
                remaining_time = deadline - time.monotonic()
                if remaining_time <= 0.0:
                    return set(remaining_names)
                self._condition.wait(timeout=remaining_time)

    def wait_for_gazebo_model_names(self, timeout: float) -> set[str]:
        """Wait until Gazebo publishes a non-empty model inventory."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._gazebo_model_names:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return set()
                self._condition.wait(timeout=remaining)
            return set(self._gazebo_model_names)

    def _fault_callback(self, message: Any) -> None:
        active = bool(message.data)
        with self._condition:
            self._motor_fault = active
            self._condition.notify_all()
        if active:
            self.publish_stop()

    @staticmethod
    def _message_stamp_ns(message: Any) -> int:
        stamp = message.header.stamp
        seconds = int(stamp.sec)
        nanoseconds = int(stamp.nanosec)
        if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
            raise ValueError("message timestamp is outside the ROS time domain")
        return seconds * 1_000_000_000 + nanoseconds

    def set_goal(self, x: float, y: float, frame_id: str, z: float = 0.0) -> None:
        """Set a goal directly while preserving its coordinate frame."""
        if not np.isfinite((x, y, z)).all() or not frame_id:
            raise ValueError("goal coordinates must be finite with a nonempty frame")
        with self._condition:
            self._goal = (float(x), float(y), float(z), frame_id)
            self._goal_sequence += 1
            self._condition.notify_all()

    def clear_goal(self) -> None:
        """Remove the current goal so no complete snapshot can be produced."""
        with self._condition:
            self._goal = None

    def publish_goal(self, x: float, y: float, frame_id: str) -> None:
        """Publish an episode goal through the same PoseStamped contract."""
        if not np.isfinite((x, y)).all() or not frame_id:
            raise ValueError("published goal must be finite with a nonempty frame")
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = frame_id
        message.pose.position.x = float(x)
        message.pose.position.y = float(y)
        message.pose.orientation.w = 1.0
        self.goal_publisher.publish(message)

    @property
    def motor_fault(self) -> bool:
        """Return the latest latched hardware motor-fault value."""
        with self._condition:
            return self._motor_fault

    def sequence_numbers(self) -> tuple[int, int, int]:
        """Return scan, odometry and Gazebo ground-truth counters."""
        with self._condition:
            return (
                self._scan_sequence,
                self._odometry_sequence,
                self._ground_truth_sequence,
            )

    def stream_activity(self) -> RosStreamActivity:
        """Return stream counters and stamps without consuming queued data."""
        with self._condition:
            return RosStreamActivity(
                scan_sequence=self._scan_sequence,
                odometry_sequence=self._odometry_sequence,
                ground_truth_sequence=self._ground_truth_sequence,
                contact_sequence=self._contact_sequence,
                scan_stamp_ns=self._scan_stamp_ns,
                odometry_stamp_ns=self._odometry_stamp_ns,
            )

    def missing_fresh_snapshot_data(
        self,
        after_sequences: tuple[int, int, int],
        *,
        use_ground_truth: bool,
        after_stamp_ns: int | None,
    ) -> tuple[str, ...]:
        """Explain why a fresh-snapshot barrier has not completed."""
        return self.stream_activity().missing_snapshot_data(
            after_sequences,
            use_ground_truth=use_ground_truth,
            after_stamp_ns=after_stamp_ns,
        )

    def goal_sequence(self) -> int:
        """Return the number of accepted internal or external goals."""
        with self._condition:
            return self._goal_sequence

    def data_age(self) -> tuple[float, float]:
        """Return wall-clock age of scan and odometry data in seconds."""
        now = time.monotonic()
        with self._condition:
            return now - self._scan_wall_time, now - self._odometry_wall_time

    def latest_laser_sectors(self) -> np.ndarray | None:
        """Return a copy of the latest canonical scan, if one is available."""
        with self._condition:
            if self._laser_sectors is None:
                return None
            return self._laser_sectors.copy()

    def _goal_in_frame(self, target_frame: str) -> tuple[float, float] | None:
        with self._condition:
            goal = self._goal
        if goal is None:
            return None
        x, y, z, source_frame = goal
        if not source_frame or source_frame == target_frame:
            return x, y
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                RosTime(),
            ).transform
        except Exception:
            return None
        rotated = _rotate_vector_by_quaternion(x, y, z, transform.rotation)
        return (
            rotated[0] + float(transform.translation.x),
            rotated[1] + float(transform.translation.y),
        )

    def snapshot(self, *, use_ground_truth: bool) -> SensorSnapshot | None:
        """Build a snapshot when sensors, pose and a resolvable goal exist."""
        with self._condition:
            laser = None if self._laser_sectors is None else self._laser_sectors.copy()
            pose = self._odom_pose
            ground_truth_pose = self._ground_truth_pose
            minimum = self._minimum_scan
            velocity = self._velocity
            scan_sequence = self._scan_sequence
            odometry_sequence = self._odometry_sequence
            ground_sequence = self._ground_truth_sequence
            scan_wall_time = self._scan_wall_time
            odometry_wall_time = self._odometry_wall_time
            scan_stamp_ns = self._scan_stamp_ns
            odometry_stamp_ns = self._odometry_stamp_ns
            motor_fault = self._motor_fault
        if (
            laser is None
            or pose is None
            or (use_ground_truth and ground_truth_pose is None)
        ):
            return None
        goal = self._goal_in_frame(self.odom_frame)
        if goal is None:
            return None
        return SensorSnapshot(
            laser_sectors=laser,
            minimum_scan=minimum,
            x=pose[0],
            y=pose[1],
            yaw=pose[2],
            velocity=velocity,
            goal_x=goal[0],
            goal_y=goal[1],
            scan_sequence=scan_sequence,
            odometry_sequence=odometry_sequence,
            ground_truth_sequence=ground_sequence,
            scan_wall_time=scan_wall_time,
            odometry_wall_time=odometry_wall_time,
            scan_stamp_ns=scan_stamp_ns,
            odometry_stamp_ns=odometry_stamp_ns,
            motor_fault=motor_fault,
            ground_truth_x=(
                None if ground_truth_pose is None else ground_truth_pose[0]
            ),
            ground_truth_y=(
                None if ground_truth_pose is None else ground_truth_pose[1]
            ),
            ground_truth_yaw=(
                None if ground_truth_pose is None else ground_truth_pose[2]
            ),
        )

    def wait_for_fresh_snapshot(
        self,
        after_sequences: tuple[int, int, int],
        timeout: float,
        *,
        use_ground_truth: bool,
        after_stamp_ns: int | None = None,
    ) -> SensorSnapshot | None:
        """Wait for sensor counters newer than ``after_sequences``."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                scan_fresh = self._scan_sequence > after_sequences[0]
                odom_fresh = self._odometry_sequence > after_sequences[1]
                ground_fresh = (
                    not use_ground_truth
                    or self._ground_truth_sequence > after_sequences[2]
                )
                stamps_fresh = (
                    after_stamp_ns is None
                    or (
                        self._scan_stamp_ns > after_stamp_ns
                        and self._odometry_stamp_ns > after_stamp_ns
                    )
                )
                if scan_fresh and odom_fresh and ground_fresh and stamps_fresh:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)
        return self.snapshot(use_ground_truth=use_ground_truth)

    def publish_command(self, command: Iterable[float]) -> None:
        """Publish a finite SI velocity vector as ``geometry_msgs/Twist``."""
        values = np.asarray(tuple(command), dtype=np.float32)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("velocity command must be a finite [vx, vy, wz] vector")
        message = Twist()
        message.linear.x = float(values[0])
        message.linear.y = float(values[1])
        message.angular.z = float(values[2])
        self.command_publisher.publish(message)

    def publish_stop(self) -> None:
        """Publish an explicit zero-velocity safety command."""
        self.publish_command((0.0, 0.0, 0.0))

    def call_service(self, key: str, request: Any, timeout: float) -> Any:
        """Call the first available namespaced or root Gazebo service."""
        clients = self._service_clients.get(key, ())
        if not clients:
            raise RuntimeError(f"Gazebo service clients are disabled for {key}")
        deadline = time.monotonic() + timeout
        selected = None
        selected_name = None
        while time.monotonic() < deadline and selected is None:
            for name, client in clients:
                if client.service_is_ready():
                    selected = client
                    selected_name = name
                    break
            if selected is None:
                time.sleep(0.05)
        if selected is None:
            names = ", ".join(self._SERVICE_CANDIDATES[key])
            raise TimeoutError(f"Gazebo service not available ({names})")

        future = selected.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _: completed.set())
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or not completed.wait(remaining):
            raise TimeoutError(f"Gazebo service {selected_name} timed out")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(
                f"Gazebo service {selected_name} failed: {exception}"
            ) from exception
        return future.result()


class _RosRuntime:
    """Run one ROS node in a private context and executor thread."""

    def __init__(self, **node_kwargs: Any) -> None:
        """Initialize a private ROS context for one Gym environment."""
        _require_ros()
        self.context = Context()
        rclpy.init(args=None, context=self.context)
        self.node = RosObservationNode(context=self.context, **node_kwargs)
        self.executor = MultiThreadedExecutor(num_threads=3, context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(
            target=self.executor.spin,
            name="martha-ppo-ros",
            daemon=True,
        )
        self.thread.start()
        self._closed = False

    def close(self) -> None:
        """Stop the robot and release the private ROS context once."""
        if self._closed:
            return
        self._closed = True
        try:
            self.node.publish_stop()
        except Exception:
            pass
        self.executor.shutdown(timeout_sec=2.0)
        self.thread.join(timeout=2.0)
        try:
            self.node.destroy_node()
        finally:
            if self.context.ok():
                self.context.shutdown()


GOAL_MODEL_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="goal_point">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><sphere><radius>0.20</radius></sphere></geometry>
        <material>
          <ambient>0 1 0 1</ambient>
          <diffuse>0 1 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


class MarthaEnv(_GymEnvBase):
    """Synchronous Gymnasium wrapper over Martha's common ROS interface."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        action_mode: str = "continuous",
        render_mode: str | None = None,
        map_mode: str = "random",
        map_index: int | None = None,
        *,
        backend: str = "gazebo",
        worlds_directory: str | Path | None = None,
        world_origins: dict[str, tuple[float, float]] | None = None,
        preloaded_worlds: bool = False,
        scan_topic: str = "/scan",
        odometry_topic: str = "/odometry/filtered",
        goal_topic: str = "/goal_pose",
        cmd_vel_topic: str = "/cmd_vel",
        contact_topic: str = "/contacts",
        fault_topic: str | None = None,
        scan_range_max: float = 8.0,
        min_scan_coverage: float = 0.90,
        odom_frame: str = "odom",
        robot_name: str = "robot",
        service_namespace: str = "",
        robot_z: float = 0.0,
        reset_settle_samples: int = 3,
        max_steps: int = 300,
        sensor_timeout: float = 1.0,
        control_timeout: float = 2.0,
        service_timeout: float = 5.0,
        goal_tolerance: float = 0.25,
        goal_distance_scale: float = DEFAULT_GOAL_DISTANCE_SCALE,
        min_goal_distance: float = 2.0,
        footprint_safety_margin: float = 0.01,
        action_limits: ActionLimits = ActionLimits(),
        reward_config: RewardConfig = RewardConfig(),
        allow_hardware_training: bool = False,
    ) -> None:
        """Configure spaces, maps, ROS topics and backend safety policy."""
        _require_gymnasium()
        _require_ros()
        super().__init__()
        if action_mode != "continuous":
            raise ValueError("MarthaEnv supports only action_mode='continuous'")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if map_mode not in ("random", "predefined"):
            raise ValueError("map_mode must be 'random' or 'predefined'")
        if backend not in ("gazebo", "hardware"):
            raise ValueError("backend must be 'gazebo' or 'hardware'")
        if fault_topic is None:
            fault_topic = fault_topic_for_backend(backend)
        elif not isinstance(fault_topic, str):
            raise TypeError("fault_topic must be a string or None")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool):
            raise TypeError("max_steps must be an integer")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if (
            not isinstance(reset_settle_samples, int)
            or isinstance(reset_settle_samples, bool)
            or reset_settle_samples <= 0
        ):
            raise ValueError("reset_settle_samples must be a positive integer")
        finite_parameters = np.asarray(
            (
                scan_range_max,
                min_scan_coverage,
                robot_z,
                sensor_timeout,
                control_timeout,
                service_timeout,
                goal_tolerance,
                goal_distance_scale,
                min_goal_distance,
                footprint_safety_margin,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(finite_parameters).all():
            raise ValueError("environment numeric parameters must be finite")
        if min(sensor_timeout, control_timeout, service_timeout) <= 0.0:
            raise ValueError("timeouts must be positive")
        if scan_range_max <= 0.0:
            raise ValueError("scan_range_max must be positive")
        if not 0.0 < min_scan_coverage <= 1.0:
            raise ValueError("min_scan_coverage must be in (0, 1]")
        if goal_tolerance <= 0.0 or goal_distance_scale <= 0.0:
            raise ValueError("goal distances must be positive")
        if min_goal_distance < 0.0:
            raise ValueError("min_goal_distance cannot be negative")
        if footprint_safety_margin < 0.0:
            raise ValueError("footprint_safety_margin cannot be negative")
        _validate_action_limits(action_limits)
        validate_reward_config(reward_config)
        if backend == "hardware" and not allow_hardware_training:
            raise RuntimeError(
                "Physical training is disabled by default. Use policy_node for "
                "inference, or pass allow_hardware_training=True only with an "
                "operator, emergency stop and a controlled test area."
            )

        self.render_mode = render_mode
        self.map_mode = map_mode
        self.map_index = map_index
        self.backend = backend
        self.preloaded_worlds = bool(preloaded_worlds)
        self.robot_name = robot_name
        self.robot_z = float(robot_z)
        self.reset_settle_samples = reset_settle_samples
        self.max_steps = int(max_steps)
        self.sensor_timeout = float(sensor_timeout)
        self.control_timeout = float(control_timeout)
        self.service_timeout = float(service_timeout)
        self.goal_tolerance = float(goal_tolerance)
        self.goal_distance_scale = float(goal_distance_scale)
        self.min_goal_distance = float(min_goal_distance)
        self.footprint_safety_margin = float(footprint_safety_margin)
        self.action_limits = action_limits
        self.reward_config = reward_config

        if worlds_directory is None:
            source_worlds = Path(__file__).resolve().parents[2] / "worlds"
            if source_worlds.is_dir():
                worlds_directory = source_worlds
            else:
                try:
                    from ament_index_python.packages import (
                        get_package_share_directory,
                    )

                    worlds_directory = (
                        Path(get_package_share_directory("martha")) / "worlds"
                    )
                except Exception as exc:
                    raise FileNotFoundError(
                        "Could not locate Martha's installed worlds directory"
                    ) from exc
        self.world_paths = discover_worlds(worlds_directory)
        if backend == "gazebo" and not self.world_paths:
            raise FileNotFoundError(
                f"No canonical training worlds found in {Path(worlds_directory)}"
            )
        local_maps = tuple(WorldMap.from_sdf(path) for path in self.world_paths)
        if world_origins is None:
            self.predefined_maps = local_maps
        else:
            missing_origins = [
                world.world_name
                for world in local_maps
                if world.world_name not in world_origins
            ]
            if missing_origins:
                raise ValueError(f"missing world origins: {missing_origins}")
            self.predefined_maps = tuple(
                world.translated(*world_origins[world.world_name])
                for world in local_maps
            )
        if map_index is not None and not 0 <= map_index < len(self.predefined_maps):
            raise IndexError(
                f"map_index {map_index} outside [0, {len(self.predefined_maps) - 1}]"
            )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_SIZE,),
            dtype=np.float32,
        )
        frame_low = np.full(OBSERVATION_FRAME_SIZE, -1.0, dtype=np.float32)
        frame_low[:LASER_SECTORS] = 0.0
        frame_low[LASER_SECTORS] = 0.0
        observation_low = np.tile(frame_low, OBSERVATION_HISTORY_FRAMES)
        observation_high = np.ones(OBSERVATION_SIZE, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )

        self._runtime = _RosRuntime(
            node_name=f"martha_ppo_env_{robot_name}",
            scan_topic=scan_topic,
            odometry_topic=odometry_topic,
            goal_topic=goal_topic,
            cmd_vel_topic=cmd_vel_topic,
            contact_topic=(contact_topic if backend == "gazebo" else ""),
            fault_topic=fault_topic,
            scan_range_max=scan_range_max,
            min_scan_coverage=min_scan_coverage,
            odom_frame=odom_frame,
            robot_name=robot_name,
            service_namespace=service_namespace,
            enable_gazebo_services=backend == "gazebo",
            accept_external_goal=backend != "gazebo",
            use_sim_time=backend == "gazebo",
        )
        self.ros = self._runtime.node
        self.policy_contract = build_policy_contract(
            self.ros.scan_range_max,
            self.goal_distance_scale,
            self.action_limits,
        )
        self._closed = False
        self._active_world_index: int | None = None
        self._scenario_entity_names: set[str] = set()
        self._goal_marker_name: str | None = None
        self._world_map: WorldMap | None = None
        self._distance_field: np.ndarray | None = None
        self._episode_sample: EpisodeSample | None = None
        self._step_count = 0
        self._previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._observation_history = ObservationHistory()
        self._reward_state: RewardState | None = None
        self._stagnation_reference_distance: float | None = None
        self._stagnation_steps = 0
        self._last_observation: np.ndarray | None = None
        self._last_snapshot: SensorSnapshot | None = None

    def _call_empty(self, key: str) -> None:
        attempts = (
            GAZEBO_CONTROL_SERVICE_ATTEMPTS
            if key in {"pause", "unpause"}
            else 1
        )
        for attempt in range(1, attempts + 1):
            try:
                self.ros.call_service(
                    key,
                    Empty.Request(),
                    self.service_timeout,
                )
                return
            except TimeoutError:
                if key == "pause":
                    self.ros.publish_stop()
                if attempt == attempts:
                    raise

    def _delete_entity(self, name: str, *, ignore_failure: bool = False) -> None:
        request = DeleteEntity.Request()
        request.name = name
        response = self.ros.call_service(
            "delete_entity",
            request,
            self.service_timeout,
        )
        if not response.success and not ignore_failure:
            raise RuntimeError(
                f"Could not delete Gazebo entity {name}: {response.status_message}"
            )

    @staticmethod
    def _spawn_payload(model_xml: str) -> tuple[str, Any]:
        model = ET.fromstring(model_xml)
        pose_values = [
            float(value)
            for value in (model.findtext("pose") or "0 0 0 0 0 0").split()
        ]
        pose_values.extend([0.0] * (6 - len(pose_values)))
        pose_element = model.find("pose")
        if pose_element is not None:
            model.remove(pose_element)
        wrapped_xml = (
            '<?xml version="1.0"?><sdf version="1.6">'
            + ET.tostring(model, encoding="unicode")
            + "</sdf>"
        )
        pose = Pose()
        pose.position.x = pose_values[0]
        pose.position.y = pose_values[1]
        pose.position.z = pose_values[2]
        quaternion = _quaternion_from_rpy(
            pose_values[3],
            pose_values[4],
            pose_values[5],
        )
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        return wrapped_xml, pose

    def _spawn_entity(self, name: str, xml: str, pose: Any) -> None:
        request = SpawnEntity.Request()
        request.name = name
        request.xml = xml
        request.robot_namespace = ""
        request.initial_pose = pose
        request.reference_frame = "world"
        response = self.ros.call_service(
            "spawn_entity",
            request,
            self.service_timeout,
        )
        if not response.success:
            raise RuntimeError(
                f"Could not spawn Gazebo entity {name}: {response.status_message}"
            )

    def _delete_entities_and_wait(self, names: set[str]) -> None:
        """Delete entities and confirm their absence before names are reused."""
        if not names:
            return
        inventory_sequence, _ = self.ros.gazebo_model_inventory()
        for name in sorted(names):
            self._delete_entity(name, ignore_failure=True)

        # Gazebo publishes model-state changes while physics is active. This
        # barrier also gives the GUI time to consume each deletion before a
        # replacement with the same stable name is created.
        self.ros.publish_stop()
        self._call_empty("unpause")
        try:
            remaining = self.ros.wait_for_gazebo_entities_absent(
                names,
                inventory_sequence,
                self.control_timeout,
            )
        finally:
            self._call_empty("pause")
        if remaining:
            formatted_names = ", ".join(sorted(remaining))
            raise RuntimeError(
                "Gazebo did not remove the previous PPO entities: "
                f"{formatted_names}"
            )

    def _switch_world(self, index: int) -> None:
        if self.preloaded_worlds:
            self._active_world_index = index
            return
        # Gazebo is shared external state: another process may have changed the
        # scenario even when this local environment cached the same index.
        source_names = {
            name
            for world_map in self.predefined_maps
            for name in world_map.scenario_model_names
        }
        names_to_delete = source_names | self._scenario_entity_names
        names_to_delete.update(
            name
            for name in self.ros.gazebo_model_names()
            if name.startswith(PPO_SCENARIO_ENTITY_PREFIX)
            or name.startswith(PPO_GOAL_ENTITY_PREFIX)
        )
        names_to_delete.add("goal_point")
        if self._goal_marker_name is not None:
            names_to_delete.add(self._goal_marker_name)
        self._delete_entities_and_wait(names_to_delete)

        self._scenario_entity_names = set()
        self._goal_marker_name = None
        world_map = self.predefined_maps[index]
        for name in world_map.scenario_model_names:
            xml, pose = self._spawn_payload(world_map.model_xml[name])
            runtime_name = f"{PPO_SCENARIO_ENTITY_PREFIX}_{name}"
            self._spawn_entity(runtime_name, xml, pose)
            self._scenario_entity_names.add(runtime_name)
        self._active_world_index = index

    def _cleanup_managed_gazebo_entities(self) -> None:
        """Best-effort removal of every scenario entity owned by PPO."""
        names = set(self._scenario_entity_names)
        if not self.preloaded_worlds:
            names.update(
                name
                for name in self.ros.gazebo_model_names()
                if name.startswith(PPO_SCENARIO_ENTITY_PREFIX)
            )
        if self._goal_marker_name is not None:
            names.add(self._goal_marker_name)
        for name in sorted(names):
            self._delete_entity(name, ignore_failure=True)
        self._scenario_entity_names.clear()
        self._goal_marker_name = None

    def _reset_world_scenario(self, index: int) -> None:
        """Reset dynamics before restoring the selected scenario geometry."""
        if self.preloaded_worlds:
            self._active_world_index = index
            return
        # Gazebo remembers the initial poses of models by name.  Calling
        # reset_world after replacing (for example) wall_left can therefore
        # move that newly spawned wall back to the pose from the launch world.
        # Always reset first and replace the scenario models afterwards so the
        # live geometry stays aligned with the WorldMap used by PPO.
        self._call_empty("reset_world")
        self._switch_world(index)

    def _ensure_gazebo_model_inventory(self) -> None:
        """Obtain model names even when a previous process left Gazebo paused."""
        if self.ros.gazebo_model_names():
            return
        self.ros.publish_stop()
        self._call_empty("unpause")
        try:
            names = self.ros.wait_for_gazebo_model_names(self.control_timeout)
        finally:
            self._call_empty("pause")
        if not names:
            raise TimeoutError("Gazebo did not publish its model inventory")

    def _spawn_goal_marker(self, x: float, y: float) -> None:
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = 0.20
        pose.orientation.w = 1.0
        marker_name = f"{PPO_GOAL_ENTITY_PREFIX}{self.robot_name}"
        self._spawn_entity(marker_name, GOAL_MODEL_SDF, pose)
        self._goal_marker_name = marker_name

    def _set_robot_state(self, x: float, y: float, yaw: float) -> None:
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = self.robot_name
        request.state.reference_frame = "world"
        request.state.pose.position.x = float(x)
        request.state.pose.position.y = float(y)
        request.state.pose.position.z = self.robot_z
        quaternion = _quaternion_from_rpy(0.0, 0.0, yaw)
        request.state.pose.orientation.x = quaternion[0]
        request.state.pose.orientation.y = quaternion[1]
        request.state.pose.orientation.z = quaternion[2]
        request.state.pose.orientation.w = quaternion[3]
        response = self.ros.call_service(
            "set_entity_state",
            request,
            self.service_timeout,
        )
        if not response.success:
            raise RuntimeError("Gazebo rejected the robot reset pose")

    def _reset_filter_pose(self) -> None:
        """Realign the EKF to an episode-local odom origin after teleporting."""
        request = SetPose.Request()
        request.pose.header.stamp = self.ros.get_clock().now().to_msg()
        request.pose.header.frame_id = self.ros.odom_frame
        request.pose.pose.pose.orientation.w = 1.0
        covariance = [0.0] * 36
        for index in (0, 7, 14, 21, 28, 35):
            covariance[index] = 1e-6
        request.pose.pose.covariance = covariance
        self.ros.call_service("set_pose", request, self.service_timeout)

    @staticmethod
    def _world_point_in_episode_odom(
        point_x: float,
        point_y: float,
        sample: EpisodeSample,
    ) -> tuple[float, float]:
        """Transform a world point into the zeroed episode odom frame."""
        delta_x = point_x - sample.start_x
        delta_y = point_y - sample.start_y
        cosine = math.cos(sample.start_yaw)
        sine = math.sin(sample.start_yaw)
        return (
            cosine * delta_x + sine * delta_y,
            -sine * delta_x + cosine * delta_y,
        )

    def _world_index_for_reset(self, options: dict[str, Any]) -> int:
        if "world_index" in options:
            index = int(options["world_index"])
        elif self.map_index is not None:
            index = self.map_index
        elif self.map_mode == "random":
            index = int(self.np_random.integers(len(self.predefined_maps)))
        else:
            index = 0 if self._active_world_index is None else (
                self._active_world_index + 1
            ) % len(self.predefined_maps)
        if not 0 <= index < len(self.predefined_maps):
            raise IndexError(
                f"world_index {index} outside [0, {len(self.predefined_maps) - 1}]"
            )
        return index

    def _episode_for_reset(
        self,
        world_map: WorldMap,
        options: dict[str, Any],
    ) -> EpisodeSample:
        has_start = "start" in options
        has_goal = "goal" in options
        if has_start != has_goal:
            raise ValueError("Gazebo reset options must provide start and goal together")
        if not has_start:
            return world_map.sample_episode(
                self.np_random,
                min_goal_distance=self.min_goal_distance,
            )

        start_values = tuple(float(value) for value in options["start"])
        goal_values = tuple(float(value) for value in options["goal"])
        if len(start_values) not in (2, 3) or len(goal_values) != 2:
            raise ValueError("start=(x,y[,yaw]) and goal=(x,y) are required")
        if (
            not np.isfinite(start_values).all()
            or not np.isfinite(goal_values).all()
        ):
            raise ValueError("Gazebo start and goal coordinates must be finite")
        start_yaw = start_values[2] if len(start_values) == 3 else 0.0
        distance_field = world_map.distance_field(goal_values[0], goal_values[1])
        shortest = world_map.path_distance(
            start_values[0],
            start_values[1],
            distance_field,
        )
        if not math.isfinite(shortest):
            raise ValueError("requested start and goal are not connected free poses")
        return EpisodeSample(
            start_x=start_values[0],
            start_y=start_values[1],
            start_yaw=start_yaw,
            goal_x=goal_values[0],
            goal_y=goal_values[1],
            shortest_path=shortest,
        )

    def _build_observation(self, snapshot: SensorSnapshot) -> tuple[np.ndarray, float]:
        # The policy consumes the goal directly: normalized distance plus the
        # sine and cosine of its bearing.  No global planner is involved -- the
        # recurrent policy must learn to route around dead ends on its own.
        goal, euclidean_distance, _ = goal_features(
            snapshot.x,
            snapshot.y,
            snapshot.yaw,
            snapshot.goal_x,
            snapshot.goal_y,
            self.goal_distance_scale,
        )
        frame = build_observation_frame(
            snapshot.laser_sectors,
            goal,
            snapshot.velocity,
            max(self.action_limits.max_vx, self.action_limits.max_vy),
            self.action_limits.max_wz,
        )
        observation = self._observation_history.push(
            frame,
            min(snapshot.scan_stamp_ns, snapshot.odometry_stamp_ns),
        )
        if observation.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(
                f"observation contract changed: expected {OBSERVATION_SIZE}, "
                f"got {observation.shape}"
            )
        return observation, euclidean_distance

    def _goal_bearing(self, snapshot: SensorSnapshot) -> float:
        """Return the policy-frame angle from Martha's front to the goal."""
        _, _, bearing = goal_features(
            snapshot.x,
            snapshot.y,
            snapshot.yaw,
            snapshot.goal_x,
            snapshot.goal_y,
            self.goal_distance_scale,
        )
        return bearing

    def _distance_for_metrics(
        self,
        snapshot: SensorSnapshot,
        euclidean_distance: float,
    ) -> float:
        """Return geodesic Gazebo distance only for route metrics and SPL."""
        if self.backend == "gazebo" and self._world_map is not None:
            assert self._distance_field is not None
            assert snapshot.ground_truth_x is not None
            assert snapshot.ground_truth_y is not None
            distance = self._world_map.path_distance(
                snapshot.ground_truth_x,
                snapshot.ground_truth_y,
                self._distance_field,
            )
            if math.isfinite(distance):
                return distance
            # Do not turn an invalid simulator pose into a shorter metric by
            # silently switching to Euclidean distance.
            return math.inf
        return euclidean_distance

    def _episode_euclidean_distance(
        self,
        snapshot: SensorSnapshot,
        policy_distance: float,
    ) -> float:
        """Use simulator truth only for episode scoring and termination."""
        if self.backend != "gazebo" or self._episode_sample is None:
            return policy_distance
        assert snapshot.ground_truth_x is not None
        assert snapshot.ground_truth_y is not None
        return math.hypot(
            self._episode_sample.goal_x - snapshot.ground_truth_x,
            self._episode_sample.goal_y - snapshot.ground_truth_y,
        )

    def _reset_preloaded_explicit_episode(
        self,
        world_map: WorldMap,
        sample: EpisodeSample,
    ) -> SensorSnapshot:
        """Reset one catalog episode without the legacy multi-attempt settle."""
        goal_x, goal_y = self._world_point_in_episode_odom(
            sample.goal_x,
            sample.goal_y,
            sample,
        )
        self.ros.set_goal(goal_x, goal_y, self.ros.odom_frame)
        self.ros.publish_goal(goal_x, goal_y, self.ros.odom_frame)
        self._reset_filter_pose()
        reset_stamp_ns = self.ros.get_clock().now().nanoseconds
        sequences = self.ros.sequence_numbers()
        contact_sequence = self.ros.clear_contacts()
        snapshot = None
        fresh_contact_message = False
        self._call_empty("unpause")
        try:
            # Two samples give the EKF one complete update after SetPose while
            # all validation still happens in a single physics interval.
            for _ in range(2):
                snapshot = self.ros.wait_for_fresh_snapshot(
                    sequences,
                    self.control_timeout,
                    use_ground_truth=True,
                    after_stamp_ns=reset_stamp_ns,
                )
                if snapshot is None:
                    break
                sequences = (
                    snapshot.scan_sequence,
                    snapshot.odometry_sequence,
                    snapshot.ground_truth_sequence,
                )
            fresh_contact_message = self.ros.wait_for_fresh_contact_message(
                contact_sequence,
                self.control_timeout,
            )
        finally:
            self._call_empty("pause")
        if snapshot is None:
            missing = self.ros.missing_fresh_snapshot_data(
                sequences,
                use_ground_truth=True,
                after_stamp_ns=reset_stamp_ns,
            )
            raise TimeoutError(
                "shared Gazebo reset timed out waiting for fresh data for "
                f"{self.robot_name}: {', '.join(missing) or 'complete snapshot'}"
            )
        if not fresh_contact_message:
            contact_activity = self.ros.stream_activity()
            raise TimeoutError(
                "shared Gazebo reset timed out waiting for fresh contact for "
                f"{self.robot_name}: contact(sequence="
                f"{contact_activity.contact_sequence}, need>{contact_sequence})"
            )

        return self._finalize_preloaded_explicit_episode(
            world_map,
            sample,
            snapshot,
        )

    def validate_shared_reset(
        self,
        world_map: WorldMap,
        sample: EpisodeSample,
        snapshot: SensorSnapshot,
        *,
        distance_field: np.ndarray | None = None,
    ) -> ResetSafetyCheck:
        """Validate one reset observation without mutating episode state."""
        assert snapshot.ground_truth_x is not None
        assert snapshot.ground_truth_y is not None
        assert snapshot.ground_truth_yaw is not None
        truth_x = snapshot.ground_truth_x
        truth_y = snapshot.ground_truth_y
        truth_yaw = snapshot.ground_truth_yaw
        wrapped_yaw = math.atan2(math.sin(snapshot.yaw), math.cos(snapshot.yaw))
        map_free = world_map.is_free_pose(truth_x, truth_y)
        scan_hit = footprint_scan_hit(
            snapshot.laser_sectors,
            self.ros.scan_range_max,
            safety_margin=self.footprint_safety_margin,
        )
        odom_position_error = math.hypot(snapshot.x, snapshot.y)
        odom_yaw_error = abs(wrapped_yaw)
        if distance_field is None:
            distance_field = world_map.distance_field(sample.goal_x, sample.goal_y)
        shortest_path = world_map.path_distance(
            truth_x,
            truth_y,
            distance_field,
        )
        return ResetSafetyCheck(
            map_free=bool(map_free),
            footprint_clear=scan_hit is None,
            odom_xy_error=float(odom_position_error),
            odom_yaw_error=float(odom_yaw_error),
            connected=math.isfinite(shortest_path),
            shortest_path=float(shortest_path),
            truth_x=float(truth_x),
            truth_y=float(truth_y),
            truth_yaw=float(truth_yaw),
            minimum_scan=float(snapshot.minimum_scan),
            scan_hit=scan_hit,
        )

    def _finalize_preloaded_explicit_episode(
        self,
        world_map: WorldMap,
        sample: EpisodeSample,
        snapshot: SensorSnapshot,
        *,
        validation: ResetSafetyCheck | None = None,
    ) -> SensorSnapshot:
        """Validate and commit a settled explicit episode while paused."""
        distance_field = world_map.distance_field(sample.goal_x, sample.goal_y)
        if validation is None:
            validation = self.validate_shared_reset(
                world_map,
                sample,
                snapshot,
                distance_field=distance_field,
            )
        if not validation.safe:
            raise RuntimeError(
                "catalog reset did not settle at a safe connected pose: "
                + validation.diagnostic()
            )
        settled_sample = EpisodeSample(
            start_x=validation.truth_x,
            start_y=validation.truth_y,
            start_yaw=validation.truth_yaw,
            goal_x=sample.goal_x,
            goal_y=sample.goal_y,
            shortest_path=validation.shortest_path,
        )
        final_goal_x, final_goal_y = self._world_point_in_episode_odom(
            sample.goal_x,
            sample.goal_y,
            settled_sample,
        )
        self.ros.set_goal(final_goal_x, final_goal_y, self.ros.odom_frame)
        self.ros.publish_goal(final_goal_x, final_goal_y, self.ros.odom_frame)
        self._world_map = world_map
        self._distance_field = distance_field
        self._episode_sample = settled_sample
        self.ros.clear_contacts()
        return snapshot

    def prepare_shared_reset(
        self,
        *,
        seed: int,
        options: dict[str, Any],
    ) -> PendingGazeboReset:
        """
        Prepare one explicit shared-Gazebo episode without running physics.

        The fleet coordinator owns pause/unpause.  This method deliberately
        performs only operations that are safe while Gazebo is paused: state
        invalidation, deterministic sampling, goal setup, teleport and EKF
        reset.  ``_reset_world_scenario`` is owned by the coordinator and must
        have run once for the complete round before this method is called.
        """
        if self._closed:
            raise RuntimeError("MarthaEnv is closed")
        if self.backend != "gazebo" or not self.preloaded_worlds:
            raise RuntimeError(
                "shared reset preparation requires preloaded Gazebo worlds"
            )
        options = dict(options)
        if "start" not in options or "goal" not in options:
            raise ValueError("shared Gazebo reset requires explicit start and goal")

        self._invalidate_episode_state()
        super().reset(seed=seed)
        index = self._world_index_for_reset(options)
        world_map = self.predefined_maps[index]
        sample = self._episode_for_reset(world_map, options)
        # Every MarthaEnv keeps its own episode metadata even though the
        # coordinator switches the shared preloaded world only once.
        self._active_world_index = index

        self.ros.publish_stop()
        if self._goal_marker_name is not None:
            self._delete_entity(self._goal_marker_name, ignore_failure=True)
            self._goal_marker_name = None
        self._spawn_goal_marker(sample.goal_x, sample.goal_y)
        self._set_robot_state(sample.start_x, sample.start_y, sample.start_yaw)

        goal_x, goal_y = self._world_point_in_episode_odom(
            sample.goal_x,
            sample.goal_y,
            sample,
        )
        self.ros.set_goal(goal_x, goal_y, self.ros.odom_frame)
        self.ros.publish_goal(goal_x, goal_y, self.ros.odom_frame)
        self._reset_filter_pose()
        reset_stamp_ns = self.ros.get_clock().now().nanoseconds
        sequences = self.ros.sequence_numbers()
        contact_sequence = self.ros.clear_contacts()
        return PendingGazeboReset(
            world_map=world_map,
            sample=sample,
            reset_stamp_ns=reset_stamp_ns,
            sequences=sequences,
            contact_sequence=contact_sequence,
        )

    def finish_shared_reset(
        self,
        pending: PendingGazeboReset,
        snapshot: SensorSnapshot,
        *,
        validation: ResetSafetyCheck | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Validate, commit and expose one coordinated reset result."""
        settled = self._finalize_preloaded_explicit_episode(
            pending.world_map,
            pending.sample,
            snapshot,
            validation=validation,
        )
        return self._finish_reset_result(settled)

    def _reset_gazebo(self, options: dict[str, Any]) -> SensorSnapshot:
        index = self._world_index_for_reset(options)
        world_map = self.predefined_maps[index]
        explicit_episode = "start" in options or "goal" in options
        self.ros.publish_stop()
        self._call_empty("pause")
        self._ensure_gazebo_model_inventory()
        if self.preloaded_worlds and self._goal_marker_name is not None:
            self._delete_entity(self._goal_marker_name, ignore_failure=True)
            self._goal_marker_name = None

        # Reset dynamics before every placement attempt.  The selected
        # scenario must be restored after reset_world because Gazebo otherwise
        # reuses the launch world's initial poses for models with the same name.
        last_problem = "unknown reset failure"
        maximum_attempts = (
            1 if explicit_episode else MAX_RANDOM_RESET_ATTEMPTS
        )
        for _ in range(maximum_attempts):
            sample = self._episode_for_reset(world_map, options)
            self._reset_world_scenario(index)
            self.ros.publish_stop()
            self._spawn_goal_marker(sample.goal_x, sample.goal_y)
            self._set_robot_state(sample.start_x, sample.start_y, sample.start_yaw)

            # A temporary goal makes the snapshot complete while the chassis
            # contacts settle. It is replaced using the measured Gazebo pose.
            goal_x, goal_y = self._world_point_in_episode_odom(
                sample.goal_x,
                sample.goal_y,
                sample,
            )
            self.ros.set_goal(goal_x, goal_y, self.ros.odom_frame)
            if self.preloaded_worlds and explicit_episode:
                return self._reset_preloaded_explicit_episode(world_map, sample)
            settle_stamp_ns = self.ros.get_clock().now().nanoseconds
            sequences = self.ros.sequence_numbers()
            contact_sequence = self.ros.clear_contacts()
            self._call_empty("unpause")
            settled_snapshot = None
            fresh_contact_message = False
            try:
                for _ in range(self.reset_settle_samples):
                    settled_snapshot = self.ros.wait_for_fresh_snapshot(
                        sequences,
                        self.control_timeout,
                        use_ground_truth=True,
                        after_stamp_ns=settle_stamp_ns,
                    )
                    if settled_snapshot is None:
                        break
                    sequences = (
                        settled_snapshot.scan_sequence,
                        settled_snapshot.odometry_sequence,
                        settled_snapshot.ground_truth_sequence,
                    )
                fresh_contact_message = self.ros.wait_for_fresh_contact_message(
                    contact_sequence,
                    self.control_timeout,
                )
            finally:
                self._call_empty("pause")
            if settled_snapshot is None:
                self.ros.publish_stop()
                missing = self.ros.missing_fresh_snapshot_data(
                    sequences,
                    use_ground_truth=True,
                    after_stamp_ns=settle_stamp_ns,
                )
                raise TimeoutError(
                    "Gazebo reset timed out waiting for fresh data for "
                    f"{self.robot_name}: "
                    f"{', '.join(missing) or 'complete snapshot'}"
                )
            if not fresh_contact_message:
                contact_activity = self.ros.stream_activity()
                raise TimeoutError(
                    "Gazebo reset timed out waiting for fresh contact for "
                    f"{self.robot_name}: contact(sequence="
                    f"{contact_activity.contact_sequence}, "
                    f"need>{contact_sequence})"
                )

            assert settled_snapshot.ground_truth_x is not None
            assert settled_snapshot.ground_truth_y is not None
            assert settled_snapshot.ground_truth_yaw is not None
            settled_x = settled_snapshot.ground_truth_x
            settled_y = settled_snapshot.ground_truth_y
            settled_yaw = settled_snapshot.ground_truth_yaw
            geometry_safe = world_map.is_free_pose(settled_x, settled_y)
            lidar_safe = not scan_hits_footprint(
                settled_snapshot.laser_sectors,
                self.ros.scan_range_max,
                safety_margin=self.footprint_safety_margin,
            )
            distance_field = world_map.distance_field(sample.goal_x, sample.goal_y)
            shortest_path = world_map.path_distance(
                settled_x,
                settled_y,
                distance_field,
            )
            connected = math.isfinite(shortest_path)
            if not geometry_safe or not lidar_safe or not connected:
                failed_checks = []
                if not geometry_safe:
                    failed_checks.append("map geometry")
                if not lidar_safe:
                    failed_checks.append("LiDAR footprint")
                if not connected:
                    failed_checks.append("goal connectivity")
                last_problem = (
                    "settled pose failed " + ", ".join(failed_checks)
                )
                if explicit_episode:
                    break
                continue

            settled_sample = EpisodeSample(
                start_x=settled_x,
                start_y=settled_y,
                start_yaw=settled_yaw,
                goal_x=sample.goal_x,
                goal_y=sample.goal_y,
                shortest_path=shortest_path,
            )
            goal_x, goal_y = self._world_point_in_episode_odom(
                settled_sample.goal_x,
                settled_sample.goal_y,
                settled_sample,
            )
            self.ros.set_goal(goal_x, goal_y, self.ros.odom_frame)
            self.ros.publish_goal(goal_x, goal_y, self.ros.odom_frame)

            # Align only after contact dynamics settle and require messages
            # produced strictly after the EKF service call.
            self._reset_filter_pose()
            alignment_stamp_ns = self.ros.get_clock().now().nanoseconds
            sequences = self.ros.sequence_numbers()
            self._call_empty("unpause")
            snapshot = None
            try:
                for _ in range(2):
                    snapshot = self.ros.wait_for_fresh_snapshot(
                        sequences,
                        self.control_timeout,
                        use_ground_truth=True,
                        after_stamp_ns=alignment_stamp_ns,
                    )
                    if snapshot is None:
                        break
                    sequences = (
                        snapshot.scan_sequence,
                        snapshot.odometry_sequence,
                        snapshot.ground_truth_sequence,
                    )
            finally:
                self._call_empty("pause")
            if snapshot is None:
                self.ros.publish_stop()
                raise TimeoutError(
                    "EKF alignment did not produce fresh Gazebo observations"
                )

            wrapped_yaw = math.atan2(math.sin(snapshot.yaw), math.cos(snapshot.yaw))
            odom_aligned = (
                math.hypot(snapshot.x, snapshot.y) <= 0.08
                and abs(wrapped_yaw) <= 0.12
            )
            final_pose_safe = (
                snapshot.ground_truth_x is not None
                and snapshot.ground_truth_y is not None
                and world_map.is_free_pose(
                    snapshot.ground_truth_x,
                    snapshot.ground_truth_y,
                )
                and not scan_hits_footprint(
                    snapshot.laser_sectors,
                    self.ros.scan_range_max,
                    safety_margin=self.footprint_safety_margin,
                )
            )
            if not odom_aligned or not final_pose_safe:
                last_problem = (
                    "EKF did not align to a safe, stationary episode origin"
                )
                if explicit_episode:
                    break
                continue

            self._world_map = world_map
            self._distance_field = distance_field
            self._episode_sample = settled_sample
            # A post-placement bumper message has now been observed.  Clear
            # it so only messages emitted after the next action can terminate.
            self.ros.clear_contacts()
            return snapshot

        self.ros.publish_stop()
        raise RuntimeError(
            f"Could not establish a safe Gazebo episode after "
            f"{maximum_attempts} attempt(s): {last_problem}"
        )

    def _reset_hardware(self, options: dict[str, Any]) -> SensorSnapshot:
        self.ros.publish_stop()
        if not bool(options.get("manual_reset", False)):
            raise RuntimeError(
                "Hardware reset requires options={'manual_reset': True, ...} "
                "after an operator has placed the robot safely."
            )
        if "goal" not in options:
            raise RuntimeError(
                "Every hardware reset requires an explicit goal=(x, y); "
                "a goal from a previous episode is never reused."
            )
        goal_values = tuple(float(value) for value in options["goal"])
        if len(goal_values) != 2 or not np.isfinite(goal_values).all():
            raise ValueError("hardware goal must be a finite (x, y)")
        goal_frame = str(options.get("goal_frame", self.ros.odom_frame)).strip()
        if not goal_frame:
            raise ValueError("hardware goal_frame cannot be empty")
        self.ros.set_goal(goal_values[0], goal_values[1], goal_frame)
        self.ros.publish_goal(goal_values[0], goal_values[1], goal_frame)
        reset_stamp_ns = self.ros.get_clock().now().nanoseconds
        sequences = self.ros.sequence_numbers()
        snapshot = self.ros.wait_for_fresh_snapshot(
            sequences,
            self.control_timeout,
            use_ground_truth=False,
            after_stamp_ns=reset_stamp_ns,
        )
        if snapshot is None:
            missing = self.ros.missing_fresh_snapshot_data(
                sequences,
                use_ground_truth=False,
                after_stamp_ns=reset_stamp_ns,
            )
            raise TimeoutError(
                "Hardware reset timed out waiting for fresh data: "
                f"{', '.join(missing) or 'complete snapshot'}"
            )
        if snapshot.motor_fault:
            raise RuntimeError("Cannot reset while /hardware/motor_fault is active")
        self._world_map = None
        self._distance_field = None
        self._episode_sample = None
        return snapshot

    def _invalidate_episode_state(self) -> None:
        """Make ``step`` impossible until the next reset fully succeeds."""
        self._step_count = 0
        self._previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._observation_history.clear()
        self._reward_state = None
        self._stagnation_reference_distance = None
        self._stagnation_steps = 0
        self._last_observation = None
        self._last_snapshot = None
        self._world_map = None
        self._distance_field = None
        self._episode_sample = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start one random or explicitly specified navigation episode."""
        if self._closed:
            raise RuntimeError("MarthaEnv is closed")
        # Invalidate the previous episode before any operation that can fail.
        # A timeout, unsafe pose or hardware fault must never leave step()
        # authorized by an observation from the prior episode.
        self._invalidate_episode_state()
        super().reset(seed=seed)
        options = dict(options or {})
        snapshot = (
            self._reset_gazebo(options)
            if self.backend == "gazebo"
            else self._reset_hardware(options)
        )
        return self._finish_reset_result(snapshot)

    def _finish_reset_result(
        self,
        snapshot: SensorSnapshot,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Initialize observation, reward and metrics from a valid reset."""
        observation, policy_distance = self._build_observation(snapshot)
        euclidean_distance = self._episode_euclidean_distance(
            snapshot,
            policy_distance,
        )
        distance = self._distance_for_metrics(snapshot, euclidean_distance)
        self._reward_state = RewardState.initial(distance)
        self._stagnation_reference_distance = distance
        self._stagnation_steps = 0
        self._last_observation = observation
        self._last_snapshot = snapshot
        info = {
            "backend": self.backend,
            "world_index": self._active_world_index,
            "start": (
                None
                if self._episode_sample is None
                else (
                    self._episode_sample.start_x,
                    self._episode_sample.start_y,
                    self._episode_sample.start_yaw,
                )
            ),
            "goal": (snapshot.goal_x, snapshot.goal_y),
            "position": (snapshot.x, snapshot.y, snapshot.yaw),
            "ground_truth_position": (
                snapshot.ground_truth_x,
                snapshot.ground_truth_y,
                snapshot.ground_truth_yaw,
            ),
            "distance": distance,
            "euclidean_distance": euclidean_distance,
            "policy_goal_distance": policy_distance,
            "shortest_path": (
                None
                if self._episode_sample is None
                else self._episode_sample.shortest_path
            ),
        }
        return observation, info

    def _timeout_transition(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.ros.publish_stop()
        reward = 0.0
        components = empty_reward_components()
        self._previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        if self._last_observation is None:
            raise TimeoutError("No valid observation exists after sensor timeout")
        observation = self._last_observation.copy()
        if self._last_snapshot is not None:
            observation, _ = self._build_observation(self._last_snapshot)
            self._last_observation = observation
        return observation, reward, False, True, {
            "reached_goal": False,
            "collision": False,
            "out_of_bounds": False,
            "stagnated": False,
            "sensor_timeout": True,
            "motor_fault": self.ros.motor_fault,
            "reward_components": components,
        }

    def prepare_step(
        self,
        action: Iterable[float],
    ) -> PendingStep:
        """Publish one action and capture freshness barriers while paused."""
        if self._closed:
            raise RuntimeError("MarthaEnv is closed")
        if self._last_observation is None:
            raise RuntimeError("Call reset() before step()")
        requested_action = sanitize_action(action)
        previous_action = self._previous_action.copy()
        limited_action = limit_action_rate(
            requested_action,
            previous_action,
            self.action_limits.max_action_delta,
        )
        command_inhibited = self.ros.motor_fault
        applied_action = (
            np.zeros(ACTION_SIZE, dtype=np.float32)
            if command_inhibited
            else limited_action
        )
        command = scale_action(applied_action, self.action_limits)
        contact_sequence = self.ros.clear_contacts()
        self.ros.publish_command(command)
        command_stamp_ns = self.ros.get_clock().now().nanoseconds
        sequences = self.ros.sequence_numbers()
        return PendingStep(
            requested_action=requested_action,
            limited_action=limited_action,
            applied_action=applied_action,
            command=command,
            command_inhibited=command_inhibited,
            command_stamp_ns=command_stamp_ns,
            sequences=sequences,
            contact_sequence=contact_sequence,
        )

    def prepare_stop_step(self) -> PendingStep:
        """Publish an immediate stop and account for its following transition."""
        if self._closed:
            raise RuntimeError("MarthaEnv is closed")
        if self._last_observation is None:
            raise RuntimeError("Call reset() before prepare_stop_step()")
        zero = np.zeros(ACTION_SIZE, dtype=np.float32)
        contact_sequence = self.ros.clear_contacts()
        self.ros.publish_stop()
        command_stamp_ns = self.ros.get_clock().now().nanoseconds
        sequences = self.ros.sequence_numbers()
        return PendingStep(
            requested_action=zero.copy(),
            limited_action=zero.copy(),
            applied_action=zero.copy(),
            command=zero.copy(),
            command_inhibited=self.ros.motor_fault,
            command_stamp_ns=command_stamp_ns,
            sequences=sequences,
            contact_sequence=contact_sequence,
        )

    def wait_for_step_snapshot(self, pending: PendingStep) -> SensorSnapshot | None:
        """Wait for the sensor transition associated with a prepared action."""
        return self.ros.wait_for_fresh_snapshot(
            pending.sequences,
            self.sensor_timeout,
            use_ground_truth=self.backend == "gazebo",
            after_stamp_ns=pending.command_stamp_ns,
        )

    def finish_step(
        self,
        pending: PendingStep,
        snapshot: SensorSnapshot | None,
        *,
        contact_collision: bool,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Finish reward and termination logic after shared physics pauses."""
        if snapshot is None:
            return self._timeout_transition()

        self._step_count += 1
        self._previous_action = pending.applied_action
        observation, policy_distance = self._build_observation(snapshot)
        euclidean_distance = self._episode_euclidean_distance(
            snapshot,
            policy_distance,
        )
        world_index = None
        out_of_bounds = False
        if self._world_map is not None:
            assert snapshot.ground_truth_x is not None
            assert snapshot.ground_truth_y is not None
            world_index = self._active_world_index
            truth_x = snapshot.ground_truth_x
            truth_y = snapshot.ground_truth_y
            # The geodesic field is defined only on cells with the configured
            # robot clearance.  Treat entering any inflated obstacle/boundary
            # cell as a terminal unsafe pose, even if Gazebo has not emitted a
            # physical contact yet.
            out_of_bounds = not self._world_map.is_free_pose(
                truth_x,
                truth_y,
            )
        motor_fault = snapshot.motor_fault
        collision = bool(contact_collision or motor_fault)
        reached_goal = euclidean_distance <= self.goal_tolerance
        distance = self._distance_for_metrics(snapshot, euclidean_distance)
        if not math.isfinite(distance):
            out_of_bounds = True
        primary_terminal = bool(reached_goal or collision or out_of_bounds)
        stagnated = False
        if not primary_terminal:
            if self._stagnation_reference_distance is None:
                raise RuntimeError(
                    "stagnation state was not initialized by reset"
                )
            (
                self._stagnation_reference_distance,
                self._stagnation_steps,
                stagnated,
            ) = update_stagnation(
                self._stagnation_reference_distance,
                self._stagnation_steps,
                distance,
                self.reward_config,
            )
        terminated = bool(primary_terminal or stagnated)
        truncated = bool(not terminated and self._step_count >= self.max_steps)
        if self._reward_state is None:
            raise RuntimeError("reward state was not initialized by reset")
        reward, components, self._reward_state = calculate_reward(
            state=self._reward_state,
            distance=distance,
            goal_bearing=self._goal_bearing(snapshot),
            minimum_scan=snapshot.minimum_scan,
            angular_velocity=float(pending.command[2]),
            reached_goal=reached_goal,
            collision=collision,
            out_of_bounds=out_of_bounds,
            timeout=truncated,
            stagnated=stagnated,
            config=self.reward_config,
        )
        if terminated or truncated:
            self.ros.publish_stop()

        self._last_observation = observation
        self._last_snapshot = snapshot
        info = {
            "backend": self.backend,
            "world_index": world_index,
            "step": self._step_count,
            "reached_goal": reached_goal,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "stagnated": stagnated,
            "sensor_timeout": False,
            "motor_fault": motor_fault,
            "distance": distance,
            "euclidean_distance": euclidean_distance,
            "policy_goal_distance": policy_distance,
            "minimum_scan": snapshot.minimum_scan,
            "position": (snapshot.x, snapshot.y, snapshot.yaw),
            "ground_truth_position": (
                snapshot.ground_truth_x,
                snapshot.ground_truth_y,
                snapshot.ground_truth_yaw,
            ),
            "goal": (snapshot.goal_x, snapshot.goal_y),
            "requested_action": pending.requested_action.copy(),
            "limited_action": pending.limited_action.copy(),
            "applied_action": pending.applied_action.copy(),
            "command_inhibited": pending.command_inhibited,
            "command": pending.command.copy(),
            "reward_components": components,
        }
        return observation, reward, terminated, truncated, info

    def step(
        self,
        action: Iterable[float],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one normalized action and await a fresh sensor transition."""
        pending = self.prepare_step(action)
        if self.backend == "gazebo":
            self._call_empty("unpause")
            try:
                snapshot = self.wait_for_step_snapshot(pending)
            finally:
                self._call_empty("pause")
        else:
            snapshot = self.wait_for_step_snapshot(pending)
        contacts = self.ros.consume_contact_pairs(
            after_sequence=pending.contact_sequence,
        )
        collision = bool(external_contact_models(contacts, self.robot_name))
        return self.finish_step(
            pending,
            snapshot,
            contact_collision=collision,
        )

    def render(self) -> None:
        """Delegate visualization to the externally launched Gazebo/RViz."""
        # Gazebo/RViz rendering is owned by their launch processes.
        return None

    def stop(self) -> None:
        """Publish an explicit zero command without closing the environment."""
        if not self._closed:
            self.ros.publish_stop()

    def ground_truth_position(self) -> tuple[float, float] | None:
        """Return the last verified Gazebo position for reset separation."""
        snapshot = self._last_snapshot
        if (
            snapshot is None
            or snapshot.ground_truth_x is None
            or snapshot.ground_truth_y is None
        ):
            return None
        return float(snapshot.ground_truth_x), float(snapshot.ground_truth_y)

    def park(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Retire this robot outside every arena without destroying its stack."""
        if self.backend != "gazebo":
            raise RuntimeError("parking is only available in Gazebo")
        self.ros.publish_stop()
        if self._goal_marker_name is not None:
            self._delete_entity(self._goal_marker_name, ignore_failure=True)
            self._goal_marker_name = None
        self._set_robot_state(x, y, yaw)
        self.ros.clear_goal()
        self.ros.clear_contacts()
        self._invalidate_episode_state()

    def close(self) -> None:
        """Publish zero velocity and release ROS resources idempotently."""
        if self._closed:
            return
        self._closed = True
        try:
            self.ros.publish_stop()
            if self.backend == "gazebo":
                try:
                    self._call_empty("pause")
                    self._cleanup_managed_gazebo_entities()
                except Exception:
                    # Shutdown must still release the ROS context if Gazebo has
                    # already failed or disappeared.
                    pass
        finally:
            self._runtime.close()

    def __enter__(self) -> "MarthaEnv":
        """Return this open environment for context-manager use."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close the environment on context-manager exit."""
        self.close()


__all__ = [
    "ACTION_SIZE",
    "LASER_SECTORS",
    "OBSERVATION_FRAME_SIZE",
    "OBSERVATION_HISTORY_FRAMES",
    "OBSERVATION_HISTORY_SECONDS",
    "OBSERVATION_SIZE",
    "POLICY_CONTRACT_VERSION",
    "MarthaEnv",
    "RosObservationNode",
    "SensorSnapshot",
    "build_policy_contract",
    "scan_hits_footprint",
]
