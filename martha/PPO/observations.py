"""Backend-independent PPO observation construction."""

from __future__ import annotations

from collections import deque
import math
from typing import Iterable

import numpy as np


LASER_SECTORS = 36
GOAL_FEATURE_SIZE = 3
VELOCITY_SIZE = 3
OBSERVATION_FRAME_SIZE = LASER_SECTORS + GOAL_FEATURE_SIZE + VELOCITY_SIZE
OBSERVATION_HISTORY_FRAMES = 4
OBSERVATION_HISTORY_SECONDS = 1.0
OBSERVATION_SIZE = OBSERVATION_FRAME_SIZE * OBSERVATION_HISTORY_FRAMES
HISTORY_AGES_SECONDS = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)
GOAL_DISTANCE_ENCODING = "rational_v1"
DEFAULT_GOAL_DISTANCE_SCALE = 3.0
GOAL_GUIDANCE_MODE = "direct_goal_v1"


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def reduce_laser_scan(
    ranges: Iterable[float],
    range_min: float,
    range_max: float,
    sectors: int = 36,
    *,
    angle_min: float,
    angle_increment: float,
) -> tuple[np.ndarray, float]:
    """
    Min-pool a scan into canonical ``[-pi, pi)`` angular sectors.

    The result does not depend on whether a LiDAR starts at zero, starts at
    ``-pi``, or reports samples clockwise.
    """
    if sectors <= 0:
        raise ValueError("sectors must be positive")
    values = np.asarray(tuple(ranges), dtype=np.float32)
    if values.size == 0:
        raise ValueError("laser scan is empty")
    valid_max = float(range_max) if math.isfinite(range_max) and range_max > 0 else 8.0
    valid_min = max(0.0, float(range_min)) if math.isfinite(range_min) else 0.0
    values = np.nan_to_num(values, nan=valid_max, posinf=valid_max, neginf=valid_min)
    values = np.clip(values, valid_min, valid_max)
    minimum_range = float(np.min(values))

    if not math.isfinite(angle_min) or not math.isfinite(angle_increment):
        raise ValueError("laser scan angular metadata must be finite")
    if abs(angle_increment) < 1e-12:
        raise ValueError("laser scan angle_increment cannot be zero")
    angles = angle_min + np.arange(values.size) * angle_increment
    canonical_angles = (angles + math.pi) % (2.0 * math.pi) - math.pi
    sector_indices = np.floor(
        (canonical_angles + math.pi) * sectors / (2.0 * math.pi)
    ).astype(np.int32)
    sector_indices = np.clip(sector_indices, 0, sectors - 1)
    sector_values = np.full(sectors, valid_max, dtype=np.float32)
    np.minimum.at(sector_values, sector_indices, values)
    return sector_values / valid_max, minimum_range


def goal_features(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    goal_x: float,
    goal_y: float,
    goal_distance_scale: float,
) -> tuple[np.ndarray, float, float]:
    """Encode goal distance without imposing or representing an upper bound."""
    if not math.isfinite(goal_distance_scale) or goal_distance_scale <= 0.0:
        raise ValueError("goal_distance_scale must be positive and finite")
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    distance = math.hypot(dx, dy)
    bearing = normalize_angle(math.atan2(dy, dx) - robot_yaw)
    normalized_distance = distance / (distance + goal_distance_scale)
    features = np.asarray(
        [normalized_distance, math.sin(bearing), math.cos(bearing)],
        dtype=np.float32,
    )
    return features, distance, bearing


def build_observation_frame(
    laser_sectors: np.ndarray,
    goal: np.ndarray,
    velocity: Iterable[float],
    max_linear_speed: float,
    max_angular_speed: float,
) -> np.ndarray:
    """Build one normalized LiDAR, goal and odometry observation frame."""
    laser_sectors = np.asarray(laser_sectors, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    velocity = np.asarray(tuple(velocity), dtype=np.float32)
    if laser_sectors.shape != (LASER_SECTORS,):
        raise ValueError(f"laser_sectors must have shape ({LASER_SECTORS},)")
    if goal.shape != (GOAL_FEATURE_SIZE,):
        raise ValueError(f"goal must have shape ({GOAL_FEATURE_SIZE},)")
    if velocity.shape != (VELOCITY_SIZE,):
        raise ValueError(f"velocity must have shape ({VELOCITY_SIZE},)")
    velocity_scale = np.asarray(
        [max_linear_speed, max_linear_speed, max_angular_speed],
        dtype=np.float32,
    )
    normalized_velocity = np.clip(
        velocity / np.maximum(velocity_scale, 1e-6),
        -1.0,
        1.0,
    )
    frame = np.concatenate(
        [
            laser_sectors,
            goal,
            normalized_velocity,
        ]
    ).astype(np.float32, copy=False)
    if frame.shape != (OBSERVATION_FRAME_SIZE,):
        raise RuntimeError("observation frame has an invalid size")
    if not np.isfinite(frame).all():
        raise FloatingPointError("observation frame contains NaN or infinity")
    return frame


class ObservationHistory:
    """Select four observation frames distributed over one ROS-time second."""

    def __init__(self) -> None:
        self._frames: deque[tuple[int, np.ndarray]] = deque()
        self._window_ns = round(OBSERVATION_HISTORY_SECONDS * 1_000_000_000)

    def clear(self) -> None:
        """Discard every frame from the current episode or navigation goal."""
        self._frames.clear()

    def push(self, frame: np.ndarray, timestamp_ns: int) -> np.ndarray:
        """Store a frame and return the canonical oldest-to-newest stack."""
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (OBSERVATION_FRAME_SIZE,):
            raise ValueError(
                f"frame must have shape ({OBSERVATION_FRAME_SIZE},)"
            )
        if not np.isfinite(frame).all():
            raise FloatingPointError("observation frame contains NaN or infinity")
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns < 0:
            raise ValueError("observation timestamp must be non-negative")

        if self._frames and timestamp_ns < self._frames[-1][0]:
            self.clear()
        if self._frames and timestamp_ns == self._frames[-1][0]:
            self._frames[-1] = (timestamp_ns, frame.copy())
        else:
            self._frames.append((timestamp_ns, frame.copy()))

        oldest_target = timestamp_ns - self._window_ns
        while len(self._frames) > 2 and self._frames[1][0] < oldest_target:
            self._frames.popleft()

        selected = []
        for age_seconds in HISTORY_AGES_SECONDS:
            target_ns = timestamp_ns - round(age_seconds * 1_000_000_000)
            _, closest = min(
                self._frames,
                key=lambda item: (abs(item[0] - target_ns), -item[0]),
            )
            selected.append(closest)
        observation = np.stack(selected).reshape(OBSERVATION_SIZE)
        return observation.astype(np.float32, copy=False)
