"""Backend-independent PPO observation construction."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def reduce_laser_scan(
    ranges: Iterable[float],
    range_min: float,
    range_max: float,
    sectors: int = 36,
    angle_min: float | None = None,
    angle_increment: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Min-pool a scan into canonical ``[-pi, pi)`` angular sectors.

    When angular metadata is supplied, the result does not depend on whether a
    LiDAR starts at zero, starts at ``-pi``, or reports samples clockwise.  The
    fallback without metadata preserves the input order for synthetic tests.
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

    if angle_min is not None and angle_increment is not None:
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
    else:
        sector_values = np.empty(sectors, dtype=np.float32)
        boundaries = np.linspace(0, values.size, sectors + 1, dtype=np.int32)
        for index in range(sectors):
            start = int(boundaries[index])
            stop = int(boundaries[index + 1])
            if stop <= start:
                source_index = min(start, values.size - 1)
                sector_values[index] = values[source_index]
            else:
                sector_values[index] = float(np.min(values[start:stop]))
    return sector_values / valid_max, minimum_range


def goal_features(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    goal_x: float,
    goal_y: float,
    max_goal_distance: float,
) -> tuple[np.ndarray, float, float]:
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    distance = math.hypot(dx, dy)
    bearing = normalize_angle(math.atan2(dy, dx) - robot_yaw)
    normalized_distance = min(distance / max(max_goal_distance, 1e-6), 1.0)
    features = np.asarray(
        [normalized_distance, math.sin(bearing), math.cos(bearing)],
        dtype=np.float32,
    )
    return features, distance, bearing


def build_observation(
    laser_sectors: np.ndarray,
    goal: np.ndarray,
    velocity: Iterable[float],
    previous_action: Iterable[float],
    max_linear_speed: float,
    max_angular_speed: float,
) -> np.ndarray:
    velocity = np.asarray(tuple(velocity), dtype=np.float32)
    previous_action = np.asarray(tuple(previous_action), dtype=np.float32)
    if velocity.shape != (3,) or previous_action.shape != (3,):
        raise ValueError("velocity and previous_action must have shape (3,)")
    velocity_scale = np.asarray(
        [max_linear_speed, max_linear_speed, max_angular_speed],
        dtype=np.float32,
    )
    normalized_velocity = np.clip(
        velocity / np.maximum(velocity_scale, 1e-6),
        -1.0,
        1.0,
    )
    observation = np.concatenate(
        [
            np.asarray(laser_sectors, dtype=np.float32),
            np.asarray(goal, dtype=np.float32),
            normalized_velocity,
            np.clip(previous_action, -1.0, 1.0),
        ]
    ).astype(np.float32, copy=False)
    if not np.isfinite(observation).all():
        raise FloatingPointError("observation contains NaN or infinity")
    return observation
