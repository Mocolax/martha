"""Shared deterministic episode metrics for training and evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def episode_seed(base_seed: int, *coordinates: int) -> int:
    """Derive a repeatable seed from an episode/map coordinate tuple."""
    value = int(base_seed) & 0xFFFFFFFF
    for coordinate in coordinates:
        value ^= (int(coordinate) + 0x9E3779B9) & 0xFFFFFFFF
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    return value


def assert_finite(name: str, value: Any) -> None:
    """Reject non-finite policy outputs and optimizer statistics."""
    if torch.is_tensor(value):
        finite = bool(torch.isfinite(value).all().item())
    else:
        finite = bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
    if not finite:
        raise FloatingPointError(f"{name} contains NaN or infinity")


def calculate_spl(
    success: bool,
    shortest_path: float | None,
    path_length: float,
) -> float:
    """Calculate Success weighted by Path Length for one episode."""
    if not success:
        return 0.0
    if shortest_path is None or not math.isfinite(float(shortest_path)):
        return math.nan
    reference = max(0.0, float(shortest_path))
    travelled = max(0.0, float(path_length))
    if reference == 0.0:
        return 1.0
    return reference / max(reference, travelled)


def evaluation_score(metrics: dict[str, float]) -> tuple[float, ...]:
    """Rank policies by success, SPL, safety and then mean reward."""
    success = float(metrics.get("eval_success_rate", -math.inf))
    spl = float(metrics.get("eval_mean_spl", -math.inf))
    collision = float(metrics.get("eval_collision_rate", math.inf))
    reward = float(metrics.get("eval_mean_reward", -math.inf))
    if not math.isfinite(success):
        success = -math.inf
    if not math.isfinite(spl):
        spl = -math.inf
    if not math.isfinite(collision):
        collision = math.inf
    if not math.isfinite(reward):
        reward = -math.inf
    return success, spl, -collision, reward


def episode_path_state(
    reset_info: dict[str, Any],
) -> tuple[float | None, tuple[float, float] | None, float]:
    """Initialize the shortest-path reference and odometry accumulator."""
    shortest_path = reset_info.get("shortest_path")
    if shortest_path is None:
        shortest_path = reset_info.get("euclidean_distance")
    start = reset_info.get("position")
    previous_position = None
    if start is not None and len(start) >= 2:
        previous_position = (float(start[0]), float(start[1]))
    return shortest_path, previous_position, 0.0


def advance_path(
    previous_position: tuple[float, float] | None,
    path_length: float,
    info: dict[str, Any],
) -> tuple[tuple[float, float] | None, float]:
    """Accumulate planar odometry distance from transition information."""
    position = info.get("position")
    if position is None or len(position) < 2:
        return previous_position, path_length
    current = (float(position[0]), float(position[1]))
    if previous_position is not None:
        path_length += math.hypot(
            current[0] - previous_position[0],
            current[1] - previous_position[1],
        )
    return current, path_length
