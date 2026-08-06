"""Reward function shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    progress_scale: float = 30.0
    step_penalty: float = 0.05
    action_penalty: float = 0.02
    action_change_penalty: float = 0.04
    clearance_penalty: float = 0.60
    clearance_distance: float = 0.65
    collision_distance: float = 0.30
    goal_reward: float = 120.0
    collision_penalty: float = 120.0
    out_of_bounds_penalty: float = 120.0


def calculate_reward(
    previous_distance: float,
    distance: float,
    minimum_scan: float,
    action: Iterable[float],
    previous_action: Iterable[float],
    reached_goal: bool,
    collision: bool,
    out_of_bounds: bool,
    config: RewardConfig = RewardConfig(),
) -> tuple[float, dict[str, float]]:
    """Calculate potential-based progress plus safety and efficiency costs."""
    action = np.asarray(tuple(action), dtype=np.float32)
    previous_action = np.asarray(tuple(previous_action), dtype=np.float32)
    if action.shape != (3,) or previous_action.shape != (3,):
        raise ValueError("action and previous_action must have shape (3,)")

    if math.isfinite(previous_distance) and math.isfinite(distance):
        progress = previous_distance - distance
    else:
        progress = 0.0
    progress_reward = config.progress_scale * progress
    effort_cost = config.action_penalty * float(np.mean(np.square(action)))
    smoothness_cost = config.action_change_penalty * float(
        np.mean(np.square(action - previous_action))
    )
    clearance_width = max(
        config.clearance_distance - config.collision_distance,
        1e-6,
    )
    proximity = np.clip(
        (config.clearance_distance - minimum_scan) / clearance_width,
        0.0,
        1.0,
    )
    clearance_cost = config.clearance_penalty * float(proximity * proximity)

    components = {
        "progress": float(progress_reward),
        "step": -float(config.step_penalty),
        "action": -float(effort_cost),
        "action_change": -float(smoothness_cost),
        "clearance": -float(clearance_cost),
        "terminal": 0.0,
    }
    # Safety failures take precedence if noisy data reports two terminal events.
    if collision:
        components["terminal"] = -float(config.collision_penalty)
    elif out_of_bounds:
        components["terminal"] = -float(config.out_of_bounds_penalty)
    elif reached_goal:
        components["terminal"] = float(config.goal_reward)

    reward = float(sum(components.values()))
    if not math.isfinite(reward):
        raise FloatingPointError("reward contains NaN or infinity")
    return reward, components
