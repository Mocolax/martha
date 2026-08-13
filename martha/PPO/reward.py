"""Reward function shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """All editable reward weights and physical safety distances."""

    # Scale for the Coulomb-like progress potential (1 / distance).
    progress_scale: float = 20.0
    # Avoid an unbounded potential at the goal.  Match the default goal tolerance.
    progress_distance_floor: float = 0.25
    # Cost paid on every environment step, even while making progress.
    step_penalty: float = 0.02
    # Cost for large normalized commands and abrupt command changes.
    action_penalty: float = 0.03
    action_change_penalty: float = 0.05
    # Quadratic obstacle-proximity cost between both distances below.
    clearance_penalty: float = 0.20
    clearance_distance: float = 0.65
    collision_distance: float = 0.30
    # One-time terminal rewards/costs.
    goal_reward: float = 110.0
    collision_penalty: float = 130.0
    out_of_bounds_penalty: float = 100.0


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
        # The potential is inversely proportional to distance.  Its gradient is
        # proportional to 1 / distance**2, as in Coulomb's law: equal advances
        # therefore receive a larger reward when they happen close to the goal.
        previous_effective_distance = max(
            previous_distance,
            config.progress_distance_floor,
        )
        effective_distance = max(distance, config.progress_distance_floor)
        progress = (
            1.0 / effective_distance
            - 1.0 / previous_effective_distance
        )
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
