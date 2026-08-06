"""Shared conversion from normalized policy actions to robot velocities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionLimits:
    """Velocity and normalized slew limits used in simulation and hardware."""

    max_vx: float = 0.35
    max_vy: float = 0.35
    max_wz: float = 0.80
    max_action_delta: float = 0.35

    def as_array(self) -> np.ndarray:
        all_values = np.asarray(
            [
                self.max_vx,
                self.max_vy,
                self.max_wz,
                self.max_action_delta,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(all_values).all():
            raise ValueError("all action limits must be finite")
        if np.any(all_values <= 0.0):
            raise ValueError("all action limits must be positive")
        return np.asarray(
            [self.max_vx, self.max_vy, self.max_wz],
            dtype=np.float32,
        )


def sanitize_action(action) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (3,):
        raise ValueError(f"expected action shape (3,), got {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError("action contains NaN or infinity")
    return np.clip(values, -1.0, 1.0)


def limit_action_rate(
    action,
    previous_action,
    max_delta: float,
) -> np.ndarray:
    """Apply the same normalized acceleration limit to both backends."""
    action = sanitize_action(action)
    previous_action = sanitize_action(previous_action)
    if not np.isfinite(max_delta) or max_delta <= 0.0:
        raise ValueError("max_delta must be finite and positive")
    delta = np.clip(action - previous_action, -max_delta, max_delta)
    return np.clip(previous_action + delta, -1.0, 1.0).astype(np.float32)


def scale_action(action, limits: ActionLimits = ActionLimits()) -> np.ndarray:
    """Convert normalized ``[vx, vy, wz]`` into SI units."""
    return sanitize_action(action) * limits.as_array()
