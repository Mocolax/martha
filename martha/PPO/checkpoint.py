"""Canonical Martha PPO checkpoint loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import numpy as np
import torch

from .actions import ActionLimits
from .martha_env import (
    ACTION_SIZE,
    LASER_SECTORS,
    OBSERVATION_SIZE,
    POLICY_ARCHITECTURE,
    POLICY_CONTRACT_VERSION,
)
from .observations import (
    OBSERVATION_FRAME_SIZE,
    OBSERVATION_HISTORY_FRAMES,
    OBSERVATION_HISTORY_SECONDS,
)
from .network import ActorCritic


REQUIRED_CONTRACT_FIELDS = (
    "version",
    "observation_size",
    "action_size",
    "laser_sectors",
    "architecture",
    "observation_layout",
    "observation_frame_size",
    "observation_history_frames",
    "observation_history_seconds",
    "scan_range_max",
    "max_goal_distance",
    "action_limits",
)


def choose_device(requested: str) -> torch.device:
    """Resolve an automatic, CPU or CUDA device request."""
    requested = requested.strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load one checkpoint using the supported PyTorch API."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
    loaded = torch.load(resolved, map_location=device, weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint must be a dictionary")
    return loaded


def checkpoint_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return the required canonical policy contract."""
    contract = checkpoint.get("policy_contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint is missing policy_contract")
    missing = [name for name in REQUIRED_CONTRACT_FIELDS if name not in contract]
    if missing:
        raise ValueError(
            "policy_contract is missing: " + ", ".join(missing)
        )
    return contract


def action_limits_from_checkpoint(
    checkpoint: dict[str, Any],
) -> ActionLimits:
    """Read the action scaling exclusively from policy_contract."""
    values = checkpoint_contract(checkpoint)["action_limits"]
    if not isinstance(values, dict):
        raise ValueError("policy_contract action_limits must be a dictionary")
    try:
        limits = ActionLimits(
            max_vx=float(values["max_vx"]),
            max_vy=float(values["max_vy"]),
            max_wz=float(values["max_wz"]),
            max_action_delta=float(values["max_action_delta"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("policy_contract action_limits are invalid") from exc
    if not np.isfinite(limits.as_array()).all() or not math.isfinite(
        limits.max_action_delta
    ) or limits.max_action_delta <= 0.0:
        raise ValueError("policy_contract action_limits must be positive")
    return limits


def validate_policy_contract(
    contract: dict[str, Any],
    expected: dict[str, Any] | None = None,
) -> None:
    """Validate the current observation and physical scaling contract."""
    required_scalars = {
        "version": POLICY_CONTRACT_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "laser_sectors": LASER_SECTORS,
        "observation_frame_size": OBSERVATION_FRAME_SIZE,
        "observation_history_frames": OBSERVATION_HISTORY_FRAMES,
    }
    for key, required in required_scalars.items():
        try:
            actual = int(contract[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"policy_contract has invalid {key}") from exc
        if actual != required:
            raise ValueError(
                f"policy_contract {key} mismatch ({actual} != {required})"
            )

    required_strings = {
        "architecture": POLICY_ARCHITECTURE,
        "observation_layout": "frame_major",
    }
    for key, required in required_strings.items():
        if contract.get(key) != required:
            raise ValueError(
                f"policy_contract {key} mismatch "
                f"({contract.get(key)!r} != {required!r})"
            )

    for key in (
        "observation_history_seconds",
        "scan_range_max",
        "max_goal_distance",
    ):
        try:
            actual = float(contract[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"policy_contract has invalid {key}") from exc
        if not math.isfinite(actual) or actual <= 0.0:
            raise ValueError(f"policy_contract {key} must be positive")
    if not math.isclose(
        float(contract["observation_history_seconds"]),
        OBSERVATION_HISTORY_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("policy_contract observation_history_seconds mismatch")

    action_limits_from_checkpoint({"policy_contract": contract})
    if expected is None:
        return

    for key in REQUIRED_CONTRACT_FIELDS[:-1]:
        actual = contract[key]
        wanted = expected[key]
        if isinstance(wanted, float):
            if not math.isclose(
                float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"policy_contract mismatch: {key}")
        elif actual != wanted:
            raise ValueError(f"policy_contract mismatch: {key}")
    actual_limits = action_limits_from_checkpoint({"policy_contract": contract})
    try:
        expected_limits = ActionLimits(**expected["action_limits"])
    except (TypeError, ValueError) as exc:
        raise ValueError("expected action_limits are invalid") from exc
    if not np.allclose(
        actual_limits.as_array(),
        expected_limits.as_array(),
        rtol=0.0,
        atol=1e-9,
    ) or not math.isclose(
        actual_limits.max_action_delta,
        expected_limits.max_action_delta,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("policy_contract mismatch: action_limits")


def validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_contract: dict[str, Any] | None = None,
    require_optimizer: bool = False,
) -> None:
    """Validate required checkpoint fields without migration or fallbacks."""
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint is missing model_state_dict")
    if require_optimizer and "optimizer_state_dict" not in checkpoint:
        raise KeyError("resume checkpoint is missing optimizer_state_dict")
    validate_policy_contract(
        checkpoint_contract(checkpoint),
        expected=expected_contract,
    )


def load_policy(
    checkpoint_path: Path,
    device: torch.device,
    *,
    expected_contract: dict[str, Any] | None = None,
) -> tuple[ActorCritic, dict[str, Any], ActionLimits]:
    """Load the current network architecture described by a checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, device)
    validate_checkpoint(checkpoint, expected_contract=expected_contract)
    network = ActorCritic().to(device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    return network, checkpoint, action_limits_from_checkpoint(checkpoint)


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Validate and persist one canonical checkpoint."""
    validate_checkpoint(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
