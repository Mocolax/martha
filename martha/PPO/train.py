"""Train Martha's continuous PPO navigation policy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch

from .actions import ActionLimits
from .buffer import RolloutBuffer
from .checkpoint import (
    action_limits_from_checkpoint,
    checkpoint_contract,
    choose_device,
    load_checkpoint as load_checkpoint_file,
    save_checkpoint as save_checkpoint_file,
    validate_checkpoint as validate_checkpoint_data,
)
from .evaluation_core import (
    advance_path,
    assert_finite,
    calculate_spl,
    episode_path_state,
    episode_seed,
    evaluation_score,
)
from .logic import PPOLogic
from .martha_env import MarthaEnv
from .network import ActorCritic
from .observations import DEFAULT_GOAL_DISTANCE_SCALE
from .reward import REWARD_COMPONENT_NAMES, RewardConfig
from .training_layout import (
    WORLD_ORIGINS,
    create_combined_training_world,
    shuffled_world_index,
)
from .world_map import discover_worlds
from martha.simulation_speed import (
    validate_physics_step_size,
    validate_sim_speed_factor,
)


# Edit this block for the normal training setup. Every value can still be
# overridden temporarily from the command line without changing this file.
@dataclass(frozen=True)
class TrainingDefaults:
    # An upper bound, not a commitment: max_wall_time_hours ends the run
    # first and the best evaluated policy is checkpointed along the way.
    episodes: int = 6000
    sim_speed_factor: float = 5.0
    physics_step_size: float = 0.002
    lidar_samples: int = 180
    training_kinematic: bool = True
    gazebo_gui: bool = False
    gazebo_startup_timeout: float = 240.0
    max_steps: int = 1400
    # One Martha trains alone, rotating through the six arenas of the combined
    # world. She stays on one arena for this many episodes before moving on, so
    # every map gets a fair, reproducible share of experience.
    episodes_per_map: int = 4
    # No curriculum: every episode samples a goal at a fully random geodesic
    # distance (>= min_goal_distance, up to whatever the arena allows).
    max_wall_time_hours: float = 24.0
    rollout_steps: int = 1024
    ppo_epochs: int = 8
    minibatch_size: int = 256
    recurrent_sequence_length: int = 64
    lr: float = 1e-4
    # Linear learning-rate decay to lr_final_fraction over the same
    # update clock as the entropy schedule, so late updates settle
    # instead of overshooting the clip range.
    lr_final_fraction: float = 0.1
    lr_decay_updates: int = 3000
    # Stop refining a rollout once mean KL crosses this trust region.
    target_kl: float = 0.03
    # At 10 Hz a 0.99 discount only reaches 10 s, far short of a 140 s
    # episode: the goal and the timeout penalty both vanish. 0.997
    # extends the effective horizon to roughly 33 s.
    gamma: float = 0.997
    lam: float = 0.95
    eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.002
    # Exploration remains constant first, then entropy encouragement fades to
    # zero so PPO can consolidate a lower variance policy when appropriate.
    # The schedule is measured in completed PPO updates, not episodes: an
    # episode lasts anywhere from a quick collision to a full timeout, so an
    # episode clock runs fastest exactly when the agent is doing worst and
    # needs exploration most.  One update covers rollout_steps environment
    # steps, so this clock tracks the experience actually collected.
    entropy_exploration_updates: int = 200
    entropy_decay_updates: int = 1000
    # Exploration is annealed towards a floor instead of exactly zero, so the
    # learned STD cannot collapse into a deterministic policy that has no way
    # back out of whatever behaviour it settled on.
    entropy_final_fraction: float = 0.30
    policy_std_initial: float = 0.40
    policy_std_final: float = 0.25
    reward_scale: float = 1
    # Resume replays the checkpoint reward by default. Enable this (or pass
    # --override-reward) to instead apply the current RewardConfig() and warn.
    override_reward: bool = False
    max_grad_norm: float = 0.5
    eval_every: int = 100
    eval_episodes: int = 2
    eval_max_steps: int = 1400
    eval_map_count: int = 6
    backend: str = "gazebo"
    map_mode: str = "random"
    map_index: int | None = None
    goal: tuple[float, float] | None = None
    goal_frame: str = "odom"
    goal_tolerance: float = 0.25
    min_goal_distance: float = 2.0
    goal_distance_scale: float = DEFAULT_GOAL_DISTANCE_SCALE
    scan_range_max: float = 8.0
    max_vx: float = 0.5
    max_vy: float = 0.5
    max_wz: float = 0.8
    max_action_delta: float = 0.35
    run_name: str | None = None
    runs_dir: Path = Path.home() / "ros2_ws/src/martha/martha/PPO/ppo_runs"
    resume: Path | None = None
    seed: int = 42
    device: str = "auto"


DEFAULTS = TrainingDefaults()


METRIC_FIELDS = [
    "episode",
    "world_index",
    "shortest_path",
    "episode_reward",
    "episode_scaled_reward",
    "reward_step",
    "reward_distance",
    "reward_orientation",
    "reward_shortest_distance",
    "reward_laser",
    "reward_wiggle",
    "reward_terminal",
    "episode_length",
    "terminated",
    "truncated",
    "reached_goal",
    "collision",
    "near_obstacle",
    "stagnated",
    "spl",
    "elapsed_wall_s",
    "training_steps",
    "training_steps_per_second",
    "physics_wall_s",
    "reset_wall_s",
    "ppo_wall_s",
    "evaluation_wall_s",
    "checkpoint_wall_s",
    "updates",
    "loss",
    "actor_loss",
    "critic_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "policy_std",
    "actor_inactive_relu",
    "critic_inactive_relu",
    "eval_mean_reward",
    "eval_success_rate",
    "eval_collision_rate",
    "eval_mean_spl",
    "best_eval_success_rate",
    "best_eval_collision_rate",
    "best_eval_mean_spl",
    "best_eval_mean_reward",
]

_EMPTY_UPDATE_STATS = PPOLogic.empty_stats()

_EMPTY_BEST_EVAL = {
    "eval_success_rate": -math.inf,
    "eval_mean_spl": -math.inf,
    "eval_collision_rate": math.inf,
    "eval_mean_reward": -math.inf,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal training command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train continuous PPO navigation with MarthaEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=DEFAULTS.resume,
        help="Resume model, optimizer and episode number from a checkpoint.",
    )
    parser.add_argument(
        "--override-reward",
        action="store_true",
        default=DEFAULTS.override_reward,
        help="On --resume, force the current RewardConfig() without the "
        "interactive prompt (for background runs); the critic must re-adapt.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid and unsafe settings before creating an environment."""
    positive_ints = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "episodes_per_map": args.episodes_per_map,
        "rollout_steps": args.rollout_steps,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "recurrent_sequence_length": args.recurrent_sequence_length,
        "eval_max_steps": args.eval_max_steps,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"TrainingDefaults.{name} must be positive")
    if args.entropy_exploration_updates < 0 or args.entropy_decay_updates <= 0:
        raise ValueError(
            "entropy_exploration_updates cannot be negative and "
            "entropy_decay_updates must be positive"
        )
    if not 0.0 <= args.entropy_final_fraction <= 1.0:
        raise ValueError("entropy_final_fraction must be in [0, 1]")
    if not 0.0 <= args.lr_final_fraction <= 1.0:
        raise ValueError("lr_final_fraction must be in [0, 1]")
    if args.lr_decay_updates <= 0:
        raise ValueError("lr_decay_updates must be positive")
    if args.target_kl <= 0.0:
        raise ValueError("target_kl must be positive")
    if (
        args.eval_every < 0
        or args.eval_episodes <= 0
        or args.eval_map_count < 0
    ):
        raise ValueError(
            "TrainingDefaults evaluation intervals must be non-negative and "
            "eval_episodes positive"
        )
    if args.map_index is not None and args.map_index < 0:
        raise ValueError("TrainingDefaults.map_index cannot be negative")
    if args.gazebo_startup_timeout <= 0.0:
        raise ValueError("gazebo_startup_timeout must be positive")
    validate_sim_speed_factor(args.sim_speed_factor)
    validate_physics_step_size(args.physics_step_size)
    if args.lidar_samples < 36 or args.lidar_samples % 36 != 0:
        raise ValueError(
            "TrainingDefaults.lidar_samples must be a positive multiple of 36"
        )
    numeric_values = {
        "lr": args.lr,
        "gamma": args.gamma,
        "lam": args.lam,
        "eps": args.eps,
        "value-coef": args.value_coef,
        "entropy-coef": args.entropy_coef,
        "entropy-final-fraction": args.entropy_final_fraction,
        "policy-std-initial": args.policy_std_initial,
        "policy-std-final": args.policy_std_final,
        "reward-scale": args.reward_scale,
        "max-grad-norm": args.max_grad_norm,
        "goal-tolerance": args.goal_tolerance,
        "min-goal-distance": args.min_goal_distance,
        "goal-distance-scale": args.goal_distance_scale,
        "scan-range-max": args.scan_range_max,
        "max-vx": args.max_vx,
        "max-vy": args.max_vy,
        "max-wz": args.max_wz,
        "max-action-delta": args.max_action_delta,
        "gazebo-startup-timeout": args.gazebo_startup_timeout,
        "physics-step-size": args.physics_step_size,
        "max-wall-time-hours": args.max_wall_time_hours,
    }
    for name, value in numeric_values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"TrainingDefaults.{name} must be finite")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.lam <= 1.0:
        raise ValueError("gamma must be in (0, 1] and lam in [0, 1]")
    positive_names = (
        "lr",
        "eps",
        "max-grad-norm",
        "goal-tolerance",
        "min-goal-distance",
        "goal-distance-scale",
        "scan-range-max",
        "max-vx",
        "max-vy",
        "max-wz",
        "max-action-delta",
        "gazebo-startup-timeout",
        "reward-scale",
        "physics-step-size",
        "max-wall-time-hours",
    )
    if any(numeric_values[name] <= 0.0 for name in positive_names):
        raise ValueError(
            "learning, navigation and sensor limits must be positive"
        )
    if args.value_coef < 0.0 or args.entropy_coef < 0.0:
        raise ValueError(
            "value_coef and entropy_coef cannot be negative"
        )
    if not 0.0 < args.policy_std_final <= args.policy_std_initial:
        raise ValueError(
            "policy STD limits must satisfy 0 < final <= initial"
        )
    if args.backend == "hardware":
        if args.goal is None:
            raise ValueError("hardware training requires TrainingDefaults.goal")
        if not all(math.isfinite(float(value)) for value in args.goal):
            raise ValueError("hardware goal coordinates must be finite")
    elif args.goal is not None:
        raise ValueError("goal is reserved for the hardware backend")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse and validate training arguments."""
    parser = build_parser()
    parsed = parser.parse_args(argv)
    values = asdict(DEFAULTS)
    values["resume"] = parsed.resume
    values["override_reward"] = parsed.override_reward
    args = argparse.Namespace(**values)
    try:
        _validate_args(args)
        _action_limits_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for repeatable initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scale_reward_for_ppo(reward: float, reward_scale: float) -> float:
    """Scale critic targets without changing environment reward reporting."""
    scaled = float(reward) * float(reward_scale)
    if not math.isfinite(scaled):
        raise FloatingPointError("scaled PPO reward is not finite")
    return scaled


def entropy_coefficient_for_update(
    base_coefficient: float,
    exploration_updates: int,
    decay_updates: int,
    final_fraction: float,
    update: int,
) -> float:
    """
    Return the three-phase entropy coefficient for one PPO update index.

    The clock is completed PPO updates, so it advances with the experience
    actually collected rather than with an episode count whose pace depends
    on how early the agent happens to be crashing.  The coefficient holds at
    its base value, decays linearly, and then settles on a floor that keeps a
    little exploration alive for the rest of the run.
    """
    if base_coefficient < 0.0:
        raise ValueError("base entropy coefficient cannot be negative")
    if decay_updates <= 0:
        raise ValueError("decay_updates must be positive")
    if not 0.0 <= final_fraction <= 1.0:
        raise ValueError("final_fraction must be in [0, 1]")
    floor = float(base_coefficient) * float(final_fraction)
    update = max(0, int(update))
    if update <= exploration_updates:
        return float(base_coefficient)
    progress = min(
        max((update - exploration_updates) / float(decay_updates), 0.0),
        1.0,
    )
    return float(base_coefficient) + progress * (floor - base_coefficient)


def policy_std_ceiling_for_update(
    initial_std: float,
    final_std: float,
    exploration_updates: int,
    decay_updates: int,
    update: int,
) -> float:
    """Return the learned-STD ceiling on the same PPO update clock."""
    if decay_updates <= 0:
        raise ValueError("decay_updates must be positive")
    update = max(0, int(update))
    progress = min(
        max((update - exploration_updates) / float(decay_updates), 0.0),
        1.0,
    )
    return float(initial_std) + progress * (float(final_std) - float(initial_std))


def apply_entropy_schedule(
    ppo: PPOLogic,
    args: argparse.Namespace,
    update: int,
) -> None:
    """Apply the entropy weight and learned-STD ceiling for one update."""
    ppo.entropy_coef = entropy_coefficient_for_update(
        args.entropy_coef,
        args.entropy_exploration_updates,
        args.entropy_decay_updates,
        args.entropy_final_fraction,
        update,
    )
    ppo.network.clamp_policy_std(
        policy_std_ceiling_for_update(
            args.policy_std_initial,
            args.policy_std_final,
            args.entropy_exploration_updates,
            args.entropy_decay_updates,
            update,
        )
    )
    lr_progress = min(max(int(update) / float(args.lr_decay_updates), 0.0), 1.0)
    ppo.set_learning_rate_fraction(
        1.0 + lr_progress * (args.lr_final_fraction - 1.0)
    )


def _action_limits_from_args(args: argparse.Namespace) -> ActionLimits:
    """Build and validate the common normalized-action scaling contract."""
    limits = ActionLimits(
        max_vx=args.max_vx,
        max_vy=args.max_vy,
        max_wz=args.max_wz,
        max_action_delta=args.max_action_delta,
    )
    if not np.isfinite(limits.as_array()).all() or not math.isfinite(
        limits.max_action_delta
    ) or limits.max_action_delta <= 0.0:
        raise ValueError("action limits must be finite and positive")
    return limits


def _runtime_value(
    checkpoint: dict[str, Any] | None,
    contract_key: str,
    fallback: float,
) -> float:
    """Prefer normalization values saved with a resumed policy."""
    if checkpoint is None:
        return float(fallback)
    return float(checkpoint_contract(checkpoint)[contract_key])


def reward_config_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> RewardConfig:
    """Restore reproducible paper-reward values from a new checkpoint."""
    if checkpoint is None:
        return RewardConfig()
    config = checkpoint.get("config", {})
    saved = config.get("reward_config") if isinstance(config, dict) else None
    if saved is None:
        return RewardConfig()
    if not isinstance(saved, dict):
        raise ValueError("checkpoint reward_config must be a dictionary")
    expected_fields = set(asdict(RewardConfig()))
    saved_fields = set(saved)
    if saved_fields != expected_fields:
        missing = sorted(expected_fields - saved_fields)
        unexpected = sorted(saved_fields - expected_fields)
        raise ValueError(
            "checkpoint reward_config schema mismatch; start a new run "
            f"(missing={missing}, unexpected={unexpected})"
        )
    try:
        return RewardConfig(**saved)
    except TypeError as exc:
        raise ValueError("checkpoint reward_config has invalid fields") from exc


def _resolve_reward_config(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
) -> RewardConfig:
    """Pick this run's reward, honoring an explicit resume override.

    A plain resume replays the checkpoint's saved reward so the restored critic
    and optimizer stay consistent with what they were fitted to. With
    ``--override-reward`` the current ``RewardConfig()`` is applied instead and
    the changed fields are printed, because the loaded critic must re-adapt to
    the new reward and a transient performance dip is expected.
    """
    if checkpoint is None or not getattr(args, "override_reward", False):
        return reward_config_from_checkpoint(checkpoint)
    new_reward = RewardConfig()
    print(
        "WARNING: --override-reward: replacing the checkpoint reward with the "
        "current RewardConfig() defaults. The resumed critic and optimizer were "
        "trained on the old reward; expect a transient dip while they re-adapt.",
        flush=True,
    )
    saved = checkpoint.get("config", {}).get("reward_config")
    if isinstance(saved, dict):
        new_fields = asdict(new_reward)
        changes = [
            f"    {key}: {saved.get(key)} -> {new_fields[key]}"
            for key in sorted(new_fields)
            if saved.get(key) != new_fields[key]
        ]
        if changes:
            print("Reward changes on resume:", flush=True)
            for line in changes:
                print(line, flush=True)
        else:
            print(
                "  (reward values are identical to the checkpoint)", flush=True
            )
    return new_reward


def _maybe_prompt_reward_override(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
) -> None:
    """Ask, when resuming interactively, whether to adopt a changed reward.

    A resume replays the checkpoint reward by default. When the current
    ``RewardConfig()`` differs and no ``--override-reward`` was passed, prompt
    once: a yes flips on the override, a no (or no interactive terminal) keeps
    the saved reward. This keeps the ordinary resume a single ``--resume``.
    """
    if checkpoint is None or getattr(args, "override_reward", False):
        return
    saved = checkpoint.get("config", {}).get("reward_config")
    current = asdict(RewardConfig())
    if not isinstance(saved, dict) or set(saved) != set(current):
        return
    changes = [
        f"    {key}: {saved[key]} -> {current[key]}"
        for key in sorted(current)
        if saved[key] != current[key]
    ]
    if not changes:
        return
    print("El reward actual difiere del guardado en el checkpoint:", flush=True)
    for line in changes:
        print(line, flush=True)
    if not sys.stdin.isatty():
        print(
            "Sin terminal interactiva: se mantiene el reward del checkpoint. "
            "Pasa --override-reward para forzar el reward nuevo.",
            flush=True,
        )
        return
    answer = input(
        "¿Continuar con el reward NUEVO? El crítico deberá re-adaptarse "
        "[y/N]: "
    ).strip().lower()
    if answer in ("y", "yes", "s", "si", "sí"):
        args.override_reward = True
    else:
        print("Manteniendo el reward del checkpoint.", flush=True)


def _warn_legacy_reward_config(checkpoint: dict[str, Any] | None) -> None:
    """Make the intentional paper-reward fallback visible for old runs."""
    if checkpoint is None:
        return
    config = checkpoint.get("config", {})
    if not isinstance(config, dict) or "reward_config" not in config:
        print(
            "WARNING: resumed checkpoint has no reward_config; using the "
            "current paper reward defaults.",
            flush=True,
        )


def make_environment(
    args: argparse.Namespace,
    resume_checkpoint: dict[str, Any] | None,
) -> MarthaEnv:
    """Create the sole training/evaluation environment for this process."""
    return MarthaEnv(**environment_kwargs(args, resume_checkpoint))


def environment_kwargs(
    args: argparse.Namespace,
    resume_checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build identical MarthaEnv settings for local or remote workers."""
    action_limits = (
        action_limits_from_checkpoint(resume_checkpoint)
        if resume_checkpoint is not None
        else _action_limits_from_args(args)
    )
    reward_config = _resolve_reward_config(args, resume_checkpoint)
    # On Gazebo the six arenas live at once inside the combined world, each
    # shifted to its WORLD_ORIGINS offset. Passing the origins moves every
    # map's free-space definition to the same offset, so the goals we sample
    # and teleport to line up with the geometry Gazebo actually renders.
    world_origins = None if args.backend == "hardware" else WORLD_ORIGINS
    return {
        "action_mode": "continuous",
        "render_mode": None,
        "map_mode": args.map_mode,
        "map_index": args.map_index,
        "backend": args.backend,
        "world_origins": world_origins,
        "scan_range_max": _runtime_value(
            resume_checkpoint,
            "scan_range_max",
            args.scan_range_max,
        ),
        "max_steps": args.max_steps,
        "goal_tolerance": args.goal_tolerance,
        "goal_distance_scale": _runtime_value(
            resume_checkpoint,
            "goal_distance_scale",
            args.goal_distance_scale,
        ),
        "min_goal_distance": args.min_goal_distance,
        "action_limits": action_limits,
        "reward_config": reward_config,
        "allow_hardware_training": args.backend == "hardware",
    }


def _package_asset_path(relative_path: str) -> Path:
    """Resolve one source-tree or installed package data file."""
    source_path = Path(__file__).resolve().parents[2] / relative_path
    if source_path.exists():
        return source_path.resolve()
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = Path(get_package_share_directory("martha")) / relative_path
    except Exception as exc:
        raise FileNotFoundError(f"could not locate package asset {relative_path}") from exc
    if not installed.exists():
        raise FileNotFoundError(f"package asset does not exist: {installed}")
    return installed.resolve()


def build_agent(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ActorCritic, PPOLogic]:
    """Build the actor-critic network and PPO optimizer."""
    network = ActorCritic().to(device)
    ppo = PPOLogic(
        network=network,
        lr=args.lr,
        eps=args.eps,
        gamma=args.gamma,
        lam=args.lam,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        recurrent_sequence_length=args.recurrent_sequence_length,
        target_kl=args.target_kl,
    )
    return network, ppo


def _validate_resume_reward_scale(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
) -> None:
    """Prevent a resumed critic from changing its target scale silently."""
    if checkpoint is None:
        return
    try:
        saved_scale = float(checkpoint.get("config", {})["reward_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "resume checkpoint has no valid config.reward_scale"
        ) from exc
    if not math.isfinite(saved_scale) or saved_scale <= 0.0:
        raise ValueError("resume checkpoint reward_scale must be positive")
    if not math.isclose(
        float(args.reward_scale),
        saved_scale,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        if getattr(args, "override_reward", False):
            print(
                "WARNING: --override-reward: reward_scale "
                f"{saved_scale} -> {args.reward_scale}; the critic's target "
                "scale changes, so its restored values are stale.",
                flush=True,
            )
            return
        raise ValueError(
            "TrainingDefaults.reward_scale must match the checkpoint "
            f"({args.reward_scale} != {saved_scale})"
        )


def _serializable_config(
    args: argparse.Namespace,
    env: MarthaEnv,
) -> dict[str, Any]:
    """Build a portable checkpoint configuration dictionary."""
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, tuple):
            config[key] = list(value)
        else:
            config[key] = value
    config.update(
        observation_size=int(env.observation_space.shape[0]),
        action_size=int(env.action_space.shape[0]),
        scan_range_max=float(env.policy_contract["scan_range_max"]),
        goal_distance_scale=float(env.goal_distance_scale),
        max_vx=float(env.action_limits.max_vx),
        max_vy=float(env.action_limits.max_vy),
        max_wz=float(env.action_limits.max_wz),
        max_action_delta=float(env.action_limits.max_action_delta),
        reward_config=asdict(env.reward_config),
    )
    return config


def save_checkpoint(
    path: Path,
    network: ActorCritic,
    ppo: PPOLogic,
    env: MarthaEnv,
    args: argparse.Namespace,
    episode: int,
    best_eval_metrics: dict[str, float],
    updates: int = 0,
) -> None:
    """Save policy, optimizer and the complete runtime contract."""
    checkpoint = {
        "episode": int(episode),
        # The entropy schedule runs on this clock, so a resumed run has to
        # continue annealing instead of restarting exploration.
        "updates": int(updates),
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": ppo.optimizer.state_dict(),
        "best_eval_metrics": dict(best_eval_metrics),
        "best_eval_score": evaluation_score(best_eval_metrics),
        "best_eval_mean_reward": float(
            best_eval_metrics["eval_mean_reward"]
        ),
        "policy_contract": dict(env.policy_contract),
        "config": _serializable_config(args, env),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    save_checkpoint_file(path, checkpoint)


def _cuda_rng_states_on_cpu(states: Any) -> list[torch.Tensor]:
    """Return CUDA generator states in the CPU ByteTensor form PyTorch needs."""
    if not isinstance(states, (list, tuple)):
        raise ValueError("checkpoint CUDA RNG state must be a list of tensors")
    normalized = []
    for state in states:
        if not torch.is_tensor(state) or state.dtype != torch.uint8:
            raise ValueError("checkpoint CUDA RNG states must be ByteTensors")
        # Loading a checkpoint with map_location=cuda also moves these state
        # bytes to CUDA, but set_rng_state_all explicitly requires CPU tensors.
        normalized.append(state.detach().cpu())
    return normalized


def _restore_resume_state(
    checkpoint: dict[str, Any],
    network: ActorCritic,
    ppo: PPOLogic,
) -> tuple[int, dict[str, float], int]:
    """Restore model, optimizer, update counter and generator states."""
    network.load_state_dict(checkpoint["model_state_dict"])
    ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rng_state = checkpoint.get("rng_state", {})
    if isinstance(rng_state, dict):
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(
                _cuda_rng_states_on_cpu(rng_state["cuda"])
            )
    start_episode = int(checkpoint.get("episode", 0)) + 1
    saved_best = checkpoint.get("best_eval_metrics")
    if isinstance(saved_best, dict):
        best_metrics = {
            key: float(saved_best.get(key, default))
            for key, default in _EMPTY_BEST_EVAL.items()
        }
    else:
        best_metrics = dict(_EMPTY_BEST_EVAL)
        best_metrics["eval_mean_reward"] = float(
            checkpoint.get("best_eval_mean_reward", -math.inf)
        )
    saved_score = checkpoint.get("best_eval_score")
    if saved_score is not None:
        try:
            normalized_score = tuple(float(value) for value in saved_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "resume checkpoint has invalid best_eval_score"
            ) from exc
        if normalized_score != evaluation_score(best_metrics):
            raise ValueError(
                "resume checkpoint best_eval_score disagrees with "
                "best_eval_metrics"
            )
    return start_episode, best_metrics, int(checkpoint.get("updates", 0))


def make_run_dir(args: argparse.Namespace) -> Path:
    """Choose a new run directory or continue beside a resume checkpoint."""
    if args.resume is not None and args.run_name is None:
        run_dir = args.resume.expanduser().resolve().parent
    else:
        run_name = args.run_name or (
            "ppo_martha_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        run_dir = args.runs_dir.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_metric(path: Path, row: dict[str, Any]) -> None:
    """Append one complete training/evaluation metric row."""
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="", encoding="utf-8") as source:
            previous_fields = csv.DictReader(source).fieldnames
        if previous_fields != METRIC_FIELDS:
            raise ValueError(
                "metrics.csv does not use the current metric schema"
            )
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _empty_reward_components() -> dict[str, float]:
    """Return per-episode accumulators matching ``RewardConfig`` terms."""
    return {f"reward_{name}": 0.0 for name in REWARD_COMPONENT_NAMES}


def _accumulate_reward_components(
    totals: dict[str, float],
    info: dict[str, Any],
) -> None:
    """Accumulate the environment's reward explanation for one transition."""
    components = info.get("reward_components", {})
    if not isinstance(components, dict):
        return
    for name in REWARD_COMPONENT_NAMES:
        value = float(components.get(name, 0.0))
        if not math.isfinite(value):
            raise FloatingPointError(f"reward component {name} is not finite")
        totals[f"reward_{name}"] += value


def _generate_training_report(metrics_path: Path) -> None:
    """Create plots without allowing reporting problems to lose a model."""
    try:
        from .analytics import generate_training_report

        report_paths = generate_training_report(metrics_path)
    except Exception as exc:
        print(f"WARNING: could not generate training report: {exc}")
        return
    for path in report_paths:
        print(f"Training report: {path}")


def _reset_options(
    args: argparse.Namespace,
    world_index: int | None,
) -> dict[str, Any]:
    """Build backend-specific reset options without changing interfaces."""
    if args.backend == "hardware":
        return {
            "manual_reset": True,
            "goal": tuple(args.goal),
            "goal_frame": args.goal_frame,
        }
    return {} if world_index is None else {"world_index": world_index}


def training_world_index(
    args: argparse.Namespace,
    world_count: int,
    episode: int,
) -> int | None:
    """Pick which arena this episode trains on (reproducible from the seed).

    The lone robot stays on one arena for ``episodes_per_map`` episodes, then
    the next block reshuffles to another arena. There is no curriculum: which
    map comes when is decided only by the seed, so a rerun is reproducible.
    """
    if args.backend == "hardware":
        return None
    if args.map_index is not None:
        return int(args.map_index)
    if world_count <= 0:
        raise ValueError("Gazebo training requires at least one world")
    if episode <= 0:
        raise ValueError("training episode must be positive")
    block_size = int(args.episodes_per_map)
    if block_size <= 0:
        raise ValueError("episodes_per_map must be positive")
    round_index = (episode - 1) // block_size
    return shuffled_world_index(args.seed, round_index, world_count)


def training_reset_options(
    args: argparse.Namespace,
    world_count: int,
    episode: int,
) -> dict[str, Any]:
    """Build the reset request for the configured global training episode."""
    return _reset_options(
        args,
        training_world_index(args, world_count, episode),
    )


def _confirm_hardware_reset(
    args: argparse.Namespace,
    episode_label: str,
) -> None:
    """Require an operator handshake before each physical episode reset."""
    if args.backend != "hardware":
        return
    message = (
        f"HARDWARE {episode_label}: place Martha safely, clear the route, "
        "verify the emergency stop, and confirm goal "
        f"({args.goal[0]:.3f}, {args.goal[1]:.3f}) in {args.goal_frame}."
    )
    print(message)
    try:
        input("Press Enter only when the physical reset is complete: ")
    except EOFError as exc:
        raise RuntimeError(
            "hardware reset confirmation requires an interactive terminal"
        ) from exc


def _evaluation_worlds(
    env: MarthaEnv,
    args: argparse.Namespace,
) -> list[int | None]:
    """Select all configured Gazebo maps or the single hardware backend."""
    if args.backend == "hardware":
        return [None]
    if args.map_index is not None:
        return [args.map_index]
    worlds = list(range(len(env.predefined_maps)))
    count = int(args.eval_map_count)
    if count == 0 or count >= len(worlds):
        return worlds
    return sorted(
        {
            int(round(index))
            for index in np.linspace(0, len(worlds) - 1, count)
        }
    )


def evaluate_policy(
    network: ActorCritic,
    env: MarthaEnv,
    args: argparse.Namespace,
    evaluation_round: int,
) -> dict[str, float]:
    """Evaluate between episodes with the same environment."""
    rewards: list[float] = []
    successes: list[float] = []
    collisions: list[float] = []
    spl_values: list[float] = []
    was_training = network.training
    network.eval()
    try:
        with torch.no_grad():
            evaluation_worlds = _evaluation_worlds(env, args)
            total_scenarios = len(evaluation_worlds) * args.eval_episodes
            scenario_number = 0
            for world_index in evaluation_worlds:
                seed_map_index = -1 if world_index is None else world_index
                for eval_episode in range(args.eval_episodes):
                    scenario_number += 1
                    map_label = "hardware" if world_index is None else str(
                        world_index + 1
                    )
                    print(
                        f"evaluation={evaluation_round} "
                        f"scenario={scenario_number}/{total_scenarios} "
                        f"map={map_label} "
                        f"episode={eval_episode + 1}/{args.eval_episodes}",
                        flush=True,
                    )
                    seed = episode_seed(
                        args.seed,
                        1,
                        evaluation_round,
                        seed_map_index + 1,
                        eval_episode,
                    )
                    _confirm_hardware_reset(
                        args,
                        f"evaluation round {evaluation_round}, "
                        f"episode {eval_episode + 1}",
                    )
                    observation, reset_info = env.reset(
                        seed=seed,
                        options=_reset_options(args, world_index),
                    )
                    recurrent_state = network.initial_recurrent_state(1)
                    episode_start = True
                    shortest, previous_position, path_length = (
                        episode_path_state(reset_info)
                    )
                    total_reward = 0.0
                    terminated = False
                    truncated = False
                    info: dict[str, Any] = {}
                    for step_number in range(1, args.eval_max_steps + 1):
                        action, _, _, recurrent_state = (
                            network.get_actions_recurrent(
                                observation,
                                recurrent_state,
                                episode_starts=[episode_start],
                                deterministic=True,
                            )
                        )
                        action = action.squeeze(0)
                        episode_start = False
                        action_array = action.numpy().astype(np.float32)
                        assert_finite("evaluation action", action_array)
                        observation, reward, terminated, truncated, info = (
                            env.step(action_array)
                        )
                        total_reward += float(reward)
                        previous_position, path_length = advance_path(
                            previous_position,
                            path_length,
                            info,
                        )
                        if step_number == args.eval_max_steps and not (
                            terminated or truncated
                        ):
                            truncated = True
                            env.stop()
                        if terminated or truncated:
                            break
                    success = bool(info.get("reached_goal", False))
                    collision = bool(info.get("collision", False))
                    spl = calculate_spl(success, shortest, path_length)
                    rewards.append(total_reward)
                    successes.append(float(success))
                    collisions.append(float(collision))
                    spl_values.append(spl)
    finally:
        network.train(was_training)

    finite_spl = [value for value in spl_values if math.isfinite(value)]
    return {
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_collision_rate": float(np.mean(collisions)),
        "eval_mean_spl": (
            float(np.mean(finite_spl)) if finite_spl else math.nan
        ),
    }


def _checkpoint_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    """Return metrics, latest-model and best-model paths for a run."""
    return (
        run_dir / "metrics.csv",
        run_dir / "last_model.pt",
        run_dir / "best_model.pt",
    )


def _stop_launch_process(process: "subprocess.Popen[Any] | None") -> None:
    """Shut down the Gazebo launch cleanly (SIGINT, then escalate)."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=15.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


def _launch_training_gazebo(
    args: argparse.Namespace,
    world_path: Path,
    log_path: Path,
) -> tuple["subprocess.Popen[Any]", Any]:
    """Start one headless Gazebo with a single Martha in the given world.

    Unlike the old fleet trainer this owns exactly one gzserver and one robot
    (``robot_count:=1``), with the plain ``/scan`` and ``/cmd_vel`` topics that
    MarthaEnv expects by default. Gazebo's stdout/stderr go to ``log_path``.
    """
    log_stream = log_path.open("a", encoding="utf-8")
    command = [
        "ros2", "launch", "martha", "simulation.launch.py",
        f"gui:={'true' if args.gazebo_gui else 'false'}",
        f"sim_speed_factor:={args.sim_speed_factor}",
        f"physics_step_size:={args.physics_step_size}",
        f"lidar_samples:={args.lidar_samples}",
        "lidar_visualize:=false",
        f"training_kinematic:={'true' if args.training_kinematic else 'false'}",
        f"world:={world_path}",
        "robot_count:=1",
    ]
    process = subprocess.Popen(
        command,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_stream


def _wait_for_gazebo_ready(
    env: MarthaEnv,
    process: "subprocess.Popen[Any]",
    timeout: float,
) -> None:
    """Block until Gazebo has spawned the robot, or fail fast if it died."""
    deadline = time.monotonic() + float(timeout)
    while not env.ros.gazebo_model_names():
        if process.poll() is not None:
            raise RuntimeError(
                "Gazebo exited before it finished starting; see the run's "
                "gazebo.log"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Gazebo did not become ready within {timeout:.0f} s"
            )
        time.sleep(0.25)


def train(args: argparse.Namespace) -> tuple[ActorCritic, PPOLogic]:
    """Train the navigation policy with a single Martha in Gazebo.

    Owns the whole run: launch one headless Gazebo with a single robot in the
    combined six-arena world, drive the PPO episode loop (random goal
    distances, no curriculum), checkpoint the best/last policy, and tear Gazebo
    down on the way out. The ``hardware`` backend reuses the same loop against a
    real robot and never launches Gazebo.
    """
    _validate_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    resume_checkpoint = (
        None
        if args.resume is None
        else load_checkpoint_file(args.resume, device)
    )
    _warn_legacy_reward_config(resume_checkpoint)
    if resume_checkpoint is not None:
        validate_checkpoint_data(resume_checkpoint, require_optimizer=True)
        _maybe_prompt_reward_override(args, resume_checkpoint)
        _validate_resume_reward_scale(args, resume_checkpoint)
    run_dir = make_run_dir(args)
    metrics_path, last_model_path, best_model_path = _checkpoint_paths(run_dir)

    # Bring up one headless Gazebo with a single Martha in the combined world
    # (six arenas at their offsets). Hardware training launches nothing.
    launch_process = None
    log_stream = None
    combined_world = None
    if args.backend == "gazebo":
        combined_world = create_combined_training_world(
            discover_worlds(_package_asset_path("worlds"))
        )
        launch_process, log_stream = _launch_training_gazebo(
            args, combined_world, run_dir / "gazebo.log"
        )

    def _shutdown_gazebo() -> None:
        _stop_launch_process(launch_process)
        if log_stream is not None:
            log_stream.close()
        if combined_world is not None:
            combined_world.unlink(missing_ok=True)

    try:
        env = make_environment(args, resume_checkpoint)
        if args.backend == "gazebo":
            _wait_for_gazebo_ready(
                env, launch_process, args.gazebo_startup_timeout
            )
    except Exception:
        _shutdown_gazebo()
        raise
    try:
        network, ppo = build_agent(args, device)
    except Exception:
        env.close()
        _shutdown_gazebo()
        raise
    buffer = RolloutBuffer()
    start_episode = 1
    completed_updates = 0
    best_eval_metrics = dict(_EMPTY_BEST_EVAL)
    if resume_checkpoint is not None:
        try:
            validate_checkpoint_data(
                resume_checkpoint,
                expected_contract=env.policy_contract,
                require_optimizer=True,
            )
            (
                start_episode,
                best_eval_metrics,
                completed_updates,
            ) = _restore_resume_state(
                resume_checkpoint,
                network,
                ppo,
            )
        except Exception:
            env.close()
            _shutdown_gazebo()
            raise
        if start_episode > args.episodes:
            env.close()
            _shutdown_gazebo()
            raise ValueError(
                "TrainingDefaults.episodes must exceed the resumed episode"
            )

    print(f"Device: {device}")
    print(f"Backend: {args.backend}")
    print(f"Run directory: {run_dir}")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"PPO reward scale: {args.reward_scale}")
    if args.backend == "hardware":
        print(
            "HARDWARE ENABLED: keep the operator at the emergency stop and "
            "manually place the robot safely before every reset."
        )

    last_update_stats = dict(_EMPTY_UPDATE_STATS)
    completed_episode = start_episode - 1
    # Stop starting new episodes past this wall-clock budget; the best and last
    # policies are already checkpointed, so a run can be capped safely.
    training_started = time.monotonic()
    wall_time_budget = args.max_wall_time_hours * 3600.0
    try:
        for episode in range(start_episode, args.episodes + 1):
            if time.monotonic() - training_started >= wall_time_budget:
                print(
                    f"Wall-time budget of {args.max_wall_time_hours:.2f} h "
                    f"reached at episode {episode}; stopping.",
                    flush=True,
                )
                break
            train_seed = episode_seed(args.seed, 0, episode)
            _confirm_hardware_reset(args, f"training episode {episode}")
            observation, reset_info = env.reset(
                seed=train_seed,
                options=training_reset_options(
                    args,
                    len(env.predefined_maps),
                    episode,
                ),
            )
            shortest, previous_position, path_length = episode_path_state(
                reset_info
            )
            episode_reward = 0.0
            episode_scaled_reward = 0.0
            episode_reward_components = _empty_reward_components()
            episode_length = 0
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            recurrent_state = network.initial_recurrent_state(1)
            episode_start = True

            for step_number in range(1, args.max_steps + 1):
                transition_recurrent_state = recurrent_state
                action, logprob, value, next_recurrent_state = (
                    network.get_actions_recurrent(
                        observation,
                        recurrent_state,
                        episode_starts=[episode_start],
                        deterministic=False,
                    )
                )
                action = action.squeeze(0)
                action_array = action.numpy().astype(np.float32)
                assert_finite("training action", action_array)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action_array)
                if step_number == args.max_steps and not (
                    terminated or truncated
                ):
                    # Remain correct if a future/custom environment forgets its
                    # own time-limit truncation at the caller's loop boundary.
                    truncated = True
                episode_end = bool(terminated or truncated)
                next_value: Any = (
                    0.0
                    if terminated
                    else network.get_value_recurrent(
                        next_observation,
                        next_recurrent_state,
                    )
                )
                buffer.store(
                    state=observation,
                    action=action_array,
                    logprob=logprob,
                    reward=scale_reward_for_ppo(reward, args.reward_scale),
                    value=value,
                    next_value=next_value,
                    terminated=terminated,
                    episode_end=episode_end,
                    recurrent_state=transition_recurrent_state,
                    episode_start=episode_start,
                )

                episode_reward += float(reward)
                episode_scaled_reward += scale_reward_for_ppo(
                    reward,
                    args.reward_scale,
                )
                _accumulate_reward_components(
                    episode_reward_components,
                    info,
                )
                episode_length = step_number
                previous_position, path_length = advance_path(
                    previous_position,
                    path_length,
                    info,
                )
                observation = next_observation
                recurrent_state = next_recurrent_state
                episode_start = False

                if len(buffer) >= args.rollout_steps:
                    apply_entropy_schedule(ppo, args, completed_updates)
                    last_update_stats = ppo.train_buffer(buffer)
                    completed_updates += 1
                    for name, stat in last_update_stats.items():
                        assert_finite(name, stat)
                if episode_end:
                    break

            success = bool(info.get("reached_goal", False))
            collision = bool(info.get("collision", False))
            episode_spl = calculate_spl(success, shortest, path_length)
            eval_stats = {
                "eval_mean_reward": math.nan,
                "eval_success_rate": math.nan,
                "eval_collision_rate": math.nan,
                "eval_mean_spl": math.nan,
            }
            should_evaluate = (
                args.eval_every > 0 and episode % args.eval_every == 0
            )
            if should_evaluate:
                eval_stats = evaluate_policy(
                    network,
                    env,
                    args,
                    evaluation_round=episode,
                )
                if evaluation_score(eval_stats) > evaluation_score(
                    best_eval_metrics
                ):
                    best_eval_metrics = dict(eval_stats)
                    save_checkpoint(
                        best_model_path,
                        network,
                        ppo,
                        env,
                        args,
                        episode,
                        best_eval_metrics,
                        completed_updates,
                    )

            save_checkpoint(
                last_model_path,
                network,
                ppo,
                env,
                args,
                episode,
                best_eval_metrics,
                completed_updates,
            )
            write_metric(
                metrics_path,
                {
                    "episode": episode,
                    "world_index": info.get("world_index"),
                    "shortest_path": shortest,
                    "episode_reward": episode_reward,
                    "episode_scaled_reward": episode_scaled_reward,
                    **episode_reward_components,
                    "episode_length": episode_length,
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "reached_goal": int(success),
                    "collision": int(collision),
                    "near_obstacle": int(
                        bool(info.get("near_obstacle", False))
                    ),
                    "stagnated": int(
                        bool(info.get("stagnated", False))
                    ),
                    "spl": episode_spl,
                    **last_update_stats,
                    **eval_stats,
                    "best_eval_success_rate": best_eval_metrics[
                        "eval_success_rate"
                    ],
                    "best_eval_collision_rate": best_eval_metrics[
                        "eval_collision_rate"
                    ],
                    "best_eval_mean_spl": best_eval_metrics[
                        "eval_mean_spl"
                    ],
                    "best_eval_mean_reward": best_eval_metrics[
                        "eval_mean_reward"
                    ],
                },
            )
            completed_episode = episode
            eval_text = ""
            if should_evaluate:
                eval_text = (
                    f" | eval={eval_stats['eval_mean_reward']:.2f}"
                    f" | success={eval_stats['eval_success_rate']:.2f}"
                    f" | collision={eval_stats['eval_collision_rate']:.2f}"
                    f" | SPL={eval_stats['eval_mean_spl']:.2f}"
                )
            print(
                f"episode={episode:5d}"
                f" | reward={episode_reward:8.2f}"
                f" | len={episode_length:3d}"
                f" | success={int(success)}"
                f" | collision={int(collision)}"
                f" | SPL={episode_spl:.3f}"
                f"{eval_text}"
            )

        if len(buffer) > 0:
            apply_entropy_schedule(ppo, args, completed_updates)
            last_update_stats = ppo.train_buffer(buffer)
            completed_updates += 1
            for name, stat in last_update_stats.items():
                assert_finite(name, stat)
        save_checkpoint(
            last_model_path,
            network,
            ppo,
            env,
            args,
            completed_episode,
            best_eval_metrics,
            completed_updates,
        )
        if not best_model_path.exists():
            save_checkpoint(
                best_model_path,
                network,
                ppo,
                env,
                args,
                completed_episode,
                best_eval_metrics,
                completed_updates,
            )
    finally:
        env.close()
        _shutdown_gazebo()

    print("Training finished.")
    print(f"Metrics: {metrics_path}")
    print(f"Last model: {last_model_path}")
    print(f"Best model: {best_model_path}")
    _generate_training_report(metrics_path)
    return network, ppo


def main(argv: Iterable[str] | None = None) -> None:
    """Run the ROS console-script training entry point."""
    train(parse_args(argv))


if __name__ == "__main__":
    main()
