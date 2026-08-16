"""Train Martha's continuous PPO navigation policy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
import random
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
from .reward import REWARD_COMPONENT_NAMES, RewardConfig
from martha.simulation_speed import validate_sim_speed_factor


# Edit this block for the normal training setup. Every value can still be
# overridden temporarily from the command line without changing this file.
@dataclass(frozen=True)
class TrainingDefaults:
    episodes: int = 2000
    num_envs: int = 1
    sim_speed_factor: float = 2.0
    gazebo_gui: bool = True
    ros_domain_base: int = 50
    gazebo_port_base: int = 11400
    worker_startup_timeout: float = 90.0
    max_steps: int = 600
    rollout_steps: int = 1024
    ppo_epochs: int = 8
    minibatch_size: int = 256
    lr: float = 2e-5
    gamma: float = 0.99
    lam: float = 0.95
    eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.002
    # Exploration remains constant first, then entropy encouragement fades to
    # zero so PPO can consolidate a lower variance policy when appropriate.
    entropy_exploration_fraction: float = 0.25
    entropy_decay_fraction: float = 0.55
    reward_scale: float = 1
    max_grad_norm: float = 0.5
    eval_every: int = 100
    eval_episodes: int = 1
    eval_max_steps: int = 400
    eval_map_count: int = 3
    backend: str = "gazebo"
    map_mode: str = "random"
    map_index: int | None = None
    # Train on one Gazebo map for a block of global episodes before rotating.
    episodes_per_map: int = 20
    goal: tuple[float, float] | None = None
    goal_frame: str = "odom"
    goal_tolerance: float = 0.25
    min_goal_distance: float = 2.0
    max_goal_distance: float = 12.0
    scan_range_max: float = 8.0
    max_vx: float = 0.35
    max_vy: float = 0.35
    max_wz: float = 0.80
    max_action_delta: float = 0.35
    run_name: str | None = None
    runs_dir: Path = Path.home() / "ros2_ws/src/martha/martha/PPO/ppo_runs"
    resume: Path | None = None
    seed: int = 42
    device: str = "auto"


DEFAULTS = TrainingDefaults()


METRIC_FIELDS = [
    "episode",
    "episode_reward",
    "episode_scaled_reward",
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
    "out_of_bounds",
    "spl",
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
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid and unsafe settings before creating an environment."""
    positive_ints = {
        "episodes": args.episodes,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "rollout_steps": args.rollout_steps,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "eval_max_steps": args.eval_max_steps,
        "episodes_per_map": args.episodes_per_map,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"TrainingDefaults.{name} must be positive")
    if (
        not 0.0 <= args.entropy_exploration_fraction <= 1.0
        or not 0.0 < args.entropy_decay_fraction <= 1.0
        or (
            args.entropy_exploration_fraction
            + args.entropy_decay_fraction
            > 1.0
        )
    ):
        raise ValueError(
            "entropy exploration and decay fractions must be in [0, 1], "
            "the decay fraction must be positive, and their sum cannot "
            "exceed 1"
        )
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
    if args.backend == "hardware" and args.num_envs != 1:
        raise ValueError("hardware training requires num_envs=1")
    if not 0 <= args.ros_domain_base <= 232:
        raise ValueError("ros_domain_base must be in [0, 232]")
    if args.ros_domain_base + args.num_envs - 1 > 232:
        raise ValueError("parallel ROS domain IDs cannot exceed 232")
    if not 1024 <= args.gazebo_port_base <= 65535:
        raise ValueError("gazebo_port_base must be in [1024, 65535]")
    if args.gazebo_port_base + args.num_envs - 1 > 65535:
        raise ValueError("parallel Gazebo ports cannot exceed 65535")
    if args.worker_startup_timeout <= 0.0:
        raise ValueError("worker_startup_timeout must be positive")
    validate_sim_speed_factor(args.sim_speed_factor)
    numeric_values = {
        "lr": args.lr,
        "gamma": args.gamma,
        "lam": args.lam,
        "eps": args.eps,
        "value-coef": args.value_coef,
        "entropy-coef": args.entropy_coef,
        "entropy-exploration-fraction": args.entropy_exploration_fraction,
        "entropy-decay-fraction": args.entropy_decay_fraction,
        "reward-scale": args.reward_scale,
        "max-grad-norm": args.max_grad_norm,
        "goal-tolerance": args.goal_tolerance,
        "min-goal-distance": args.min_goal_distance,
        "max-goal-distance": args.max_goal_distance,
        "scan-range-max": args.scan_range_max,
        "max-vx": args.max_vx,
        "max-vy": args.max_vy,
        "max-wz": args.max_wz,
        "max-action-delta": args.max_action_delta,
        "worker-startup-timeout": args.worker_startup_timeout,
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
        "max-goal-distance",
        "scan-range-max",
        "max-vx",
        "max-vy",
        "max-wz",
        "max-action-delta",
        "worker-startup-timeout",
        "reward-scale",
    )
    if any(numeric_values[name] <= 0.0 for name in positive_names):
        raise ValueError(
            "learning, navigation and sensor limits must be positive"
        )
    if args.value_coef < 0.0 or args.entropy_coef < 0.0:
        raise ValueError("value_coef and entropy_coef cannot be negative")
    if args.min_goal_distance >= args.max_goal_distance:
        raise ValueError(
            "min_goal_distance must be below max_goal_distance"
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


def entropy_coefficient_for_episode(
    base_coefficient: float,
    total_episodes: int,
    exploration_fraction: float,
    decay_fraction: float,
    episode: int,
) -> float:
    """Return the deterministic three-phase entropy coefficient for PPO.

    The phase durations are percentages of the configured total episode count,
    not fixed episode counts, wall-clock time, or noisy evaluation results.
    This makes the schedule scale with run length and remain deterministic for
    sequential or parallel Gazebo collection.
    """
    exploration_end = total_episodes * exploration_fraction
    decay_duration = total_episodes * decay_fraction
    if episode <= exploration_end:
        return float(base_coefficient)
    if episode >= exploration_end + decay_duration:
        return 0.0
    decay_progress = min(
        max((episode - exploration_end) / decay_duration, 0.0),
        1.0,
    )
    return float(base_coefficient) * (1.0 - decay_progress)


def apply_entropy_schedule(
    ppo: PPOLogic,
    args: argparse.Namespace,
    episode: int,
) -> None:
    """Set PPO's current entropy weight immediately before an update."""
    ppo.entropy_coef = entropy_coefficient_for_episode(
        args.entropy_coef,
        args.episodes,
        args.entropy_exploration_fraction,
        args.entropy_decay_fraction,
        max(1, episode),
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
    try:
        return RewardConfig(**saved)
    except TypeError as exc:
        raise ValueError("checkpoint reward_config has invalid fields") from exc


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
    reward_config = reward_config_from_checkpoint(resume_checkpoint)
    return {
        "action_mode": "continuous",
        "render_mode": None,
        "map_mode": args.map_mode,
        "map_index": args.map_index,
        "backend": args.backend,
        "scan_range_max": _runtime_value(
            resume_checkpoint,
            "scan_range_max",
            args.scan_range_max,
        ),
        "max_steps": args.max_steps,
        "goal_tolerance": args.goal_tolerance,
        "max_goal_distance": _runtime_value(
            resume_checkpoint,
            "max_goal_distance",
            args.max_goal_distance,
        ),
        "min_goal_distance": args.min_goal_distance,
        "action_limits": action_limits,
        "reward_config": reward_config,
        "allow_hardware_training": args.backend == "hardware",
    }


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
        max_goal_distance=float(env.max_goal_distance),
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
) -> None:
    """Save policy, optimizer and the complete runtime contract."""
    checkpoint = {
        "episode": int(episode),
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
) -> tuple[int, dict[str, float]]:
    """Restore model, optimizer and optional random-generator states."""
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
    return start_episode, best_metrics


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
    """Choose the reproducible Gazebo map for one global training episode."""
    if args.backend == "hardware":
        return None
    if args.map_index is not None:
        return int(args.map_index)
    if world_count <= 0:
        raise ValueError("Gazebo training requires at least one world")
    if episode <= 0:
        raise ValueError("training episode must be positive")

    block_index = (episode - 1) // args.episodes_per_map
    cycle_index, map_position = divmod(block_index, world_count)
    cycle_seed = episode_seed(args.seed, 2, cycle_index)
    map_cycle = np.random.default_rng(cycle_seed).permutation(world_count)
    return int(map_cycle[map_position])


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


def _parallel_reset_batch(
    environments: list[Any],
    requests: list[tuple[int, int, dict[str, Any]]],
) -> dict[int, tuple[np.ndarray, dict[str, Any]]]:
    """Reset isolated workers concurrently and drain every sent request."""
    sent_indices: list[int] = []
    send_error: tuple[int, Exception] | None = None
    for environment_index, seed, options in requests:
        try:
            environments[environment_index].send_reset(
                seed=seed,
                options=options,
            )
            sent_indices.append(environment_index)
        except Exception as exc:
            send_error = (environment_index, exc)
            break

    results: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    receive_errors: list[tuple[int, Exception]] = []
    for environment_index in sent_indices:
        try:
            results[environment_index] = environments[
                environment_index
            ].receive_reset()
        except Exception as exc:
            receive_errors.append((environment_index, exc))

    if send_error is not None:
        environment_index, error = send_error
        raise RuntimeError(
            f"could not send reset to Gazebo worker {environment_index}"
        ) from error
    if receive_errors:
        indices = ", ".join(str(index) for index, _ in receive_errors)
        raise RuntimeError(
            f"Gazebo reset failed in worker(s): {indices}"
        ) from receive_errors[0][1]
    return results


def _parallel_episode_state(
    observation: np.ndarray,
    reset_info: dict[str, Any],
) -> dict[str, Any]:
    """Build the mutable accounting state for one parallel episode."""
    shortest, previous_position, path_length = episode_path_state(reset_info)
    return {
        "observation": observation,
        "shortest": shortest,
        "previous_position": previous_position,
        "path_length": path_length,
        "reward": 0.0,
        "scaled_reward": 0.0,
        "reward_components": _empty_reward_components(),
        "length": 0,
    }


def _parallel_replacement_requests(
    args: argparse.Namespace,
    environment_indices: Iterable[int],
    next_episode: int,
    world_count: int,
) -> tuple[list[tuple[int, int, dict[str, Any]]], int]:
    """Allocate at most the remaining configured parallel episodes."""
    requests = []
    for environment_index in environment_indices:
        if next_episode > args.episodes:
            break
        requests.append(
            (
                environment_index,
                episode_seed(args.seed, 0, next_episode),
                training_reset_options(args, world_count, next_episode),
            )
        )
        next_episode += 1
    return requests, next_episode


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
                    shortest, previous_position, path_length = (
                        episode_path_state(reset_info)
                    )
                    total_reward = 0.0
                    terminated = False
                    truncated = False
                    info: dict[str, Any] = {}
                    for step_number in range(1, args.eval_max_steps + 1):
                        action, _, _ = network.get_action(
                            observation,
                            deterministic=True,
                        )
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


def train(args: argparse.Namespace) -> tuple[ActorCritic, PPOLogic]:
    """Run managed Gazebo training or explicitly opted-in hardware training."""
    if args.backend == "gazebo":
        return train_gazebo(args)
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
        _validate_resume_reward_scale(args, resume_checkpoint)
    run_dir = make_run_dir(args)
    metrics_path, last_model_path, best_model_path = _checkpoint_paths(run_dir)
    env = make_environment(args, resume_checkpoint)
    try:
        network, ppo = build_agent(args, device)
    except Exception:
        env.close()
        raise
    buffer = RolloutBuffer()
    start_episode = 1
    best_eval_metrics = dict(_EMPTY_BEST_EVAL)
    if resume_checkpoint is not None:
        try:
            validate_checkpoint_data(
                resume_checkpoint,
                expected_contract=env.policy_contract,
                require_optimizer=True,
            )
            start_episode, best_eval_metrics = _restore_resume_state(
                resume_checkpoint,
                network,
                ppo,
            )
        except Exception:
            env.close()
            raise
        if start_episode > args.episodes:
            env.close()
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
    try:
        for episode in range(start_episode, args.episodes + 1):
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

            for step_number in range(1, args.max_steps + 1):
                action, logprob, value = network.get_action(
                    observation,
                    deterministic=False,
                )
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
                    else network.get_value(next_observation)
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

                if len(buffer) >= args.rollout_steps:
                    apply_entropy_schedule(ppo, args, episode)
                    last_update_stats = ppo.train_buffer(buffer)
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
                    )

            save_checkpoint(
                last_model_path,
                network,
                ppo,
                env,
                args,
                episode,
                best_eval_metrics,
            )
            write_metric(
                metrics_path,
                {
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "episode_scaled_reward": episode_scaled_reward,
                    **episode_reward_components,
                    "episode_length": episode_length,
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "reached_goal": int(success),
                    "collision": int(collision),
                    "out_of_bounds": int(
                        bool(info.get("out_of_bounds", False))
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
            apply_entropy_schedule(ppo, args, completed_episode)
            last_update_stats = ppo.train_buffer(buffer)
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
            )
    finally:
        env.close()

    print("Training finished.")
    print(f"Metrics: {metrics_path}")
    print(f"Last model: {last_model_path}")
    print(f"Best model: {best_model_path}")
    _generate_training_report(metrics_path)
    return network, ppo


def train_gazebo(args: argparse.Namespace) -> tuple[ActorCritic, PPOLogic]:
    """Train with one or more isolated, trainer-managed Gazebo workers."""
    from .parallel_env import ParallelGazeboEnvironments

    _validate_args(args)
    if args.backend != "gazebo":
        raise ValueError("parallel training requires backend='gazebo'")
    if args.num_envs > 4:
        print(
            "WARNING: more than four Gazebo workers can exhaust system RAM; "
            "monitor memory and swap.",
            flush=True,
        )

    set_seed(args.seed)
    device = choose_device(args.device)
    resume_checkpoint = (
        None
        if args.resume is None
        else load_checkpoint_file(args.resume, device)
    )
    _warn_legacy_reward_config(resume_checkpoint)
    if resume_checkpoint is not None:
        _validate_resume_reward_scale(args, resume_checkpoint)
    run_dir = make_run_dir(args)
    metrics_path, last_model_path, best_model_path = _checkpoint_paths(run_dir)
    group = ParallelGazeboEnvironments(
        count=args.num_envs,
        ros_domain_base=args.ros_domain_base,
        gazebo_port_base=args.gazebo_port_base,
        sim_speed_factor=args.sim_speed_factor,
        show_gui=args.gazebo_gui,
        startup_timeout=args.worker_startup_timeout,
        run_directory=run_dir,
        environment_kwargs=environment_kwargs(args, resume_checkpoint),
    )
    environments = group.environments
    reference_env = environments[0]
    try:
        network, ppo = build_agent(args, device)
        start_episode = 1
        best_eval_metrics = dict(_EMPTY_BEST_EVAL)
        if resume_checkpoint is not None:
            validate_checkpoint_data(
                resume_checkpoint,
                expected_contract=reference_env.policy_contract,
                require_optimizer=True,
            )
            start_episode, best_eval_metrics = _restore_resume_state(
                resume_checkpoint,
                network,
                ppo,
            )
        if start_episode > args.episodes:
            raise ValueError(
                "TrainingDefaults.episodes must exceed the resumed episode"
            )

        print(f"Device: {device}")
        print("Backend: gazebo")
        print(f"Managed Gazebo environments: {args.num_envs}")
        print(
            "Visible Gazebo worker: "
            + ("0" if args.gazebo_gui else "none")
        )
        print(f"Run directory: {run_dir}")
        print(f"Observation shape: {reference_env.observation_space.shape}")
        print(f"Action shape: {reference_env.action_space.shape}")
        print(f"PPO reward scale: {args.reward_scale}")

        buffers = [RolloutBuffer() for _ in environments]
        active: dict[int, dict[str, Any]] = {}
        seed_cursor = start_episode
        initial_count = min(
            len(environments),
            args.episodes - start_episode + 1,
        )
        initial_requests = []
        for index in range(initial_count):
            seed = episode_seed(args.seed, 0, seed_cursor)
            seed_cursor += 1
            initial_requests.append(
                (
                    index,
                    seed,
                    training_reset_options(
                        args,
                        len(reference_env.predefined_maps),
                        seed_cursor - 1,
                    ),
                )
            )
        initial_results = _parallel_reset_batch(
            environments,
            initial_requests,
        )
        for index, (observation, reset_info) in initial_results.items():
            active[index] = _parallel_episode_state(
                observation,
                reset_info,
            )

        last_update_stats = dict(_EMPTY_UPDATE_STATS)
        completed_episode = start_episode - 1
        while completed_episode < args.episodes and active:
            indices = sorted(active)
            observations = np.stack(
                [active[index]["observation"] for index in indices]
            )
            actions, logprobs, values = network.get_actions(
                observations,
                deterministic=False,
            )
            action_values = actions.numpy().astype(np.float32)
            for batch_index, environment_index in enumerate(indices):
                environments[environment_index].send_step(
                    action_values[batch_index]
                )

            transitions = []
            for environment_index in indices:
                transitions.append(
                    environments[environment_index].receive_step()
                )
            next_observations = np.stack(
                [transition[0] for transition in transitions]
            )
            next_values = network.get_value(next_observations).numpy().reshape(-1)
            ended = []
            for batch_index, environment_index in enumerate(indices):
                state = active[environment_index]
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = transitions[batch_index]
                episode_end = bool(terminated or truncated)
                buffers[environment_index].store(
                    state=state["observation"],
                    action=action_values[batch_index],
                    logprob=logprobs[batch_index],
                    reward=scale_reward_for_ppo(reward, args.reward_scale),
                    value=values[batch_index],
                    next_value=(
                        0.0 if terminated else next_values[batch_index]
                    ),
                    terminated=terminated,
                    episode_end=episode_end,
                )
                state["reward"] += float(reward)
                state["scaled_reward"] += scale_reward_for_ppo(
                    reward,
                    args.reward_scale,
                )
                _accumulate_reward_components(
                    state["reward_components"],
                    info,
                )
                state["length"] += 1
                state["previous_position"], state["path_length"] = (
                    advance_path(
                        state["previous_position"],
                        state["path_length"],
                        info,
                    )
                )
                state["observation"] = next_observation
                if episode_end:
                    state["terminated"] = bool(terminated)
                    state["truncated"] = bool(truncated)
                    state["info"] = info
                    ended.append(environment_index)

            if sum(len(buffer) for buffer in buffers) >= args.rollout_steps:
                apply_entropy_schedule(ppo, args, completed_episode + 1)
                last_update_stats = ppo.train_buffers(buffers)
                for name, stat in last_update_stats.items():
                    assert_finite(name, stat)

            for environment_index in ended:
                state = active.pop(environment_index)
                info = state["info"]
                completed_episode += 1
                success = bool(info.get("reached_goal", False))
                collision = bool(info.get("collision", False))
                episode_spl = calculate_spl(
                    success,
                    state["shortest"],
                    state["path_length"],
                )
                eval_stats = {
                    "eval_mean_reward": math.nan,
                    "eval_success_rate": math.nan,
                    "eval_collision_rate": math.nan,
                    "eval_mean_spl": math.nan,
                }
                should_evaluate = (
                    args.eval_every > 0
                    and completed_episode % args.eval_every == 0
                )
                if should_evaluate:
                    eval_stats = evaluate_policy(
                        network,
                        environments[environment_index],
                        args,
                        evaluation_round=completed_episode,
                    )
                    if evaluation_score(eval_stats) > evaluation_score(
                        best_eval_metrics
                    ):
                        best_eval_metrics = dict(eval_stats)
                        save_checkpoint(
                            best_model_path,
                            network,
                            ppo,
                            reference_env,
                            args,
                            completed_episode,
                            best_eval_metrics,
                        )

                save_checkpoint(
                    last_model_path,
                    network,
                    ppo,
                    reference_env,
                    args,
                    completed_episode,
                    best_eval_metrics,
                )
                write_metric(
                    metrics_path,
                    {
                        "episode": completed_episode,
                        "episode_reward": state["reward"],
                        "episode_scaled_reward": state["scaled_reward"],
                        **state["reward_components"],
                        "episode_length": state["length"],
                        "terminated": int(state["terminated"]),
                        "truncated": int(state["truncated"]),
                        "reached_goal": int(success),
                        "collision": int(collision),
                        "out_of_bounds": int(
                            bool(info.get("out_of_bounds", False))
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
                print(
                    f"episode={completed_episode:5d}"
                    f" | env={environment_index}"
                    f" | reward={state['reward']:8.2f}"
                    f" | len={state['length']:3d}"
                    f" | success={int(success)}"
                    f" | collision={int(collision)}"
                    f" | SPL={episode_spl:.3f}",
                    flush=True,
                )

            # ``seed_cursor`` tracks episodes already launched.  Counting only
            # completed episodes would oversubscribe the final batch because
            # other workers can still have active episodes.
            reset_requests, seed_cursor = _parallel_replacement_requests(
                args,
                ended,
                seed_cursor,
                len(reference_env.predefined_maps),
            )
            reset_results = _parallel_reset_batch(
                environments,
                reset_requests,
            )
            for environment_index, result in reset_results.items():
                observation, reset_info = result
                active[environment_index] = _parallel_episode_state(
                    observation,
                    reset_info,
                )

        if sum(len(buffer) for buffer in buffers) > 0:
            apply_entropy_schedule(ppo, args, completed_episode)
            last_update_stats = ppo.train_buffers(buffers)
            for name, stat in last_update_stats.items():
                assert_finite(name, stat)
        save_checkpoint(
            last_model_path,
            network,
            ppo,
            reference_env,
            args,
            completed_episode,
            best_eval_metrics,
        )
        if not best_model_path.exists():
            save_checkpoint(
                best_model_path,
                network,
                ppo,
                reference_env,
                args,
                completed_episode,
                best_eval_metrics,
            )
    finally:
        group.close()

    print("Gazebo training finished.")
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
