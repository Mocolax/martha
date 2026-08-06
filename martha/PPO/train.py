"""Train Martha's continuous PPO navigation policy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch

from .actions import ActionLimits
from .buffer import RolloutBuffer
from .logic import PPOLogic
from .martha_env import (
    POLICY_CONTRACT_VERSION,
    MarthaEnv,
)
from .network import ActorCritic


METRIC_FIELDS = [
    "episode",
    "episode_reward",
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
    "eval_mean_reward",
    "eval_success_rate",
    "eval_collision_rate",
    "eval_mean_spl",
    "best_eval_success_rate",
    "best_eval_collision_rate",
    "best_eval_mean_spl",
    "best_eval_mean_reward",
]

_EMPTY_UPDATE_STATS = {
    "loss": 0.0,
    "actor_loss": 0.0,
    "critic_loss": 0.0,
    "entropy": 0.0,
    "updates": 0,
}

_EMPTY_BEST_EVAL = {
    "eval_success_rate": -math.inf,
    "eval_mean_spl": -math.inf,
    "eval_collision_rate": math.inf,
    "eval_mean_reward": -math.inf,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without starting ROS or Gazebo."""
    parser = argparse.ArgumentParser(
        description="Train continuous PPO navigation with MarthaEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=2,
        help="Deterministic evaluation episodes per selected Gazebo map.",
    )
    parser.add_argument(
        "--backend",
        choices=("gazebo", "hardware"),
        default="gazebo",
    )
    parser.add_argument(
        "--map-mode",
        choices=("predefined", "random"),
        default="random",
    )
    parser.add_argument("--map-index", type=int, default=None)
    parser.add_argument(
        "--allow-hardware",
        "--allow-hardware-training",
        dest="allow_hardware",
        action="store_true",
        help=(
            "Explicitly acknowledge physical training risk, operator "
            "presence, emergency stop and a controlled test area."
        ),
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Required fixed goal for the hardware backend.",
    )
    parser.add_argument("--goal-frame", default="odom")
    parser.add_argument(
        "--non-interactive-hardware-reset",
        action="store_true",
        help=(
            "Skip the per-episode Enter prompt after accepting responsibility "
            "for an external physical reset procedure."
        ),
    )
    parser.add_argument("--goal-tolerance", type=float, default=0.25)
    parser.add_argument("--min-goal-distance", type=float, default=2.0)
    parser.add_argument("--max-goal-distance", type=float, default=12.0)
    parser.add_argument("--scan-range-max", type=float, default=8.0)
    parser.add_argument("--max-vx", type=float, default=0.35)
    parser.add_argument("--max-vy", type=float, default=0.35)
    parser.add_argument("--max-wz", type=float, default=0.80)
    parser.add_argument("--max-action-delta", type=float, default=0.35)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path.cwd() / "ppo_runs",
        help="Base output directory; defaults outside the Python package.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model, optimizer and episode number from a checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--render-eval", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid and unsafe settings before creating an environment."""
    positive_ints = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "rollout_steps": args.rollout_steps,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "hidden_dim": args.hidden_dim,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.eval_every < 0 or args.eval_episodes <= 0:
        raise ValueError(
            "--eval-every must be non-negative and --eval-episodes positive"
        )
    if args.map_index is not None and args.map_index < 0:
        raise ValueError("--map-index cannot be negative")
    numeric_values = {
        "lr": args.lr,
        "gamma": args.gamma,
        "lam": args.lam,
        "eps": args.eps,
        "value-coef": args.value_coef,
        "entropy-coef": args.entropy_coef,
        "max-grad-norm": args.max_grad_norm,
        "goal-tolerance": args.goal_tolerance,
        "min-goal-distance": args.min_goal_distance,
        "max-goal-distance": args.max_goal_distance,
        "scan-range-max": args.scan_range_max,
        "max-vx": args.max_vx,
        "max-vy": args.max_vy,
        "max-wz": args.max_wz,
        "max-action-delta": args.max_action_delta,
    }
    for name, value in numeric_values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"--{name} must be finite")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.lam <= 1.0:
        raise ValueError("--gamma must be in (0, 1] and --lam in [0, 1]")
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
    )
    if any(numeric_values[name] <= 0.0 for name in positive_names):
        raise ValueError(
            "learning, navigation and sensor limits must be positive"
        )
    if args.value_coef < 0.0 or args.entropy_coef < 0.0:
        raise ValueError("--value-coef and --entropy-coef cannot be negative")
    if args.min_goal_distance >= args.max_goal_distance:
        raise ValueError(
            "--min-goal-distance must be below --max-goal-distance"
        )
    if args.backend == "hardware":
        if not args.allow_hardware:
            raise ValueError(
                "hardware is disabled by default; pass --allow-hardware only "
                "with an operator and emergency stop"
            )
        if args.goal is None:
            raise ValueError("the hardware backend requires --goal X Y")
        if not all(math.isfinite(float(value)) for value in args.goal):
            raise ValueError("hardware goal coordinates must be finite")
    elif args.goal is not None:
        raise ValueError("--goal is reserved for the hardware backend")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse and validate training arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        _action_limits_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def choose_device(requested: str) -> torch.device:
    """Resolve an automatic, CPU or CUDA device request."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for repeatable initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_seed(base_seed: int, *coordinates: int) -> int:
    """Derive a deterministic seed from an episode/map coordinate tuple."""
    value = int(base_seed) & 0xFFFFFFFF
    for coordinate in coordinates:
        value ^= (int(coordinate) + 0x9E3779B9) & 0xFFFFFFFF
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    return value


def assert_finite(name: str, value: Any) -> None:
    """Raise immediately when a policy or optimizer value is non-finite."""
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
    """Rank policies by success, SPL, safety and only then mean reward."""
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


def _episode_path_state(
    reset_info: dict[str, Any],
) -> tuple[float | None, tuple[float, float] | None, float]:
    """Initialize the SPL reference and travelled-path accumulator."""
    shortest_path = reset_info.get("shortest_path")
    if shortest_path is None:
        # Hardware has no Gazebo occupancy grid.  Its initial straight-line
        # distance is the best available lower bound for an online SPL metric.
        shortest_path = reset_info.get("euclidean_distance")
    # ``position`` and every step's ``position`` share the episode-local odom
    # frame.  ``start`` is Gazebo-world metadata and must not be mixed with it.
    start = reset_info.get("position")
    previous_position = None
    if start is not None and len(start) >= 2:
        previous_position = (float(start[0]), float(start[1]))
    return shortest_path, previous_position, 0.0


def _advance_path(
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


def _action_limits_from_checkpoint(checkpoint: dict[str, Any]) -> ActionLimits:
    """Read action scaling from a versioned checkpoint."""
    contract_values = checkpoint.get("policy_contract", {}).get(
        "action_limits"
    )
    compatibility_values = checkpoint.get("action_limits")
    values = contract_values
    if values is None:
        values = compatibility_values
    if not isinstance(values, dict):
        raise KeyError("checkpoint is missing action_limits")
    try:
        limits = ActionLimits(
            max_vx=float(values["max_vx"]),
            max_vy=float(values["max_vy"]),
            max_wz=float(values["max_wz"]),
            max_action_delta=float(values["max_action_delta"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint action_limits are invalid") from exc
    if not np.isfinite(limits.as_array()).all() or not math.isfinite(
        limits.max_action_delta
    ) or limits.max_action_delta <= 0.0:
        raise ValueError(
            "checkpoint action_limits must be finite and positive"
        )
    if compatibility_values is not None:
        if not isinstance(compatibility_values, dict):
            raise ValueError("checkpoint top-level action_limits are invalid")
        for key, expected in asdict(limits).items():
            try:
                actual = float(compatibility_values[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "checkpoint top-level action_limits are invalid"
                ) from exc
            if not math.isfinite(actual) or not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "checkpoint action_limits disagree with policy_contract"
                )
    return limits


def _policy_contract(env: MarthaEnv) -> dict[str, Any]:
    """Return the environment's canonical inference contract."""
    contract = getattr(env, "policy_contract", None)
    if not isinstance(contract, dict):
        contract = {
            "version": POLICY_CONTRACT_VERSION,
            "observation_size": int(env.observation_space.shape[0]),
            "action_limits": asdict(env.action_limits),
        }
    return dict(contract)


def _checkpoint_version(checkpoint: dict[str, Any]) -> int:
    """Extract the canonical or compatibility policy contract version."""
    contract = checkpoint.get("policy_contract", {})
    value = contract.get(
        "version",
        checkpoint.get("policy_contract_version"),
    )
    if value is None:
        raise KeyError("checkpoint is missing the policy contract version")
    return int(value)


def validate_checkpoint(
    checkpoint: dict[str, Any],
    env: MarthaEnv,
    *,
    require_optimizer: bool,
) -> None:
    """Validate a checkpoint before loading it into this environment."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    if _checkpoint_version(checkpoint) != POLICY_CONTRACT_VERSION:
        raise ValueError(
            "checkpoint policy contract version does not match this package"
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint is missing model_state_dict")
    if require_optimizer and "optimizer_state_dict" not in checkpoint:
        raise KeyError("resume checkpoint is missing optimizer_state_dict")

    observation_shape = tuple(checkpoint.get("observation_shape", ()))
    action_shape = tuple(checkpoint.get("action_shape", ()))
    if observation_shape != tuple(env.observation_space.shape):
        raise ValueError(
            f"checkpoint observation shape {observation_shape} does not match "
            f"{tuple(env.observation_space.shape)}"
        )
    if action_shape != tuple(env.action_space.shape):
        raise ValueError(
            f"checkpoint action shape {action_shape} does not match "
            f"{tuple(env.action_space.shape)}"
        )

    saved_limits = _action_limits_from_checkpoint(checkpoint)
    if not np.allclose(
        saved_limits.as_array(),
        env.action_limits.as_array(),
        rtol=0.0,
        atol=1e-9,
    ) or not math.isclose(
        saved_limits.max_action_delta,
        env.action_limits.max_action_delta,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "checkpoint action limits do not match the environment"
        )

    saved_contract = checkpoint.get("policy_contract")
    expected_contract = _policy_contract(env)
    if isinstance(saved_contract, dict):
        for key in (
            "version",
            "observation_size",
            "action_size",
            "laser_sectors",
            "scan_range_max",
            "max_goal_distance",
        ):
            if key not in expected_contract:
                continue
            saved = saved_contract.get(key)
            expected = expected_contract[key]
            if isinstance(expected, float):
                if saved is None or not math.isclose(
                    float(saved), expected, rel_tol=0.0, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"checkpoint policy contract mismatch: {key}"
                    )
            elif saved != expected:
                raise ValueError(f"checkpoint policy contract mismatch: {key}")
        saved_contract_limits = saved_contract.get("action_limits")
        expected_contract_limits = expected_contract.get("action_limits")
        if not isinstance(saved_contract_limits, dict) or not isinstance(
            expected_contract_limits,
            dict,
        ):
            raise ValueError(
                "policy_contract action_limits must be dictionaries"
            )
        for key, expected in expected_contract_limits.items():
            try:
                saved = float(saved_contract_limits[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"checkpoint policy contract mismatch: action_limits.{key}"
                ) from exc
            if not math.isclose(
                saved,
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"checkpoint policy contract mismatch: action_limits.{key}"
                )


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint with compatibility for older PyTorch releases."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location=device)
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint must be a dictionary")
    return loaded


def _runtime_value(
    checkpoint: dict[str, Any] | None,
    contract_key: str,
    fallback: float,
) -> float:
    """Prefer normalization values saved with a resumed policy."""
    if checkpoint is None:
        return float(fallback)
    contract = checkpoint.get("policy_contract", {})
    return float(contract.get(contract_key, fallback))


def make_environment(
    args: argparse.Namespace,
    resume_checkpoint: dict[str, Any] | None,
) -> MarthaEnv:
    """Create the sole training/evaluation environment for this process."""
    action_limits = (
        _action_limits_from_checkpoint(resume_checkpoint)
        if resume_checkpoint is not None
        else _action_limits_from_args(args)
    )
    return MarthaEnv(
        action_mode="continuous",
        render_mode="human" if args.render_eval else None,
        map_mode=args.map_mode,
        map_index=args.map_index,
        backend=args.backend,
        scan_range_max=_runtime_value(
            resume_checkpoint,
            "scan_range_max",
            args.scan_range_max,
        ),
        max_steps=args.max_steps,
        goal_tolerance=args.goal_tolerance,
        max_goal_distance=_runtime_value(
            resume_checkpoint,
            "max_goal_distance",
            args.max_goal_distance,
        ),
        min_goal_distance=args.min_goal_distance,
        action_limits=action_limits,
        allow_hardware_training=args.allow_hardware,
    )


def build_agent(
    env: MarthaEnv,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ActorCritic, PPOLogic]:
    """Build the actor-critic network and PPO optimizer."""
    network = ActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=args.hidden_dim,
    ).to(device)
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


def _resume_hidden_dim(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
) -> int:
    """Use the saved architecture automatically when resuming training."""
    if checkpoint is None:
        return int(args.hidden_dim)
    try:
        hidden_dim = int(checkpoint.get("config", {})["hidden_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "resume checkpoint has no valid config.hidden_dim"
        ) from exc
    if hidden_dim <= 0:
        raise ValueError(
            "resume checkpoint config.hidden_dim must be positive"
        )
    return hidden_dim


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
        hidden_dim=int(args.hidden_dim),
        scan_range_max=float(env.policy_contract["scan_range_max"]),
        max_goal_distance=float(env.max_goal_distance),
        max_vx=float(env.action_limits.max_vx),
        max_vy=float(env.action_limits.max_vy),
        max_wz=float(env.action_limits.max_wz),
        max_action_delta=float(env.action_limits.max_action_delta),
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
    action_limits = asdict(env.action_limits)
    checkpoint = {
        "episode": int(episode),
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": ppo.optimizer.state_dict(),
        "best_eval_metrics": dict(best_eval_metrics),
        "best_eval_score": evaluation_score(best_eval_metrics),
        "best_eval_mean_reward": float(
            best_eval_metrics["eval_mean_reward"]
        ),
        "policy_contract": _policy_contract(env),
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "observation_shape": tuple(env.observation_space.shape),
        "action_shape": tuple(env.action_space.shape),
        "action_limits": action_limits,
        "config": _serializable_config(args, env),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


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
            torch.cuda.set_rng_state_all(rng_state["cuda"])
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
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
    if getattr(args, "non_interactive_hardware_reset", False):
        print(
            "Continuing under the external reset procedure acknowledged by "
            "CLI."
        )
        return
    try:
        input("Press Enter only when the physical reset is complete: ")
    except EOFError as exc:
        raise RuntimeError(
            "hardware reset confirmation requires an interactive terminal or "
            "--non-interactive-hardware-reset with an external safety "
            "procedure"
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
    return list(range(len(env.predefined_maps)))


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
            for world_index in _evaluation_worlds(env, args):
                seed_map_index = -1 if world_index is None else world_index
                for eval_episode in range(args.eval_episodes):
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
                        _episode_path_state(reset_info)
                    )
                    total_reward = 0.0
                    terminated = False
                    truncated = False
                    info: dict[str, Any] = {}
                    for step_number in range(1, args.max_steps + 1):
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
                        previous_position, path_length = _advance_path(
                            previous_position,
                            path_length,
                            info,
                        )
                        if step_number == args.max_steps and not (
                            terminated or truncated
                        ):
                            truncated = True
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
    """Run PPO training and close the sole environment on every exit path."""
    _validate_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    resume_checkpoint = (
        None if args.resume is None else _torch_load(args.resume, device)
    )
    run_dir = make_run_dir(args)
    metrics_path, last_model_path, best_model_path = _checkpoint_paths(run_dir)
    env = make_environment(args, resume_checkpoint)
    try:
        args.hidden_dim = _resume_hidden_dim(args, resume_checkpoint)
        network, ppo = build_agent(env, args, device)
    except Exception:
        env.close()
        raise
    buffer = RolloutBuffer()
    start_episode = 1
    best_eval_metrics = dict(_EMPTY_BEST_EVAL)
    if resume_checkpoint is not None:
        try:
            validate_checkpoint(resume_checkpoint, env, require_optimizer=True)
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
                "--episodes must exceed the episode stored in --resume"
            )

    print(f"Device: {device}")
    print(f"Backend: {args.backend}")
    print(f"Run directory: {run_dir}")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Action shape: {env.action_space.shape}")
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
                options=_reset_options(args, None),
            )
            shortest, previous_position, path_length = _episode_path_state(
                reset_info
            )
            episode_reward = 0.0
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
                    reward=reward,
                    value=value,
                    next_value=next_value,
                    terminated=terminated,
                    episode_end=episode_end,
                )

                episode_reward += float(reward)
                episode_length = step_number
                previous_position, path_length = _advance_path(
                    previous_position,
                    path_length,
                    info,
                )
                observation = next_observation

                if len(buffer) >= args.rollout_steps:
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
    return network, ppo


def main(argv: Iterable[str] | None = None) -> None:
    """Run the ROS console-script training entry point."""
    train(parse_args(argv))


if __name__ == "__main__":
    main()
