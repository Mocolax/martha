"""Evaluate a Martha PPO checkpoint on Gazebo maps or opted-in hardware."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .actions import ActionLimits
from .checkpoint import (
    checkpoint_contract,
    choose_device,
    load_policy,
    validate_checkpoint,
)
from .evaluation_core import (
    advance_path,
    assert_finite,
    calculate_spl,
    episode_path_state,
    episode_seed,
)
from .martha_env import MarthaEnv
from .network import ActorCritic
from .reward import RewardConfig


# Edit this block to configure standalone evaluation.  The checkpoint is the
# only temporary command-line override.
@dataclass(frozen=True)
class EvaluationDefaults:
    checkpoint: Path | None = None
    episodes: int = 1
    max_steps: int = 300
    map_index: int | None = None
    backend: str = "gazebo"
    goal: tuple[float, float] | None = None
    goal_frame: str = "odom"
    seed: int = 123
    device: str = "auto"
    csv: Path | None = None


DEFAULTS = EvaluationDefaults()


RESULT_FIELDS = [
    "map_index",
    "episode",
    "seed",
    "reward",
    "steps",
    "terminated",
    "truncated",
    "reached_goal",
    "collision",
    "out_of_bounds",
    "path_length",
    "shortest_path",
    "spl",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone checkpoint-evaluation argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a continuous Martha PPO checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULTS.checkpoint,
        help="Path to best_model.pt or last_model.pt.",
    )
    return parser


def _validate_args(args: EvaluationDefaults) -> None:
    """Reject invalid or unsafe evaluation options before starting ROS."""
    if args.checkpoint is None:
        raise ValueError(
            "set EvaluationDefaults.checkpoint or pass --checkpoint"
        )
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episode and step counts must be positive")
    if args.map_index is not None and args.map_index < 0:
        raise ValueError("EvaluationDefaults.map_index cannot be negative")
    if args.backend not in {"gazebo", "hardware"}:
        raise ValueError("backend must be gazebo or hardware")
    if args.backend == "hardware":
        if args.goal is None:
            raise ValueError("hardware evaluation requires EvaluationDefaults.goal")
        if not all(math.isfinite(float(value)) for value in args.goal):
            raise ValueError("hardware goal coordinates must be finite")
        if args.map_index is not None:
            raise ValueError("map_index is only valid for Gazebo")
    elif args.goal is not None:
        raise ValueError("goal is reserved for the hardware backend")


def parse_args(argv: Iterable[str] | None = None) -> EvaluationDefaults:
    """Parse and validate evaluation arguments."""
    parser = build_parser()
    parsed = parser.parse_args(argv)
    args = replace(DEFAULTS, checkpoint=parsed.checkpoint)
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def make_environment(
    args: EvaluationDefaults,
    checkpoint: dict[str, Any],
    action_limits: ActionLimits,
) -> MarthaEnv:
    """Create the one environment reused for every evaluation episode."""
    contract = checkpoint_contract(checkpoint)
    config = checkpoint["config"]
    saved_reward_config = config.get("reward_config")
    if saved_reward_config is None:
        print(
            "WARNING: checkpoint has no reward_config; using the current "
            "paper reward defaults.",
            flush=True,
        )
        reward_config = RewardConfig()
    elif not isinstance(saved_reward_config, dict):
        raise ValueError("checkpoint reward_config must be a dictionary")
    else:
        try:
            reward_config = RewardConfig(**saved_reward_config)
        except TypeError as exc:
            raise ValueError("checkpoint reward_config has invalid fields") from exc
    return MarthaEnv(
        action_mode="continuous",
        render_mode=None,
        map_mode="predefined",
        backend=args.backend,
        scan_range_max=float(contract["scan_range_max"]),
        max_steps=args.max_steps,
        goal_tolerance=float(config["goal_tolerance"]),
        max_goal_distance=float(contract["max_goal_distance"]),
        min_goal_distance=float(config["min_goal_distance"]),
        action_limits=action_limits,
        reward_config=reward_config,
        allow_hardware_training=args.backend == "hardware",
    )


def validate_environment_contract(
    checkpoint: dict[str, Any],
    env: MarthaEnv,
) -> None:
    """Ensure the checkpoint contract matches the sole environment."""
    validate_checkpoint(
        checkpoint,
        expected_contract=env.policy_contract,
    )


def _reset_options(
    args: EvaluationDefaults,
    map_index: int | None,
) -> dict[str, Any]:
    """Build the explicit reset request for either backend."""
    if args.backend == "hardware":
        return {
            "manual_reset": True,
            "goal": tuple(args.goal),
            "goal_frame": args.goal_frame,
        }
    return {"world_index": map_index}


def _confirm_hardware_reset(
    args: EvaluationDefaults,
    episode: int,
) -> None:
    """Require an operator handshake before each physical episode."""
    if args.backend != "hardware":
        return
    print(
        f"HARDWARE evaluation episode {episode}: place Martha safely, clear "
        "the route, verify the emergency stop, and confirm goal "
        f"({args.goal[0]:.3f}, {args.goal[1]:.3f}) in {args.goal_frame}."
    )
    try:
        input("Press Enter only when the physical reset is complete: ")
    except EOFError as exc:
        raise RuntimeError(
            "hardware reset confirmation requires an interactive terminal"
        ) from exc


def _map_indices(env: MarthaEnv, args: EvaluationDefaults) -> list[int | None]:
    """Return the requested Gazebo maps or one hardware evaluation target."""
    if args.backend == "hardware":
        return [None]
    if args.map_index is not None:
        if args.map_index >= len(env.predefined_maps):
            raise IndexError(
                f"map index {args.map_index} outside the available map range"
            )
        return [args.map_index]
    return list(range(len(env.predefined_maps)))


def evaluate_episode(
    network: ActorCritic,
    env: MarthaEnv,
    args: EvaluationDefaults,
    map_index: int | None,
    episode: int,
) -> dict[str, Any]:
    """Run one deterministic episode on the already-created environment."""
    seed_index = -1 if map_index is None else map_index
    seed = episode_seed(args.seed, seed_index, episode)
    _confirm_hardware_reset(args, episode)
    observation, reset_info = env.reset(
        seed=seed,
        options=_reset_options(args, map_index),
    )
    shortest_path, previous_position, path_length = episode_path_state(
        reset_info
    )
    total_reward = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    steps = 0
    for steps in range(1, args.max_steps + 1):
        action, _, _ = network.get_action(observation, deterministic=True)
        action_array = action.numpy().astype(np.float32)
        assert_finite("evaluation action", action_array)
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action_array)
        total_reward += float(reward)
        previous_position, path_length = advance_path(
            previous_position,
            path_length,
            info,
        )
        if steps == args.max_steps and not (terminated or truncated):
            truncated = True
        if terminated or truncated:
            break
    success = bool(info.get("reached_goal", False))
    return {
        "seed": seed,
        "reward": total_reward,
        "steps": steps,
        "terminated": int(terminated),
        "truncated": int(truncated),
        "reached_goal": int(success),
        "collision": int(bool(info.get("collision", False))),
        "out_of_bounds": int(bool(info.get("out_of_bounds", False))),
        "path_length": path_length,
        "shortest_path": shortest_path,
        "spl": calculate_spl(success, shortest_path, path_length),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write complete per-episode results to CSV."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    """Calculate a finite mean for one result column."""
    values = [
        float(row[key])
        for row in rows
        if math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else math.nan


def print_summary(rows: list[dict[str, Any]], checkpoint_path: Path) -> None:
    """Print per-map and aggregate reward, success, collision and SPL."""
    print(f"\nCheckpoint: {checkpoint_path.expanduser().resolve()}")
    print("map | episodes | reward | success | collision | SPL")
    print("----|----------|--------|---------|-----------|------")
    map_indices = sorted(
        {row["map_index"] for row in rows},
        key=lambda value: -1 if value is None else value,
    )
    for map_index in map_indices:
        map_rows = [row for row in rows if row["map_index"] == map_index]
        label = "hw" if map_index is None else str(map_index)
        print(
            f"{label:>3} |"
            f" {len(map_rows):>8} |"
            f" {_mean(map_rows, 'reward'):>6.2f} |"
            f" {_mean(map_rows, 'reached_goal'):>7.2f} |"
            f" {_mean(map_rows, 'collision'):>9.2f} |"
            f" {_mean(map_rows, 'spl'):>4.2f}"
        )
    print("----|----------|--------|---------|-----------|------")
    print(
        f"all |"
        f" {len(rows):>8} |"
        f" {_mean(rows, 'reward'):>6.2f} |"
        f" {_mean(rows, 'reached_goal'):>7.2f} |"
        f" {_mean(rows, 'collision'):>9.2f} |"
        f" {_mean(rows, 'spl'):>4.2f}"
    )


def evaluate(args: EvaluationDefaults) -> list[dict[str, Any]]:
    """Evaluate all selected episodes with one reusable environment."""
    _validate_args(args)
    device = choose_device(args.device)
    network, checkpoint, action_limits = load_policy(args.checkpoint, device)
    env = make_environment(args, checkpoint, action_limits)
    rows: list[dict[str, Any]] = []
    try:
        validate_environment_contract(checkpoint, env)
        map_indices = _map_indices(env, args)
        print(f"Device: {device}")
        print(f"Backend: {args.backend}")
        print(f"Checkpoint episode: {checkpoint.get('episode', 'unknown')}")
        print(f"Evaluating {len(map_indices)} map/backend target(s).")
        if args.backend == "hardware":
            print(
                "HARDWARE ENABLED: keep the operator at the emergency stop "
                "and "
                "manually place the robot safely before every reset."
            )
        with torch.no_grad():
            for map_index in map_indices:
                for episode in range(1, args.episodes + 1):
                    result = evaluate_episode(
                        network,
                        env,
                        args,
                        map_index,
                        episode,
                    )
                    row = {
                        "map_index": map_index,
                        "episode": episode,
                        **result,
                    }
                    rows.append(row)
                    label = (
                        "hardware"
                        if map_index is None
                        else f"map={map_index:02d}"
                    )
                    print(
                        f"{label}"
                        f" episode={episode:02d}"
                        f" reward={row['reward']:8.2f}"
                        f" steps={row['steps']:3d}"
                        f" goal={row['reached_goal']}"
                        f" collision={row['collision']}"
                        f" SPL={row['spl']:.3f}"
                    )
    finally:
        env.close()

    print_summary(rows, args.checkpoint)
    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"\nCSV: {args.csv.expanduser().resolve()}")
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    """Run the ROS console-script evaluation entry point."""
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
