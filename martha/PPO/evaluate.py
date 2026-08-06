"""Evaluate a Martha PPO checkpoint on Gazebo maps or opted-in hardware."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .actions import ActionLimits
from .martha_env import POLICY_CONTRACT_VERSION, MarthaEnv
from .network import ActorCritic


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
        required=True,
        help="Path to best_model.pt or last_model.pt.",
    )
    parser.add_argument("--episodes-per-map", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--map-index", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=("gazebo", "hardware"),
        default="gazebo",
    )
    parser.add_argument(
        "--allow-hardware",
        action="store_true",
        help=(
            "Explicitly acknowledge physical evaluation risk, operator "
            "presence and emergency-stop readiness."
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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid or unsafe evaluation options before starting ROS."""
    if args.episodes_per_map <= 0 or args.max_steps <= 0:
        raise ValueError("episode and step counts must be positive")
    if args.map_index is not None and args.map_index < 0:
        raise ValueError("--map-index cannot be negative")
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
        if args.map_index is not None:
            raise ValueError("--map-index is only valid for Gazebo")
    elif args.goal is not None:
        raise ValueError("--goal is reserved for the hardware backend")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse and validate evaluation arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def choose_device(requested: str) -> torch.device:
    """Resolve an automatic, CPU or CUDA evaluation device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def episode_seed(base_seed: int, map_index: int, episode: int) -> int:
    """Derive a distinct deterministic seed for every map/episode pair."""
    value = int(base_seed) & 0xFFFFFFFF
    for coordinate in (map_index + 1, episode):
        value ^= (int(coordinate) + 0x9E3779B9) & 0xFFFFFFFF
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    return value


def assert_finite(name: str, value: Any) -> None:
    """Reject non-finite deterministic policy actions."""
    if not np.isfinite(np.asarray(value, dtype=np.float64)).all():
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


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint while supporting PyTorch before weights_only."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    return checkpoint


def _checkpoint_version(checkpoint: dict[str, Any]) -> int:
    """Extract the canonical or compatibility policy-contract version."""
    contract = checkpoint.get("policy_contract", {})
    value = contract.get(
        "version",
        checkpoint.get("policy_contract_version"),
    )
    if value is None:
        raise KeyError("checkpoint is missing the policy contract version")
    return int(value)


def action_limits_from_checkpoint(checkpoint: dict[str, Any]) -> ActionLimits:
    """Reconstruct the exact simulation/hardware action scaling contract."""
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
        for key, expected in vars(limits).items():
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


def load_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ActorCritic, dict[str, Any], ActionLimits]:
    """Load policy metadata without constructing an environment."""
    checkpoint = _torch_load(checkpoint_path, device)
    if _checkpoint_version(checkpoint) != POLICY_CONTRACT_VERSION:
        raise ValueError(
            "checkpoint policy contract version does not match this package"
        )
    observation_shape = tuple(checkpoint.get("observation_shape", ()))
    action_shape = tuple(checkpoint.get("action_shape", ()))
    if len(observation_shape) != 1 or len(action_shape) != 1:
        raise ValueError(
            "checkpoint observation/action shapes must be one-dimensional"
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint is missing model_state_dict")
    config = checkpoint.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 256))
    network = ActorCritic(
        state_dim=observation_shape[0],
        action_dim=action_shape[0],
        hidden_dim=hidden_dim,
    ).to(device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    return network, checkpoint, action_limits_from_checkpoint(checkpoint)


def make_environment(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    action_limits: ActionLimits,
) -> MarthaEnv:
    """Create the one environment reused for every evaluation episode."""
    contract = checkpoint.get("policy_contract", {})
    config = checkpoint.get("config", {})
    return MarthaEnv(
        action_mode="continuous",
        render_mode="human" if args.render else None,
        map_mode="predefined",
        backend=args.backend,
        scan_range_max=float(contract.get("scan_range_max", 8.0)),
        max_steps=args.max_steps,
        goal_tolerance=float(config.get("goal_tolerance", 0.25)),
        max_goal_distance=float(contract.get("max_goal_distance", 12.0)),
        min_goal_distance=float(config.get("min_goal_distance", 2.0)),
        action_limits=action_limits,
        allow_hardware_training=args.allow_hardware,
    )


def validate_environment_contract(
    checkpoint: dict[str, Any],
    env: MarthaEnv,
) -> None:
    """Ensure the checkpoint contract matches the sole environment."""
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
    expected = getattr(env, "policy_contract", None)
    saved = checkpoint.get("policy_contract")
    if isinstance(expected, dict) and isinstance(saved, dict):
        for key in (
            "version",
            "observation_size",
            "action_size",
            "laser_sectors",
            "scan_range_max",
            "max_goal_distance",
        ):
            expected_value = expected.get(key)
            saved_value = saved.get(key)
            if isinstance(expected_value, float):
                matches = saved_value is not None and math.isclose(
                    float(saved_value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            else:
                matches = saved_value == expected_value
            if not matches:
                raise ValueError(f"checkpoint policy contract mismatch: {key}")
        saved_limits = saved.get("action_limits")
        expected_limits = expected.get("action_limits")
        if not isinstance(saved_limits, dict) or not isinstance(
            expected_limits,
            dict,
        ):
            raise ValueError(
                "policy_contract action_limits must be dictionaries"
            )
        for key, expected_value in expected_limits.items():
            try:
                saved_value = float(saved_limits[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"checkpoint policy contract mismatch: action_limits.{key}"
                ) from exc
            if not math.isclose(
                saved_value,
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"checkpoint policy contract mismatch: action_limits.{key}"
                )


def _reset_options(
    args: argparse.Namespace,
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
    args: argparse.Namespace,
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


def _map_indices(env: MarthaEnv, args: argparse.Namespace) -> list[int | None]:
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


def _initial_path_state(
    reset_info: dict[str, Any],
) -> tuple[float | None, tuple[float, float] | None, float]:
    """Initialize path tracking for SPL."""
    shortest_path = reset_info.get("shortest_path")
    if shortest_path is None:
        shortest_path = reset_info.get("euclidean_distance")
    # Reset and step ``position`` values share the episode-local odom frame.
    # Gazebo-world ``start`` metadata is intentionally not used here.
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
    """Add one odometry segment to an episode path."""
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


def evaluate_episode(
    network: ActorCritic,
    env: MarthaEnv,
    args: argparse.Namespace,
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
    shortest_path, previous_position, path_length = _initial_path_state(
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
        previous_position, path_length = _advance_path(
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


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
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
                for episode in range(1, args.episodes_per_map + 1):
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
