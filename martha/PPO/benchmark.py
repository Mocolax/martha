"""Reproducible readout of shared-Gazebo collection throughput."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from .evaluation_core import episode_seed
from .train import DEFAULTS, _package_asset_path, environment_kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fixed shared-Gazebo steps without updating a policy. "
            "Refuses to run while ppo_train is active."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot-counts", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument(
        "--physics-step-size",
        type=float,
        default=DEFAULTS.physics_step_size,
    )
    parser.add_argument(
        "--lidar-samples",
        type=int,
        default=DEFAULTS.lidar_samples,
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Use the detailed roller model instead of training kinematics.",
    )
    parser.add_argument(
        "--recycle-smoke",
        action="store_true",
        help=(
            "Also recycle robot 0 while the remaining robots receive "
            "accounted stop transitions."
        ),
    )
    return parser


def active_training_pids() -> tuple[int, ...]:
    """Return ppo_train PIDs visible in the current execution environment."""
    result = subprocess.run(
        ["pgrep", "-x", "ppo_train"],
        check=False,
        capture_output=True,
        text=True,
    )
    return tuple(
        int(value)
        for value in result.stdout.split()
        if value.isdigit()
    )


def _benchmark_args(
    *,
    robot_count: int,
    steps: int,
    physics_step_size: float,
    lidar_samples: int,
    training_kinematic: bool,
) -> SimpleNamespace:
    values = asdict(DEFAULTS)
    values.update(
        num_envs=robot_count,
        max_steps=steps + 10,
        physics_step_size=physics_step_size,
        lidar_samples=lidar_samples,
        training_kinematic=training_kinematic,
    )
    return SimpleNamespace(**values)


def run_case(
    *,
    robot_count: int,
    steps: int,
    physics_step_size: float,
    lidar_samples: int,
    training_kinematic: bool,
) -> dict[str, float | int | bool]:
    """Run fixed zero-command steps and report aggregate collection speed."""
    from .shared_gazebo import SharedGazeboEnvironments

    if robot_count <= 0 or steps <= 0:
        raise ValueError("robot_count and steps must be positive")
    args = _benchmark_args(
        robot_count=robot_count,
        steps=steps,
        physics_step_size=physics_step_size,
        lidar_samples=lidar_samples,
        training_kinematic=training_kinematic,
    )
    with tempfile.TemporaryDirectory(prefix="martha_ppo_benchmark_") as directory:
        group = SharedGazeboEnvironments(
            count=robot_count,
            sim_speed_factor=args.sim_speed_factor,
            physics_step_size=physics_step_size,
            lidar_samples=lidar_samples,
            training_kinematic=training_kinematic,
            show_gui=False,
            startup_timeout=args.gazebo_startup_timeout,
            run_directory=Path(directory),
            worlds_directory=_package_asset_path("worlds"),
            points_path=_package_asset_path("config/training_points.yaml"),
            environment_kwargs=environment_kwargs(args, None),
        )
        try:
            seeds = [episode_seed(args.seed, 0, index + 1) for index in range(robot_count)]
            reset_started = time.monotonic()
            group.reset_round(world_index=0, seeds=seeds)
            reset_wall_s = time.monotonic() - reset_started
            sim_started_ns = group.reference_env.ros.get_clock().now().nanoseconds
            rollout_started = time.monotonic()
            action = np.zeros(3, dtype=np.float32)
            for _ in range(steps):
                transitions = group.step_batch(
                    {index: action for index in range(robot_count)}
                )
                if any(value[2] or value[3] for value in transitions.values()):
                    raise RuntimeError(
                        "benchmark episode ended before its fixed step budget"
                    )
            rollout_wall_s = time.monotonic() - rollout_started
            sim_finished_ns = group.reference_env.ros.get_clock().now().nanoseconds
        finally:
            group.close()

    transitions = robot_count * steps
    sim_advance_s = max(0.0, (sim_finished_ns - sim_started_ns) / 1e9)
    return {
        "robot_count": robot_count,
        "steps_per_robot": steps,
        "transitions": transitions,
        "training_kinematic": training_kinematic,
        "physics_step_size": physics_step_size,
        "lidar_samples": lidar_samples,
        "reset_wall_s": reset_wall_s,
        "rollout_wall_s": rollout_wall_s,
        "transitions_per_second": transitions / max(rollout_wall_s, 1e-9),
        "sim_real_time_factor": sim_advance_s / max(rollout_wall_s, 1e-9),
    }


def run_recycle_smoke(
    *,
    robot_count: int,
    physics_step_size: float,
    lidar_samples: int,
    training_kinematic: bool,
) -> dict[str, float | int | bool]:
    """Exercise one live slot reset without silently advancing its peers."""
    from .shared_gazebo import SharedGazeboEnvironments

    if robot_count < 2:
        raise ValueError("recycle smoke requires at least two robots")
    args = _benchmark_args(
        robot_count=robot_count,
        steps=10,
        physics_step_size=physics_step_size,
        lidar_samples=lidar_samples,
        training_kinematic=training_kinematic,
    )
    with tempfile.TemporaryDirectory(
        prefix="martha_ppo_recycle_smoke_"
    ) as directory:
        group = SharedGazeboEnvironments(
            count=robot_count,
            sim_speed_factor=args.sim_speed_factor,
            physics_step_size=physics_step_size,
            lidar_samples=lidar_samples,
            training_kinematic=training_kinematic,
            show_gui=False,
            startup_timeout=args.gazebo_startup_timeout,
            run_directory=Path(directory),
            worlds_directory=_package_asset_path("worlds"),
            points_path=_package_asset_path("config/training_points.yaml"),
            environment_kwargs=environment_kwargs(args, None),
        )
        try:
            group.reset_round(
                world_index=0,
                seeds=[
                    episode_seed(args.seed, 0, index + 1)
                    for index in range(robot_count)
                ],
            )
            action = np.zeros(3, dtype=np.float32)
            group.step_batch({index: action for index in range(robot_count)})
            active = list(range(1, robot_count))
            started = time.monotonic()
            recycled, passive = group.reset_slots(
                world_index=0,
                assignments={0: episode_seed(args.seed, 0, robot_count + 1)},
                active_indices=active,
            )
            reset_wall_s = time.monotonic() - started
            if set(recycled) != {0} or set(passive) != set(active):
                raise RuntimeError("recycle smoke returned an incomplete fleet")
            if any(value[2] or value[3] for value in passive.values()):
                raise RuntimeError("a passive peer ended during recycle smoke")
            following = group.step_batch({
                index: action for index in range(robot_count)
            })
            if set(following) != set(range(robot_count)):
                raise RuntimeError("fleet did not resume after slot recycling")
        finally:
            group.close()

    return {
        "robot_count": robot_count,
        "training_kinematic": training_kinematic,
        "physics_step_size": physics_step_size,
        "lidar_samples": lidar_samples,
        "recycled_slots": len(recycled),
        "accounted_active_peers": len(passive),
        "reset_wall_s": reset_wall_s,
        "fleet_resumed": True,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pids = active_training_pids()
    if pids:
        raise RuntimeError(
            "ppo_benchmark refuses to compete with active ppo_train PIDs: "
            + ", ".join(str(pid) for pid in pids)
        )
    results = [
        run_case(
            robot_count=count,
            steps=args.steps,
            physics_step_size=args.physics_step_size,
            lidar_samples=args.lidar_samples,
            training_kinematic=not args.detailed,
        )
        for count in args.robot_counts
    ]
    output: object = results
    if args.recycle_smoke:
        output = {
            "benchmarks": results,
            "recycle_smoke": run_recycle_smoke(
                robot_count=max(2, max(args.robot_counts)),
                physics_step_size=args.physics_step_size,
                lidar_samples=args.lidar_samples,
                training_kinematic=not args.detailed,
            ),
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
