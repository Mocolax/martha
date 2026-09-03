"""Pure contract tests for the guarded Gazebo throughput benchmark."""

import pytest


pytest.importorskip("torch")

from martha.PPO.benchmark import _benchmark_args, build_parser  # noqa: E402
from martha.PPO.train import DEFAULTS  # noqa: E402


def test_benchmark_defaults_cover_the_supported_fleet_sizes():
    args = build_parser().parse_args([])

    assert args.robot_counts == [1, 2, 4]
    assert args.steps == 240
    assert args.physics_step_size == 0.002
    assert args.lidar_samples == 180
    assert args.detailed is False
    assert args.recycle_smoke is False


def test_benchmark_can_enable_live_slot_recycling_smoke():
    args = build_parser().parse_args(["--recycle-smoke"])

    assert args.recycle_smoke is True


def test_benchmark_profile_does_not_change_learning_hyperparameters():
    args = _benchmark_args(
        robot_count=2,
        steps=20,
        physics_step_size=0.002,
        lidar_samples=180,
        training_kinematic=True,
    )

    assert args.num_envs == 2
    assert args.max_steps == 30
    assert args.lr == DEFAULTS.lr
    assert args.ppo_epochs == 8
    assert args.reward_scale == 1
