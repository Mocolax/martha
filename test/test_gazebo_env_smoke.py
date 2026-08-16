"""Opt-in live Gazebo smoke test for the PPO environment."""

import os

import numpy as np
import pytest

from martha.PPO.observations import OBSERVATION_SIZE


pytest.importorskip("gymnasium")
pytest.importorskip("rclpy")

from martha.PPO.martha_env import MarthaEnv  # noqa: E402


@pytest.mark.skipif(
    os.environ.get("MARTHA_GAZEBO_SMOKE") != "1",
    reason="requires the canonical Martha Gazebo bringup",
)
def test_all_gazebo_worlds_reset_ekf_and_produce_fresh_transitions():
    """Swap every world, align its EKF and execute one bounded transition."""
    env = MarthaEnv(
        backend="gazebo",
        map_mode="predefined",
        map_index=0,
        max_steps=5,
    )
    try:
        assert len(env.predefined_maps) == 9
        for world_index in range(len(env.predefined_maps)):
            observation, reset_info = env.reset(
                seed=7 + world_index,
                options={"world_index": world_index},
            )
            assert observation.shape == (OBSERVATION_SIZE,)
            assert np.isfinite(observation).all()
            assert reset_info["world_index"] == world_index
            assert reset_info["position"][:2] == pytest.approx(
                (0.0, 0.0),
                abs=0.08,
            ), reset_info
            assert abs(reset_info["position"][2]) <= 0.12, reset_info

            next_observation, reward, terminated, truncated, info = env.step(
                np.asarray((0.1, 0.0, 0.0), dtype=np.float32)
            )
            assert next_observation.shape == (OBSERVATION_SIZE,)
            assert np.isfinite(next_observation).all()
            assert np.isfinite(reward)
            assert info["world_index"] == world_index
            assert info["sensor_timeout"] is False
            assert info["motor_fault"] is False
            assert not terminated, info
            assert not truncated
    finally:
        env.close()
