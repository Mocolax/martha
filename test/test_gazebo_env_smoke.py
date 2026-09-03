"""Opt-in live test for one gzserver and the eight-robot fleet."""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("gymnasium")
pytest.importorskip("rclpy")

from martha.PPO.shared_gazebo import SharedGazeboEnvironments  # noqa: E402
from martha.PPO.world_map import WorldMap, discover_worlds  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAZEBO_LIDAR_RANGE_MIN_METERS = 0.12


def _live_catalog(world_paths):
    worlds = {}
    for path in world_paths:
        world = WorldMap.from_sdf(path)
        labels, sizes = np.unique(
            world.components[world.free],
            return_counts=True,
        )
        component = int(labels[int(np.argmax(sizes))])
        points = []
        for row, column in np.argwhere(world.components == component):
            x = float(world.x_coordinates[column])
            y = float(world.y_coordinates[row])
            if all(
                np.hypot(x - point["x"], y - point["y"]) >= 1.5
                for point in points
            ):
                points.append(
                    {
                        "id": f"{world.world_name}_{len(points):02d}",
                        "x": x,
                        "y": y,
                        "yaw": None,
                    }
                )
            if len(points) == 10:
                break
        assert len(points) == 10
        worlds[world.world_name] = {"points": points}
    return {"worlds": worlds}


@pytest.mark.skipif(
    os.environ.get("MARTHA_GAZEBO_SMOKE") != "1",
    reason="requires the Martha ROS 2/Gazebo container",
)
def test_one_gzserver_eight_robots_six_arenas_and_contact_topics(tmp_path):
    """Prove fleet readiness plus a wheel-aligned terminal wall contact."""
    worlds_directory = PROJECT_ROOT / "worlds"
    world_paths = discover_worlds(worlds_directory)
    catalog = _live_catalog(world_paths)
    points_path = tmp_path / "training_points.yaml"
    points_path.write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    group = SharedGazeboEnvironments(
        count=8,
        sim_speed_factor=1.0,
        physics_step_size=0.002,
        lidar_samples=180,
        training_kinematic=True,
        show_gui=False,
        startup_timeout=180.0,
        run_directory=tmp_path,
        worlds_directory=worlds_directory,
        points_path=points_path,
        environment_kwargs={
            "backend": "gazebo",
            "map_mode": "predefined",
            "max_steps": 5,
            "min_goal_distance": 1.0,
        },
    )
    try:
        assert group._launch_process is not None
        assert group._launch_process.poll() is None
        assert len(group.environments) == 8
        assert {
            environment.ros.contact_subscription.topic_name
            for environment in group.environments
        } == {f"/martha_{index}/contacts" for index in range(8)}
        model_names = group.reference_env.ros.gazebo_model_names()
        assert {f"martha_{index}" for index in range(8)} <= model_names
        for world_name in catalog["worlds"]:
            assert any(name.startswith(f"{world_name}__") for name in model_names)

        activity_before_reset = [
            environment.ros.stream_activity()
            for environment in group.environments
        ]
        reset_results = group.reset_round(
            world_index=0,
            seeds=[10_000 + index for index in range(8)],
        )

        assert set(reset_results) == set(range(8))
        for index, environment in enumerate(group.environments):
            observation, info = reset_results[index]
            snapshot = environment._last_snapshot
            assert environment.observation_space.contains(observation)
            assert np.isfinite(observation).all()
            assert info["world_index"] == 0
            assert snapshot is not None
            assert snapshot.scan_sequence > activity_before_reset[index].scan_sequence
            assert (
                snapshot.odometry_sequence
                > activity_before_reset[index].odometry_sequence
            )
            assert (
                snapshot.ground_truth_sequence
                > activity_before_reset[index].ground_truth_sequence
            )
            assert snapshot.scan_stamp_ns > activity_before_reset[index].scan_stamp_ns
            assert (
                snapshot.odometry_stamp_ns
                > activity_before_reset[index].odometry_stamp_ns
            )
            assert np.isfinite(snapshot.laser_sectors).all()
            assert np.isfinite(snapshot.minimum_scan)
            assert snapshot.minimum_scan > GAZEBO_LIDAR_RANGE_MIN_METERS

        # four_rooms has its right inner wall face at local x=9.925 m.  At
        # x=9.640 the front wheel reaches that plane while the recessed LiDAR
        # body remains 5 mm clear.  The contact shell must still terminate the
        # episode, proving that wheel-first contacts are covered.
        first = group.environments[0]
        first._set_robot_state(-30.0 + 9.640, 15.0, 0.0)
        transition = group.step_batch(
            {0: np.zeros(3, dtype=np.float32)}
        )[0]
        _, _, terminated, truncated, info = transition
        assert terminated is True
        assert truncated is False
        assert info["collision"] is True
        assert info["out_of_bounds"] is False

    finally:
        group.close()
