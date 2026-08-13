"""Pure tests for PPO geometry, observations, and reward shaping."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from martha.PPO.actions import ActionLimits, limit_action_rate
from martha.PPO.observations import (
    build_observation,
    goal_features,
    reduce_laser_scan,
)
from martha.PPO.reward import RewardConfig, calculate_reward
from martha.PPO.martha_env import MarthaEnv, fault_topic_for_backend
from martha.PPO.world_map import BoxObstacle, WorldMap, discover_worlds
from martha.simulation_speed import (
    MAX_SIM_SPEED_FACTOR,
    create_scaled_world,
    validate_sim_speed_factor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLDS_DIRECTORY = PROJECT_ROOT / "worlds"
TRAINING_WORLDS = tuple(
    WORLDS_DIRECTORY / f"mundo_{index}.world" for index in range(1, 10)
)


@pytest.mark.parametrize("value", (0.0, -1.0, np.inf, np.nan, "invalid"))
def test_sim_speed_factor_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="sim_speed_factor"):
        validate_sim_speed_factor(value)


def test_sim_speed_factor_has_a_safety_ceiling():
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_sim_speed_factor(MAX_SIM_SPEED_FACTOR + 0.1)


def test_scaled_world_changes_wall_speed_not_physics_step(tmp_path):
    source = TRAINING_WORLDS[0]
    scaled_path = create_scaled_world(source, 4.0, directory=tmp_path)

    source_physics = ET.parse(source).getroot().find("./world/physics")
    scaled_physics = ET.parse(scaled_path).getroot().find("./world/physics")
    assert source_physics is not None
    assert scaled_physics is not None
    assert float(source_physics.findtext("max_step_size")) == 0.001
    assert float(scaled_physics.findtext("max_step_size")) == 0.001
    assert float(scaled_physics.findtext("real_time_update_rate")) == 4000.0
    assert float(scaled_physics.findtext("real_time_factor")) == 4.0


def test_discovers_all_training_worlds_in_numeric_order():
    discovered = discover_worlds(WORLDS_DIRECTORY)

    assert tuple(path.name for path in discovered) == tuple(
        f"mundo_{index}.world" for index in range(1, 10)
    )


def test_motor_fault_topic_is_hardware_only():
    assert fault_topic_for_backend("gazebo") == ""
    assert fault_topic_for_backend("hardware") == "/hardware/motor_fault"
    with pytest.raises(ValueError, match="backend"):
        fault_topic_for_backend("invalid")


def test_failed_reset_state_cannot_authorize_an_old_transition():
    env = object.__new__(MarthaEnv)
    env._closed = False
    env._step_count = 12
    env._previous_action = np.ones(3, dtype=np.float32)
    env._previous_distance = 4.0
    env._last_observation = np.ones(45, dtype=np.float32)
    env._last_snapshot = object()
    env._world_map = object()
    env._distance_field = np.ones((2, 2), dtype=np.float32)
    env._episode_sample = object()

    env._invalidate_episode_state()

    assert env._step_count == 0
    np.testing.assert_array_equal(env._previous_action, np.zeros(3))
    assert math.isinf(env._previous_distance)
    assert env._last_observation is None
    assert env._last_snapshot is None
    assert env._world_map is None
    assert env._distance_field is None
    assert env._episode_sample is None
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        env.step(np.zeros(3, dtype=np.float32))


def test_gazebo_dynamics_reset_precedes_scenario_replacement():
    env = object.__new__(MarthaEnv)
    calls = []
    env._call_empty = lambda service: calls.append(("service", service))
    env._switch_world = lambda index: calls.append(("world", index))

    env._reset_world_scenario(4)

    assert calls == [("service", "reset_world"), ("world", 4)]


def test_pause_service_timeout_retries_and_keeps_robot_stopped():
    env = object.__new__(MarthaEnv)
    env.service_timeout = 0.1

    class FakeRos:
        def __init__(self):
            self.calls = 0
            self.stops = 0

        def call_service(self, key, request, timeout):
            self.calls += 1
            if self.calls < 3:
                raise TimeoutError("transient pause timeout")

        def publish_stop(self):
            self.stops += 1

    env.ros = FakeRos()

    env._call_empty("pause")

    assert env.ros.calls == 3
    assert env.ros.stops == 2


def test_repeated_world_switches_confirm_deletion_and_reuse_stable_names():
    env = object.__new__(MarthaEnv)
    world = type(
        "FakeWorld",
        (),
        {
            "scenario_model_names": ("wall_left", "obstacle"),
            "model_xml": {"wall_left": "wall xml", "obstacle": "obs xml"},
        },
    )()
    env.predefined_maps = (world,)
    env._scenario_entity_names = set()
    env._goal_marker_name = None
    stale_entities = {
        "martha_ppo_s58_wall_left",
        "martha_ppo_goal_58",
    }
    events = []

    class FakeRos:
        def gazebo_model_names(self):
            return stale_entities

        def gazebo_model_inventory(self):
            return 7, stale_entities

        def wait_for_gazebo_entities_absent(
            self, names, after_sequence, timeout
        ):
            events.append(("confirmed", set(names), after_sequence, timeout))
            return set()

        def publish_stop(self):
            events.append(("stop",))

    env.ros = FakeRos()
    env.control_timeout = 2.0
    env._call_empty = lambda name: events.append((name,))
    deleted = []
    spawned = []
    env._delete_entity = lambda name, **_: deleted.append(name)
    env._spawn_payload = lambda xml: (xml, object())
    env._spawn_entity = lambda name, xml, pose: spawned.append(name)

    env._switch_world(0)
    env._switch_world(0)

    assert spawned == [
        "martha_ppo_s_wall_left",
        "martha_ppo_s_obstacle",
        "martha_ppo_s_wall_left",
        "martha_ppo_s_obstacle",
    ]
    assert set(spawned) <= set(deleted)
    assert stale_entities <= set(deleted)
    confirmations = [event for event in events if event[0] == "confirmed"]
    assert len(confirmations) == 2
    assert all(event[2] == 7 for event in confirmations)


@pytest.mark.parametrize("world_path", TRAINING_WORLDS, ids=lambda path: path.stem)
def test_every_world_samples_a_connected_start_and_goal(world_path):
    world = WorldMap.from_sdf(world_path)
    seed = int(world_path.stem.rsplit("_", 1)[1])
    sample = world.sample_episode(
        np.random.default_rng(seed),
        min_goal_distance=1.0,
        attempts=2_000,
    )

    start_index = world.grid_index(sample.start_x, sample.start_y)
    goal_index = world.grid_index(sample.goal_x, sample.goal_y)

    assert start_index is not None
    assert goal_index is not None
    assert world.free[start_index]
    assert world.free[goal_index]
    assert world.components[start_index] >= 0
    assert world.components[start_index] == world.components[goal_index]
    assert -np.pi <= sample.start_yaw <= np.pi

    distance_field = world.distance_field(sample.goal_x, sample.goal_y)
    measured_distance = world.path_distance(
        sample.start_x,
        sample.start_y,
        distance_field,
    )
    euclidean_distance = np.hypot(
        sample.goal_x - sample.start_x,
        sample.goal_y - sample.start_y,
    )

    assert np.isfinite(measured_distance)
    assert measured_distance == pytest.approx(sample.shortest_path)
    assert measured_distance >= euclidean_distance - 1e-6
    assert world.path_distance(
        sample.goal_x,
        sample.goal_y,
        distance_field,
    ) == pytest.approx(0.0)


def test_geodesic_distance_routes_around_an_inflated_obstacle():
    world = WorldMap(
        path=Path("synthetic_barrier.world"),
        world_name="synthetic_barrier",
        bounds=(0.0, 4.0, 0.0, 4.0),
        obstacles=(
            BoxObstacle(
                name="barrier",
                x=2.0,
                y=2.0,
                size_x=0.20,
                size_y=3.0,
            ),
        ),
        model_xml={},
        resolution=0.10,
        robot_clearance=0.10,
    )
    start = (1.0, 2.0)
    goal = (3.0, 2.0)
    distance_field = world.distance_field(*goal)
    geodesic_distance = world.path_distance(*start, distance_field)
    direct_distance = np.hypot(goal[0] - start[0], goal[1] - start[1])

    assert np.isfinite(geodesic_distance)
    assert geodesic_distance > direct_distance + 0.5


def test_geodesic_potential_interpolates_between_grid_centers():
    world = WorldMap(
        path=Path("synthetic_open.world"),
        world_name="synthetic_open",
        bounds=(0.0, 4.0, 0.0, 4.0),
        obstacles=(),
        model_xml={},
        resolution=0.10,
        robot_clearance=0.10,
    )
    goal_x = float(world.x_coordinates[20])
    goal_y = float(world.y_coordinates[20])
    start_x = float(world.x_coordinates[10])
    start_y = goal_y
    distance_field = world.distance_field(goal_x, goal_y)

    at_center = world.path_distance(start_x, start_y, distance_field)
    between_centers = world.path_distance(
        start_x + 0.5 * world.resolution,
        start_y,
        distance_field,
    )

    assert between_centers == pytest.approx(
        at_center - 0.5 * world.resolution,
        abs=1e-5,
    )


def test_action_limits_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        ActionLimits(max_vx=np.nan).as_array()
    with pytest.raises(ValueError, match="finite"):
        limit_action_rate((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), np.inf)


def test_scan_reduction_sanitizes_nan_and_infinities():
    sectors, minimum_range = reduce_laser_scan(
        [np.nan, np.inf, -np.inf, 0.50, 99.0],
        range_min=0.12,
        range_max=8.0,
        sectors=5,
        angle_min=-np.pi,
        angle_increment=2.0 * np.pi / 5.0,
    )

    np.testing.assert_allclose(
        sectors,
        np.asarray([1.0, 1.0, 0.12 / 8.0, 0.50 / 8.0, 1.0]),
    )
    assert minimum_range == pytest.approx(0.12)
    assert np.isfinite(sectors).all()
    assert np.all((0.0 <= sectors) & (sectors <= 1.0))


def test_scan_reduction_requires_angular_metadata():
    with pytest.raises(TypeError, match="angle_min"):
        reduce_laser_scan(
            [1.0, 2.0, 3.0],
            range_min=0.12,
            range_max=8.0,
            sectors=3,
        )


def test_scan_reduction_canonicalizes_start_angle_and_direction():
    counter_clockwise, _ = reduce_laser_scan(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        range_min=0.12,
        range_max=8.0,
        sectors=8,
        angle_min=-np.pi,
        angle_increment=np.pi / 4.0,
    )
    clockwise_from_zero, _ = reduce_laser_scan(
        [5.0, 4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0],
        range_min=0.12,
        range_max=8.0,
        sectors=8,
        angle_min=0.0,
        angle_increment=-np.pi / 4.0,
    )

    np.testing.assert_allclose(counter_clockwise, clockwise_from_zero)


@pytest.mark.parametrize("samples", (360, 720, 1440))
@pytest.mark.parametrize("direction", (-1.0, 1.0))
def test_scan_reduction_is_independent_of_a2m8_sample_density(
    samples,
    direction,
):
    expected = np.linspace(0.4, 7.6, 36, dtype=np.float32)
    angle_min = -np.pi if direction > 0.0 else np.pi
    angle_increment = direction * 2.0 * np.pi / samples
    angles = angle_min + np.arange(samples) * angle_increment
    canonical = (angles + np.pi) % (2.0 * np.pi) - np.pi
    indices = np.floor(
        (canonical + np.pi) * expected.size / (2.0 * np.pi)
    ).astype(np.int32)
    indices = np.clip(indices, 0, expected.size - 1)

    reduced, _ = reduce_laser_scan(
        expected[indices],
        range_min=0.12,
        range_max=8.0,
        sectors=expected.size,
        angle_min=angle_min,
        angle_increment=angle_increment,
    )

    np.testing.assert_allclose(reduced, expected / 8.0)


def test_observation_has_backend_independent_shape_and_finite_values():
    scan, _ = reduce_laser_scan(
        np.linspace(0.20, 8.0, 360),
        range_min=0.12,
        range_max=8.0,
        sectors=36,
        angle_min=-np.pi,
        angle_increment=2.0 * np.pi / 360.0,
    )
    goal, distance, bearing = goal_features(
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=np.pi / 2.0,
        goal_x=2.0,
        goal_y=0.0,
        max_goal_distance=10.0,
    )
    observation = build_observation(
        laser_sectors=scan,
        goal=goal,
        velocity=(0.5, -0.5, 0.25),
        previous_action=(0.25, -0.25, 0.5),
        max_linear_speed=1.0,
        max_angular_speed=0.5,
    )

    assert distance == pytest.approx(2.0)
    assert bearing == pytest.approx(-np.pi / 2.0)
    assert observation.shape == (45,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    np.testing.assert_allclose(observation[-6:-3], [0.5, -0.5, 0.5])


def test_reward_orders_progress_stationary_and_regression():
    arguments = {
        "minimum_scan": 8.0,
        "action": (0.0, 0.0, 0.0),
        "previous_action": (0.0, 0.0, 0.0),
        "reached_goal": False,
        "collision": False,
        "out_of_bounds": False,
    }

    progress, progress_components = calculate_reward(5.0, 4.0, **arguments)
    stationary, _ = calculate_reward(5.0, 5.0, **arguments)
    regression, regression_components = calculate_reward(5.0, 6.0, **arguments)

    assert progress > stationary > regression
    assert progress_components["progress"] > 0.0
    assert regression_components["progress"] < 0.0


def test_coulomb_progress_reward_grows_near_the_goal_and_is_bounded():
    arguments = {
        "minimum_scan": 8.0,
        "action": (0.0, 0.0, 0.0),
        "previous_action": (0.0, 0.0, 0.0),
        "reached_goal": False,
        "collision": False,
        "out_of_bounds": False,
    }
    config = RewardConfig(progress_distance_floor=0.25)

    _, far_components = calculate_reward(5.0, 4.0, config=config, **arguments)
    _, near_components = calculate_reward(1.0, 0.5, config=config, **arguments)
    _, capped_components = calculate_reward(
        0.25,
        0.0,
        config=config,
        **arguments,
    )

    assert near_components["progress"] > far_components["progress"]
    assert capped_components["progress"] == pytest.approx(0.0)


def test_goal_bonus_and_collision_penalty_have_terminal_precedence():
    config = RewardConfig()
    arguments = {
        "previous_distance": 0.2,
        "distance": 0.2,
        "minimum_scan": 8.0,
        "action": (0.0, 0.0, 0.0),
        "previous_action": (0.0, 0.0, 0.0),
        "out_of_bounds": False,
        "config": config,
    }

    neutral, neutral_components = calculate_reward(
        reached_goal=False,
        collision=False,
        **arguments,
    )
    goal, goal_components = calculate_reward(
        reached_goal=True,
        collision=False,
        **arguments,
    )
    collision, collision_components = calculate_reward(
        reached_goal=True,
        collision=True,
        **arguments,
    )

    assert goal_components["terminal"] == pytest.approx(config.goal_reward)
    assert collision_components["terminal"] == pytest.approx(
        -config.collision_penalty
    )
    assert neutral_components["terminal"] == 0.0
    assert goal > neutral > collision


def test_invalid_geodesic_cannot_cancel_a_collision_penalty():
    reward, components = calculate_reward(
        previous_distance=5.0,
        distance=math.inf,
        minimum_scan=0.1,
        action=(1.0, 0.0, 0.0),
        previous_action=(0.0, 0.0, 0.0),
        reached_goal=False,
        collision=True,
        out_of_bounds=False,
    )

    assert components["progress"] == 0.0
    assert components["terminal"] == -RewardConfig().collision_penalty
    assert reward < -RewardConfig().collision_penalty
