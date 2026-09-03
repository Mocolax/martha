"""Pure tests for PPO geometry, observations, and reward shaping."""

import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import martha.PPO.martha_env as martha_env_module
from martha.PPO.actions import ActionLimits, limit_action_rate
from martha.PPO.observations import (
    OBSERVATION_FRAME_SIZE,
    OBSERVATION_SIZE,
    ObservationHistory,
    build_observation_frame,
    goal_features,
    reduce_laser_scan,
)
from martha.PPO.reward import (
    RewardConfig,
    RewardState,
    calculate_reward,
    update_stagnation,
)
from martha.PPO.martha_env import (
    MarthaEnv,
    fault_topic_for_backend,
    footprint_scan_hit,
)
from martha.PPO.world_map import (
    TRAINING_WORLD_NAMES,
    BoxObstacle,
    WorldMap,
    discover_worlds,
)
from martha.simulation_speed import (
    MAX_PHYSICS_STEP_SIZE,
    MAX_SIM_SPEED_FACTOR,
    create_scaled_world,
    validate_physics_step_size,
    validate_sim_speed_factor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLDS_DIRECTORY = PROJECT_ROOT / "worlds"
TRAINING_WORLDS = tuple(
    WORLDS_DIRECTORY / f"{name}.world" for name in TRAINING_WORLD_NAMES
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
    scaled_world = ET.parse(scaled_path).getroot().find("./world")
    state_plugins = scaled_world.findall(
        "plugin[@filename='libgazebo_ros_state.so']"
    )
    assert len(state_plugins) == 1
    assert state_plugins[0].findtext("./ros/namespace") == "/gazebo"


@pytest.mark.parametrize("value", (0.0, -1.0, np.inf, np.nan, "invalid"))
def test_physics_step_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="physics_step_size"):
        validate_physics_step_size(value)


def test_training_world_can_override_physics_step_without_editing_source(tmp_path):
    source = TRAINING_WORLDS[0]
    scaled_path = create_scaled_world(
        source,
        3.0,
        directory=tmp_path,
        physics_step_size=0.002,
    )

    source_physics = ET.parse(source).getroot().find("./world/physics")
    scaled_physics = ET.parse(scaled_path).getroot().find("./world/physics")
    assert float(source_physics.findtext("max_step_size")) == 0.001
    assert float(scaled_physics.findtext("max_step_size")) == 0.002
    assert float(scaled_physics.findtext("real_time_update_rate")) == 1500.0
    assert float(scaled_physics.findtext("real_time_factor")) == 3.0

    with pytest.raises(ValueError, match="cannot exceed"):
        validate_physics_step_size(MAX_PHYSICS_STEP_SIZE + 0.001)


def test_discovers_exactly_the_six_training_worlds_in_layout_order():
    discovered = discover_worlds(WORLDS_DIRECTORY)

    assert tuple(path.stem for path in discovered) == TRAINING_WORLD_NAMES


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
    env._observation_history = ObservationHistory()
    env._reward_state = RewardState.initial(4.0)
    env._stagnation_reference_distance = 4.0
    env._stagnation_steps = 12
    env._last_observation = np.ones(OBSERVATION_SIZE, dtype=np.float32)
    env._last_snapshot = object()
    env._world_map = object()
    env._distance_field = np.ones((2, 2), dtype=np.float32)
    env._episode_sample = object()

    env._invalidate_episode_state()

    assert env._step_count == 0
    np.testing.assert_array_equal(env._previous_action, np.zeros(3))
    assert env._reward_state is None
    assert env._stagnation_reference_distance is None
    assert env._stagnation_steps == 0
    assert env._last_observation is None
    assert env._last_snapshot is None
    assert env._world_map is None
    assert env._distance_field is None
    assert env._episode_sample is None
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        env.step(np.zeros(3, dtype=np.float32))


def test_gazebo_dynamics_reset_precedes_scenario_replacement():
    env = object.__new__(MarthaEnv)
    env.preloaded_worlds = False
    calls = []
    env._call_empty = lambda service: calls.append(("service", service))
    env._switch_world = lambda index: calls.append(("world", index))

    env._reset_world_scenario(4)

    assert calls == [("service", "reset_world"), ("world", 4)]


def test_pause_service_timeout_retries_and_keeps_robot_stopped(monkeypatch):
    monkeypatch.setattr(
        martha_env_module,
        "Empty",
        type("FakeEmpty", (), {"Request": object}),
    )
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


def test_non_navigable_map_pose_terminates_instead_of_crashing_stagnation():
    env = object.__new__(MarthaEnv)
    env._closed = False
    env.backend = "gazebo"
    env._last_observation = np.zeros(168, dtype=np.float32)
    env._step_count = 0
    env._previous_action = np.zeros(3, dtype=np.float32)
    env._active_world_index = 0
    env._world_map = SimpleNamespace(is_free_pose=lambda x, y: False)
    env._distance_field = np.zeros((1, 1), dtype=np.float64)
    env._reward_state = RewardState.initial(3.0)
    env._stagnation_reference_distance = 3.0
    env._stagnation_steps = 0
    env.max_steps = 800
    env.goal_tolerance = 0.25
    env.reward_config = RewardConfig()
    env._build_observation = lambda snapshot: (
        np.zeros(168, dtype=np.float32),
        3.0,
    )
    env._episode_euclidean_distance = lambda snapshot, distance: distance
    env._distance_for_metrics = lambda snapshot, distance: math.inf
    env._goal_bearing = lambda snapshot: 0.0

    class FakeRos:
        motor_fault = False

        def __init__(self):
            self.stop_calls = 0

        def publish_stop(self):
            self.stop_calls += 1

    env.ros = FakeRos()
    pending = SimpleNamespace(
        requested_action=np.zeros(3, dtype=np.float32),
        limited_action=np.zeros(3, dtype=np.float32),
        applied_action=np.zeros(3, dtype=np.float32),
        command=np.zeros(3, dtype=np.float32),
        command_inhibited=False,
    )
    snapshot = SimpleNamespace(
        x=0.0,
        y=0.0,
        yaw=0.0,
        goal_x=3.0,
        goal_y=0.0,
        ground_truth_x=0.0,
        ground_truth_y=0.0,
        ground_truth_yaw=0.0,
        minimum_scan=0.3,
        motor_fault=False,
    )

    _, reward, terminated, truncated, info = env.finish_step(
        pending,
        snapshot,
        contact_collision=False,
    )

    assert terminated is True
    assert truncated is False
    assert info["out_of_bounds"] is True
    assert info["stagnated"] is False
    assert reward == pytest.approx(-RewardConfig().out_of_bounds_penalty)
    assert env.ros.stop_calls == 1


def test_repeated_world_switches_confirm_deletion_and_reuse_stable_names():
    env = object.__new__(MarthaEnv)
    env.preloaded_worlds = False
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
    seed = TRAINING_WORLDS.index(world_path) + 1
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
    assert measured_distance == pytest.approx(sample.shortest_path, abs=2e-4)
    assert measured_distance >= euclidean_distance - 1e-6
    assert world.path_distance(
        sample.goal_x,
        sample.goal_y,
        distance_field,
    ) == pytest.approx(0.0, abs=2e-4)


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


def test_footprint_scan_diagnostic_identifies_sector_angle_and_threshold():
    sectors = np.ones(36, dtype=np.float32)
    sectors[0] = 0.12 / 8.0

    hit = footprint_scan_hit(sectors, 8.0, safety_margin=0.01)

    assert hit is not None
    assert hit.sector_index == 0
    assert math.degrees(hit.angle_radians) == pytest.approx(-175.0)
    assert hit.measured_range == pytest.approx(0.12)
    assert hit.threshold > hit.measured_range
    assert footprint_scan_hit(np.ones(36), 8.0, safety_margin=0.01) is None


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


def test_observation_frame_has_backend_independent_values():
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
        goal_distance_scale=3.0,
    )
    frame = build_observation_frame(
        laser_sectors=scan,
        goal=goal,
        velocity=(0.5, -0.5, 0.25),
        max_linear_speed=1.0,
        max_angular_speed=0.5,
    )

    assert distance == pytest.approx(2.0)
    assert bearing == pytest.approx(-np.pi / 2.0)
    assert goal[0] == pytest.approx(2.0 / 5.0)
    assert frame.shape == (OBSERVATION_FRAME_SIZE,)
    assert frame.dtype == np.float32
    assert np.isfinite(frame).all()
    np.testing.assert_allclose(frame[-3:], [0.5, -0.5, 0.5])


def test_rational_goal_distance_has_no_finite_maximum_or_clipping():
    near, _, _ = goal_features(0.0, 0.0, 0.0, 13.0, 0.0, 3.0)
    far, _, _ = goal_features(0.0, 0.0, 0.0, 22.0, 0.0, 3.0)

    assert near[0] == pytest.approx(13.0 / 16.0)
    assert far[0] == pytest.approx(22.0 / 25.0)
    assert 0.0 < near[0] < far[0] < 1.0


def test_observation_history_returns_the_newest_frame_only():
    history = ObservationHistory()
    frames = [
        np.full(OBSERVATION_FRAME_SIZE, value, dtype=np.float32)
        for value in range(4)
    ]

    first = history.push(frames[0], 0)
    history.push(frames[1], 333_333_333)
    history.push(frames[2], 666_666_667)
    observation = history.push(frames[3], 1_000_000_000)

    np.testing.assert_array_equal(first, frames[0])
    np.testing.assert_array_equal(observation, frames[3])


def test_observation_history_clears_when_ros_time_moves_backwards():
    history = ObservationHistory()
    old = np.ones(OBSERVATION_FRAME_SIZE, dtype=np.float32)
    current = np.full(OBSERVATION_FRAME_SIZE, 2.0, dtype=np.float32)
    history.push(old, 2_000_000_000)

    observation = history.push(current, 1_000_000_000)

    np.testing.assert_array_equal(observation, current)


def _paper_reward(
    state: RewardState,
    distance: float,
    *,
    bearing: float = 0.0,
    minimum_scan: float = 8.0,
    angular_velocity: float = 0.0,
    reached_goal: bool = False,
    collision: bool = False,
    out_of_bounds: bool = False,
    timeout: bool = False,
    stagnated: bool = False,
    config: RewardConfig = RewardConfig(),
):
    return calculate_reward(
        state=state,
        distance=distance,
        goal_bearing=bearing,
        minimum_scan=minimum_scan,
        angular_velocity=angular_velocity,
        reached_goal=reached_goal,
        collision=collision,
        out_of_bounds=out_of_bounds,
        timeout=timeout,
        stagnated=stagnated,
        config=config,
    )


def test_stagnation_requires_a_full_window_without_meaningful_progress():
    config = RewardConfig(
        stagnation_window_steps=3,
        stagnation_min_progress=0.12,
    )
    reference = 5.0
    steps = 0

    reference, steps, stagnated = update_stagnation(
        reference, steps, 4.95, config
    )
    assert (reference, steps, stagnated) == (5.0, 1, False)
    reference, steps, stagnated = update_stagnation(
        reference, steps, 4.87, config
    )
    assert (reference, steps, stagnated) == (4.87, 0, False)
    for expected_steps in range(1, 4):
        reference, steps, stagnated = update_stagnation(
            reference, steps, 4.87, config
        )
        assert steps == expected_steps
        assert stagnated is (expected_steps == 3)


def test_stagnation_has_an_exclusive_terminal_penalty():
    state = RewardState.initial(5.0)
    reward, components, _ = _paper_reward(
        state,
        5.0,
        stagnated=True,
    )

    assert reward == pytest.approx(-2.0)
    assert components["terminal"] == pytest.approx(-2.0)
    assert all(
        value == 0.0
        for key, value in components.items()
        if key != "terminal"
    )


def test_paper_distance_reward_uses_asymmetric_linear_scales():
    config = RewardConfig()
    _, approaching, _ = _paper_reward(RewardState.initial(5.0), 4.0)
    _, retreating, _ = _paper_reward(RewardState.initial(5.0), 6.0)

    assert approaching["distance"] == pytest.approx(
        config.distance_positive_scale
    )
    assert retreating["distance"] == pytest.approx(
        -config.distance_negative_scale
    )


def test_paper_orientation_rewards_front_without_penalizing_back():
    config = RewardConfig()
    _, front, _ = _paper_reward(RewardState.initial(5.0), 5.0, bearing=0.0)
    _, back, _ = _paper_reward(
        RewardState.initial(5.0),
        5.0,
        bearing=math.pi,
    )

    assert front["orientation"] == pytest.approx(
        config.orientation_positive_scale
    )
    assert back["orientation"] == 0.0


def test_paper_nonterminal_transition_has_small_step_cost():
    config = RewardConfig()
    reward, components, _ = _paper_reward(
        RewardState.initial(5.0),
        5.0,
        bearing=math.pi,
    )

    assert components["step"] == pytest.approx(-config.step_penalty)
    assert reward == pytest.approx(-config.step_penalty)


def test_paper_shortest_distance_rewards_only_new_episode_records():
    config = RewardConfig()
    _, first, state = _paper_reward(RewardState.initial(5.0), 4.0)
    _, retreat, state = _paper_reward(state, 4.5)
    _, new_record, state = _paper_reward(state, 3.5)

    assert first["shortest_distance"] == pytest.approx(
        config.shortest_distance_scale
    )
    assert retreat["shortest_distance"] == 0.0
    assert new_record["shortest_distance"] == pytest.approx(
        0.5 * config.shortest_distance_scale
    )
    assert state.best_distance == pytest.approx(3.5)


def test_paper_laser_penalty_is_linear_below_clearance_threshold():
    config = RewardConfig()
    _, safe, _ = _paper_reward(RewardState.initial(5.0), 5.0, minimum_scan=0.65)
    _, close, _ = _paper_reward(RewardState.initial(5.0), 5.0, minimum_scan=0.15)

    assert safe["laser"] == 0.0
    assert close["laser"] == pytest.approx(
        -(config.laser_clearance_distance - 0.15)
        * config.laser_penalty_scale
    )


def test_paper_wiggle_penalty_uses_direct_reversals_in_a_ten_step_window():
    config = RewardConfig()
    state = RewardState.initial(5.0)
    components = {}
    for angular_velocity in (0.3, -0.3, 0.3, -0.3, 0.3):
        _, components, state = _paper_reward(
            state,
            5.0,
            angular_velocity=angular_velocity,
        )

    assert sum(state.reversals) == 4
    assert components["wiggle"] == pytest.approx(-config.wiggle_penalty)
    _, straight, state = _paper_reward(state, 5.0, angular_velocity=0.0)
    _, resumed, _ = _paper_reward(state, 5.0, angular_velocity=-0.3)
    assert straight["wiggle"] == pytest.approx(-config.wiggle_penalty)
    assert resumed["wiggle"] == pytest.approx(-config.wiggle_penalty)


def test_paper_terminal_and_timeout_rewards_are_exclusive():
    config = RewardConfig()
    state = RewardState.initial(0.2)
    goal, goal_components, _ = _paper_reward(
        state,
        0.1,
        minimum_scan=0.1,
        reached_goal=True,
    )
    collision, collision_components, _ = _paper_reward(
        state,
        0.1,
        reached_goal=True,
        collision=True,
    )
    timeout, timeout_components, _ = _paper_reward(
        state,
        0.1,
        timeout=True,
    )

    assert goal == pytest.approx(config.goal_reward)
    assert collision == pytest.approx(-config.collision_penalty)
    assert timeout == pytest.approx(-config.timeout_penalty)
    assert all(value == 0.0 for key, value in goal_components.items() if key != "terminal")
    assert all(value == 0.0 for key, value in collision_components.items() if key != "terminal")
    assert timeout_components["terminal"] == pytest.approx(
        -config.timeout_penalty
    )
    assert all(
        value == 0.0
        for key, value in timeout_components.items()
        if key != "terminal"
    )
