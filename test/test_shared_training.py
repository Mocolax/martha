"""Pure tests for the one-server, multi-Martha training coordinator."""

from pathlib import Path
from types import SimpleNamespace
import threading
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from martha.PPO.martha_env import RosObservationNode, external_contact_models
from martha.PPO.shared_gazebo import SharedGazeboEnvironments
from martha.PPO.training_layout import (
    MIN_START_SEPARATION,
    WORLD_ORIGINS,
    TrainingPoint,
    create_combined_training_world,
    load_training_points,
    parking_pose,
    sample_round_episodes,
    shuffled_world_index,
    validate_training_points,
)
from martha.PPO.world_map import TRAINING_WORLD_NAMES, WorldMap, discover_worlds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLDS_DIRECTORY = PROJECT_ROOT / "worlds"


def _well_spaced_points(world_map, count=10):
    labels, sizes = np.unique(
        world_map.components[world_map.free],
        return_counts=True,
    )
    component = int(labels[int(np.argmax(sizes))])
    selected = []
    for row, column in np.argwhere(world_map.components == component):
        point = TrainingPoint(
            point_id=f"{world_map.world_name}_{len(selected):02d}",
            x=float(world_map.x_coordinates[column]),
            y=float(world_map.y_coordinates[row]),
        )
        if all(
            np.hypot(point.x - other.x, point.y - other.y) >= 1.5
            for other in selected
        ):
            selected.append(point)
        if len(selected) == count:
            return tuple(selected)
    raise AssertionError(f"not enough test points in {world_map.world_name}")


def _actual_maps():
    return tuple(
        WorldMap.from_sdf(path) for path in discover_worlds(WORLDS_DIRECTORY)
    )


def test_catalog_loader_rejects_missing_worlds_and_duplicate_ids(tmp_path):
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("worlds:\n  room:\n    points: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="world mismatch"):
        load_training_points(incomplete)

    document = {
        "worlds": {
            name: {
                "points": [
                    {"id": f"{name}_same", "x": 0.0, "y": 0.0},
                    {"id": f"{name}_same", "x": 1.0, "y": 1.0},
                ]
            }
            for name in TRAINING_WORLD_NAMES
        }
    }
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="repeats point id"):
        load_training_points(duplicate)


def test_empty_shipped_catalog_fails_before_gazebo_with_world_name():
    maps = _actual_maps()
    catalog = load_training_points(PROJECT_ROOT / "config/training_points.yaml")

    with pytest.raises(ValueError, match="world four_rooms needs at least 8"):
        validate_training_points(
            catalog,
            maps,
            robot_count=8,
            min_goal_distance=2.0,
        )


def test_catalog_validation_and_sampling_use_unique_starts_and_distinct_goals():
    maps = _actual_maps()
    catalog = {world.world_name: _well_spaced_points(world) for world in maps}
    validate_training_points(
        catalog,
        maps,
        robot_count=8,
        min_goal_distance=1.0,
    )

    world = maps[0]
    episodes = sample_round_episodes(
        world,
        catalog[world.world_name],
        WORLD_ORIGINS[world.world_name],
        robot_count=8,
        min_goal_distance=1.0,
        rng=np.random.default_rng(17),
    )
    origin_x, origin_y = WORLD_ORIGINS[world.world_name]
    starts = {(sample.start_x, sample.start_y) for sample in episodes}

    assert len(starts) == 8
    for sample in episodes:
        assert (sample.start_x, sample.start_y) != (
            sample.goal_x,
            sample.goal_y,
        )
        assert sample.shortest_path >= 1.0
        assert world.is_free_pose(
            sample.start_x - origin_x,
            sample.start_y - origin_y,
        )
    for first_index, first in enumerate(episodes):
        for second in episodes[first_index + 1:]:
            assert np.hypot(
                first.start_x - second.start_x,
                first.start_y - second.start_y,
            ) >= MIN_START_SEPARATION


def test_combined_world_has_six_offset_islands_and_unique_model_names(tmp_path):
    world_paths = discover_worlds(WORLDS_DIRECTORY)
    combined_path = create_combined_training_world(
        world_paths,
        directory=tmp_path,
    )
    world = ET.parse(combined_path).getroot().find("world")
    assert world is not None
    names = [model.get("name") for model in world.findall("model")]

    assert len(names) == len(set(names))
    for world_name in TRAINING_WORLD_NAMES:
        assert any(name.startswith(f"{world_name}__") for name in names)
    assert world.find("plugin[@filename='libgazebo_ros_state.so']") is not None


def test_parking_row_is_unique_and_outside_every_arena():
    poses = [parking_pose(index, 8) for index in range(8)]

    assert len(set(poses)) == 8
    assert all(y == pytest.approx(-43.0) for _, y, _ in poses)
    assert all(
        abs(first[0] - second[0]) >= 4.0
        for index, first in enumerate(poses)
        for second in poses[index + 1:]
    )


def test_contact_filter_ignores_ground_and_self_but_reports_other_robot():
    pairs = {
        ("martha_0::base_link::shell", "ground_plane::link::collision"),
        ("martha_0::base_link::shell", "martha_0::wheel::collision"),
        ("martha_0::base_link::shell", "martha_1::base_link::shell"),
        ("wall::link::collision", "martha_0::base_link::shell"),
    }

    assert external_contact_models(pairs, "martha_0") == {"martha_1", "wall"}


def test_contact_barrier_discards_stale_events_and_accepts_fresh_empty_message():
    node = object.__new__(RosObservationNode)
    node._condition = threading.Condition()
    node._contact_sequence = 0
    node._contact_events = []
    contact = SimpleNamespace(
        collision1_name="martha_0::base_link::shell",
        collision2_name="wall::link::collision",
    )
    node._contact_callback(SimpleNamespace(states=[contact]))
    barrier = node.clear_contacts()
    node._contact_callback(SimpleNamespace(states=[]))

    assert node.wait_for_fresh_contact_message(barrier, 0.01)
    assert node.consume_contact_pairs(after_sequence=barrier) == set()


class _FakeRos:
    def __init__(self, pairs=()):
        self.pairs = set(pairs)

    def consume_contact_pairs(self, *, after_sequence):
        assert after_sequence == 7
        return set(self.pairs)


class _FakeEnvironment:
    def __init__(self, name, pairs=(), calls=None):
        self.robot_name = name
        self.ros = _FakeRos(pairs)
        self.calls = calls

    def _call_empty(self, operation):
        if self.calls is not None:
            self.calls.append(operation)

    def prepare_step(self, action):
        return SimpleNamespace(contact_sequence=7, action=action)

    def wait_for_step_snapshot(self, pending):
        return f"snapshot-{self.robot_name}"

    def finish_step(self, pending, snapshot, *, contact_collision):
        info = {"collision": contact_collision}
        return snapshot, 0.0, contact_collision, False, info


def test_vector_step_unpauses_and_pauses_once_and_ends_both_contact_robots():
    calls = []
    pair = (
        "martha_0::base_link::contact_shell_collision",
        "martha_1::base_link::contact_shell_collision",
    )
    group = object.__new__(SharedGazeboEnvironments)
    group.environments = [
        _FakeEnvironment("martha_0", [pair], calls),
        _FakeEnvironment("martha_1"),
    ]
    parked = []
    group.park = parked.append

    results = group.step_batch(
        {0: np.zeros(3, dtype=np.float32), 1: np.zeros(3, dtype=np.float32)}
    )

    assert calls == ["unpause", "pause"]
    assert results[0][2] and results[1][2]
    assert parked == [0, 1]


def test_world_shuffle_is_reproducible_and_has_no_repeats_per_cycle():
    first = [shuffled_world_index(42, index, 6) for index in range(12)]
    repeat = [shuffled_world_index(42, index, 6) for index in range(12)]
    different = [shuffled_world_index(43, index, 6) for index in range(12)]

    assert first == repeat
    assert first != different
    assert set(first[:6]) == set(range(6))
    assert set(first[6:]) == set(range(6))
