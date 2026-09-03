"""Unit tests for the coordinated shared-Gazebo reset barrier."""

from types import SimpleNamespace

import numpy as np
import pytest

from martha.PPO import martha_env
from martha.PPO.martha_env import MarthaEnv, PendingGazeboReset, PendingStep
from martha.PPO.shared_gazebo import (
    RECYCLE_PLACEMENT_ATTEMPTS,
    RecyclePlacementUnavailable,
    SharedGazeboEnvironments,
)


def _snapshot(sequence, *, safe=True, minimum_scan=2.0):
    return SimpleNamespace(
        scan_sequence=sequence,
        odometry_sequence=sequence,
        ground_truth_sequence=sequence,
        safe=safe,
        minimum_scan=minimum_scan,
    )


class _BatchRos:
    def __init__(self, *, ready=True, safety_sequence=()):
        self.ready = ready
        self.safety_sequence = tuple(safety_sequence)
        self.wait_calls = []

    def wait_for_fresh_snapshot(
        self,
        sequences,
        timeout,
        *,
        use_ground_truth,
        after_stamp_ns,
    ):
        self.wait_calls.append((sequences, timeout, after_stamp_ns))
        if not self.ready:
            return None
        index = len(self.wait_calls) - 1
        safe = (
            self.safety_sequence[index]
            if index < len(self.safety_sequence)
            else True
        )
        return _snapshot(
            index + 1,
            safe=safe,
            minimum_scan=2.0 if safe else 0.12,
        )

    def stream_activity(self):
        return SimpleNamespace(contact_sequence=1 if self.ready else 0)

    def missing_fresh_snapshot_data(self, *args, **kwargs):
        return ("scan(sequence=0, need>0)",)

    def consume_contact_pairs(self, *, after_sequence):
        return set()


class _BatchEnvironment:
    def __init__(self, index, events, *, ready=True, safety_sequence=()):
        self.index = index
        self.robot_name = f"martha_{index}"
        self.min_goal_distance = 2.0
        self.control_timeout = 0.20
        self.reset_settle_samples = 3
        self.ros = _BatchRos(
            ready=ready,
            safety_sequence=safety_sequence,
        )
        self.events = events

    def _call_empty(self, operation):
        self.events.append(operation)

    def ground_truth_position(self):
        return 10.0 + self.index, 10.0

    def prepare_stop_step(self):
        self.events.append(("stop_step", self.index))
        return PendingStep(
            requested_action=np.zeros(3),
            limited_action=np.zeros(3),
            applied_action=np.zeros(3),
            command=np.zeros(3),
            command_inhibited=False,
            command_stamp_ns=10,
            sequences=(0, 0, 0),
            contact_sequence=0,
        )

    def wait_for_step_snapshot(self, pending):
        return _snapshot(20)

    def finish_step(self, pending, snapshot, *, contact_collision):
        self.events.append(("passive_finish", self.index))
        return (
            f"passive-observation-{self.index}",
            0.0,
            contact_collision,
            False,
            {"collision": contact_collision},
        )

    def _reset_world_scenario(self, world_index):
        self.events.append(("world", self.index, world_index))

    def prepare_shared_reset(self, *, seed, options):
        self.events.append(("prepare", self.index, seed, options))
        return PendingGazeboReset(
            world_map=SimpleNamespace(
                world_name="room",
                distance_field=lambda x, y: "distance-field",
            ),
            sample=SimpleNamespace(
                start_x=options["start"][0],
                start_y=options["start"][1],
                start_yaw=options["start"][2],
                goal_x=options["goal"][0],
                goal_y=options["goal"][1],
            ),
            reset_stamp_ns=10,
            sequences=(0, 0, 0),
            contact_sequence=0,
        )

    def validate_shared_reset(
        self,
        world_map,
        sample,
        snapshot,
        *,
        distance_field,
    ):
        return SimpleNamespace(
            safe=snapshot.safe,
            diagnostic=lambda: (
                f"minimum_scan={snapshot.minimum_scan:.6f}, "
                "lidar_sector=18, lidar_angle_deg=5.000, "
                "lidar_range=0.120000, lidar_threshold=0.265000"
            ),
        )

    def finish_shared_reset(self, pending, snapshot, *, validation=None):
        assert validation is not None and validation.safe
        self.events.append(("finish", self.index, snapshot.scan_sequence))
        return f"observation-{self.index}", {"robot": self.index}


def _group(events, *, ready=(True, True), safety_sequences=None):
    group = object.__new__(SharedGazeboEnvironments)
    group.count = len(ready)
    if safety_sequences is None:
        safety_sequences = [() for _ in ready]
    group.environments = [
        _BatchEnvironment(
            index,
            events,
            ready=is_ready,
            safety_sequence=safety_sequences[index],
        )
        for index, is_ready in enumerate(ready)
    ]
    group.local_maps = [SimpleNamespace(world_name="room")]
    group.catalog = {"room": ()}
    group._distance_field_caches = [{}]
    group.park_all = lambda: events.append("park_all")
    group.park = lambda index: events.append(("park", index))
    return group


def _episodes(count):
    return [
        SimpleNamespace(
            start_x=float(index),
            start_y=0.0,
            start_yaw=0.0,
            goal_x=float(index),
            goal_y=3.0,
        )
        for index in range(count)
    ]


def test_environment_prepare_reset_keeps_physics_paused_and_sets_every_barrier(
    monkeypatch,
):
    events = []
    world_map = SimpleNamespace()
    sample = _episodes(1)[0]
    ros = SimpleNamespace(
        odom_frame="martha_0/odom",
        publish_stop=lambda: events.append("stop"),
        set_goal=lambda x, y, frame: events.append(("set_goal", x, y, frame)),
        publish_goal=lambda x, y, frame: events.append(
            ("publish_goal", x, y, frame)
        ),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1234)
        ),
        sequence_numbers=lambda: (4, 5, 6),
        clear_contacts=lambda: 7,
    )
    environment = object.__new__(MarthaEnv)
    environment._closed = False
    environment.backend = "gazebo"
    environment.preloaded_worlds = True
    environment.predefined_maps = [world_map]
    environment._goal_marker_name = "old-goal"
    environment.ros = ros
    environment._invalidate_episode_state = lambda: events.append("invalidate")
    environment._episode_for_reset = lambda selected, options: sample
    environment._delete_entity = lambda name, ignore_failure: events.append(
        ("delete", name, ignore_failure)
    )
    environment._spawn_goal_marker = lambda x, y: events.append(
        ("spawn_goal", x, y)
    )
    environment._set_robot_state = lambda x, y, yaw: events.append(
        ("teleport", x, y, yaw)
    )
    environment._reset_filter_pose = lambda: events.append("reset_ekf")
    monkeypatch.setattr(
        martha_env._GymEnvBase,
        "reset",
        lambda self, *, seed=None: events.append(("seed", seed)),
        raising=False,
    )

    pending = environment.prepare_shared_reset(
        seed=42,
        options={"world_index": 0, "start": (0, 0, 0), "goal": (0, 3)},
    )

    assert pending.world_map is world_map
    assert pending.sample is sample
    assert pending.reset_stamp_ns == 1234
    assert pending.sequences == (4, 5, 6)
    assert pending.contact_sequence == 7
    assert environment._active_world_index == 0
    assert events == [
        "invalidate",
        ("seed", 42),
        "stop",
        ("delete", "old-goal", True),
        ("spawn_goal", 0.0, 3.0),
        ("teleport", 0.0, 0.0, 0.0),
        ("set_goal", 0.0, 3.0, "martha_0/odom"),
        ("publish_goal", 0.0, 3.0, "martha_0/odom"),
        "reset_ekf",
    ]
    assert "pause" not in events and "unpause" not in events


def test_reset_round_prepares_all_then_uses_one_physics_window(monkeypatch):
    events = []
    group = _group(events)
    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.sample_round_episodes",
        lambda *args, **kwargs: _episodes(2),
    )

    results = group.reset_round(world_index=0, seeds=[11, 12])

    assert results == {
        0: ("observation-0", {"robot": 0}),
        1: ("observation-1", {"robot": 1}),
    }
    assert events == [
        "park_all",
        ("world", 0, 0),
        ("prepare", 0, 11, {
            "world_index": 0,
            "start": (0.0, 0.0, 0.0),
            "goal": (0.0, 3.0),
        }),
        ("prepare", 1, 12, {
            "world_index": 0,
            "start": (1.0, 0.0, 0.0),
            "goal": (1.0, 3.0),
        }),
        "unpause",
        "pause",
        ("finish", 0, 3),
        ("finish", 1, 3),
    ]
    assert [len(environment.ros.wait_calls) for environment in group.environments] == [
        3,
        3,
    ]
    assert all(
        timeout == 0.0
        for environment in group.environments
        for _, timeout, _ in environment.ros.wait_calls
    )


def test_recycle_slot_accounts_one_passive_transition_for_active_robot(
    monkeypatch,
):
    events = []
    group = _group(events)
    monkeypatch.setattr(
        group,
        "_sample_recycled_episodes",
        lambda **kwargs: tuple(_episodes(1)),
    )

    recycled, passive = group.reset_slots(
        world_index=0,
        assignments={0: 101},
        active_indices=[1],
    )

    assert recycled == {0: ("observation-0", {"robot": 0})}
    assert passive[1][0] == "passive-observation-1"
    assert passive[1][2] is False
    assert events == [
        ("park", 0),
        ("prepare", 0, 101, {
            "world_index": 0,
            "start": (0.0, 0.0, 0.0),
            "goal": (0.0, 3.0),
        }),
        ("stop_step", 1),
        "unpause",
        "pause",
        ("passive_finish", 1),
        ("finish", 0, 3),
    ]


def test_recycle_placement_unavailable_has_a_distinct_retryable_error(
    monkeypatch,
):
    events = []
    group = _group(events)
    attempts = []

    def always_occupied(**kwargs):
        attempts.append(kwargs["resample_index"])
        return tuple(_episodes(1))

    monkeypatch.setattr(group, "_sample_reset_episodes", always_occupied)
    group.environments[1].ground_truth_position = lambda: (0.0, 0.0)

    with pytest.raises(
        RecyclePlacementUnavailable,
        match=rf"after {RECYCLE_PLACEMENT_ATTEMPTS} deterministic samples",
    ):
        group._sample_recycled_episodes(
            world_index=0,
            seeds=[101],
            active_indices=[1],
        )

    assert attempts == list(range(RECYCLE_PLACEMENT_ATTEMPTS))


def test_reset_round_timeout_retries_without_partial_commit(monkeypatch):
    events = []
    group = _group(events, ready=(True, False))
    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.sample_round_episodes",
        lambda *args, **kwargs: _episodes(2),
    )

    with pytest.raises(RuntimeError) as error:
        group.reset_round(world_index=0, seeds=[21, 22])

    assert "martha_1=[scan(sequence=0, need>0), contact(" in str(error.value)
    assert "attempt=5/5" in str(error.value)
    assert events.count("unpause") == 5
    assert events.count("pause") == 5
    assert not any(
        isinstance(event, tuple) and event[0] == "finish"
        for event in events
    )


def test_transient_unsafe_scan_requires_three_following_safe_samples(monkeypatch):
    events = []
    group = _group(
        events,
        ready=(True,),
        safety_sequences=((False, True, True, True),),
    )
    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.sample_round_episodes",
        lambda *args, **kwargs: _episodes(1),
    )

    results = group.reset_round(world_index=0, seeds=[31])

    assert results == {0: ("observation-0", {"robot": 0})}
    assert len(group.environments[0].ros.wait_calls) == 4
    assert events.count("park_all") == 1
    assert ("finish", 0, 4) in events


def test_persistent_unsafe_scan_uses_three_retries_then_two_resamples(monkeypatch):
    events = []
    group = _group(
        events,
        ready=(True,),
        safety_sequences=((False,) * 60,),
    )
    sample_calls = []

    def sample_episodes(
        *,
        world_index,
        seeds,
        resample_index,
        max_goal_distance,
    ):
        sample_calls.append(resample_index)
        assert max_goal_distance is None
        return tuple(_episodes(1))

    group._sample_reset_episodes = sample_episodes

    with pytest.raises(RuntimeError) as error:
        group.reset_round(world_index=0, seeds=[41])

    message = str(error.value)
    assert sample_calls == [0, 1, 2]
    assert "attempt=5/5" in message
    assert "map=room" in message
    assert "minimum_scan=0.120000" in message
    assert "lidar_sector=18" in message
    assert len(group.environments[0].ros.wait_calls) == 60
    assert events.count("park_all") == 5
    assert not any(
        isinstance(event, tuple) and event[0] == "finish"
        for event in events
    )


def test_alternate_round_sampling_is_deterministic_and_distinct(monkeypatch):
    group = _group([])

    def sampled_value(*args, rng, **kwargs):
        return (int(rng.integers(0, 2**31)),)

    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.sample_round_episodes",
        sampled_value,
    )

    original = group._sample_reset_episodes(
        world_index=0,
        seeds=[51, 52],
        resample_index=0,
    )
    alternate = group._sample_reset_episodes(
        world_index=0,
        seeds=[51, 52],
        resample_index=1,
    )

    assert original == group._sample_reset_episodes(
        world_index=0,
        seeds=[51, 52],
        resample_index=0,
    )
    assert alternate == group._sample_reset_episodes(
        world_index=0,
        seeds=[51, 52],
        resample_index=1,
    )
    assert original != alternate
