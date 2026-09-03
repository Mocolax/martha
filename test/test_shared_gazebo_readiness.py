"""Focused tests for shared-Gazebo message readiness and diagnostics."""

from pathlib import Path
import threading

import pytest

from martha.PPO.martha_env import RosObservationNode, RosStreamActivity
from martha.PPO.shared_gazebo import SharedGazeboEnvironments


def _activity(
    scan=0,
    odometry=0,
    model_state=0,
    contact=0,
    scan_stamp=-1,
    odometry_stamp=-1,
):
    return RosStreamActivity(
        scan_sequence=scan,
        odometry_sequence=odometry,
        ground_truth_sequence=model_state,
        contact_sequence=contact,
        scan_stamp_ns=scan_stamp,
        odometry_stamp_ns=odometry_stamp,
    )


def test_stream_activity_is_read_only_and_reports_each_missing_stream():
    node = object.__new__(RosObservationNode)
    node._condition = threading.Condition()
    node._scan_sequence = 12
    node._odometry_sequence = 0
    node._ground_truth_sequence = 7
    node._contact_sequence = 0
    node._scan_stamp_ns = 1_500
    node._odometry_stamp_ns = -1
    queued_contact = (1, ("robot::shell", "wall::collision"))
    node._contact_events = [queued_contact]

    baseline = _activity(scan=11, model_state=6)
    missing = node.stream_activity().missing_after(
        baseline,
        require_contact=True,
    )

    assert missing == (
        "odometry(sequence=0, need>0, stamp_ns=-1)",
        "contact(sequence=0, need>0)",
    )
    assert node._contact_events == [queued_contact]


def test_snapshot_diagnostic_distinguishes_sequences_from_ros_stamps():
    activity = _activity(
        scan=5,
        odometry=4,
        model_state=9,
        scan_stamp=90,
        odometry_stamp=80,
    )

    assert activity.missing_snapshot_data(
        (4, 4, 8),
        use_ground_truth=True,
        after_stamp_ns=100,
    ) == (
        "odometry(sequence=4, need>4)",
        "scan_stamp(stamp_ns=90, need>100)",
        "odometry_stamp(stamp_ns=80, need>100)",
    )


class _LaunchProcess:
    def poll(self):
        return None


class _StartupRos:
    def __init__(self, activity, topics=()):
        self.activity = activity
        self.topics = set(topics)

    def stream_activity(self):
        return self.activity

    def gazebo_model_names(self):
        return {"martha_0"}

    def get_topic_names_and_types(self):
        return [(topic, ()) for topic in self.topics]


class _StartupEnvironment:
    robot_name = "martha_0"

    def __init__(self, ros):
        self.ros = ros
        self.calls = []

    def _call_empty(self, operation):
        self.calls.append(operation)


def test_shared_startup_requires_messages_even_when_topics_exist():
    topics = {
        "/martha_0/scan",
        "/martha_0/contacts",
        "/martha_0/odometry/filtered",
    }
    ros = _StartupRos(_activity(), topics)
    environment = _StartupEnvironment(ros)
    group = object.__new__(SharedGazeboEnvironments)
    group.count = 1
    group.environments = [environment]
    group._launch_process = _LaunchProcess()

    with pytest.raises(TimeoutError) as error:
        group._wait_until_ready(0.01, Path("gazebo.log"))

    message = str(error.value)
    assert "martha_0=[scan(" in message
    assert "odometry(" in message
    assert "model_state(" in message
    assert "contact(" in message
    assert environment.calls == ["unpause"]


def test_shared_startup_pauses_after_every_stream_advances(monkeypatch):
    topics = {
        "/martha_0/scan",
        "/martha_0/contacts",
        "/martha_0/odometry/filtered",
    }
    ros = _StartupRos(_activity(), topics)
    environment = _StartupEnvironment(ros)
    group = object.__new__(SharedGazeboEnvironments)
    group.count = 1
    group.environments = [environment]
    group._launch_process = _LaunchProcess()

    def advance(_duration):
        ros.activity = _activity(
            scan=1,
            odometry=1,
            model_state=1,
            contact=1,
            scan_stamp=10,
            odometry_stamp=10,
        )

    monkeypatch.setattr("martha.PPO.shared_gazebo.time.sleep", advance)

    group._wait_until_ready(1.0, Path("gazebo.log"))

    assert environment.calls == ["unpause", "pause"]


def test_constructor_cleans_launch_when_startup_is_interrupted(
    monkeypatch,
    tmp_path,
):
    world_path = tmp_path / "world.world"
    combined_world = tmp_path / "combined.world"
    combined_world.write_text("world", encoding="utf-8")
    launch_process = object()
    stopped = []
    environments = []

    class _Environment:
        def __init__(self, **_kwargs):
            self.closed = False
            environments.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.discover_worlds",
        lambda _directory: (world_path,),
    )
    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.WorldMap.from_sdf",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.load_training_points",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.validate_training_points",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.create_combined_training_world",
        lambda _paths: combined_world,
    )
    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.subprocess.Popen",
        lambda *_args, **_kwargs: launch_process,
    )
    monkeypatch.setattr(
        "martha.PPO.shared_gazebo.MarthaEnv",
        _Environment,
    )
    monkeypatch.setattr(
        "martha.PPO.shared_gazebo._stop_launch_process",
        stopped.append,
    )
    monkeypatch.setattr(
        SharedGazeboEnvironments,
        "_wait_until_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        SharedGazeboEnvironments(
            count=1,
            sim_speed_factor=1.0,
            physics_step_size=0.002,
            lidar_samples=180,
            training_kinematic=True,
            show_gui=False,
            startup_timeout=1.0,
            run_directory=tmp_path / "run",
            worlds_directory=tmp_path,
            points_path=tmp_path / "points.yaml",
            environment_kwargs={"min_goal_distance": 1.0},
        )

    assert stopped == [launch_process]
    assert len(environments) == 1 and environments[0].closed
    assert not combined_world.exists()
